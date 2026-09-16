"""GUI app for drowning detection: live camera preview or batch video testing.

Supports one USB webcam (OpenCV) or one Raspberry Pi CSI camera (Picamera2) at
a time -- only one capture device may run at once, the Pi 4 cannot drive more.

main.py is kept as-is (headless benchmarking script for the OV5647 camera);
this app is a separate GUI front-end sharing the same NCNN inference code.
"""
from __future__ import annotations

import argparse
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import ncnn
import numpy as np
from PIL import Image, ImageTk

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None


REPO_DIR = Path(__file__).resolve().parent
MODEL_ROOT = REPO_DIR / "model"
NCNN_MEAN = [0.0, 0.0, 0.0]
NCNN_NORM = [1 / 255.0, 1 / 255.0, 1 / 255.0]
BOX_COLORS = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 128, 0),
    (255, 0, 255),
    (0, 255, 255),
]


# --------------------------------------------------------------------------
# NCNN inference (same approach as main.py)
# --------------------------------------------------------------------------

def load_class_names(model_dir: Path) -> dict[int, str]:
    defaults = {0: "drowning", 1: "out of water", 2: "swimming"}
    path = model_dir / "metadata.yaml"
    if not path.is_file():
        return defaults
    names: dict[int, str] = {}
    in_names = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("names:"):
            in_names = True
            continue
        if not in_names:
            continue
        if line and not line[0].isspace():
            break
        stripped = line.strip()
        if not stripped or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        names[int(key)] = value.strip()
    return names or defaults


def load_export_imgsz(model_dir: Path, fallback: int) -> int:
    path = model_dir / "metadata.yaml"
    if not path.is_file():
        return fallback
    in_imgsz = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("imgsz:"):
            in_imgsz = True
            continue
        if in_imgsz:
            stripped = line.strip().lstrip("- ")
            if stripped.isdigit():
                return int(stripped)
            break
    return fallback


def letterbox(image: np.ndarray, imgsz: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    height, width = image.shape[:2]
    scale = min(imgsz / height, imgsz / width)
    new_width, new_height = int(round(width * scale)), int(round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    pad_w = imgsz - new_width
    pad_h = imgsz - new_height
    left = pad_w // 2
    top = pad_h // 2
    padded = cv2.copyMakeBorder(
        resized, top, pad_h - top, left, pad_w - left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return padded, scale, (left, top)


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = np.empty_like(boxes)
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        index = int(order[0])
        keep.append(index)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[index], x1[rest])
        yy1 = np.maximum(y1[index], y1[rest])
        xx2 = np.minimum(x2[index], x2[rest])
        yy2 = np.minimum(y2[index], y2[rest])
        intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = intersection / (areas[index] + areas[rest] - intersection + 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


def parse_ncnn_output(raw: np.ndarray, num_classes: int) -> np.ndarray:
    array = np.array(raw)
    if array.ndim == 3:
        array = array.squeeze(0)
    channels = 4 + num_classes
    if array.ndim != 2:
        raise RuntimeError(f"Unexpected NCNN output shape: {array.shape}")
    if array.shape[0] == channels:
        return array
    if array.shape[1] == channels:
        return array.T
    raise RuntimeError(f"Unexpected NCNN output shape {array.shape}; expected {channels} channels")


class NcnnDetector:
    """YOLO NCNN runtime without PyTorch/Ultralytics on the hot path."""

    def __init__(self, model_dir: Path, imgsz: int, conf: float, iou: float, threads: int) -> None:
        param_path = model_dir / "model.ncnn.param"
        bin_path = model_dir / "model.ncnn.bin"
        if not param_path.is_file() or not bin_path.is_file():
            raise FileNotFoundError(f"NCNN model files not found in {model_dir}")

        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.names = load_class_names(model_dir)
        self.net = ncnn.Net()
        self.net.opt.num_threads = max(1, threads)
        self.net.opt.use_vulkan_compute = False
        self.net.opt.lightmode = True
        self.net.opt.use_winograd_convolution = True
        self.net.opt.use_sgemm_convolution = True
        self.net.opt.use_packing_layout = True
        self.net.load_param(str(param_path))
        self.net.load_model(str(bin_path))

    def predict(self, image: np.ndarray) -> list[tuple[np.ndarray, float, int]]:
        padded, scale, (pad_left, pad_top) = letterbox(image, self.imgsz)
        mat_in = ncnn.Mat.from_pixels(padded, ncnn.Mat.PixelType.PIXEL_BGR2RGB, self.imgsz, self.imgsz)
        mat_in.substract_mean_normalize(NCNN_MEAN, NCNN_NORM)

        extractor = self.net.create_extractor()
        extractor.input("in0", mat_in)
        _, mat_out = extractor.extract("out0")
        prediction = parse_ncnn_output(np.array(mat_out), len(self.names))

        boxes_xywh = prediction[:4].T
        class_scores = prediction[4:].T
        class_ids = class_scores.argmax(axis=1)
        scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
        mask = scores >= self.conf
        if not np.any(mask):
            return []

        boxes_xyxy = xywh_to_xyxy(boxes_xywh[mask])
        scores = scores[mask]
        class_ids = class_ids[mask]
        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_left) / scale
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_top) / scale
        height, width = image.shape[:2]
        boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clip(0, width)
        boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clip(0, height)

        detections: list[tuple[np.ndarray, float, int]] = []
        for class_id in np.unique(class_ids):
            class_mask = class_ids == class_id
            keep = nms(boxes_xyxy[class_mask], scores[class_mask], self.iou)
            for box, score in zip(boxes_xyxy[class_mask][keep], scores[class_mask][keep]):
                detections.append((box, float(score), int(class_id)))
        return detections


def draw_detections(frame: np.ndarray, detections: list[tuple[np.ndarray, float, int]], names: dict[int, str]) -> np.ndarray:
    for box, confidence, class_id in detections:
        label_name = str(names[class_id])
        color = BOX_COLORS[class_id % len(BOX_COLORS)]
        points = box.astype(int)
        cv2.rectangle(frame, tuple(points[:2]), tuple(points[2:]), color, 2)
        cv2.putText(
            frame, f"{label_name} {confidence:.2f}", (int(points[0]), max(20, int(points[1]) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )
    return frame


# --------------------------------------------------------------------------
# Camera discovery + uniform capture wrappers
# --------------------------------------------------------------------------

def list_usb_cameras(max_index: int = 6) -> list[int]:
    found: list[int] = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened() and sys.platform == "win32":
            cap.release()
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            found.append(index)
        cap.release()
    return found


def list_pi_cameras() -> list[int]:
    if Picamera2 is None:
        return []
    try:
        infos = Picamera2.global_camera_info()
    except Exception:
        return []
    return list(range(len(infos)))


class UsbCamera:
    def __init__(self, index: int, width: int, height: int) -> None:
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open USB camera {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Unable to read a frame from the USB camera")
        return frame

    def release(self) -> None:
        self.cap.release()


class PiCamera:
    def __init__(self, camera_num: int, width: int, height: int) -> None:
        if Picamera2 is None:
            raise RuntimeError("picamera2 is not installed")
        self.camera = Picamera2(camera_num=camera_num)
        self.camera.configure(
            self.camera.create_preview_configuration(main={"size": (width, height), "format": "RGB888"}, buffer_count=4)
        )
        self.camera.start()
        time.sleep(0.4)

    def read(self) -> np.ndarray:
        request = self.camera.capture_request()
        try:
            return request.make_array("main").copy()
        finally:
            request.release()

    def release(self) -> None:
        self.camera.stop()


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.title("Drowning Detection")
        self.geometry("1000x700")

        self.args = args
        self.detector: NcnnDetector | None = None
        self.camera = None
        self.camera_running = False
        self.video_queue: queue.Queue = queue.Queue()

        self._build_model_bar()
        self._build_tabs()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- shared controls -------------------------------------------------

    def _build_model_bar(self) -> None:
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=8, pady=6)

        ttk.Label(bar, text="Model:").pack(side="left")
        models = sorted(
            p.name for p in MODEL_ROOT.iterdir() if p.is_dir() and (p / "model.ncnn.param").is_file()
        ) if MODEL_ROOT.is_dir() else []
        default_model = self.args.model if self.args.model in models else (models[0] if models else "")
        self.model_var = tk.StringVar(value=default_model)
        ttk.Combobox(bar, textvariable=self.model_var, values=models, width=28, state="readonly").pack(side="left", padx=4)

        ttk.Label(bar, text="Conf:").pack(side="left", padx=(12, 0))
        self.conf_var = tk.StringVar(value=str(self.args.conf))
        ttk.Entry(bar, textvariable=self.conf_var, width=6).pack(side="left", padx=4)

        ttk.Label(bar, text="IoU:").pack(side="left", padx=(12, 0))
        self.iou_var = tk.StringVar(value=str(self.args.iou))
        ttk.Entry(bar, textvariable=self.iou_var, width=6).pack(side="left", padx=4)

        ttk.Label(bar, text="Threads:").pack(side="left", padx=(12, 0))
        self.threads_var = tk.StringVar(value=str(self.args.threads))
        ttk.Entry(bar, textvariable=self.threads_var, width=4).pack(side="left", padx=4)

    def _get_detector(self) -> NcnnDetector:
        model_name = self.model_var.get()
        if not model_name:
            raise RuntimeError("No exported NCNN model found under model/")
        model_dir = MODEL_ROOT / model_name
        conf = float(self.conf_var.get())
        iou = float(self.iou_var.get())
        threads = int(self.threads_var.get())
        imgsz = load_export_imgsz(model_dir, 320)
        return NcnnDetector(model_dir, imgsz=imgsz, conf=conf, iou=iou, threads=threads)

    def _build_tabs(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=8, pady=8)

        self.camera_tab = ttk.Frame(notebook)
        self.video_tab = ttk.Frame(notebook)
        notebook.add(self.camera_tab, text="Camera")
        notebook.add(self.video_tab, text="Test video")

        self._build_camera_tab()
        self._build_video_tab()

    # -- camera tab --------------------------------------------------------

    def _build_camera_tab(self) -> None:
        controls = ttk.Frame(self.camera_tab)
        controls.pack(fill="x", pady=4)

        ttk.Button(controls, text="Quet camera", command=self._refresh_cameras).pack(side="left")
        self.start_btn = ttk.Button(controls, text="Bat dau", command=self._start_camera)
        self.start_btn.pack(side="left", padx=6)
        self.stop_btn = ttk.Button(controls, text="Dung", command=self._stop_camera, state="disabled")
        self.stop_btn.pack(side="left")

        self.camera_list_frame = ttk.LabelFrame(self.camera_tab, text="Camera phat hien (chi chon duoc 1)")
        self.camera_list_frame.pack(fill="x", pady=6)
        self.camera_var = tk.StringVar(value="")
        self.camera_radios: list[ttk.Radiobutton] = []

        self.camera_view = ttk.Label(self.camera_tab)
        self.camera_view.pack(fill="both", expand=True, pady=4)

        self.camera_status = ttk.Label(self.camera_tab, text="Chua chon camera.")
        self.camera_status.pack(fill="x")

        self._refresh_cameras()

    def _refresh_cameras(self) -> None:
        if self.camera_running:
            messagebox.showinfo("Drowning Detection", "Dung camera dang chay truoc khi quet lai.")
            return
        for radio in self.camera_radios:
            radio.destroy()
        self.camera_radios.clear()

        entries: list[tuple[str, str]] = []
        for cam_num in list_pi_cameras():
            entries.append((f"pi:{cam_num}", f"Pi CSI camera #{cam_num}"))
        for index in list_usb_cameras():
            entries.append((f"usb:{index}", f"USB webcam #{index}"))

        if not entries:
            ttk.Label(self.camera_list_frame, text="Khong phat hien camera nao.").pack(anchor="w")
            return

        for key, label in entries:
            radio = ttk.Radiobutton(self.camera_list_frame, text=label, value=key, variable=self.camera_var)
            radio.pack(anchor="w")
            self.camera_radios.append(radio)
        if not self.camera_var.get():
            self.camera_var.set(entries[0][0])

    def _start_camera(self) -> None:
        if self.camera_running:
            return
        selection = self.camera_var.get()
        if not selection:
            messagebox.showwarning("Drowning Detection", "Chua co camera nao duoc chon.")
            return
        try:
            self.detector = self._get_detector()
        except Exception as error:
            messagebox.showerror("Drowning Detection", f"Khong tai duoc model: {error}")
            return

        kind, raw_id = selection.split(":", 1)
        try:
            if kind == "pi":
                self.camera = PiCamera(int(raw_id), 640, 480)
            else:
                self.camera = UsbCamera(int(raw_id), 640, 480)
        except Exception as error:
            messagebox.showerror("Drowning Detection", f"Khong mo duoc camera: {error}")
            return

        self.camera_running = True
        self.start_btn.state(["disabled"])
        self.stop_btn.state(["!disabled"])
        for radio in self.camera_radios:
            radio.state(["disabled"])
        self.camera_status.configure(text=f"Dang chay: {selection}")
        self._camera_loop()

    def _camera_loop(self) -> None:
        if not self.camera_running or self.camera is None:
            return
        try:
            started = time.monotonic()
            frame = self.camera.read()
            detections = self.detector.predict(frame) if self.detector else []
            frame = draw_detections(frame, detections, self.detector.names if self.detector else {})
            infer_ms = (time.monotonic() - started) * 1000.0
            cv2.putText(frame, f"{infer_ms:.0f}ms  {len(detections)} det", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            self._show_frame(self.camera_view, frame)
        except Exception as error:
            self.camera_status.configure(text=f"Loi: {error}")
            self._stop_camera()
            return
        self.after(10, self._camera_loop)

    def _stop_camera(self) -> None:
        self.camera_running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.start_btn.state(["!disabled"])
        self.stop_btn.state(["disabled"])
        for radio in self.camera_radios:
            radio.state(["!disabled"])
        self.camera_status.configure(text="Da dung.")

    @staticmethod
    def _show_frame(label: ttk.Label, frame_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = ImageTk.PhotoImage(Image.fromarray(rgb))
        label.configure(image=image)
        label.image = image  # keep a reference, tkinter drops it otherwise

    # -- video test tab ----------------------------------------------------

    def _build_video_tab(self) -> None:
        controls = ttk.Frame(self.video_tab)
        controls.pack(fill="x", pady=4)

        ttk.Button(controls, text="Mo video...", command=self._open_video).pack(side="left")
        self.video_path_var = tk.StringVar(value="Chua chon video.")
        ttk.Label(controls, textvariable=self.video_path_var).pack(side="left", padx=8)

        self.run_video_btn = ttk.Button(controls, text="Chay xu ly", command=self._run_video, state="disabled")
        self.run_video_btn.pack(side="left", padx=8)

        self.video_progress = ttk.Progressbar(self.video_tab, mode="determinate")
        self.video_progress.pack(fill="x", pady=6)

        self.video_preview = ttk.Label(self.video_tab)
        self.video_preview.pack(fill="both", expand=True, pady=4)

        result_frame = ttk.Frame(self.video_tab)
        result_frame.pack(fill="x", pady=4)
        ttk.Label(result_frame, text="Ket qua:").pack(side="left")
        self.result_path_var = tk.StringVar(value="(chua co)")
        ttk.Entry(result_frame, textvariable=self.result_path_var, state="readonly", width=70).pack(
            side="left", padx=6, fill="x", expand=True
        )
        ttk.Button(result_frame, text="Mo thu muc", command=self._open_result_folder).pack(side="left")

        self._video_path: Path | None = None
        self._result_path: Path | None = None

    def _open_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Chon video",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("Tat ca", "*.*")],
        )
        if not path:
            return
        self._video_path = Path(path)
        self.video_path_var.set(str(self._video_path))
        self.run_video_btn.state(["!disabled"])

    def _run_video(self) -> None:
        if self._video_path is None:
            return
        try:
            detector = self._get_detector()
        except Exception as error:
            messagebox.showerror("Drowning Detection", f"Khong tai duoc model: {error}")
            return

        self.run_video_btn.state(["disabled"])
        self.video_progress["value"] = 0
        self.result_path_var.set("(dang xu ly...)")

        thread = threading.Thread(target=self._video_worker, args=(detector, self._video_path), daemon=True)
        thread.start()
        self.after(50, self._poll_video_queue)

    def _video_worker(self, detector: NcnnDetector, video_path: Path) -> None:
        output_dir = REPO_DIR / "runs" / "gui"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{video_path.stem}.avi"

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            self.video_queue.put(("error", f"Khong mo duoc video: {video_path}"))
            return

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))

        processed = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                detections = detector.predict(frame)
                frame = draw_detections(frame, detections, detector.names)
                writer.write(frame)
                processed += 1
                if processed % 5 == 0 or processed == total_frames:
                    progress = processed / total_frames if total_frames else 0.0
                    self.video_queue.put(("progress", progress, frame.copy()))
        finally:
            capture.release()
            writer.release()

        self.video_queue.put(("done", output_path))

    def _poll_video_queue(self) -> None:
        try:
            while True:
                message = self.video_queue.get_nowait()
                kind = message[0]
                if kind == "progress":
                    _, progress, frame = message
                    self.video_progress["value"] = progress * 100
                    self._show_frame(self.video_preview, frame)
                elif kind == "done":
                    self._result_path = message[1]
                    self.video_progress["value"] = 100
                    self.result_path_var.set(str(self._result_path))
                    self.run_video_btn.state(["!disabled"])
                    return
                elif kind == "error":
                    messagebox.showerror("Drowning Detection", message[1])
                    self.result_path_var.set("(loi)")
                    self.run_video_btn.state(["!disabled"])
                    return
        except queue.Empty:
            pass
        self.after(50, self._poll_video_queue)

    def _open_result_folder(self) -> None:
        if self._result_path is None:
            return
        folder = self._result_path.parent
        try:
            if sys.platform == "win32":
                subprocess.run(["explorer", str(folder)], check=False)
            elif sys.platform == "darwin":
                subprocess.run(["open", str(folder)], check=False)
            else:
                subprocess.run(["xdg-open", str(folder)], check=False)
        except Exception:
            pass

    def _on_close(self) -> None:
        self._stop_camera()
        self.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo26n_drowning-2", help="Model folder name under model/")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    app = App(args)
    app.mainloop()


if __name__ == "__main__":
    main()
