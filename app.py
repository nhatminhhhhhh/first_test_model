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
import tkinter.font as tkfont
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

# UI palette
PAD = 10
COLOR_BG = "#1e2530"
COLOR_PANEL = "#262e3d"
COLOR_ACCENT = "#3aa0ff"
COLOR_OK = "#33cc77"
COLOR_WARN = "#ff5555"
COLOR_TEXT = "#e8ecf1"
COLOR_MUTED = "#8b95a5"


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


def draw_hud(frame: np.ndarray, fps: float, infer_ms: float, count: int) -> np.ndarray:
    """Overlay FPS / inference time / detection count in the top-left corner."""
    text = f"FPS {fps:4.1f}   infer {infer_ms:5.0f} ms   det {count}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(frame, (6, 6), (6 + tw + 16, 6 + th + 16), (20, 20, 20), -1)
    cv2.putText(frame, text, (14, 14 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 220, 60), 2)
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
        self.geometry("1150x760")
        self.minsize(900, 600)
        self.configure(bg=COLOR_BG)

        self.args = args
        self.detector: NcnnDetector | None = None
        self.camera = None
        self.camera_running = False
        self.video_queue: queue.Queue = queue.Queue()
        self._video_path: Path | None = None
        self._result_path: Path | None = None
        self._prev_frame_time = 0.0

        self._setup_style()
        self._build_header()
        self._build_model_bar()
        self._build_tabs()
        self._build_status_bar()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- look & feel -------------------------------------------------------

    def _setup_style(self) -> None:
        for name in ("TkDefaultFont", "TkTextFont", "TkHeadingFont", "TkMenuFont"):
            try:
                tkfont.nametofont(name).configure(size=10)
            except tk.TclError:
                pass

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", background=COLOR_PANEL, foreground=COLOR_TEXT, font=("TkDefaultFont", 10))
        style.configure("TFrame", background=COLOR_PANEL)
        style.configure("Header.TFrame", background=COLOR_BG)
        style.configure("TLabelframe", background=COLOR_PANEL, foreground=COLOR_TEXT, bordercolor="#3a4457")
        style.configure("TLabelframe.Label", background=COLOR_PANEL, foreground=COLOR_MUTED, font=("TkDefaultFont", 9, "bold"))
        style.configure("TLabel", background=COLOR_PANEL, foreground=COLOR_TEXT)
        style.configure("Header.TLabel", background=COLOR_BG, foreground=COLOR_TEXT, font=("TkDefaultFont", 16, "bold"))
        style.configure("Sub.TLabel", background=COLOR_BG, foreground=COLOR_MUTED, font=("TkDefaultFont", 9))
        style.configure("Status.TLabel", background="#141a24", foreground=COLOR_MUTED, font=("TkDefaultFont", 9))
        style.configure("Muted.TLabel", foreground=COLOR_MUTED)
        style.configure("TNotebook", background=COLOR_BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(16, 8), font=("TkDefaultFont", 10, "bold"))
        style.map("TNotebook.Tab", background=[("selected", COLOR_PANEL)], foreground=[("selected", COLOR_TEXT)])
        style.configure("TRadiobutton", background=COLOR_PANEL, foreground=COLOR_TEXT)
        style.configure("TEntry", fieldbackground="#141a24", foreground=COLOR_TEXT, insertcolor=COLOR_TEXT)
        style.configure("TCombobox", fieldbackground="#141a24", foreground=COLOR_TEXT)

        style.configure("TButton", padding=(10, 6), background="#3a4457", foreground=COLOR_TEXT, borderwidth=0)
        style.map("TButton", background=[("active", "#465370"), ("disabled", "#2c3444")])
        style.configure("Start.TButton", background=COLOR_OK, foreground="#0c1710")
        style.map("Start.TButton", background=[("active", "#3fe08a"), ("disabled", "#2c3444")])
        style.configure("Stop.TButton", background=COLOR_WARN, foreground="#1a0505")
        style.map("Stop.TButton", background=[("active", "#ff7777"), ("disabled", "#2c3444")])
        style.configure("Accent.TButton", background=COLOR_ACCENT, foreground="#04121f")
        style.map("Accent.TButton", background=[("active", "#5fb4ff"), ("disabled", "#2c3444")])
        style.configure("Horizontal.TProgressbar", background=COLOR_ACCENT, troughcolor="#141a24", bordercolor="#141a24")

    def _build_header(self) -> None:
        header = ttk.Frame(self, style="Header.TFrame")
        header.pack(fill="x", padx=0, pady=0)
        inner = ttk.Frame(header, style="Header.TFrame")
        inner.pack(fill="x", padx=PAD * 2, pady=(PAD, 4))
        ttk.Label(inner, text="Drowning Detection", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            inner, text="Nhan dien duoi nuoc / boi / ngoai nuoc bang model NCNN", style="Sub.TLabel"
        ).pack(anchor="w")

    # -- shared controls -----------------------------------------------------

    def _build_model_bar(self) -> None:
        bar = ttk.LabelFrame(self, text="Cau hinh model")
        bar.pack(fill="x", padx=PAD * 2, pady=(6, 4))
        for col in range(8):
            bar.columnconfigure(col, weight=0)

        models = sorted(
            p.name for p in MODEL_ROOT.iterdir() if p.is_dir() and (p / "model.ncnn.param").is_file()
        ) if MODEL_ROOT.is_dir() else []
        default_model = self.args.model if self.args.model in models else (models[0] if models else "")

        ttk.Label(bar, text="Model").grid(row=0, column=0, padx=(10, 4), pady=8, sticky="w")
        self.model_var = tk.StringVar(value=default_model)
        ttk.Combobox(bar, textvariable=self.model_var, values=models, width=24, state="readonly").grid(
            row=0, column=1, padx=4, pady=8
        )

        ttk.Label(bar, text="Conf").grid(row=0, column=2, padx=(16, 4), pady=8, sticky="w")
        self.conf_var = tk.StringVar(value=str(self.args.conf))
        ttk.Entry(bar, textvariable=self.conf_var, width=6).grid(row=0, column=3, padx=4, pady=8)

        ttk.Label(bar, text="IoU").grid(row=0, column=4, padx=(16, 4), pady=8, sticky="w")
        self.iou_var = tk.StringVar(value=str(self.args.iou))
        ttk.Entry(bar, textvariable=self.iou_var, width=6).grid(row=0, column=5, padx=4, pady=8)

        ttk.Label(bar, text="Threads").grid(row=0, column=6, padx=(16, 4), pady=8, sticky="w")
        self.threads_var = tk.StringVar(value=str(self.args.threads))
        ttk.Entry(bar, textvariable=self.threads_var, width=4).grid(row=0, column=7, padx=4, pady=8)

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
        notebook.pack(fill="both", expand=True, padx=PAD * 2, pady=(4, 4))

        self.camera_tab = ttk.Frame(notebook)
        self.video_tab = ttk.Frame(notebook)
        notebook.add(self.camera_tab, text="  Camera  ")
        notebook.add(self.video_tab, text="  Test video  ")

        self._build_camera_tab()
        self._build_video_tab()

    def _build_status_bar(self) -> None:
        bar = ttk.Frame(self, style="TFrame")
        bar.configure(style="TFrame")
        status = tk.Frame(self, bg="#141a24", height=28)
        status.pack(fill="x", side="bottom")
        self.status_var = tk.StringVar(value="San sang.")
        tk.Label(
            status, textvariable=self.status_var, bg="#141a24", fg=COLOR_MUTED,
            font=("TkDefaultFont", 9), anchor="w",
        ).pack(fill="x", padx=PAD * 2, pady=4)

    # -- camera tab ----------------------------------------------------------

    def _build_camera_tab(self) -> None:
        self.camera_tab.columnconfigure(0, weight=0, minsize=260)
        self.camera_tab.columnconfigure(1, weight=1)
        self.camera_tab.rowconfigure(0, weight=1)

        # left sidebar: camera list + controls
        sidebar = ttk.Frame(self.camera_tab)
        sidebar.grid(row=0, column=0, sticky="nsw", padx=(0, PAD), pady=PAD)

        self.camera_list_frame = ttk.LabelFrame(sidebar, text="Camera phat hien (chon 1)")
        self.camera_list_frame.pack(fill="x", pady=(0, 10))
        self.camera_var = tk.StringVar(value="")
        self.camera_radios: list[ttk.Radiobutton] = []

        ttk.Button(sidebar, text="Quet lai camera", command=self._refresh_cameras).pack(fill="x", pady=(0, 6))
        self.start_btn = ttk.Button(sidebar, text="Bat dau", style="Start.TButton", command=self._start_camera)
        self.start_btn.pack(fill="x", pady=(0, 6))
        self.stop_btn = ttk.Button(sidebar, text="Dung", style="Stop.TButton", command=self._stop_camera, state="disabled")
        self.stop_btn.pack(fill="x")

        # main: live preview
        main = ttk.Frame(self.camera_tab)
        main.grid(row=0, column=1, sticky="nsew", pady=PAD)
        main.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)

        preview_holder = tk.Frame(main, bg="#0b0f16")
        preview_holder.grid(row=0, column=0, sticky="nsew")
        self.camera_view = tk.Label(preview_holder, bg="#0b0f16", text="Chua bat dau camera", fg=COLOR_MUTED)
        self.camera_view.pack(expand=True)

        self.camera_status = ttk.Label(main, text="Chua chon camera.", style="Muted.TLabel")
        self.camera_status.grid(row=1, column=0, sticky="w", pady=(8, 0))

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
            ttk.Label(self.camera_list_frame, text="Khong phat hien camera nao.", style="Muted.TLabel").pack(
                anchor="w", padx=8, pady=6
            )
            return

        for key, label in entries:
            radio = ttk.Radiobutton(self.camera_list_frame, text=label, value=key, variable=self.camera_var)
            radio.pack(anchor="w", padx=8, pady=3)
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
        self._prev_frame_time = time.monotonic()
        self.start_btn.state(["disabled"])
        self.stop_btn.state(["!disabled"])
        for radio in self.camera_radios:
            radio.state(["disabled"])
        self.camera_status.configure(text=f"Dang chay: {selection}")
        self.status_var.set(f"Camera dang chay | model={self.model_var.get()}")
        self._camera_loop()

    def _camera_loop(self) -> None:
        if not self.camera_running or self.camera is None:
            return
        try:
            now = time.monotonic()
            infer_started = now
            frame = self.camera.read()
            detections = self.detector.predict(frame) if self.detector else []
            infer_ms = (time.monotonic() - infer_started) * 1000.0
            frame = draw_detections(frame, detections, self.detector.names if self.detector else {})

            elapsed = now - self._prev_frame_time
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            self._prev_frame_time = now
            draw_hud(frame, fps, infer_ms, len(detections))

            self._show_frame(self.camera_view, frame)
            self.status_var.set(
                f"Camera dang chay | model={self.model_var.get()} | FPS={fps:.1f} | infer={infer_ms:.0f}ms | det={len(detections)}"
            )
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
        self.status_var.set("San sang.")

    @staticmethod
    def _show_frame(label: tk.Label, frame_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = ImageTk.PhotoImage(Image.fromarray(rgb))
        label.configure(image=image, text="")
        label.image = image  # keep a reference, tkinter drops it otherwise

    # -- video test tab ------------------------------------------------------

    def _build_video_tab(self) -> None:
        self.video_tab.columnconfigure(0, weight=1)
        self.video_tab.rowconfigure(2, weight=1)

        controls = ttk.LabelFrame(self.video_tab, text="Nguon video")
        controls.grid(row=0, column=0, sticky="ew", pady=(PAD, 6))
        controls.columnconfigure(1, weight=1)

        ttk.Button(controls, text="Mo video...", style="Accent.TButton", command=self._open_video).grid(
            row=0, column=0, padx=8, pady=8
        )
        self.video_path_var = tk.StringVar(value="Chua chon video.")
        ttk.Entry(controls, textvariable=self.video_path_var, state="readonly").grid(
            row=0, column=1, sticky="ew", padx=8, pady=8
        )
        self.run_video_btn = ttk.Button(
            controls, text="Chay xu ly", style="Start.TButton", command=self._run_video, state="disabled"
        )
        self.run_video_btn.grid(row=0, column=2, padx=8, pady=8)

        self.video_progress = ttk.Progressbar(self.video_tab, mode="determinate")
        self.video_progress.grid(row=1, column=0, sticky="ew", pady=(0, 6))

        preview_holder = tk.Frame(self.video_tab, bg="#0b0f16")
        preview_holder.grid(row=2, column=0, sticky="nsew", pady=(0, 6))
        self.video_preview = tk.Label(preview_holder, bg="#0b0f16", text="Chua xu ly video", fg=COLOR_MUTED)
        self.video_preview.pack(expand=True)

        result_frame = ttk.LabelFrame(self.video_tab, text="Ket qua")
        result_frame.grid(row=3, column=0, sticky="ew")
        result_frame.columnconfigure(1, weight=1)
        ttk.Label(result_frame, text="File output:").grid(row=0, column=0, padx=8, pady=8, sticky="w")
        self.result_path_var = tk.StringVar(value="(chua co)")
        ttk.Entry(result_frame, textvariable=self.result_path_var, state="readonly").grid(
            row=0, column=1, sticky="ew", padx=8, pady=8
        )
        ttk.Button(result_frame, text="Mo thu muc", command=self._open_result_folder).grid(
            row=0, column=2, padx=8, pady=8
        )

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
        self.status_var.set(f"Dang xu ly video: {self._video_path.name}")

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
        prev_time = time.monotonic()
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                started = time.monotonic()
                detections = detector.predict(frame)
                infer_ms = (time.monotonic() - started) * 1000.0
                frame = draw_detections(frame, detections, detector.names)

                now = time.monotonic()
                elapsed = now - prev_time
                prev_time = now
                processing_fps = 1.0 / elapsed if elapsed > 0 else 0.0
                draw_hud(frame, processing_fps, infer_ms, len(detections))

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
                    self.status_var.set(f"Dang xu ly video: {progress * 100:.0f}%")
                elif kind == "done":
                    self._result_path = message[1]
                    self.video_progress["value"] = 100
                    self.result_path_var.set(str(self._result_path))
                    self.run_video_btn.state(["!disabled"])
                    self.status_var.set(f"Da xong: {self._result_path}")
                    return
                elif kind == "error":
                    messagebox.showerror("Drowning Detection", message[1])
                    self.result_path_var.set("(loi)")
                    self.run_video_btn.state(["!disabled"])
                    self.status_var.set("San sang.")
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
