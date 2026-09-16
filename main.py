"""Run drowning detection from a Raspberry Pi OV5647 camera."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

# Two threads by default: 4-core 100% load overheats / brownouts a Pi 4 in ~2 minutes.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

import cv2
import ncnn
import numpy as np

try:
    from picamera2 import Picamera2
except ImportError as error:
    raise RuntimeError(
        "Picamera2 is required on Raspberry Pi. Install it with "
        "'sudo apt install -y python3-picamera2'."
    ) from error


REPO_DIR = Path(__file__).resolve().parent
BOX_COLORS = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 128, 0),
    (255, 0, 255),
    (0, 255, 255),
]
NCNN_MEAN = [0.0, 0.0, 0.0]
NCNN_NORM = [1 / 255.0, 1 / 255.0, 1 / 255.0]
THERMAL_PATH = Path("/sys/class/thermal/thermal_zone0/temp")
THROTTLE_BITS = {
    0: "under-voltage now",
    1: "ARM freq capped now",
    2: "throttled now",
    3: "soft temp limit now",
    16: "under-voltage occurred",
    17: "ARM freq capped occurred",
    18: "throttled occurred",
    19: "soft temp limit occurred",
}


def read_vcgencmd(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["vcgencmd", *args],
            capture_output=True,
            text=True,
            timeout=0.5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = result.stdout.strip()
    return text or None


def read_cpu_temp_c() -> float | None:
    measured = read_vcgencmd("measure_temp")
    if measured and "temp=" in measured:
        try:
            return float(measured.split("=", 1)[1].rstrip("'C"))
        except ValueError:
            pass
    try:
        return int(THERMAL_PATH.read_text().strip()) / 1000.0
    except OSError:
        return None


def read_core_volts() -> str | None:
    measured = read_vcgencmd("measure_volts", "core")
    if not measured:
        return None
    return measured.replace("volt=", "core=")


def read_throttle_flags() -> tuple[int | None, list[str]]:
    raw = read_vcgencmd("get_throttled")
    if not raw or "throttled=" not in raw:
        return None, []
    value = int(raw.split("=", 1)[1], 16)
    flags = [name for bit, name in THROTTLE_BITS.items() if value & (1 << bit)]
    return value, flags


def pi_status_text() -> str:
    temp = read_cpu_temp_c()
    value, flags = read_throttle_flags()
    volts = read_core_volts()
    parts: list[str] = []
    if temp is not None:
        parts.append(f"temp={temp:.1f}C")
    if volts:
        parts.append(volts)
    if value is not None:
        parts.append(f"throttled=0x{value:x}")
        if flags:
            parts.append(", ".join(flags))
        else:
            parts.append("ok")
    return "  ".join(parts) if parts else "no Pi thermal/throttle sysfs"


def load_class_names(model_dir: Path) -> dict[int, str]:
    """Read class names from Ultralytics metadata, with drowning-detection defaults."""
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
    """Resize and pad to a square while keeping aspect ratio (YOLO-style)."""
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
        resized,
        top,
        pad_h - top,
        left,
        pad_w - left,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
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
    """Return predictions as (4 + num_classes, num_boxes)."""
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
    raise RuntimeError(
        f"Unexpected NCNN output shape {array.shape}; expected {channels} channels"
    )


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
        mat_in = ncnn.Mat.from_pixels(
            padded,
            ncnn.Mat.PixelType.PIXEL_BGR2RGB,
            self.imgsz,
            self.imgsz,
        )
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
            selected_boxes = boxes_xyxy[class_mask][keep]
            selected_scores = scores[class_mask][keep]
            for box, score in zip(selected_boxes, selected_scores):
                detections.append((box, float(score), int(class_id)))
        return detections


def grab_frame(camera: Picamera2) -> np.ndarray:
    """Take one camera frame. Avoid a capture thread: it contends for Pi 4 RAM bandwidth."""
    request = camera.capture_request()
    try:
        return request.make_array("main").copy()
    finally:
        request.release()


def draw_detections(
    frame: np.ndarray,
    detections: list[tuple[np.ndarray, float, int]],
    names: dict[int, str],
) -> np.ndarray:
    """Draw each detection's box, class label, and confidence."""
    for box, confidence, class_id in detections:
        label_name = str(names[class_id])
        color = BOX_COLORS[class_id % len(BOX_COLORS)]
        points = box.astype(int)
        cv2.rectangle(frame, tuple(points[:2]), tuple(points[2:]), color, 2)
        cv2.putText(
            frame,
            f"{label_name} {confidence:.2f}",
            (int(points[0]), max(20, int(points[1]) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="model/yolo26n_drowning-2",
        help="NCNN model directory exported by Ultralytics",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=0, help="0 = use export size from metadata.yaml")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="NCNN threads. 1 draws less current on a weak Pi 4 PSU",
    )
    parser.add_argument(
        "--max-fps",
        type=float,
        default=10.0,
        help="Cap loop rate so the CPU can idle. 0 disables the cap",
    )
    parser.add_argument(
        "--thermal-limit",
        type=float,
        default=58.0,
        help="Start extra idle above this SoC temp. Pi dying near 63C is PSU sag, not thermal shutdown",
    )
    parser.add_argument("--output", default="runs/detect/ov5647.avi")
    parser.add_argument("--save", action="store_true", help="Write annotated video (slow on Pi 4)")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--bench", type=int, default=0, help="Run N frames, print average, then exit")
    parser.add_argument("--log-every", type=int, default=10, help="Print FPS/infer to the terminal every N frames")
    args = parser.parse_args()

    cv2.setNumThreads(2) # 2 is the default for OpenCV

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = REPO_DIR / model_path
    if not model_path.is_dir():
        raise FileNotFoundError(f"NCNN model directory not found: {model_path}")
    if not 0 <= args.conf <= 1:
        raise ValueError("--conf must be between 0 and 1")
    if args.width <= 0 or args.height <= 0 or args.fps <= 0 or args.threads <= 0:
        raise ValueError("width, height, fps and threads must be greater than zero")

    export_imgsz = load_export_imgsz(model_path, 416)
    imgsz = args.imgsz if args.imgsz > 0 else export_imgsz
    if imgsz != export_imgsz:
        print(
            f"Warning: this NCNN graph is exported at {export_imgsz}. "
            f"Using --imgsz {imgsz} can be inaccurate; re-export the model at that size."
        )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_DIR / output_path
    if args.save:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    detector = NcnnDetector(
        model_path,
        imgsz=imgsz,
        conf=args.conf,
        iou=args.iou,
        threads=args.threads,
    )

    camera = Picamera2()
    camera.configure(
        camera.create_preview_configuration(
            main={"size": (args.width, args.height), "format": "RGB888"},
            buffer_count=4,
        )
    )
    camera.start()
    time.sleep(0.4)

    writer: cv2.VideoWriter | None = None
    if args.save:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            args.fps,
            (args.width, args.height),
        )
        if not writer.isOpened():
            camera.stop()
            raise RuntimeError(f"Unable to open output video: {output_path}")

    dummy = grab_frame(camera)
    for _ in range(max(0, args.warmup)):
        detector.predict(dummy)

    previous_time = time.monotonic()
    infer_total_ms = 0.0
    frame_count = 0
    last_backoff_log = 0.0
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    print(
        f"Native NCNN imgsz={imgsz} threads={args.threads} max-fps={args.max_fps}. "
        f"{pi_status_text()}. Ctrl+C to stop.",
        flush=True,
    )
    try:
        while True:
            loop_started = time.monotonic()
            image = grab_frame(camera)
            infer_started = time.monotonic()
            detections = detector.predict(image)
            infer_ms = (time.monotonic() - infer_started) * 1000.0
            if args.no_display and writer is None:
                frame = image
            else:
                frame = draw_detections(image, detections, detector.names)
            current_time = time.monotonic()
            elapsed = current_time - previous_time
            previous_time = current_time
            actual_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            infer_total_ms += infer_ms
            frame_count += 1
            if args.log_every > 0 and frame_count % args.log_every == 0:
                avg_infer = infer_total_ms / frame_count
                print(
                    f"[{frame_count}] fps={actual_fps:.1f}  infer={infer_ms:.0f}ms  "
                    f"avg_infer={avg_infer:.0f}ms  dets={len(detections)}  {pi_status_text()}",
                    flush=True,
                )
            if not args.no_display:
                cv2.putText(
                    frame,
                    f"FPS: {actual_fps:.1f}  infer: {infer_ms:.0f}ms",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
            if writer is not None:
                writer.write(frame)
            if not args.no_display:
                cv2.imshow("OV5647 drowning detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.bench > 0 and frame_count >= args.bench:
                break
            if args.bench <= 0:
                remaining = 0.0
                if args.max_fps > 0:
                    remaining = (1.0 / args.max_fps) - (time.monotonic() - loop_started)
                temp = read_cpu_temp_c()
                if args.thermal_limit > 0 and temp is not None:
                    if temp >= args.thermal_limit:
                        remaining = max(remaining, 0.4 + (temp - args.thermal_limit) * 0.08)
                    if temp >= 58.0:
                        remaining = max(remaining, 1.5)
                    if temp >= 61.0:
                        remaining = max(remaining, 3.0)
                        now = time.monotonic()
                        if now - last_backoff_log >= 5.0:
                            last_backoff_log = now
                            print(
                                f"Load backoff at {temp:.1f}C to avoid PSU cutoff. "
                                f"{pi_status_text()}",
                                flush=True,
                            )
                if remaining > 0:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        print("Stopped.", flush=True)
    finally:
        if frame_count:
            print(
                f"Average infer {infer_total_ms / frame_count:.0f}ms "
                f"over {frame_count} frames ({1000.0 * frame_count / infer_total_ms:.1f} infer FPS).",
                flush=True,
            )
        if writer is not None:
            writer.release()
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
