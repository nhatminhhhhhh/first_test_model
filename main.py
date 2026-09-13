"""Run drowning detection from a Raspberry Pi OV5647 camera."""

from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path

# Keep OpenCV from competing with NCNN for the four Cortex-A72 cores.
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
CONFIRMATION_WINDOW_SECONDS = 2.0
MATCH_IOU_THRESHOLD = 0.5
NCNN_MEAN = [0.0, 0.0, 0.0]
NCNN_NORM = [1 / 255.0, 1 / 255.0, 1 / 255.0]


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


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    """Return the intersection-over-union of two xyxy boxes."""
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def letterbox(image: np.ndarray, imgsz: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize and pad to a square while keeping aspect ratio (YOLO-style)."""
    height, width = image.shape[:2]
    scale = min(imgsz / height, imgsz / width)
    new_width, new_height = int(round(width * scale)), int(round(height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
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


class LatestFrame:
    """Always keep the newest camera frame so inference is not serialized with capture."""

    def __init__(self, camera: Picamera2) -> None:
        self._camera = camera
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            frame = self._camera.capture_array()
            with self._lock:
                self._frame = frame

    def get(self) -> np.ndarray:
        while self._running:
            with self._lock:
                frame = self._frame
            if frame is not None:
                return frame.copy()
            time.sleep(0.001)
        raise RuntimeError("Camera capture thread has stopped")

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)


def draw_detections(
    frame: np.ndarray,
    detections: list[tuple[np.ndarray, float, int]],
    names: dict[int, str],
    confirmation_boxes: list[dict[str, object]],
    timestamp: float,
) -> np.ndarray:
    """Draw detections and update the two-second drowning confirmations."""
    confirmation_boxes[:] = [
        item
        for item in confirmation_boxes
        if timestamp - float(item["started_at"]) <= CONFIRMATION_WINDOW_SECONDS
    ]

    for box, confidence, class_id in detections:
        label_name = str(names[class_id])
        is_drowning = label_name.lower() == "drowning"
        confirmed = False
        if is_drowning:
            for item in confirmation_boxes:
                if (
                    confidence > 0.6
                    and box_iou(box, item["box"]) >= MATCH_IOU_THRESHOLD
                ):
                    item["box"] = box
                    item["confirmed"] = True
                    confirmed = True
                    break
            if confidence > 0.6 and not confirmed:
                confirmation_boxes.append(
                    {"box": box, "started_at": timestamp, "confirmed": False}
                )

        color = (
            (0, 0, 255)
            if confirmed
            else (0, 255, 255)
            if is_drowning
            else (0, 255, 0)
        )
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
        default="model_ncnn_model",
        help="NCNN model directory exported by Ultralytics",
    )
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=0, help="0 = use export size from metadata.yaml")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", default="runs/detect/ov5647.avi")
    parser.add_argument("--save", action="store_true", help="Write annotated video (slow on Pi 4)")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    cv2.setNumThreads(1)

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
            main={"size": (args.width, args.height), "format": "BGR888"},
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

    latest = LatestFrame(camera)
    dummy = latest.get()
    for _ in range(max(0, args.warmup)):
        detector.predict(dummy)

    confirmation_boxes: list[dict[str, object]] = []
    previous_time = time.monotonic()
    print(
        "Running native NCNN. Tips: --no-display, avoid --save, "
        "and re-export with `yolo export format=ncnn imgsz=320 int8=True` for more FPS."
    )
    try:
        while True:
            image = latest.get()
            infer_started = time.monotonic()
            detections = detector.predict(image)
            infer_ms = (time.monotonic() - infer_started) * 1000.0
            timestamp = time.monotonic()
            frame = draw_detections(
                image,
                detections,
                detector.names,
                confirmation_boxes,
                timestamp,
            )
            current_time = time.monotonic()
            elapsed = current_time - previous_time
            previous_time = current_time
            actual_fps = 1.0 / elapsed if elapsed > 0 else 0.0
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
    finally:
        latest.stop()
        if writer is not None:
            writer.release()
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
