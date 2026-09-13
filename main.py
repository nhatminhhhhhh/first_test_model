"""Run drowning detection from a Raspberry Pi OV5647 camera."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

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


def draw_detections(
    result: Results,
    confirmation_boxes: list[dict[str, object]],
    timestamp: float,
) -> np.ndarray:
    """Draw detections and update the two-second drowning confirmations."""
    frame = result.orig_img.copy()
    detections = [
        (box.cpu().numpy(), float(confidence), int(class_id))
        for box, confidence, class_id in zip(
            result.boxes.xyxy, result.boxes.conf, result.boxes.cls
        )
    ]
    confirmation_boxes[:] = [
        item
        for item in confirmation_boxes
        if timestamp - float(item["started_at"]) <= CONFIRMATION_WINDOW_SECONDS
    ]

    for box, confidence, class_id in detections:
        label_name = str(result.names[class_id])
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
        help="NCNN model directory or compatible Ultralytics model path",
    )
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--output", default="runs/detect/ov5647.avi")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = REPO_DIR / model_path
    if not model_path.is_file() and not model_path.is_dir():
        raise FileNotFoundError(f"Model file or directory not found: {model_path}")
    if not 0 <= args.conf <= 1:
        raise ValueError("--conf must be between 0 and 1")
    if args.imgsz <= 0 or args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise ValueError("imgsz, width, height and fps must be greater than zero")

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    camera = Picamera2()
    camera.configure(
        camera.create_preview_configuration(
            main={"size": (args.width, args.height), "format": "BGR888"}
        )
    )
    camera.start()
    time.sleep(1)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        camera.stop()
        raise RuntimeError(f"Unable to open output video: {output_path}")

    confirmation_boxes: list[dict[str, object]] = []
    previous_time = time.monotonic()
    try:
        while True:
            image = camera.capture_array()
            result = model.predict(
                source=image,
                conf=args.conf,
                imgsz=args.imgsz,
                verbose=False,
            )[0]
            timestamp = time.monotonic()
            frame = draw_detections(result, confirmation_boxes, timestamp)
            current_time = time.monotonic()
            elapsed = current_time - previous_time
            previous_time = current_time
            actual_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            cv2.putText(
                frame,
                f"FPS: {actual_fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            writer.write(frame)
            if not args.no_display:
                cv2.imshow("OV5647 drowning detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        writer.release()
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
