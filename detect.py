"""Run the trained drowning-detection model on an image, video, or webcam."""

from __future__ import annotations

import argparse
from itertools import chain
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results


REPO_DIR = Path(__file__).resolve().parent
CONFIRMATION_WINDOW_SECONDS = 2.0
MATCH_IOU_THRESHOLD = 0.5


def parse_source(value: str) -> int | str:
    """Use an integer for a webcam index and a path/URL otherwise."""
    return int(value) if value.isdigit() else value


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
    names = result.names
    boxes = result.boxes
    detections = [
        (box.cpu().numpy(), float(conf), int(cls))
        for box, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls)
    ]

    confirmation_boxes[:] = [
        item for item in confirmation_boxes
        if timestamp - float(item["started_at"]) <= CONFIRMATION_WINDOW_SECONDS
    ]
    for box, confidence, class_id in detections:
        label_name = str(names[class_id])
        is_drowning = label_name.lower() == "drowning"
        confirmed = False

        if is_drowning:
            for item in confirmation_boxes:
                if confidence > 0.6 and box_iou(box, item["box"]) >= MATCH_IOU_THRESHOLD: 
                    item["box"] = box
                    item["confirmed"] = True
                    confirmed = True
                    break

            if confidence > 0.6 and not confirmed:
                confirmation_boxes.append(
                    {"box": box, "started_at": timestamp, "confirmed": False}
                )

        color = (0, 0, 255) if confirmed else (0, 255, 255) if is_drowning else (0, 255, 0)
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
    parser.add_argument("--source", required=True, help="Image/video path, URL, or webcam index (0)")
    parser.add_argument(
        "--model",
        default="model_ncnn_model",
        help="Path to an NCNN model directory or compatible YOLO weights file",
    )
    parser.add_argument("--conf", type=float, default=0.2, help="Detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=320, help="Inference image size")
    parser.add_argument("--output", default="runs/", help="Output directory")
    parser.add_argument("--camera-width", type=int, default=640, help="Webcam capture width")
    parser.add_argument("--camera-height", type=int, default=480, help="Webcam capture height")
    parser.add_argument("--camera-fps", type=float, default=30, help="Requested webcam capture FPS")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = REPO_DIR / model_path
    if not model_path.is_file() and not model_path.is_dir():
        raise FileNotFoundError(f"Model file or directory not found: {model_path}")
    if not 0 <= args.conf <= 1:
        raise ValueError("--conf must be between 0 and 1")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be greater than 0")

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = REPO_DIR / output_dir

    source = parse_source(args.source)
    model = YOLO(str(model_path))
    output_path = output_dir 
    output_path.mkdir(parents=True, exist_ok=True)
    confirmation_boxes: list[dict[str, object]] = []

    if isinstance(source, int):
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"Unable to open camera {source}. "
                "Check that the camera is connected and not being used by another application."
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
        capture.set(cv2.CAP_PROP_FPS, args.camera_fps)

        height, width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(output_path / "camera.avi"),
            cv2.VideoWriter_fourcc(*"XVID"),
            args.camera_fps,
            (width, height),
        )
        prev_time = time.monotonic()
        try:
            while True:
                success, image = capture.read()
                if not success:
                    raise RuntimeError("Unable to read a frame from the webcam")

                result = model.predict(
                    source=image,
                    conf=args.conf,
                    imgsz=args.imgsz,
                    verbose=False,
                )[0]
                current_time = time.monotonic()
                frame = draw_detections(result, confirmation_boxes, current_time)
                writer.write(frame)

                actual_fps = 1.0 / (current_time - prev_time) if current_time > prev_time else 0
                prev_time = current_time
                cv2.putText(frame, f"FPS: {actual_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Drowning detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()
        return

    results = model.predict(
        source=source,
        conf=args.conf,
        imgsz=args.imgsz,
        stream=True,
        verbose=False,
    )

    first_result = next(results)
    source_path = Path(args.source) if isinstance(source, str) else None
    is_image = source_path is not None and source_path.suffix.lower() in {
        ".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"
    }
    if is_image:
        frame = draw_detections(first_result, confirmation_boxes, time.monotonic())
        cv2.imwrite(str(output_path / source_path.name), frame)
        return

    capture = cv2.VideoCapture(source)
    fps = capture.get(cv2.CAP_PROP_FPS) if capture.isOpened() else 0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else 0
    capture.release()
    fps = fps if fps > 0 else 30.0
    if source_path is not None:
        video_name = source_path.stem + ".avi"
    else:
        video_name = "webcam.avi"
    video_path = output_path / video_name
    height, width = first_result.orig_img.shape[:2]
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (width, height),
    )
    start_time = time.monotonic()
    processed_frames = 0
    last_progress = -1
    if total_frames > 0:
        print("\rTiến trình xử lý video:   0%", end="", flush=True)
        last_progress = 0
    try:
        for result in chain((first_result,), results):
            processed_frames += 1
            timestamp = time.monotonic() - start_time
            frame = draw_detections(result, confirmation_boxes, timestamp)
            writer.write(frame)
            if total_frames > 0:
                progress = min(100, int(processed_frames * 100 / total_frames))
                if progress != last_progress:
                    print(
                        f"\rTiến trình xử lý video: {progress:3d}%",
                        end="",
                        flush=True,
                    )
                    last_progress = progress
    finally:
        writer.release()
        if total_frames > 0:
            print()


if __name__ == "__main__":
    main()
