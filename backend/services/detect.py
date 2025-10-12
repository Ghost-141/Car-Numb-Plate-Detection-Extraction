import argparse
from pathlib import Path
from typing import Callable, Dict, Optional
from uuid import uuid4

import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO  # noqa: F401

from backend.services.config import get_config
from backend.services.utils import get_car, initialize_video_writer, read_license_plate, write_csv


def process_video(
    input_path: Path,
    output_dir: Path,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> Dict[str, Path]:
    """
    Run license plate detection on a video file and store the processed outputs.
    """
    license_detector, model, vehicle_id, results = get_config()
    tracker = DeepSort()

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    unique_id = uuid4().hex[:6]
    output_video_path = output_dir / f"processed_{unique_id}.mp4"
    output_csv_path = output_dir / f"plates_{unique_id}.csv"

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {input_path}")
    out = initialize_video_writer(cap, str(output_video_path))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or 1
    frame_num = -1
    if progress_cb:
        progress_cb(0)

    try:
        while True:
            frame_num += 1
            ret, frame = cap.read()
            if not ret:
                break

            results[frame_num] = {}

            detections = model(frame, verbose=False)[0]
            detection_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicle_id:
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    detection_.append((bbox, score, int(class_id)))

            track_ids = tracker.update_tracks(detection_, frame=frame)

            license_plates = license_detector(frame, imgsz=640)[0]
            for license_plate_data in license_plates.boxes.data.tolist():
                lp_x1, lp_y1, lp_x2, lp_y2, lp_score, lp_class_id = license_plate_data
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate_data, track_ids)

                if car_id != -1:
                    license_plate_crop = frame[int(lp_y1): int(lp_y2), int(lp_x1): int(lp_x2), :]
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    license_plate_crop_thres = cv2.adaptiveThreshold(
                        license_plate_crop_gray,
                        255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY_INV,
                        11,
                        2,
                    )
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thres)

                    if license_plate_text is not None and license_plate_text != 0:
                        results[frame_num][car_id] = {
                            "car": {"bbox": [xcar1, ycar1, xcar2, ycar2]},
                            "license_plate": {
                                "text": license_plate_text,
                                "text_score": license_plate_text_score,
                            },
                        }
                        cv2.rectangle(frame, (int(lp_x1), int(lp_y1)), (int(lp_x2), int(lp_y2)), (0, 0, 255), 2)
                        cv2.putText(
                            frame,
                            license_plate_text,
                            (int(lp_x1), int(lp_y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 0, 255),
                            2,
                        )

            out.write(frame)
            if progress_cb:
                progress_cb(min(99, int((frame_num + 1) / total_frames * 100)))
    finally:
        out.release()
        cap.release()

    write_csv(results, str(output_csv_path))
    if progress_cb:
        progress_cb(100)

    return {"video": output_video_path, "csv": output_csv_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run license plate detection on a video file.")
    parser.add_argument("--input", default="input_video.mp4", help="Path to the input video.")
    parser.add_argument("--output-dir", default="outputs", help="Directory to save processed results.")
    args = parser.parse_args()

    process_video(Path(args.input), Path(args.output_dir))


if __name__ == "__main__":
    main()
