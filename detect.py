import cv2
from utils import get_car, read_license_plate, write_csv, initialize_video_writer
from config import get_config
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def main():
    license_detector, model, vehicle_id, results = get_config()
    tracker = DeepSort()

    # Open video
    cap = cv2.VideoCapture("input_video.mp4")
    out = initialize_video_writer(cap, 'Detection_video.mp4')

    frame_num = -1
    while True:
        frame_num += 1
        ret, frame = cap.read()
        if not ret:
            break

        results[frame_num] = {}

        # Car detection
        detections = model(frame, verbose=False)[0]
        detection_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicle_id:
                bbox = [x1, y1, x2 - x1, y2 - y1]
                detection_.append((bbox, score, int(class_id)))

        # track vehicle
        track_ids = tracker.update_tracks(detection_, frame=frame)

        # detect license plate
        license_plates = license_detector(frame, imgsz=640)[0]
        for license_plate_data in license_plates.boxes.data.tolist():
            lp_x1, lp_y1, lp_x2, lp_y2, lp_score, lp_class_id = license_plate_data

            # assign car to license plate
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate_data, track_ids)

            if car_id != -1:
                # crop license plate
                license_plate_crop = frame[int(lp_y1):int(lp_y2), int(lp_x1):int(lp_x2), :]

                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                license_plate_crop_thres = cv2.adaptiveThreshold(license_plate_crop_gray, 255,
                                                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                                 cv2.THRESH_BINARY_INV, 11, 2)
                # read license plate
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thres)

                if license_plate_text is not None and license_plate_text != 0:
                    results[frame_num][car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                        'license_plate': {
                            'text': license_plate_text,
                            'text_score': license_plate_text_score
                        }
                    }
                    # Draw the currently read license plate text
                    cv2.rectangle(frame, (int(lp_x1), int(lp_y1)), (int(lp_x2), int(lp_y2)), (0, 0, 255), 2)
                    cv2.putText(frame, license_plate_text, (int(lp_x1), int(lp_y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 0, 255), 2)

        # Write the frame to the output video
        out.write(frame)

    # write results
    write_csv(results, 'test.csv')

    # Release resources
    out.release()
    cap.release()

if __name__ == "__main__":
    main()
