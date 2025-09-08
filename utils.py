import easyocr
import csv
import cv2

reader = easyocr.Reader(['en'], gpu=True)

def get_car(license_plate, vehicle_track_ids):

    x1, y1, x2, y2, score, class_id = license_plate

    found_lcn = False

    car_idx = -1

    for j in range(len(vehicle_track_ids)):
        track = vehicle_track_ids[j]
        xcar1, ycar1, xcar2, ycar2 = track.to_tlbr()

        # check if license plate is inside car bounding box
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_idx = j
            found_lcn = True
            break

    if found_lcn:
        # We found a car, now get its details and return them
        found_track = vehicle_track_ids[car_idx]
        car_id = found_track.track_id
        xcar1, ycar1, xcar2, ycar2 = found_track.to_tlbr()
        return xcar1, ycar1, xcar2, ycar2, car_id

    return 0, 0, 0, 0, -1



def read_license_plate(license_plate_crop):

    detections = reader.readtext(license_plate_crop, detail=1, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    if not detections:
        return 0, 0

    # Find the detection with the highest score
    best_detection = max(detections, key=lambda x: x[2])
    bbox, text, score = best_detection

    return text, score


def write_csv(results, output_path):
    """
    Write unique car_id entries to CSV.
    For each car_id, keep only the license plate text with the highest confidence score,
    along with car bounding box and frame number.
    """
    best_entries = {}

    for frame_nmr, frame_results in results.items():
        for car_id, car_data in frame_results.items():
            lp_data = car_data.get('license_plate', {})
            if 'text' in lp_data and 'text_score' in lp_data:
                text = lp_data['text']
                score = lp_data['text_score']

                if car_id not in best_entries or score > best_entries[car_id]['score']:
                    best_entries[car_id] = {
                        'text': text,
                        'score': score,
                        'frame': frame_nmr,
                        'bbox': car_data.get('car', {}).get('bbox')
                    }

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_nmr', 'car_id', 'car_bbox', 'license_number', 'license_plate_conf_score'])

        # Sort by frame number
        sorted_best_entries = sorted(best_entries.items(), key=lambda item: item[1]['frame'])

        for car_id, data in sorted_best_entries:
            bbox_str = ','.join(map(str, data['bbox'])) if data['bbox'] else ''
            writer.writerow([data['frame'], car_id, bbox_str, data['text'], data['score']])



def initialize_video_writer(cap, output_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))