from ultralytics import YOLO
def get_config():
    license_detector = YOLO("./backend/models/best.pt")
    model = YOLO('./backend/models/yolo11n.pt')
    vehicle_id = [2, 3, 5, 7]
    results = {}
    return license_detector, model, vehicle_id, results