# Car Number Plate Detection and Recognition

This project is a Python application for detecting and recognizing car number plates from a video stream. It uses a combination of object detection, tracking, and optical character recognition (OCR) to identify vehicles, track them, and read their license plates.

## Features

*   **Vehicle Detection:** Detects cars, trucks, and other vehicles in a video.
*   **Vehicle Tracking:** Tracks the detected vehicles across frames to maintain a unique ID for each vehicle.
*   **License Plate Detection:** Locates and extracts license plates from the detected vehicles.
*   **License Plate Recognition:** Reads the characters from the license plates using OCR.
*   **CSV Output:** Saves the results, including the vehicle ID, license plate number, and confidence score, to a CSV file.
*   **Video Output:** Creates a new video with bounding boxes drawn around the detected vehicles and their license plates.

## Technologies Used

*   **Python:** The core programming language.
*   **OpenCV:** For video processing and drawing bounding boxes.
*   **YOLO (You Only Look Once):** For object detection (vehicles and license plates). The specific models used are `yolo11n.pt` for general object detection and `anpr-demo-model.pt` for license plate detection.
*   **DeepSort:** For tracking the detected vehicles.
*   **EasyOCR:** For recognizing the text on the license plates.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **Install the required Python libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file. See the "Creating `requirements.txt`" section below.)*

## Usage

1.  **Place your input video** in the root directory of the project and name it `anpr-demo-video.mp4`.
2.  **Run the `detect.py` script:**
    ```bash
    python detect.py
    ```

## Output

*   **`Detection_video.mp4`:** A video file with bounding boxes around the detected cars and license plates. The recognized license plate text is also displayed.
*   **`test.csv`:** A CSV file containing the following columns:
    *   `frame_nmr`: The frame number where the license plate was first confidently recognized.
    *   `car_id`: The unique ID assigned to the vehicle.
    *   `car_bbox`: The bounding box of the car.
    *   `license_number`: The recognized license plate number.
    *   `license_plate_conf_score`: The confidence score of the license plate recognition.

## Creating `requirements.txt`

You can create a `requirements.txt` file by running the following command in your terminal. This will list all the Python packages you have installed in your current environment. It's recommended to use a virtual environment for this.

```bash
pip freeze > requirements.txt
```

A typical `requirements.txt` for this project might look like this:

```
opencv-python
ultralytics
deepsort-realtime
easyocr
torch
torchvision
torchaudio
```
