# Car Number Plate Detection and Extraction

<img width="960" height="450" alt="License Plate Detection & Extraction" src="https://github.com/user-attachments/assets/1328599c-bcc1-445d-8449-f6792a25531f" />


This project is based on python for detecting and extracting car license plate number from a video stream. It uses a combination of object detection, tracking, and optical character recognition (OCR) to identify vehicles, track them, and read their license plates.

## Features

*   **License Plate Detection:** Detects license plates from the detected vehicles.
*   **License Plate Recognition:** Extract the characters from the license plates using OCR.
*   **CSV Output:** Saves the results, including the vehicle ID, license plate number, and confidence score, to a CSV file.
*   **Video Output:** Creates a new video with bounding boxes drawn around the detected license plates followed by extracted text.

## Libraries Used

*   **Python:** The core programming language.
*   **OpenCV:** For video processing and saving.
*   **Model:**
    * **Vechicle Detection:** For detecting vechiles, I have used `yolo11n` pre-trained model.
    * **License Plate Detection:** For license plate detection, I have fine-tuned `yolo11n` on lincense plate [dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e). 
*   **DeepSort:** For tracking the detected vehicles.
*   **EasyOCR:** For extracting the text of detected license plates.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Ghost-141/Car-Numb-Plate-Detection-Extraction.git
    cd Car-Numb-Plate-Detection-Extraction
    ```

2.  **Create a virtual environment and activate it:**

    *   **Using `conda`:**
        ```bash
        conda create -n anpr python=3.10
        conda activate anpr
        ```

    *   **Using `venv`:**
        ```bash
        python -m venv anpr
        source anpr/bin/activate # for linux
        anpr\Scripts\activate # for windows
        ```

3.  **Install the required Python libraries:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Use:

1.  Replace your video file directory in place of `input_video.mp4` in detect.py` file.
2.  Run the `detect.py` script.

## Output

*   **`Detection_video.mp4`:** A video file with bounding boxes around license plates, followed by license plate text.
*   **`test.csv`:** A CSV file containing the following columns:

frame_mmr|car_id|car_bbox|license_number|license_plate_conf_score
-----|-----|-----|-----|------|
-----|-----|-----|-----|------|
|                             |

*   `frame_nmr`: The frame number where the license plate was first confidently recognized.
*   `car_id`: The unique ID assigned to the vehicle.
*   `car_bbox`: The bounding box of the car.
*   `license_number`: The recognized license plate number.
*   `license_plate_conf_score`: The confidence score of the license plate recognition.


## Future Work

- Improve model accuracy detection in low light 
- Import & optimize model for edge devices
