# Object Detection System

An image-based object detection web app built with Python, Streamlit and YOLOv8. Upload an image and the app detects objects, draws bounding boxes and shows the count of each detected object.

## Features

- Upload an image and detect objects using the YOLOv8 nano model (`yolov8n.pt`) from Ultralytics
- View the detected image with bounding boxes
- See the count of each detected object
- Simple web interface built with Streamlit

## Note

This project works on uploaded images only. It does not support webcam-based real-time tracking.

## Technologies Used

- Python
- Streamlit
- YOLOv8 (Ultralytics)
- Pillow

## How to Run

1. Install the required libraries:

```bash
pip install -r requirements.txt
```

2. Run the app:

```bash
streamlit run app.py
```
