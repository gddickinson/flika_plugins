# OpenCV Camera

Live webcam capture and AVI file import using OpenCV. Stream camera frames directly into flika windows for real-time processing.

## Features

- **Live camera capture** -- stream webcam frames into flika
- **AVI file import** -- load video files via OpenCV
- **Basic filters** -- Canny edge detection, background subtraction

## Requirements

- [flika](https://github.com/flika-org/flika) >= 0.2.23
- **opencv-python** (`pip install opencv-python`)

## Usage

1. Go to **Plugins > OpenCV**
2. Choose **Start Camera** to begin live capture, or **Open File** to load an AVI video
3. Frames are loaded into a new flika window
