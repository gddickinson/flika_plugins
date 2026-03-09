# Timestamp

Adds a dynamic elapsed-time label (in milliseconds) to any flika image window. The timestamp updates automatically as you scroll through frames.

## Features

- Text overlay showing elapsed time in milliseconds
- Auto-updates as you navigate through frames
- Uses the window's configured frame rate for timing

## Usage

1. Open a 3D image stack (time series) in flika
2. Go to **Plugins > Timestamp**
3. The timestamp overlay appears on the current window
4. Scroll through frames to see the timestamp update

## Requirements

- [flika](https://github.com/flika-org/flika) >= 0.2.23

## Notes

The frame rate is read from the window's settings. Ensure the frame rate is correctly configured via **File > Settings** for accurate time display.
