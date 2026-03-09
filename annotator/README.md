# Annotator

Burns frame-number text annotations into each frame of a 3D image stack using PIL/Pillow text rendering.

## Features

- Renders frame index (0, 1, 2, ...) as text on each frame
- High-quality text rendering via PIL/Pillow
- Creates a new annotated window (non-destructive)

## Requirements

- [flika](https://github.com/flika-org/flika) >= 0.2.23
- **Pillow** (`pip install Pillow`)

## Usage

1. Open a 3D image stack in flika
2. Go to **Plugins > Annotator**
3. A new window is created with frame numbers burned into each frame

## Notes

- Requires a system font (Arial on Windows; may need adjustment on macOS/Linux)
- Works with 3D stacks (time series); 2D images are not supported
