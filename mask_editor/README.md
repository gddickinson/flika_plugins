# Mask Editor

Interactive binary mask painting tool with a dockable control panel and mouse/keyboard-driven drawing and erasing.

## Features

- **Brush drawing** -- paint mask regions with configurable brush size
- **Eraser** -- remove mask regions
- **Undo** -- frame-level undo support
- **Threshold** -- generate initial mask from intensity threshold
- **Invert** -- invert the current mask
- **Fill/Clear** -- fill or clear the entire mask
- **Save/Load** -- save masks to file and reload later

## Requirements

- [flika](https://github.com/flika-org/flika) >= 0.2.23
- scikit-image

## Usage

1. Open an image in flika
2. Go to **Plugins > Mask Editor**
3. Use the brush tool to paint mask regions (left-click to draw)
4. Hold Shift + click to erase
5. Adjust brush size with the slider
6. Use the toolbar buttons for threshold, invert, fill, clear
7. Save the mask when done

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| D | Draw mode |
| E | Erase mode |
| Ctrl+Z | Undo |
