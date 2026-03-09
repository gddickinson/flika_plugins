# Center-Surround ROI

Creates linked center and surround ROIs for local background subtraction with real-time trace display.

## Features

- Linked center and surround ROI regions
- Real-time trace display (center, surround, and difference)
- Adjustable center and surround sizes
- Configurable ROI shape (circle, ellipse, square)
- Live background-subtracted signal

## Requirements

- [flika](https://github.com/flika-org/flika) >= 0.2.23

## Usage

1. Open a time-series image in flika
2. Go to **Plugins > Center-Surround ROI**
3. Click **Start** to place the ROI
4. Drag the ROI to the region of interest
5. Adjust center/surround widths using the sliders
6. The trace plots update in real time

## Parameters

| Parameter | Description |
|-----------|-------------|
| Center width | Radius of the inner ROI |
| Surround width | Width of the outer ring |
| Inner ratio | Ratio of inner to outer region |

## Use Cases

- Calcium imaging with local background correction
- Fluorescence measurements where nearby background varies spatially
- Any signal extraction requiring local baseline subtraction
