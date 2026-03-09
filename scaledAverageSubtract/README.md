# Scaled Average Subtract

Removes the global calcium response from imaging stacks by detecting the peak via rolling average, creating a template from peak frames, and subtracting a scaled version from every frame.

## Features

- Rolling average peak detection
- Template image generation from peak frames
- Scaled subtraction to isolate local events from global transient
- ROI-based analysis region selection

## Requirements

- [flika](https://github.com/flika-org/flika) >= 0.2.23

## Usage

1. Open a calcium imaging time series in flika
2. Optionally draw an ROI to define the analysis region
3. Go to **Plugins > Scaled Average Subtract**
4. Set window size and average size
5. Click OK to generate the subtracted stack

## Parameters

| Parameter | Description |
|-----------|-------------|
| Window size | Rolling average window for peak detection |
| Average size | Number of frames around peak to average for template |

## Use Cases

- Isolating local calcium puffs/sparks from the global calcium wave
- Removing global photobleaching artifacts
- Background trend removal in fluorescence time series
