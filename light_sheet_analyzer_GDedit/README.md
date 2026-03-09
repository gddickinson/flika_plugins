# Light Sheet Analyzer (GD Edit)

Converts 3D image stacks from light-sheet microscopy into proper 4D volumes with shear correction, projection, dF/F0 computation, and interactive 3D viewing.

## Features

- **3D to 4D reshaping** -- specify number of Z steps per volume
- **Shear transform correction** -- correct for oblique light-sheet geometry
- **Volume projection** -- max or mean intensity projection
- **dF/F0** -- compute fluorescence ratio relative to baseline
- **Volume Viewer** -- interactive 3D visualization with slice navigation
- **Frame trimming** -- remove incomplete volumes at end of acquisition

## Requirements

- [flika](https://github.com/flika-org/flika) >= 0.2.23

## Usage

1. Open a light-sheet TIFF stack in flika
2. Go to **Plugins > Light Sheet Analyzer**
3. Set the number of Z steps per volume
4. Apply shear correction if needed
5. Generate projections or open the Volume Viewer

## Parameters

| Parameter | Description |
|-----------|-------------|
| Number of steps | Z planes per volume |
| Shear factor | Correction factor for oblique geometry |
| Baseline frames | Frame range for F0 in dF/F0 calculation |
| Trim last frame | Remove incomplete final volume |
