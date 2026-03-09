# Flash Remover

Removes flash artifacts (e.g., UV uncaging flashes) from image stacks using linear interpolation or noise-based intensity scaling.

## Features

- **Automatic flash detection** -- moving average peak detection
- **Linear interpolation** -- smoothly interpolate across flash frames
- **Noise-based correction** -- intensity scaling using pre-flash statistics
- **Manual boundaries** -- specify flash start/end frames manually

## Requirements

- [flika](https://github.com/flika-org/flika) >= 0.2.23

## Usage

1. Open an image stack containing flash artifacts
2. Go to **Plugins > Flash Remover**
3. Choose detection mode (automatic or manual)
4. Select removal method (interpolation or noise scaling)
5. Apply to generate a corrected stack

## Parameters

| Parameter | Description |
|-----------|-------------|
| Flash start | First frame of the flash (auto-detected or manual) |
| Flash end | Last frame of the flash |
| Method | Interpolation or noise-based correction |
| Detection threshold | Sensitivity for automatic flash detection |
