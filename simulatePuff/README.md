# Puff Simulator

Adds synthetic calcium puffs (2D Gaussian blips) to image stacks at controlled positions, times, amplitudes, and durations.

## Features

- **Single puff** -- add one puff at a specified position and time
- **Random puff train** -- generate multiple puffs at one site with random timing
- **Multi-site random** -- distribute puffs across sites within an ROI
- **Full simulation** -- generate a complete synthetic test movie with known ground truth
- Configurable amplitude, sigma (width), and duration

## Requirements

- [flika](https://github.com/flika-org/flika) >= 0.2.23

## Usage

1. Open an image stack (or create a blank one) in flika
2. Go to **Plugins > Simulate Puff**
3. Choose simulation mode:
   - **Add Single**: click on image to place a puff
   - **Random Train**: generates puffs at one site with Poisson timing
   - **Multi-site**: distributes puff sites randomly within an ROI
   - **Full Simulation**: creates an entire synthetic movie
4. Set parameters (amplitude, sigma, duration, rate)
5. Run simulation

## Parameters

| Parameter | Description |
|-----------|-------------|
| Amplitude | Peak intensity of the synthetic puff |
| Sigma | Width of the 2D Gaussian (pixels) |
| Duration | Number of frames for the puff event |
| Rate | Average puffs per second (for random modes) |
| N sites | Number of puff sites (multi-site mode) |
