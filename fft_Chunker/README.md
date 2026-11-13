# FFT Chunker

Loads CSV time-series data, splits traces into chunks, performs FFT on each chunk, averages power spectra across user-defined groups, and exports log10-transformed results.

## Features

- Load time-series data from CSV files
- Split traces into configurable chunk sizes
- Compute FFT power spectra per chunk
- Average spectra across baseline and up to 4 experimental groups
- Export log10-transformed power spectra to CSV

## Requirements

- [flika](https://github.com/flika-org/flika) >= 0.2.23
- pandas

## Usage

1. Go to **Plugins > FFT Chunker**
2. Load a CSV file with time-series data
3. Set chunk size and group boundaries (baseline, puff groups)
4. Run the FFT analysis
5. Export averaged power spectra to CSV

## Parameters

| Parameter | Description |
|-----------|-------------|
| Chunk size | Number of frames per FFT chunk |
| Baseline start/stop | Chunks averaged for the baseline (chunk numbers from 1, stop included) |
| Puff 1-4 start/stop | Chunks averaged for each experimental group (from 1, stop included; start = stop is one chunk) |

A group is used when stop > 0 and stop >= start. Before 2026-11-12 the stop chunk was left out of each group and
start = stop stopped with an error, so averages exported earlier cover one chunk fewer than the stop value suggests.
