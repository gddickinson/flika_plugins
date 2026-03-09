# Light Sheet TIFF Loader

Specialized file loader for multi-channel light-sheet microscopy TIFF and CZI files. Automatically detects axis ordering and splits interleaved channels into separate flika windows.

## Features

- Load multi-channel TIFF stacks with automatic axis detection
- Load Zeiss CZI files via bundled czifile reader
- Split interleaved channels into separate windows
- Handle up to 5D data (T, Z, C, Y, X)

## Requirements

- [flika](https://github.com/flika-org/flika) >= 0.2.23
- tifffile

## Usage

1. Go to **Plugins > Light Sheet TIFF Loader**
2. Select a TIFF or CZI file
3. Channels are automatically split into separate flika windows
