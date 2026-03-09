# Frame Remover

Removes periodic blocks of frames from image stacks. Useful for removing stimulus frames or other periodic artifacts from time-series data.

## Features

- Remove frames at regular intervals throughout a stack
- Configurable start frame, end frame, block length, and interval
- Creates a new window with frames removed (non-destructive)

## Requirements

- [flika](https://github.com/flika-org/flika) >= 0.2.23

## Usage

1. Open a 3D image stack in flika
2. Go to **Plugins > Frame Remover**
3. Set the start frame, end frame, block length, and interval
4. Click OK to generate a new stack with the specified frames removed

## Parameters

| Parameter | Description |
|-----------|-------------|
| Start frame | First frame to begin removing |
| End frame | Last frame in the removal range |
| Block length | Number of consecutive frames to remove per interval |
| Interval | Spacing between removal blocks |

## Example

To remove 2 frames every 10 frames from frame 0 to 200:
- Start: 0, End: 200, Block length: 2, Interval: 10
- Removes frames 0-1, 10-11, 20-21, ..., 190-191
