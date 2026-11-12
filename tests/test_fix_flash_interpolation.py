"""Regression: flash_remover's linear interpolation did not reach the frame after the flash.

removeFlash_interpolate replaces frames flashStart-1 .. flashEnd+1 (one buffer frame on each side) with a straight
line between the two buffer frames, but interpolated n frames over xp=[0, n] instead of [0, n-1]. The last buffer
frame (a good frame after the flash) was therefore overwritten with a value short of its own, and a step was left
between it and the next frame.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from web.source_loader import load  # noqa: E402


def test_interpolation_joins_the_frames_either_side_of_the_flash():
    shown = {}
    fn = load(ROOT / "flash_remover/flash_remover.py", ["Flash_remover.removeFlash_interpolate"],
              {"Window": lambda img, name: shown.setdefault("img", img)})["Flash_remover.removeFlash_interpolate"]
    movie = np.full((30, 2, 2), 10.0)
    movie[14:] = 20.0          # baseline is higher after the flash
    movie[10:14] = 500.0       # the flash, frames 10-13; frames 9 and 14 are the buffer frames
    values = {"window": SimpleNamespace(image=movie), "manualFlash": True, "flashStart": 10, "flashEnd": 13,
              "addNoise": False}
    me = SimpleNamespace(getValue=values.get)
    fn(me)
    out = shown["img"]
    np.testing.assert_array_equal(out[:10], movie[:10])
    np.testing.assert_array_equal(out[14:], movie[14:])   # frame 14 (the buffer after the flash) is unchanged
    np.testing.assert_allclose(out[9:15, 0, 0], np.linspace(10, 20, 6))
