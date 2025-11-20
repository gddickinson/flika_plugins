"""
Image Transforms Plugin for FLIKA
==================================
A comprehensive plugin providing various geometric and intensity transformations
for FLIKA image stacks, optimized for TIRF microscopy and experimental imaging.

Author: George (via Claude)
Version: 1.0.0
"""

import numpy as np
from qtpy import QtWidgets, QtCore, QtGui
from scipy import ndimage
import flika
from distutils.version import StrictVersion
flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, WindowSelector, SliderLabel, CheckBox, ComboBox
else:
    from flika.utils.BaseProcess import BaseProcess, WindowSelector, SliderLabel, CheckBox, ComboBox

from flika import global_vars as g
from flika.window import Window




class Rotate_90(BaseProcess):
    """rotate_90(direction='clockwise', keepSourceWindow=False)

    Rotates the image stack by 90 degrees.

    Parameters:
        direction (str): 'clockwise' or 'counterclockwise'
        keepSourceWindow (bool): Keep the source window open

    Returns:
        newWindow: Window containing the rotated image
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()

        direction = ComboBox()
        direction.addItem('clockwise')
        direction.addItem('counterclockwise')

        self.items.append({'name': 'direction', 'string': 'Direction', 'object': direction})
        super().gui()

    def get_init_settings_dict(self):
        s = dict()
        s['direction'] = 'clockwise'
        return s

    def __call__(self, direction='clockwise', keepSourceWindow=False):
        self.start(keepSourceWindow)

        # Determine rotation direction
        k = 1 if direction == 'clockwise' else -1  # k=1 for CW (90°), k=-1 for CCW (270°)

        # Rotate each frame
        if self.tif.ndim == 3:
            self.newtif = np.array([np.rot90(frame, k=k) for frame in self.tif])
        else:
            self.newtif = np.rot90(self.tif, k=k)

        self.newname = f"{self.oldname} - Rotate 90° {direction}"
        return self.end()


class Rotate_Custom(BaseProcess):
    """rotate_custom(angle=0.0, reshape=True, order=1, keepSourceWindow=False)

    Rotates the image stack by a custom angle.

    Parameters:
        angle (float): Rotation angle in degrees (positive = counterclockwise)
        reshape (bool): Reshape output to hold entire rotated image
        order (int): Interpolation order (0=nearest, 1=linear, 3=cubic)
        keepSourceWindow (bool): Keep the source window open

    Returns:
        newWindow: Window containing the rotated image
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()

        angle = SliderLabel(0)
        angle.setRange(-180, 180)

        reshape = CheckBox()
        reshape.setChecked(True)

        order = ComboBox()
        order.addItem('0 - Nearest neighbor')
        order.addItem('1 - Linear')
        order.addItem('3 - Cubic')
        order.setCurrentIndex(1)

        self.items.append({'name': 'angle', 'string': 'Angle (degrees)', 'object': angle})
        self.items.append({'name': 'reshape', 'string': 'Reshape to fit', 'object': reshape})
        self.items.append({'name': 'order', 'string': 'Interpolation', 'object': order})
        super().gui()

    def get_init_settings_dict(self):
        s = dict()
        s['angle'] = 0.0
        s['reshape'] = True
        s['order'] = 1
        return s

    def __call__(self, angle=0.0, reshape=True, order=1, keepSourceWindow=False):
        self.start(keepSourceWindow)

        # Extract order value if it's a string
        if isinstance(order, str):
            order = int(order.split(' - ')[0])

        # Rotate each frame
        if self.tif.ndim == 3:
            self.newtif = np.array([
                ndimage.rotate(frame, angle, reshape=reshape, order=order, mode='constant', cval=0)
                for frame in self.tif
            ])
        else:
            self.newtif = ndimage.rotate(self.tif, angle, reshape=reshape, order=order, mode='constant', cval=0)

        self.newname = f"{self.oldname} - Rotated {angle}°"
        return self.end()


class Flip_Image(BaseProcess):
    """flip_image(direction='horizontal', keepSourceWindow=False)

    Flips the image stack horizontally or vertically.

    Parameters:
        direction (str): 'horizontal' or 'vertical'
        keepSourceWindow (bool): Keep the source window open

    Returns:
        newWindow: Window containing the flipped image
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()

        direction = ComboBox()
        direction.addItem('horizontal')
        direction.addItem('vertical')

        self.items.append({'name': 'direction', 'string': 'Flip Direction', 'object': direction})
        super().gui()

    def get_init_settings_dict(self):
        s = dict()
        s['direction'] = 'horizontal'
        return s

    def __call__(self, direction='horizontal', keepSourceWindow=False):
        self.start(keepSourceWindow)

        # Determine flip axis
        if direction == 'horizontal':
            axis = -1  # Flip along x-axis (left-right)
        else:
            axis = -2  # Flip along y-axis (up-down)

        self.newtif = np.flip(self.tif, axis=axis)

        self.newname = f"{self.oldname} - Flip {direction}"
        return self.end()


class Transpose_Image(BaseProcess):
    """transpose_image(keepSourceWindow=False)

    Transposes the image (swaps x and y axes).

    Parameters:
        keepSourceWindow (bool): Keep the source window open

    Returns:
        newWindow: Window containing the transposed image
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        super().gui()

    def get_init_settings_dict(self):
        return {}

    def __call__(self, keepSourceWindow=False):
        self.start(keepSourceWindow)

        # Transpose each frame
        if self.tif.ndim == 3:
            self.newtif = np.array([frame.T for frame in self.tif])
        else:
            self.newtif = self.tif.T

        self.newname = f"{self.oldname} - Transposed"
        return self.end()


class Invert_Intensity(BaseProcess):
    """invert_intensity(keepSourceWindow=False)

    Inverts the intensity values of the image (negative).

    Parameters:
        keepSourceWindow (bool): Keep the source window open

    Returns:
        newWindow: Window containing the inverted image
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        super().gui()

    def get_init_settings_dict(self):
        return {}

    def __call__(self, keepSourceWindow=False):
        self.start(keepSourceWindow)

        # Invert intensity
        max_val = np.iinfo(self.tif.dtype).max if np.issubdtype(self.tif.dtype, np.integer) else self.tif.max()
        self.newtif = max_val - self.tif

        self.newname = f"{self.oldname} - Inverted"
        return self.end()


class Normalize_Intensity(BaseProcess):
    """normalize_intensity(method='minmax', keepSourceWindow=False)

    Normalizes the intensity values of the image.

    Parameters:
        method (str): Normalization method ('minmax', 'zscore', 'percentile')
        percentile_low (float): Lower percentile for clipping (0-100)
        percentile_high (float): Upper percentile for clipping (0-100)
        keepSourceWindow (bool): Keep the source window open

    Returns:
        newWindow: Window containing the normalized image
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()

        method = ComboBox()
        method.addItem('minmax')
        method.addItem('zscore')
        method.addItem('percentile')

        percentile_low = SliderLabel(0)
        percentile_low.setRange(0, 50)

        percentile_high = SliderLabel(0)
        percentile_high.setRange(50, 100)
        percentile_high.setValue(100)

        self.items.append({'name': 'method', 'string': 'Normalization Method', 'object': method})
        self.items.append({'name': 'percentile_low', 'string': 'Lower Percentile', 'object': percentile_low})
        self.items.append({'name': 'percentile_high', 'string': 'Upper Percentile', 'object': percentile_high})
        super().gui()

    def get_init_settings_dict(self):
        s = dict()
        s['method'] = 'minmax'
        s['percentile_low'] = 0
        s['percentile_high'] = 100
        return s

    def __call__(self, method='minmax', percentile_low=0, percentile_high=100, keepSourceWindow=False):
        self.start(keepSourceWindow)

        img = self.tif.astype(np.float64)

        if method == 'minmax':
            # Min-max normalization to [0, 1]
            img_min = img.min()
            img_max = img.max()
            if img_max > img_min:
                normalized = (img - img_min) / (img_max - img_min)
            else:
                normalized = img

        elif method == 'zscore':
            # Z-score normalization
            mean = img.mean()
            std = img.std()
            if std > 0:
                normalized = (img - mean) / std
            else:
                normalized = img - mean

        elif method == 'percentile':
            # Percentile-based normalization
            p_low = np.percentile(img, percentile_low)
            p_high = np.percentile(img, percentile_high)
            img_clipped = np.clip(img, p_low, p_high)
            if p_high > p_low:
                normalized = (img_clipped - p_low) / (p_high - p_low)
            else:
                normalized = img_clipped

        # Convert back to original dtype range
        if np.issubdtype(self.tif.dtype, np.integer):
            max_val = np.iinfo(self.tif.dtype).max
            self.newtif = (normalized * max_val).astype(self.tif.dtype)
        else:
            self.newtif = normalized.astype(self.tif.dtype)

        self.newname = f"{self.oldname} - Normalized ({method})"
        return self.end()


class Crop_To_Square(BaseProcess):
    """crop_to_square(keepSourceWindow=False)

    Crops the image to the largest centered square.
    Useful for preparing images for certain analysis pipelines.

    Parameters:
        keepSourceWindow (bool): Keep the source window open

    Returns:
        newWindow: Window containing the cropped image
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        super().gui()

    def get_init_settings_dict(self):
        return {}

    def __call__(self, keepSourceWindow=False):
        self.start(keepSourceWindow)

        if self.tif.ndim == 3:
            t, h, w = self.tif.shape
        else:
            h, w = self.tif.shape

        # Calculate crop parameters
        size = min(h, w)
        y_start = (h - size) // 2
        x_start = (w - size) // 2

        # Crop
        if self.tif.ndim == 3:
            self.newtif = self.tif[:, y_start:y_start+size, x_start:x_start+size]
        else:
            self.newtif = self.tif[y_start:y_start+size, x_start:x_start+size]

        self.newname = f"{self.oldname} - Cropped to Square"
        return self.end()


class Bin_Pixels(BaseProcess):
    """bin_pixels(bin_factor=2, method='mean', keepSourceWindow=False)

    Bins (downsamples) pixels by averaging or summing groups.
    Useful for reducing noise or file size.

    Parameters:
        bin_factor (int): Number of pixels to bin (2 = 2x2 binning)
        method (str): Binning method ('mean' or 'sum')
        keepSourceWindow (bool): Keep the source window open

    Returns:
        newWindow: Window containing the binned image
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()

        bin_factor = SliderLabel(0)
        bin_factor.setRange(2, 8)
        bin_factor.setValue(2)

        method = ComboBox()
        method.addItem('mean')
        method.addItem('sum')

        self.items.append({'name': 'bin_factor', 'string': 'Bin Factor', 'object': bin_factor})
        self.items.append({'name': 'method', 'string': 'Binning Method', 'object': method})
        super().gui()

    def get_init_settings_dict(self):
        s = dict()
        s['bin_factor'] = 2
        s['method'] = 'mean'
        return s

    def __call__(self, bin_factor=2, method='mean', keepSourceWindow=False):
        self.start(keepSourceWindow)

        bin_factor = int(bin_factor)

        if self.tif.ndim == 3:
            t, h, w = self.tif.shape
            new_h = h // bin_factor
            new_w = w // bin_factor

            # Trim to multiple of bin_factor
            h_trim = new_h * bin_factor
            w_trim = new_w * bin_factor
            trimmed = self.tif[:, :h_trim, :w_trim]

            # Reshape and bin
            reshaped = trimmed.reshape(t, new_h, bin_factor, new_w, bin_factor)
            if method == 'mean':
                self.newtif = reshaped.mean(axis=(2, 4))
            else:
                self.newtif = reshaped.sum(axis=(2, 4))
        else:
            h, w = self.tif.shape
            new_h = h // bin_factor
            new_w = w // bin_factor

            h_trim = new_h * bin_factor
            w_trim = new_w * bin_factor
            trimmed = self.tif[:h_trim, :w_trim]

            reshaped = trimmed.reshape(new_h, bin_factor, new_w, bin_factor)
            if method == 'mean':
                self.newtif = reshaped.mean(axis=(1, 3))
            else:
                self.newtif = reshaped.sum(axis=(1, 3))

        # Preserve dtype for sum, convert for mean
        if method == 'sum':
            self.newtif = self.newtif.astype(self.tif.dtype)
        else:
            self.newtif = self.newtif.astype(self.tif.dtype)

        self.newname = f"{self.oldname} - Binned {bin_factor}x{bin_factor} ({method})"
        return self.end()


class Pad_Image(BaseProcess):
    """pad_image(pad_width=10, mode='constant', constant_value=0, keepSourceWindow=False)

    Adds padding around the image.

    Parameters:
        pad_width (int): Number of pixels to pad on each side
        mode (str): Padding mode ('constant', 'edge', 'reflect', 'symmetric')
        constant_value (float): Value for constant padding
        keepSourceWindow (bool): Keep the source window open

    Returns:
        newWindow: Window containing the padded image
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()

        pad_width = SliderLabel(0)
        pad_width.setRange(1, 100)
        pad_width.setValue(10)

        mode = ComboBox()
        mode.addItem('constant')
        mode.addItem('edge')
        mode.addItem('reflect')
        mode.addItem('symmetric')

        constant_value = SliderLabel(0)
        constant_value.setRange(0, 65535)
        constant_value.setValue(0)

        self.items.append({'name': 'pad_width', 'string': 'Pad Width (pixels)', 'object': pad_width})
        self.items.append({'name': 'mode', 'string': 'Padding Mode', 'object': mode})
        self.items.append({'name': 'constant_value', 'string': 'Constant Value', 'object': constant_value})
        super().gui()

    def get_init_settings_dict(self):
        s = dict()
        s['pad_width'] = 10
        s['mode'] = 'constant'
        s['constant_value'] = 0
        return s

    def __call__(self, pad_width=10, mode='constant', constant_value=0, keepSourceWindow=False):
        self.start(keepSourceWindow)

        pad_width = int(pad_width)

        # Set up padding parameters
        if self.tif.ndim == 3:
            pad_params = ((0, 0), (pad_width, pad_width), (pad_width, pad_width))
        else:
            pad_params = ((pad_width, pad_width), (pad_width, pad_width))

        # Apply padding
        if mode == 'constant':
            self.newtif = np.pad(self.tif, pad_params, mode=mode, constant_values=constant_value)
        else:
            self.newtif = np.pad(self.tif, pad_params, mode=mode)

        self.newname = f"{self.oldname} - Padded ({mode})"
        return self.end()


# Create instances
rotate_90 = Rotate_90()
rotate_custom = Rotate_Custom()
flip_image = Flip_Image()
transpose_image = Transpose_Image()
invert_intensity = Invert_Intensity()
normalize_intensity = Normalize_Intensity()
crop_to_square = Crop_To_Square()
bin_pixels = Bin_Pixels()
pad_image = Pad_Image()


def launch_docs():
    """Open the plugin documentation"""
    url = 'https://github.com/flika-org/flika'
    QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))
