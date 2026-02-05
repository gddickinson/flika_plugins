"""
Calcium Noise Analysis Plugin for FLIKA
========================================
Comprehensive calcium imaging noise analysis toolkit integrating traditional signal
processing algorithms, noise characterization, and event detection methods.

Based on research from:
- CASCADE (Calibrated spike inference from calcium imaging)
- Swaminathan et al. 2020 - Power spectral density analysis
- Lock & Parker 2020 - SD fluctuation analysis (FLIKA)
- Suite2p baseline estimation methods

Author: George
Version: 2.0.0
"""

import numpy as np
from qtpy import QtWidgets, QtCore, QtGui
from scipy import ndimage, signal
from scipy.ndimage import gaussian_filter, percentile_filter, minimum_filter, maximum_filter, median_filter, uniform_filter, label
from scipy.signal import welch, butter, sosfiltfilt, filtfilt
from scipy.optimize import curve_fit
import flika
from distutils.version import StrictVersion

flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, WindowSelector, SliderLabel, CheckBox, ComboBox
else:
    from flika.utils.BaseProcess import BaseProcess, WindowSelector, SliderLabel, CheckBox, ComboBox

from flika import global_vars as g
from flika.window import Window

# Import core modular analysis functions
from . import power_spectrum_module
from . import fluctuation_module
from . import shot_noise_module
from . import baseline_module
from . import event_detection_module
from . import correlation_module

# Import advanced analysis modules
from . import advanced_correlation_module
from . import psm_crm_module
from . import flux_analysis_module
from . import fitting_module

# Import spatial analysis modules
from . import activity_detection_module
from . import spatial_analysis_module

# Import temporal analysis modules (new)
from . import temporal_fluctuation_module
from . import event_kinetics_module
from . import oscillation_analysis_module

# Import wave propagation module (new)
from . import wave_propagation_module

# Import quality control module (new)
from . import quality_control_module

# Import batch comparison module (new)
from . import batch_comparison_module


class Power_Spectrum_Map(BaseProcess):
    """power_spectrum_map(low_freq_min=0.1, low_freq_max=5.0, high_freq_cutoff=50.0,
                          nperseg=256, keepSourceWindow=False)

    Generate 2D map identifying Ca²⁺-active pixels via power spectral density analysis.
    Separates calcium signals (0.1-5 Hz) from high-frequency shot noise (>50 Hz).

    Based on Swaminathan et al. 2020 methodology.

    Parameters:
        low_freq_min (float): Lower bound of calcium signal frequency band (Hz)
        low_freq_max (float): Upper bound of calcium signal frequency band (Hz)
        high_freq_cutoff (float): Frequency above which to measure shot noise (Hz)
        nperseg (int): Length of each segment for Welch's method (power of 2)
        keepSourceWindow (bool): Keep the source window open

    Returns:
        newWindow: Window containing the power spectrum map
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()

        low_freq_min = SliderLabel(0)
        low_freq_min.setRange(1, 50)
        low_freq_min.setValue(1)

        low_freq_max = SliderLabel(0)
        low_freq_max.setRange(1, 100)
        low_freq_max.setValue(50)

        high_freq_cutoff = SliderLabel(0)
        high_freq_cutoff.setRange(10, 200)
        high_freq_cutoff.setValue(50)

        nperseg = ComboBox()
        nperseg.addItem('64')
        nperseg.addItem('128')
        nperseg.addItem('256')
        nperseg.addItem('512')
        nperseg.addItem('1024')
        nperseg.setCurrentIndex(2)  # Default to 256

        self.items.append({'name': 'low_freq_min', 'string': 'Low Freq Min (Hz)', 'object': low_freq_min})
        self.items.append({'name': 'low_freq_max', 'string': 'Low Freq Max (Hz)', 'object': low_freq_max})
        self.items.append({'name': 'high_freq_cutoff', 'string': 'High Freq Cutoff (Hz)', 'object': high_freq_cutoff})
        self.items.append({'name': 'nperseg', 'string': 'Segment Length', 'object': nperseg})
        super().gui()

    def get_init_settings_dict(self):
        s = dict()
        s['low_freq_min'] = 1
        s['low_freq_max'] = 50
        s['high_freq_cutoff'] = 50
        s['nperseg'] = 256
        return s

    def __call__(self, low_freq_min=1, low_freq_max=50, high_freq_cutoff=50,
                 nperseg=256, keepSourceWindow=False):
        self.start(keepSourceWindow)

        # Convert slider integers to float frequencies (divide by 10)
        low_freq_min = float(low_freq_min) / 10.0
        low_freq_max = float(low_freq_max) / 10.0

        # Get framerate from current window
        if hasattr(g.win, 'framerate') and g.win.framerate is not None:
            fs = g.win.framerate
        else:
            fs = 30.0
            g.m.statusBar().showMessage(f'Using default framerate: {fs} Hz', 5000)

        # Convert nperseg if it's a string
        if isinstance(nperseg, str):
            nperseg = int(nperseg)

        # Compute power spectrum map
        psm = power_spectrum_module.compute_power_spectrum_map(
            self.tif, fs,
            low_freq=(low_freq_min, low_freq_max),
            high_freq=high_freq_cutoff,
            nperseg=nperseg
        )

        # Convert to displayable format
        self.newtif = psm.astype(np.float32)
        self.newname = f"{self.oldname} - Power Spectrum Map"

        return self.end()


class Fluctuation_Analysis(BaseProcess):
    """fluctuation_analysis(spatial_sigma=1.0, highpass_cutoff=0.5, window_size=30,
                           shot_noise_factor=None, keepSourceWindow=False)

    FLIKA-style SD fluctuation analysis for detecting local Ca²⁺ signals.
    Applies spatial smoothing, temporal high-pass filtering, computes running variance,
    and optionally corrects for shot noise.

    Based on Lock & Parker 2020 methodology.

    Parameters:
        spatial_sigma (float): Gaussian blur sigma for spatial smoothing
        highpass_cutoff (float): High-pass filter cutoff frequency (Hz)
        window_size (int): Window size for running variance (frames)
        shot_noise_factor (float): Factor for shot noise correction (None to disable)
        keepSourceWindow (bool): Keep the source window open

    Returns:
        newWindow: Window containing the fluctuation analysis result (SD map)
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()

        spatial_sigma = SliderLabel(0)
        spatial_sigma.setRange(1, 100)
        spatial_sigma.setValue(10)

        highpass_cutoff = SliderLabel(0)
        highpass_cutoff.setRange(1, 100)
        highpass_cutoff.setValue(5)

        window_size = SliderLabel(0)
        window_size.setRange(5, 100)
        window_size.setValue(30)

        shot_noise_factor = SliderLabel(0)
        shot_noise_factor.setRange(0, 2000)
        shot_noise_factor.setValue(0)

        apply_shot_noise = CheckBox()
        apply_shot_noise.setChecked(False)

        self.items.append({'name': 'spatial_sigma', 'string': 'Spatial Sigma', 'object': spatial_sigma})
        self.items.append({'name': 'highpass_cutoff', 'string': 'Highpass Cutoff (Hz)', 'object': highpass_cutoff})
        self.items.append({'name': 'window_size', 'string': 'Window Size (frames)', 'object': window_size})
        self.items.append({'name': 'apply_shot_noise', 'string': 'Apply Shot Noise Correction', 'object': apply_shot_noise})
        self.items.append({'name': 'shot_noise_factor', 'string': 'Shot Noise Factor', 'object': shot_noise_factor})
        super().gui()

    def get_init_settings_dict(self):
        s = dict()
        s['spatial_sigma'] = 10
        s['highpass_cutoff'] = 5
        s['window_size'] = 30
        s['apply_shot_noise'] = False
        s['shot_noise_factor'] = 0
        return s

    def __call__(self, spatial_sigma=10, highpass_cutoff=5, window_size=30,
                 apply_shot_noise=False, shot_noise_factor=0, keepSourceWindow=False):
        self.start(keepSourceWindow)

        # Convert slider integers to floats
        spatial_sigma = float(spatial_sigma) / 10.0
        highpass_cutoff = float(highpass_cutoff) / 10.0
        shot_noise_factor = float(shot_noise_factor) / 1000.0

        # Get framerate from current window
        if hasattr(g.win, 'framerate') and g.win.framerate is not None:
            fs = g.win.framerate
        else:
            fs = 30.0
            g.m.statusBar().showMessage(f'Using default framerate: {fs} Hz', 5000)

        # Convert window size to int
        window_size = int(window_size)

        # Apply shot noise factor if enabled
        shot_noise = shot_noise_factor if apply_shot_noise else None

        # Compute fluctuation analysis
        sd_stack = fluctuation_module.fluctuation_analysis(
            self.tif, fs,
            spatial_sigma=spatial_sigma,
            highpass_cutoff=highpass_cutoff,
            window_size=window_size,
            shot_noise_factor=shot_noise
        )

        # Convert to displayable format
        self.newtif = sd_stack.astype(np.float32)
        self.newname = f"{self.oldname} - SD Fluctuation"

        return self.end()


class Anscombe_Transform(BaseProcess):
    """anscombe_transform(gain=1.0, offset=0.0, read_noise=0.0, keepSourceWindow=False)

    Apply Anscombe variance-stabilizing transform for Poisson-distributed photon counts.

    This transform makes the variance approximately constant and Gaussian, enabling
    standard signal processing methods on photon-counting data.

    Parameters:
        gain (float): Camera gain (electrons/ADU)
        offset (float): Camera offset/bias (ADU)
        read_noise (float): Read noise standard deviation (electrons)
        keepSourceWindow (bool): Keep the source window open

    Returns:
        newWindow: Window containing Anscombe-transformed data
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()

        gain = SliderLabel(0)
        gain.setRange(1, 100)
        gain.setValue(10)

        offset = SliderLabel(0)
        offset.setRange(0, 500)
        offset.setValue(100)

        read_noise = SliderLabel(0)
        read_noise.setRange(0, 100)
        read_noise.setValue(10)

        self.items.append({'name': 'gain', 'string': 'Gain (e-/ADU)', 'object': gain})
        self.items.append({'name': 'offset', 'string': 'Offset (ADU)', 'object': offset})
        self.items.append({'name': 'read_noise', 'string': 'Read Noise (e-)', 'object': read_noise})
        super().gui()

    def get_init_settings_dict(self):
        s = dict()
        s['gain'] = 10
        s['offset'] = 100
        s['read_noise'] = 10
        return s

    def __call__(self, gain=10, offset=100, read_noise=10, keepSourceWindow=False):
        self.start(keepSourceWindow)

        gain = float(gain) / 10.0
        offset = float(offset)
        read_noise = float(read_noise) / 10.0

        # Apply Anscombe transform
        transformed = shot_noise_module.anscombe_transform(
            self.tif.astype(np.float64),
            gain=gain,
            offset=offset,
            read_noise=read_noise
        )

        self.newtif = transformed.astype(np.float32)
        self.newname = f"{self.oldname} - Anscombe"

        return self.end()


class Estimate_Baseline_F0(BaseProcess):
    """estimate_baseline_f0(method='percentile', window=300, percentile=8,
                           keepSourceWindow=False)

    Estimate baseline fluorescence (F0) for each pixel using various methods.

    Parameters:
        method (str): 'percentile', 'mode', 'minimum_filter', or 'median'
        window (int): Window size for local estimation (frames)
        percentile (float): Percentile value for percentile method (0-100)
        keepSourceWindow (bool): Keep the source window open

    Returns:
        newWindow: Window containing estimated F0 values
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()

        method = ComboBox()
        method.addItem('percentile')
        method.addItem('mode')
        method.addItem('minimum_filter')
        method.addItem('median')
        method.setCurrentIndex(0)

        window = SliderLabel(0)
        window.setRange(50, 1000)
        window.setValue(300)

        percentile = SliderLabel(0)
        percentile.setRange(1, 50)
        percentile.setValue(8)

        self.items.append({'name': 'method', 'string': 'Estimation Method', 'object': method})
        self.items.append({'name': 'window', 'string': 'Window Size (frames)', 'object': window})
        self.items.append({'name': 'percentile', 'string': 'Percentile', 'object': percentile})
        super().gui()

    def get_init_settings_dict(self):
        s = dict()
        s['method'] = 'percentile'
        s['window'] = 300
        s['percentile'] = 8
        return s

    def __call__(self, method='percentile', window=300, percentile=8, keepSourceWindow=False):
        self.start(keepSourceWindow)

        window = int(window)
        percentile = int(percentile)

        if self.tif.ndim == 3:
            T, H, W = self.tif.shape
            f0_map = np.zeros((H, W), dtype=np.float32)

            for i in range(H):
                for j in range(W):
                    trace = self.tif[:, i, j].astype(np.float64)

                    if method == 'percentile':
                        f0_map[i, j] = baseline_module.estimate_f0_percentile(trace, window, percentile)
                    elif method == 'mode':
                        f0_map[i, j] = baseline_module.estimate_f0_mode(trace, window)
                    elif method == 'minimum_filter':
                        f0_map[i, j] = baseline_module.estimate_f0_minimum_filter(trace, window)
                    elif method == 'median':
                        f0_map[i, j] = baseline_module.estimate_f0_median(trace, window)

                if i % 10 == 0:
                    g.m.statusBar().showMessage(f'Processing row {i+1}/{H}', 1000)

            self.newtif = f0_map
            self.newname = f"{self.oldname} - F0 ({method})"
        else:
            g.m.statusBar().showMessage('F0 estimation requires time series data', 5000)
            return None

        return self.end()


class Compute_DFF(BaseProcess):
    """compute_dff(f0_method='percentile', window=300, percentile=8, epsilon=1.0,
                   keepSourceWindow=False)

    Compute normalized fluorescence change (ΔF/F₀) for each pixel.

    Parameters:
        f0_method (str): Method for F0 estimation
        window (int): Window size for F0 estimation (frames)
        percentile (float): Percentile value for percentile method
        epsilon (float): Small constant to prevent division by zero
        keepSourceWindow (bool): Keep the source window open

    Returns:
        newWindow: Window containing ΔF/F₀ values
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()

        f0_method = ComboBox()
        f0_method.addItem('percentile')
        f0_method.addItem('mode')
        f0_method.addItem('minimum_filter')
        f0_method.addItem('median')
        f0_method.setCurrentIndex(0)

        window = SliderLabel(0)
        window.setRange(50, 1000)
        window.setValue(300)

        percentile = SliderLabel(0)
        percentile.setRange(1, 50)
        percentile.setValue(8)

        epsilon = SliderLabel(0)
        epsilon.setRange(1, 100)
        epsilon.setValue(10)

        self.items.append({'name': 'f0_method', 'string': 'F0 Method', 'object': f0_method})
        self.items.append({'name': 'window', 'string': 'Window Size (frames)', 'object': window})
        self.items.append({'name': 'percentile', 'string': 'Percentile', 'object': percentile})
        self.items.append({'name': 'epsilon', 'string': 'Epsilon', 'object': epsilon})
        super().gui()

    def get_init_settings_dict(self):
        s = dict()
        s['f0_method'] = 'percentile'
        s['window'] = 300
        s['percentile'] = 8
        s['epsilon'] = 10
        return s

    def __call__(self, f0_method='percentile', window=300, percentile=8, epsilon=10,
                 keepSourceWindow=False):
        self.start(keepSourceWindow)

        window = int(window)
        percentile = int(percentile)
        epsilon = float(epsilon) / 10.0

        if self.tif.ndim == 3:
            T, H, W = self.tif.shape
            dff_stack = np.zeros_like(self.tif, dtype=np.float64)

            for i in range(H):
                for j in range(W):
                    trace = self.tif[:, i, j].astype(np.float64)

                    if f0_method == 'percentile':
                        f0 = baseline_module.estimate_f0_percentile(trace, window, percentile)
                    elif f0_method == 'mode':
                        f0 = baseline_module.estimate_f0_mode(trace, window)
                    elif f0_method == 'minimum_filter':
                        f0 = baseline_module.estimate_f0_minimum_filter(trace, window)
                    elif f0_method == 'median':
                        f0 = baseline_module.estimate_f0_median(trace, window)

                    dff_stack[:, i, j] = baseline_module.compute_dff(trace, f0, epsilon)

                if i % 10 == 0:
                    g.m.statusBar().showMessage(f'Processing row {i+1}/{H}', 1000)

            self.newtif = dff_stack.astype(np.float32)
            self.newname = f"{self.oldname} - dF/F"
        else:
            g.m.statusBar().showMessage('dF/F computation requires time series data', 5000)
            return None

        return self.end()


class Detect_Calcium_Sparks(BaseProcess):
    """detect_calcium_sparks(detection_method='template', amplitude_threshold=3.0,
                            template_width=10, keepSourceWindow=False)

    Detect calcium sparks/puffs using threshold-based detection.

    First estimates baseline F0, then detects events using dual thresholds
    for intensity and peak confirmation.

    Parameters:
        detection_method (str): 'template' (uses detect_sparks) or 'threshold' (simple ΔF/F)
        amplitude_threshold (float): Threshold in standard deviations
        template_width (int): Minimum event size (pixels) for template method
        keepSourceWindow (bool): Keep the source window open

    Returns:
        newWindow: Window containing detection mask (2D image showing event locations)
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()

        detection_method = ComboBox()
        detection_method.addItem('template')
        detection_method.addItem('threshold')
        detection_method.setCurrentIndex(0)

        amplitude_threshold = SliderLabel(0)
        amplitude_threshold.setRange(10, 100)
        amplitude_threshold.setValue(30)

        template_width = SliderLabel(0)
        template_width.setRange(3, 50)
        template_width.setValue(10)

        self.items.append({'name': 'detection_method', 'string': 'Detection Method', 'object': detection_method})
        self.items.append({'name': 'amplitude_threshold', 'string': 'Amplitude Threshold (SD)', 'object': amplitude_threshold})
        self.items.append({'name': 'template_width', 'string': 'Template Width (frames)', 'object': template_width})
        super().gui()

    def get_init_settings_dict(self):
        s = dict()
        s['detection_method'] = 'template'
        s['amplitude_threshold'] = 30
        s['template_width'] = 10
        return s

    def __call__(self, detection_method='template', amplitude_threshold=30, template_width=10,
                 keepSourceWindow=False):
        self.start(keepSourceWindow)

        amplitude_threshold = float(amplitude_threshold) / 10.0
        template_width = int(template_width)

        if self.tif.ndim != 3:
            g.m.statusBar().showMessage('Event detection requires time series data', 5000)
            return None

        T, H, W = self.tif.shape

        # Estimate F0 first (using global histogram method for single scalar per pixel)
        g.m.statusBar().showMessage('Estimating baseline F0...', 2000)
        f0 = np.zeros((H, W))
        for i in range(H):
            for j in range(W):
                trace = self.tif[:, i, j].astype(np.float64)
                # Use histogram method which returns a single scalar
                f0[i, j] = baseline_module.estimate_f0_from_histogram(trace, percentile=10)

            # Progress update every 10 rows
            if i % 10 == 0:
                g.m.statusBar().showMessage(f'Estimating F0... {i+1}/{H} rows', 1000)

        # Detect sparks using actual function
        g.m.statusBar().showMessage('Detecting calcium events...', 2000)

        if detection_method == 'template':
            # Use detect_sparks with threshold parameters
            sparks = event_detection_module.detect_sparks(
                self.tif, f0,
                intensity_thresh=amplitude_threshold,
                peak_thresh=amplitude_threshold * 1.5,
                min_size=int(template_width)
            )

            # Convert spark list to binary mask
            detection_mask = np.zeros((H, W), dtype=np.float32)
            for spark_mask in sparks:
                if spark_mask.ndim == 2:
                    detection_mask += spark_mask.astype(np.float32)

            # Normalize
            if detection_mask.max() > 0:
                detection_mask = detection_mask / detection_mask.max()

            n_sparks = len(sparks)
            g.m.statusBar().showMessage(f'Detected {n_sparks} calcium events', 8000)

        else:
            # Simple threshold detection on ΔF/F
            dff_stack = np.zeros_like(self.tif, dtype=np.float64)
            for i in range(H):
                for j in range(W):
                    trace = self.tif[:, i, j].astype(np.float64)
                    dff_stack[:, i, j] = baseline_module.compute_dff(trace, f0[i, j], epsilon=1.0)

            # Threshold
            detection_mask = (np.max(dff_stack, axis=0) > amplitude_threshold).astype(np.float32)
            n_pixels = np.sum(detection_mask)
            g.m.statusBar().showMessage(f'Detected {n_pixels} active pixels', 8000)

        self.newtif = detection_mask.astype(np.float32)
        self.newname = f"{self.oldname} - Event Detection"

        return self.end()


class Local_Correlation_Image(BaseProcess):
    """local_correlation_image(neighborhood=8, keepSourceWindow=False)

    Compute local correlation image showing pixel similarity to neighbors.

    Useful for identifying synchronized calcium activity and detecting artifacts.

    Parameters:
        neighborhood (int): Size of neighborhood for correlation (pixels)
        keepSourceWindow (bool): Keep the source window open

    Returns:
        newWindow: Window containing correlation image
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()

        neighborhood = SliderLabel(0)
        neighborhood.setRange(3, 20)
        neighborhood.setValue(8)

        self.items.append({'name': 'neighborhood', 'string': 'Neighborhood Size', 'object': neighborhood})
        super().gui()

    def get_init_settings_dict(self):
        s = dict()
        s['neighborhood'] = 8
        return s

    def __call__(self, neighborhood=8, keepSourceWindow=False):
        self.start(keepSourceWindow)

        neighborhood = int(neighborhood)

        if self.tif.ndim != 3:
            g.m.statusBar().showMessage('Local correlation requires time series data', 5000)
            return None

        corr_img = correlation_module.local_correlation_image(self.tif, neighborhood=neighborhood)

        self.newtif = corr_img.astype(np.float32)
        self.newname = f"{self.oldname} - Correlation Image"

        return self.end()


class Compute_Noise_Metric(BaseProcess):
    """compute_noise_metric(baseline_frames=100, keepSourceWindow=False)

    Compute CASCADE noise metric: ν = σ_ΔF/F × √(frame_rate)

    This standardizes noise comparisons across recording conditions.
    Typical values range from ~1 (low noise) to ~8-9 (high noise).

    Parameters:
        baseline_frames (int): Number of frames to use for baseline estimation
        keepSourceWindow (bool): Keep the source window open

    Returns:
        newWindow: Window containing noise metric map (ν value per pixel)
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()

        baseline_frames = SliderLabel(0)
        baseline_frames.setRange(50, 1000)
        baseline_frames.setValue(100)

        self.items.append({'name': 'baseline_frames', 'string': 'Baseline Frames', 'object': baseline_frames})
        super().gui()

    def get_init_settings_dict(self):
        s = dict()
        s['baseline_frames'] = 100
        return s

    def __call__(self, baseline_frames=100, keepSourceWindow=False):
        self.start(keepSourceWindow)

        baseline_frames = int(baseline_frames)

        if hasattr(g.win, 'framerate') and g.win.framerate is not None:
            fs = g.win.framerate
        else:
            fs = 30.0
            g.m.statusBar().showMessage(f'Using default framerate: {fs} Hz', 5000)

        if self.tif.ndim != 3:
            g.m.statusBar().showMessage('Noise metric requires time series data', 5000)
            return None

        T, H, W = self.tif.shape
        noise_map = np.zeros((H, W), dtype=np.float32)

        for i in range(H):
            for j in range(W):
                trace = self.tif[:, i, j].astype(np.float64)

                f0 = baseline_module.estimate_f0_percentile(trace, window=baseline_frames, percentile=8)
                dff = baseline_module.compute_dff(trace, f0, epsilon=1.0)

                sigma_dff = np.std(dff)
                noise_metric = sigma_dff * np.sqrt(fs)
                noise_map[i, j] = noise_metric

            if i % 10 == 0:
                g.m.statusBar().showMessage(f'Processing row {i+1}/{H}', 1000)

        mean_noise = np.mean(noise_map)
        median_noise = np.median(noise_map)
        g.m.statusBar().showMessage(
            f'Noise metric: mean={mean_noise:.2f}, median={median_noise:.2f}',
            10000
        )

        self.newtif = noise_map
        self.newname = f"{self.oldname} - Noise Metric (ν)"

        return self.end()


class Butterworth_Bandpass(BaseProcess):
    """butterworth_bandpass(low_freq=0.1, high_freq=5.0, order=2, keepSourceWindow=False)

    Apply Butterworth bandpass filter to isolate calcium transient frequencies.

    Typical calcium signal frequencies are 0.1-5 Hz. This filter removes both
    low-frequency baseline drift and high-frequency shot noise.

    Parameters:
        low_freq (float): Low frequency cutoff (Hz)
        high_freq (float): High frequency cutoff (Hz)
        order (int): Filter order (higher = steeper cutoff)
        keepSourceWindow (bool): Keep the source window open

    Returns:
        newWindow: Window containing bandpass filtered data
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()

        low_freq = SliderLabel(0)
        low_freq.setRange(1, 1000)
        low_freq.setValue(10)

        high_freq = SliderLabel(0)
        high_freq.setRange(1, 500)
        high_freq.setValue(50)

        order = SliderLabel(0)
        order.setRange(1, 10)
        order.setValue(2)

        self.items.append({'name': 'low_freq', 'string': 'Low Freq (Hz)', 'object': low_freq})
        self.items.append({'name': 'high_freq', 'string': 'High Freq (Hz)', 'object': high_freq})
        self.items.append({'name': 'order', 'string': 'Filter Order', 'object': order})
        super().gui()

    def get_init_settings_dict(self):
        s = dict()
        s['low_freq'] = 10
        s['high_freq'] = 50
        s['order'] = 2
        return s

    def __call__(self, low_freq=10, high_freq=50, order=2, keepSourceWindow=False):
        self.start(keepSourceWindow)

        low_freq = float(low_freq) / 100.0
        high_freq = float(high_freq) / 10.0
        order = int(order)

        if hasattr(g.win, 'framerate') and g.win.framerate is not None:
            fs = g.win.framerate
        else:
            fs = 30.0
            g.m.statusBar().showMessage(f'Using default framerate: {fs} Hz', 5000)

        if self.tif.ndim == 3:
            T, H, W = self.tif.shape
            filtered = np.zeros_like(self.tif, dtype=np.float64)

            for i in range(H):
                for j in range(W):
                    trace = self.tif[:, i, j].astype(np.float64)
                    filtered[:, i, j] = power_spectrum_module.butterworth_bandpass(
                        trace, fs, low=low_freq, high=high_freq, order=order
                    )

                if i % 10 == 0:
                    g.m.statusBar().showMessage(f'Processing row {i+1}/{H}', 1000)

            self.newtif = filtered.astype(self.tif.dtype)
        else:
            g.m.statusBar().showMessage('Bandpass filter requires time series data', 5000)
            return None

        self.newname = f"{self.oldname} - Bandpass {low_freq}-{high_freq} Hz"

        return self.end()


# ============================================================================
# NEW ADVANCED PROCESSES (v2.0)
# ============================================================================

class Generate_PSM(BaseProcess):
    """generate_psm(eta_threshold=0.5, nperseg=256, keepSourceWindow=False)

    Generate Power Spectrum Map (PSM) showing η values (Swaminathan et al. 2020).

    η = (signal band power) / (noise band power)
    Highlights pixels with strong Ca²⁺ signal relative to noise.

    Parameters:
        eta_threshold (float): Display threshold for η values
        nperseg (int): Welch segment length
        keepSourceWindow (bool): Keep the source window open

    Returns:
        newWindow: Window containing PSM (η values)
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()

        eta_threshold = SliderLabel(0)
        eta_threshold.setRange(0, 100)
        eta_threshold.setValue(5)

        nperseg = ComboBox()
        nperseg.addItem('128')
        nperseg.addItem('256')
        nperseg.addItem('512')
        nperseg.setCurrentIndex(1)

        self.items.append({'name': 'eta_threshold', 'string': 'η Threshold', 'object': eta_threshold})
        self.items.append({'name': 'nperseg', 'string': 'Segment Length', 'object': nperseg})
        super().gui()

    def get_init_settings_dict(self):
        return {'eta_threshold': 5, 'nperseg': 256}

    def __call__(self, eta_threshold=5, nperseg=256, keepSourceWindow=False):
        self.start(keepSourceWindow)

        eta_threshold = float(eta_threshold) / 10.0
        if isinstance(nperseg, str):
            nperseg = int(nperseg)

        if hasattr(g.win, 'framerate') and g.win.framerate is not None:
            fs = g.win.framerate
        else:
            fs = 30.0
            g.m.statusBar().showMessage(f'Using default framerate: {fs} Hz', 5000)

        # Generate PSM (returns list of PSMs)
        T = self.tif.shape[0]
        g.m.statusBar().showMessage(f'Computing PSM for {T} frames with subsection_length={nperseg}...', 3000)

        # Check if stack is long enough
        if T < nperseg:
            g.m.statusBar().showMessage(
                f'Stack too short! Need at least {nperseg} frames, have {T}. Reduce segment length.',
                10000
            )
            return None

        psm_list = psm_crm_module.compute_power_spectrum_map(self.tif, fs, subsection_length=nperseg)

        # Compute mean PSM from list
        if len(psm_list) == 0:
            g.m.statusBar().showMessage('No PSM subsections generated - check stack length', 10000)
            return None

        g.m.statusBar().showMessage(f'Averaging {len(psm_list)} PSM subsections...', 2000)
        psm = psm_crm_module.mean_power_spectrum_map(psm_list)

        # Apply threshold for display
        psm_display = np.where(psm >= eta_threshold, psm, 0)

        # Report statistics
        n_active = np.sum(psm_display > 0)
        mean_eta = np.mean(psm[psm > 0]) if np.any(psm > 0) else 0
        g.m.statusBar().showMessage(
            f'PSM: {n_active} active pixels, mean η={mean_eta:.2f}',
            8000
        )

        self.newtif = psm_display.astype(np.float32)
        self.newname = f"{self.oldname} - PSM (η)"

        return self.end()


class Generate_CRM(BaseProcess):
    """generate_crm(xi_threshold=0.3, lag_frames=10, keepSourceWindow=False)

    Generate Correlation Map (CRM) showing ξ values (Swaminathan et al. 2020).

    ξ = cross-correlation coefficient with neighbors
    Highlights pixels with synchronized Ca²⁺ activity.

    Parameters:
        xi_threshold (float): Display threshold for ξ values
        lag_frames (int): Maximum lag for correlation
        keepSourceWindow (bool): Keep the source window open

    Returns:
        newWindow: Window containing CRM (ξ values)
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()

        xi_threshold = SliderLabel(0)
        xi_threshold.setRange(0, 100)
        xi_threshold.setValue(30)

        lag_frames = SliderLabel(0)
        lag_frames.setRange(1, 50)
        lag_frames.setValue(10)

        self.items.append({'name': 'xi_threshold', 'string': 'ξ Threshold', 'object': xi_threshold})
        self.items.append({'name': 'lag_frames', 'string': 'Max Lag (frames)', 'object': lag_frames})
        super().gui()

    def get_init_settings_dict(self):
        return {'xi_threshold': 30, 'lag_frames': 10}

    def __call__(self, xi_threshold=30, lag_frames=10, keepSourceWindow=False):
        self.start(keepSourceWindow)

        xi_threshold = float(xi_threshold) / 100.0
        lag_frames = int(lag_frames)

        if hasattr(g.win, 'framerate') and g.win.framerate is not None:
            fs = g.win.framerate
        else:
            fs = 30.0
            g.m.statusBar().showMessage(f'Using default framerate: {fs} Hz', 5000)

        # Generate CRM (returns list of CRMs)
        T = self.tif.shape[0]
        subsection_length = 500  # Default from module
        g.m.statusBar().showMessage(f'Computing CRM for {T} frames with max_lag={lag_frames}...', 3000)

        # Check if stack is long enough
        if T < subsection_length:
            g.m.statusBar().showMessage(
                f'Stack too short! Need at least {subsection_length} frames, have {T}. CRM requires longer recordings.',
                10000
            )
            return None

        crm_list = psm_crm_module.compute_correlation_map_from_stack(self.tif, max_lag=lag_frames)

        # Compute mean CRM from list
        if len(crm_list) == 0:
            g.m.statusBar().showMessage('No CRM subsections generated - check stack length', 10000)
            return None

        g.m.statusBar().showMessage(f'Averaging {len(crm_list)} CRM subsections...', 2000)
        crm = psm_crm_module.mean_correlation_map(crm_list)

        # Apply threshold for display
        crm_display = np.where(crm >= xi_threshold, crm, 0)

        # Report statistics
        n_active = np.sum(crm_display > 0)
        mean_xi = np.mean(crm[crm > 0]) if np.any(crm > 0) else 0
        g.m.statusBar().showMessage(
            f'CRM: {n_active} active pixels, mean ξ={mean_xi:.2f}',
            8000
        )

        self.newtif = crm_display.astype(np.float32)
        self.newname = f"{self.oldname} - CRM (ξ)"

        return self.end()


class SD_Image_Stack(BaseProcess):
    """sd_image_stack(spatial_sigma=1.0, highpass_cutoff=0.5, window_frames=20,
                      keepSourceWindow=False)

    Generate SD (standard deviation) image stack for Lock & Parker 2020 analysis.
    Highlights transient local Ca²⁺ signals during global elevations.

    Parameters:
        spatial_sigma (float): Spatial smoothing sigma (pixels)
        highpass_cutoff (float): Temporal high-pass cutoff (Hz)
        window_frames (int): Window size for variance calculation
        keepSourceWindow (bool): Keep the source window open

    Returns:
        newWindow: Window containing SD stack
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()

        spatial_sigma = SliderLabel(0)
        spatial_sigma.setRange(1, 50)
        spatial_sigma.setValue(10)

        highpass_cutoff = SliderLabel(0)
        highpass_cutoff.setRange(1, 100)
        highpass_cutoff.setValue(5)

        window_frames = SliderLabel(0)
        window_frames.setRange(5, 100)
        window_frames.setValue(20)

        self.items.append({'name': 'spatial_sigma', 'string': 'Spatial Sigma', 'object': spatial_sigma})
        self.items.append({'name': 'highpass_cutoff', 'string': 'Highpass (Hz)', 'object': highpass_cutoff})
        self.items.append({'name': 'window_frames', 'string': 'Window (frames)', 'object': window_frames})
        super().gui()

    def get_init_settings_dict(self):
        return {'spatial_sigma': 10, 'highpass_cutoff': 5, 'window_frames': 20}

    def __call__(self, spatial_sigma=10, highpass_cutoff=5, window_frames=20,
                 keepSourceWindow=False):
        self.start(keepSourceWindow)

        spatial_sigma = float(spatial_sigma) / 10.0
        highpass_cutoff = float(highpass_cutoff) / 10.0
        window_frames = int(window_frames)

        if hasattr(g.win, 'framerate') and g.win.framerate is not None:
            fs = g.win.framerate
        else:
            fs = 30.0
            g.m.statusBar().showMessage(f'Using default framerate: {fs} Hz', 5000)

        # Generate SD stack
        sd_stack = temporal_fluctuation_module.compute_sd_image_stack(
            self.tif, fs,
            spatial_sigma=spatial_sigma,
            highpass_cutoff=highpass_cutoff,
            window_frames=window_frames
        )

        self.newtif = sd_stack.astype(np.float32)
        self.newname = f"{self.oldname} - SD Stack"

        return self.end()


class Detect_Wave_Sites(BaseProcess):
    """detect_wave_sites(threshold=0.3, min_separation=10, keepSourceWindow=False)

    Detect Ca²⁺ wave initiation sites.

    Identifies pixels where Ca²⁺ signal rises earliest and propagates outward.

    Parameters:
        threshold (float): Threshold for signal onset (ΔF/F₀)
        min_separation (int): Minimum distance between sites (pixels)
        keepSourceWindow (bool): Keep the source window open

    Returns:
        newWindow: Window showing initiation sites
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()

        threshold = SliderLabel(0)
        threshold.setRange(1, 100)
        threshold.setValue(30)

        min_separation = SliderLabel(0)
        min_separation.setRange(5, 50)
        min_separation.setValue(10)

        self.items.append({'name': 'threshold', 'string': 'Threshold (ΔF/F₀)', 'object': threshold})
        self.items.append({'name': 'min_separation', 'string': 'Min Separation (px)', 'object': min_separation})
        super().gui()

    def get_init_settings_dict(self):
        return {'threshold': 30, 'min_separation': 10}

    def __call__(self, threshold=30, min_separation=10, keepSourceWindow=False):
        self.start(keepSourceWindow)

        threshold = float(threshold) / 100.0
        min_separation = int(min_separation)

        # Detect initiation sites
        sites = wave_propagation_module.detect_wave_initiation_sites(
            self.tif, threshold=threshold, min_separation=min_separation
        )

        # Create visualization
        onset_map = sites['onset_map']

        # Mark initiation sites
        site_map = np.copy(onset_map)
        for (y, x) in sites['initiation_sites']:
            # Mark with a cross
            site_map[max(0,y-2):min(site_map.shape[0],y+3), x] = site_map.min()
            site_map[y, max(0,x-2):min(site_map.shape[1],x+3)] = site_map.min()

        g.m.statusBar().showMessage(
            f"Detected {len(sites['initiation_sites'])} wave initiation sites",
            10000
        )

        self.newtif = site_map.astype(np.float32)
        self.newname = f"{self.oldname} - Wave Sites"

        return self.end()


class Spatiotemporal_Map(BaseProcess):
    """spatiotemporal_map(axis='x', keepSourceWindow=False)

    Create kymograph-style spatiotemporal map for wave visualization.

    Shows Ca²⁺ signal evolution along one spatial dimension over time.

    Parameters:
        axis (str): 'x' or 'y' for spatial axis
        keepSourceWindow (bool): Keep the source window open

    Returns:
        newWindow: Window containing spatiotemporal map (time vs position)
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()

        axis = ComboBox()
        axis.addItem('x')
        axis.addItem('y')
        axis.setCurrentIndex(0)

        self.items.append({'name': 'axis', 'string': 'Spatial Axis', 'object': axis})
        super().gui()

    def get_init_settings_dict(self):
        return {'axis': 'x'}

    def __call__(self, axis='x', keepSourceWindow=False):
        self.start(keepSourceWindow)

        # Create spatiotemporal map
        st_map = wave_propagation_module.create_spatiotemporal_map(self.tif, axis=axis)

        self.newtif = st_map.astype(np.float32)
        self.newname = f"{self.oldname} - Kymograph ({axis})"

        return self.end()


class Quality_Report(BaseProcess):
    """quality_report(keepSourceWindow=True)

    Generate comprehensive quality control report.

    Computes SNR metrics, detects photobleaching and motion artifacts,
    calculates CASCADE noise metric (ν).

    Parameters:
        keepSourceWindow (bool): Keep the source window open

    Returns:
        Prints report to status bar and creates SNR map
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        super().gui()

    def get_init_settings_dict(self):
        return {}

    def __call__(self, keepSourceWindow=True):
        self.start(keepSourceWindow)

        if hasattr(g.win, 'framerate') and g.win.framerate is not None:
            fs = g.win.framerate
        else:
            fs = 30.0
            g.m.statusBar().showMessage(f'Using default framerate: {fs} Hz', 5000)

        # Compute dFF for CASCADE metric
        T, H, W = self.tif.shape
        dff_stack = np.zeros_like(self.tif, dtype=np.float64)

        g.m.statusBar().showMessage('Computing ΔF/F...', 2000)
        for i in range(H):
            for j in range(W):
                trace = self.tif[:, i, j].astype(np.float64)
                f0 = baseline_module.estimate_f0_percentile(trace, window=100, percentile=8)
                dff_stack[:, i, j] = baseline_module.compute_dff(trace, f0, epsilon=1.0)

        # Generate report
        g.m.statusBar().showMessage('Generating quality report...', 2000)
        report = quality_control_module.generate_quality_report(
            self.tif, fs, dff_stack=dff_stack
        )

        # Print summary
        print("\n" + "="*70)
        print("QUALITY CONTROL REPORT")
        print("="*70)
        print(f"Overall Quality: {report['overall_quality']:.1%}")
        print(f"Category: {report['overall_category'].upper()}")
        print(f"\nSNR Metrics:")
        print(f"  Temporal SNR: {report['snr']['temporal_snr']:.2f}")
        print(f"  Spatial SNR: {report['snr']['spatial_snr']:.2f}")
        print(f"  Peak SNR: {report['snr']['peak_snr']:.2f}")
        print(f"  Quality Score: {report['snr']['quality_score']:.2%}")

        if 'cascade' in report:
            print(f"\nCASCADE Noise Metric:")
            print(f"  ν = {report['cascade']['nu']:.2f}")
            print(f"  Quality: {report['cascade']['quality_category']}")

        print(f"\nPhotobleaching:")
        print(f"  Detected: {report['photobleaching']['bleach_detected']}")
        if report['photobleaching']['bleach_detected']:
            print(f"  Bleach: {report['photobleaching']['bleach_percent']:.1f}%")

        print(f"\nMotion Artifacts:")
        print(f"  Quality frames: {report['motion']['quality_frames']:.1f}%")

        if report['warnings']:
            print(f"\nWarnings:")
            for warning in report['warnings']:
                print(f"  ⚠ {warning}")

        print("="*70 + "\n")

        g.m.statusBar().showMessage(
            f"Quality: {report['overall_category']} ({report['overall_quality']:.1%})",
            15000
        )

        # Return CASCADE nu map
        if 'cascade' in report:
            self.newtif = report['cascade']['nu_map'].astype(np.float32)
            self.newname = f"{self.oldname} - Quality (ν map)"
        else:
            self.newtif = np.ones((H, W), dtype=np.float32)
            self.newname = f"{self.oldname} - Quality Report"

        return self.end()


# Create instances for menu access
power_spectrum_map = Power_Spectrum_Map()
fluctuation_analysis = Fluctuation_Analysis()
anscombe_transform = Anscombe_Transform()
estimate_baseline_f0 = Estimate_Baseline_F0()
compute_dff = Compute_DFF()
detect_calcium_sparks = Detect_Calcium_Sparks()
local_correlation_image = Local_Correlation_Image()
compute_noise_metric = Compute_Noise_Metric()
butterworth_bandpass = Butterworth_Bandpass()

# Advanced processes (v2.0)
generate_psm = Generate_PSM()
generate_crm = Generate_CRM()
sd_image_stack = SD_Image_Stack()
detect_wave_sites = Detect_Wave_Sites()
spatiotemporal_map = Spatiotemporal_Map()
quality_report = Quality_Report()


def launch_docs():
    """Open the plugin documentation"""
    import os
    import webbrowser

    plugin_dir = os.path.dirname(os.path.abspath(__file__))
    about_path = os.path.join(plugin_dir, 'about.html')

    if os.path.exists(about_path):
        webbrowser.open('file://' + about_path)
    else:
        url = 'https://github.com/flika-org/flika'
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))
