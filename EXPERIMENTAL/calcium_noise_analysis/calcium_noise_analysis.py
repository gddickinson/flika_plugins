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
Version: 1.0.0
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

# Import modular analysis functions
from . import power_spectrum_module
from . import fluctuation_module
from . import shot_noise_module
from . import baseline_module
from . import event_detection_module
from . import correlation_module


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
        nperseg.addItem('128')
        nperseg.addItem('256')
        nperseg.addItem('512')
        nperseg.addItem('1024')
        nperseg.setCurrentIndex(1)  # Default to 256

        self.items.append({'name': 'low_freq_min', 'string': 'Low Freq Min (Hz)', 'object': low_freq_min})
        self.items.append({'name': 'low_freq_max', 'string': 'Low Freq Max (Hz)', 'object': low_freq_max})
        self.items.append({'name': 'high_freq_cutoff', 'string': 'High Freq Cutoff (Hz)', 'object': high_freq_cutoff})
        self.items.append({'name': 'nperseg', 'string': 'Segment Length', 'object': nperseg})
        super().gui()

    def get_init_settings_dict(self):
        s = dict()
        s['low_freq_min'] = 1  # Integer for SliderLabel(0)
        s['low_freq_max'] = 50  # Integer for SliderLabel(0)
        s['high_freq_cutoff'] = 50  # Integer for SliderLabel(0)
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
            fs = 30.0  # Default framerate
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
        s['spatial_sigma'] = 10  # Will be divided by 10 to get 1.0
        s['highpass_cutoff'] = 5  # Will be divided by 10 to get 0.5
        s['window_size'] = 30
        s['apply_shot_noise'] = False
        s['shot_noise_factor'] = 0  # Will be divided by 1000 to get 0.0
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
            fs = 30.0  # Default framerate
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

        self.newtif = sd_stack.astype(self.tif.dtype)
        self.newname = f"{self.oldname} - SD Fluctuation"

        return self.end()


class Anscombe_Transform(BaseProcess):
    """anscombe_transform(inverse=False, keepSourceWindow=False)

    Apply the Anscombe variance-stabilizing transform for Poisson noise.

    The Anscombe transform converts Poisson-distributed data to approximately
    Gaussian with unit variance: y = 2√(x + 3/8)

    This stabilization enables standard denoising techniques on photon shot noise.

    Parameters:
        inverse (bool): Apply inverse transform (from stabilized back to original)
        keepSourceWindow (bool): Keep the source window open

    Returns:
        newWindow: Window containing the transformed image
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()

        inverse = CheckBox()
        inverse.setChecked(False)

        self.items.append({'name': 'inverse', 'string': 'Apply Inverse Transform', 'object': inverse})
        super().gui()

    def get_init_settings_dict(self):
        s = dict()
        s['inverse'] = False
        return s

    def __call__(self, inverse=False, keepSourceWindow=False):
        self.start(keepSourceWindow)

        if inverse:
            self.newtif = shot_noise_module.inverse_anscombe(self.tif.astype(np.float64))
            self.newname = f"{self.oldname} - Inverse Anscombe"
        else:
            self.newtif = shot_noise_module.anscombe_transform(self.tif.astype(np.float64))
            self.newname = f"{self.oldname} - Anscombe"

        # Convert back to original dtype range
        if np.issubdtype(self.tif.dtype, np.integer):
            max_val = np.iinfo(self.tif.dtype).max
            min_val = self.newtif.min()
            max_val_new = self.newtif.max()
            if max_val_new > min_val:
                normalized = (self.newtif - min_val) / (max_val_new - min_val)
                self.newtif = (normalized * max_val).astype(self.tif.dtype)

        return self.end()


class Estimate_Baseline_F0(BaseProcess):
    """estimate_baseline_f0(method='percentile', window_frames=1000, percentile=8,
                            keepSourceWindow=False)

    Estimate baseline fluorescence (F₀) using various methods.

    Methods:
    - 'percentile': Sliding percentile (8th percentile over 15-30s windows) - robust to transients
    - 'maximin': Suite2p method (maximum of sliding minimum) - fast and effective
    - 'photobleaching': Double exponential fit for photobleaching correction

    Parameters:
        method (str): Baseline estimation method
        window_frames (int): Window size in frames (for percentile/maximin)
        percentile (int): Percentile value (0-100, for percentile method)
        keepSourceWindow (bool): Keep the source window open

    Returns:
        newWindow: Window containing the estimated baseline F₀
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()

        method = ComboBox()
        method.addItem('percentile')
        method.addItem('maximin')
        method.addItem('photobleaching')

        window_frames = SliderLabel(0)
        window_frames.setRange(50, 3000)
        window_frames.setValue(1000)

        percentile = SliderLabel(0)
        percentile.setRange(1, 50)
        percentile.setValue(8)

        self.items.append({'name': 'method', 'string': 'Method', 'object': method})
        self.items.append({'name': 'window_frames', 'string': 'Window Size (frames)', 'object': window_frames})
        self.items.append({'name': 'percentile', 'string': 'Percentile (%)', 'object': percentile})
        super().gui()

    def get_init_settings_dict(self):
        s = dict()
        s['method'] = 'percentile'
        s['window_frames'] = 1000
        s['percentile'] = 8
        return s

    def __call__(self, method='percentile', window_frames=1000, percentile=8, keepSourceWindow=False):
        self.start(keepSourceWindow)

        window_frames = int(window_frames)
        percentile = int(percentile)

        if self.tif.ndim == 3:
            T, H, W = self.tif.shape
            f0 = np.zeros_like(self.tif, dtype=np.float64)

            # Process each pixel
            for i in range(H):
                for j in range(W):
                    trace = self.tif[:, i, j].astype(np.float64)

                    if method == 'percentile':
                        f0[:, i, j] = baseline_module.estimate_f0_percentile(
                            trace, window=window_frames, percentile=percentile
                        )
                    elif method == 'maximin':
                        f0[:, i, j] = baseline_module.estimate_f0_maximin(
                            trace, window=window_frames
                        )
                    elif method == 'photobleaching':
                        try:
                            f0[:, i, j] = baseline_module.photobleaching_baseline(trace)
                        except:
                            # Fallback to percentile if fitting fails
                            f0[:, i, j] = baseline_module.estimate_f0_percentile(
                                trace, window=window_frames, percentile=percentile
                            )

                # Progress update
                if i % 10 == 0:
                    g.m.statusBar().showMessage(f'Processing row {i+1}/{H}', 1000)
        else:
            # Single frame - just return the image
            f0 = self.tif.astype(np.float64)

        self.newtif = f0.astype(self.tif.dtype)
        self.newname = f"{self.oldname} - F0 ({method})"

        return self.end()


class Compute_DFF(BaseProcess):
    """compute_dff(f0_window=100, baseline_percentile=8, epsilon=1.0, keepSourceWindow=False)

    Calculate ΔF/F (normalized fluorescence change).

    ΔF/F = (F - F₀) / max(F₀, ε)

    The epsilon parameter prevents division artifacts when F₀ is near zero.
    F₀ is estimated using sliding percentile method if not provided separately.

    Parameters:
        f0_window (int): Window size for baseline estimation (frames)
        baseline_percentile (int): Percentile for baseline (0-100)
        epsilon (float): Minimum baseline value to prevent division errors
        keepSourceWindow (bool): Keep the source window open

    Returns:
        newWindow: Window containing ΔF/F values
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()

        f0_window = SliderLabel(0)
        f0_window.setRange(50, 3000)
        f0_window.setValue(100)

        baseline_percentile = SliderLabel(0)
        baseline_percentile.setRange(1, 50)
        baseline_percentile.setValue(8)

        epsilon = SliderLabel(0)
        epsilon.setRange(1, 1000)
        epsilon.setValue(10)

        self.items.append({'name': 'f0_window', 'string': 'Baseline Window (frames)', 'object': f0_window})
        self.items.append({'name': 'baseline_percentile', 'string': 'Baseline Percentile (%)', 'object': baseline_percentile})
        self.items.append({'name': 'epsilon', 'string': 'Epsilon (min baseline)', 'object': epsilon})
        super().gui()

    def get_init_settings_dict(self):
        s = dict()
        s['f0_window'] = 100
        s['baseline_percentile'] = 8
        s['epsilon'] = 10  # Will be divided by 10 to get 1.0
        return s

    def __call__(self, f0_window=100, baseline_percentile=8, epsilon=10, keepSourceWindow=False):
        self.start(keepSourceWindow)

        f0_window = int(f0_window)
        baseline_percentile = int(baseline_percentile)
        epsilon = float(epsilon) / 10.0  # Convert slider value to float

        if self.tif.ndim == 3:
            T, H, W = self.tif.shape
            dff = np.zeros_like(self.tif, dtype=np.float64)

            # Process each pixel
            for i in range(H):
                for j in range(W):
                    trace = self.tif[:, i, j].astype(np.float64)
                    f0 = baseline_module.estimate_f0_percentile(
                        trace, window=f0_window, percentile=baseline_percentile
                    )
                    dff[:, i, j] = baseline_module.compute_dff(trace, f0, epsilon=epsilon)

                # Progress update
                if i % 10 == 0:
                    g.m.statusBar().showMessage(f'Processing row {i+1}/{H}', 1000)
        else:
            # Single frame - cannot compute ΔF/F
            g.m.statusBar().showMessage('ΔF/F requires time series data', 5000)
            return None

        self.newtif = dff.astype(np.float32)
        self.newname = f"{self.oldname} - ΔF/F"

        return self.end()


class Detect_Calcium_Sparks(BaseProcess):
    """detect_sparks(intensity_thresh=2.0, peak_thresh=3.8, min_size=40,
                     median_filter_size=3, uniform_filter_size=3, keepSourceWindow=False)

    Detect calcium sparks using dual-threshold approach.

    Uses variance stabilization for Poisson noise handling:
    normalized = (F - F₀) / √F₀

    Then applies dual thresholds:
    - Intensity threshold (2σ) defines event boundaries
    - Peak threshold (3.8σ) confirms true events

    Parameters:
        intensity_thresh (float): Intensity threshold in units of σ (standard deviations)
        peak_thresh (float): Peak threshold in units of σ (must be > intensity_thresh)
        min_size (int): Minimum event size in pixels
        median_filter_size (int): Median filter size for smoothing
        uniform_filter_size (int): Uniform filter size for smoothing
        keepSourceWindow (bool): Keep the source window open

    Returns:
        newWindow: Window containing detected sparks mask (labeled regions)
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()

        intensity_thresh = SliderLabel(0)
        intensity_thresh.setRange(5, 100)
        intensity_thresh.setValue(20)

        peak_thresh = SliderLabel(0)
        peak_thresh.setRange(10, 100)
        peak_thresh.setValue(38)

        min_size = SliderLabel(0)
        min_size.setRange(10, 200)
        min_size.setValue(40)

        median_filter_size = SliderLabel(0)
        median_filter_size.setRange(1, 11)
        median_filter_size.setValue(3)

        uniform_filter_size = SliderLabel(0)
        uniform_filter_size.setRange(1, 11)
        uniform_filter_size.setValue(3)

        self.items.append({'name': 'intensity_thresh', 'string': 'Intensity Threshold (σ)', 'object': intensity_thresh})
        self.items.append({'name': 'peak_thresh', 'string': 'Peak Threshold (σ)', 'object': peak_thresh})
        self.items.append({'name': 'min_size', 'string': 'Min Event Size (pixels)', 'object': min_size})
        self.items.append({'name': 'median_filter_size', 'string': 'Median Filter Size', 'object': median_filter_size})
        self.items.append({'name': 'uniform_filter_size', 'string': 'Uniform Filter Size', 'object': uniform_filter_size})
        super().gui()

    def get_init_settings_dict(self):
        s = dict()
        s['intensity_thresh'] = 20  # Will be divided by 10 to get 2.0
        s['peak_thresh'] = 38  # Will be divided by 10 to get 3.8
        s['min_size'] = 40
        s['median_filter_size'] = 3
        s['uniform_filter_size'] = 3
        return s

    def __call__(self, intensity_thresh=20, peak_thresh=38, min_size=40,
                 median_filter_size=3, uniform_filter_size=3, keepSourceWindow=False):
        self.start(keepSourceWindow)

        # Convert slider integers to floats
        intensity_thresh = float(intensity_thresh) / 10.0
        peak_thresh = float(peak_thresh) / 10.0
        min_size = int(min_size)
        median_filter_size = int(median_filter_size)
        uniform_filter_size = int(uniform_filter_size)

        if self.tif.ndim == 3:
            T, H, W = self.tif.shape

            # Estimate baseline (use first 100 frames or available frames)
            baseline_frames = min(100, T)
            f0 = np.mean(self.tif[:baseline_frames], axis=0)

            # Detect sparks frame by frame
            labels_stack = np.zeros_like(self.tif, dtype=np.int32)
            total_sparks = 0

            for t in range(T):
                frame = self.tif[t]
                sparks_mask = event_detection_module.detect_sparks(
                    frame[np.newaxis, :, :], f0,
                    intensity_thresh=intensity_thresh,
                    peak_thresh=peak_thresh,
                    min_size=min_size,
                    median_filter_size=median_filter_size,
                    uniform_filter_size=uniform_filter_size
                )

                if len(sparks_mask) > 0:
                    # Combine all spark regions into labeled mask
                    for i, spark in enumerate(sparks_mask):
                        labels_stack[t][spark[0]] = total_sparks + i + 1
                    total_sparks += len(sparks_mask)

                # Progress update
                if t % 50 == 0:
                    g.m.statusBar().showMessage(f'Processing frame {t+1}/{T}', 1000)

            self.newtif = labels_stack.astype(np.uint16)
            self.newname = f"{self.oldname} - Sparks Detected ({total_sparks} events)"
            g.m.statusBar().showMessage(f'Detected {total_sparks} sparks', 5000)
        else:
            # Single frame detection
            f0 = self.tif.astype(np.float64)
            sparks_mask = event_detection_module.detect_sparks(
                self.tif[np.newaxis, :, :], f0,
                intensity_thresh=intensity_thresh,
                peak_thresh=peak_thresh,
                min_size=min_size,
                median_filter_size=median_filter_size,
                uniform_filter_size=uniform_filter_size
            )

            labels_img = np.zeros_like(self.tif, dtype=np.int32)
            for i, spark in enumerate(sparks_mask):
                labels_img[spark[0]] = i + 1

            self.newtif = labels_img.astype(np.uint16)
            self.newname = f"{self.oldname} - Sparks ({len(sparks_mask)} events)"
            g.m.statusBar().showMessage(f'Detected {len(sparks_mask)} sparks', 5000)

        return self.end()


class Local_Correlation_Image(BaseProcess):
    """local_correlation_image(neighborhood=1, keepSourceWindow=False)

    Compute local correlation image for neuron/ROI detection.

    For each pixel, computes average Pearson correlation with its spatial neighbors
    across time. High correlation indicates co-active regions (neurons, dendrites).

    Based on CaImAn/Suite2p correlation image methodology.

    Parameters:
        neighborhood (int): Radius of neighborhood (1 = 3x3, 2 = 5x5, etc.)
        keepSourceWindow (bool): Keep the source window open

    Returns:
        newWindow: Window containing the local correlation image
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()

        neighborhood = SliderLabel(0)
        neighborhood.setRange(1, 5)
        neighborhood.setValue(1)

        self.items.append({'name': 'neighborhood', 'string': 'Neighborhood Radius', 'object': neighborhood})
        super().gui()

    def get_init_settings_dict(self):
        s = dict()
        s['neighborhood'] = 1
        return s

    def __call__(self, neighborhood=1, keepSourceWindow=False):
        self.start(keepSourceWindow)

        neighborhood = int(neighborhood)

        if self.tif.ndim != 3:
            g.m.statusBar().showMessage('Local correlation requires time series data', 5000)
            return None

        # Compute local correlation image
        corr_img = correlation_module.local_correlation_image(self.tif, neighborhood=neighborhood)

        # Convert to displayable format
        self.newtif = corr_img.astype(np.float32)
        self.newname = f"{self.oldname} - Correlation Image"

        return self.end()


class Compute_Noise_Metric(BaseProcess):
    """compute_noise_metric(baseline_frames=100, keepSourceWindow=False)

    Compute CASCADE noise metric: ν = σ_ΔF/F × √(frame_rate)

    This standardizes noise comparisons across recording conditions.
    Typical values range from ~1 (low noise) to ~8-9 (high noise).

    Creates a summary image showing the noise metric for each pixel.

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

        # Get framerate from current window
        if hasattr(g.win, 'framerate') and g.win.framerate is not None:
            fs = g.win.framerate
        else:
            fs = 30.0  # Default framerate
            g.m.statusBar().showMessage(f'Using default framerate: {fs} Hz', 5000)

        if self.tif.ndim != 3:
            g.m.statusBar().showMessage('Noise metric requires time series data', 5000)
            return None

        T, H, W = self.tif.shape
        noise_map = np.zeros((H, W), dtype=np.float32)

        # Compute noise metric for each pixel
        for i in range(H):
            for j in range(W):
                trace = self.tif[:, i, j].astype(np.float64)

                # Estimate baseline
                f0 = baseline_module.estimate_f0_percentile(
                    trace, window=baseline_frames, percentile=8
                )

                # Compute ΔF/F
                dff = baseline_module.compute_dff(trace, f0, epsilon=1.0)

                # Compute noise metric
                sigma_dff = np.std(dff)
                noise_metric = sigma_dff * np.sqrt(fs)
                noise_map[i, j] = noise_metric

            # Progress update
            if i % 10 == 0:
                g.m.statusBar().showMessage(f'Processing row {i+1}/{H}', 1000)

        # Report statistics
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
        s['low_freq'] = 10  # Will be divided by 100 to get 0.1
        s['high_freq'] = 50  # Will be divided by 10 to get 5.0
        s['order'] = 2
        return s

    def __call__(self, low_freq=10, high_freq=50, order=2, keepSourceWindow=False):
        self.start(keepSourceWindow)

        # Convert slider integers to floats
        low_freq = float(low_freq) / 100.0
        high_freq = float(high_freq) / 10.0
        order = int(order)

        # Get framerate from current window
        if hasattr(g.win, 'framerate') and g.win.framerate is not None:
            fs = g.win.framerate
        else:
            fs = 30.0  # Default framerate
            g.m.statusBar().showMessage(f'Using default framerate: {fs} Hz', 5000)

        if self.tif.ndim == 3:
            T, H, W = self.tif.shape
            filtered = np.zeros_like(self.tif, dtype=np.float64)

            # Filter each pixel's time series
            for i in range(H):
                for j in range(W):
                    trace = self.tif[:, i, j].astype(np.float64)
                    filtered[:, i, j] = power_spectrum_module.butterworth_bandpass(
                        trace, fs, low=low_freq, high=high_freq, order=order
                    )

                # Progress update
                if i % 10 == 0:
                    g.m.statusBar().showMessage(f'Processing row {i+1}/{H}', 1000)

            self.newtif = filtered.astype(self.tif.dtype)
        else:
            g.m.statusBar().showMessage('Bandpass filter requires time series data', 5000)
            return None

        self.newname = f"{self.oldname} - Bandpass {low_freq}-{high_freq} Hz"

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


def launch_docs():
    """Open the plugin documentation"""
    import os
    import webbrowser

    # Try to open the about.html file
    plugin_dir = os.path.dirname(os.path.abspath(__file__))
    about_path = os.path.join(plugin_dir, 'about.html')

    if os.path.exists(about_path):
        webbrowser.open('file://' + about_path)
    else:
        # Fallback to GitHub
        url = 'https://github.com/flika-org/flika'
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))
