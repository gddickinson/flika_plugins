from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks, peak_widths
from scipy.stats import zscore
from distutils.version import StrictVersion
import flika
from flika.window import Window
import flika.global_vars as g
import pyqtgraph as pg
from time import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector
    from flika.roi import ROI_Drawing, makeROI
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector
    from flika.roi import ROI_Drawing, makeROI


class KymographAnalyzer(BaseProcess_noPriorWindow):
    """
    Advanced Kymograph Analysis for studying spatiotemporal dynamics in TIRF microscopy.

    Perfect for analyzing:
    - Ca2+ flickers and calcium signaling dynamics
    - Membrane protein movements (PIEZO1, ion channels)
    - Vesicle trafficking and exocytosis
    - Protein recruitment dynamics
    - Wave propagation along membranes
    - Temporal correlation of events

    Features:
    - Multiple ROI types (line, polyline, freehand)
    - Automatic event detection (peaks, waves)
    - Intensity profiling and background subtraction
    - Velocity/speed measurements
    - Dwell time analysis
    - Statistical quantification
    - Export capabilities

    Draw an ROI (line or path) along your region of interest,
    then click 'Generate Kymograph' to visualize dynamics over time.
    """

    def __init__(self):
        BaseProcess_noPriorWindow.__init__(self)
        self.kymograph = None
        self.roi = None
        self.source_window = None
        self.kymo_window = None
        self.analysis_results = {}

    def __call__(self, window, roi_type='line', width=5, temporal_binning=1,
                 normalize=True, detrend=False, gaussian_sigma=0):
        """
        Generate kymograph from ROI in source window.

        Parameters:
        -----------
        window : Window
            Source image window (must be 3D time series)
        roi_type : str
            Type of ROI ('line', 'polyline', 'freehand')
        width : int
            Width of the line ROI in pixels
        temporal_binning : int
            Bin frames together (1 = no binning)
        normalize : bool
            Normalize intensity to [0, 1]
        detrend : bool
            Remove linear trend
        gaussian_sigma : float
            Gaussian smoothing sigma (0 = no smoothing)
        """
        if window is None:
            g.m.statusBar().showMessage("Error: No window selected")
            return

        if window.image.ndim != 3:
            g.m.statusBar().showMessage("Error: Image must be a 3D time series")
            return

        g.m.statusBar().showMessage("Generating kymograph...")
        t_start = time()

        self.source_window = window
        image_stack = window.imageArray()  # Use imageArray() for proper 3D array format

        # Get ROI coordinates
        if not hasattr(window, 'rois') or len(window.rois) == 0:
            g.m.statusBar().showMessage("Error: No ROI found. Draw a line ROI first.")
            return

        self.roi = window.rois[-1]  # Use most recent ROI

        # Extract line coordinates from ROI
        coords = self.get_roi_coordinates(self.roi, width)

        if coords is None:
            g.m.statusBar().showMessage("Error: Could not extract ROI coordinates")
            return

        # Generate kymograph
        kymo = self.generate_kymograph(image_stack, coords, temporal_binning)

        # Apply processing
        if detrend:
            kymo = self.detrend_kymograph(kymo)

        if normalize:
            kymo = self.normalize_kymograph(kymo)

        if gaussian_sigma > 0:
            kymo = gaussian_filter1d(kymo, sigma=gaussian_sigma, axis=0)

        self.kymograph = kymo

        # Create kymograph window
        self.kymo_window = Window(kymo, name=f"Kymograph_{window.name}")

        # Note: setAspectLocked not available on FLIKA's ImageView
        # The kymograph will display with default aspect ratio

        elapsed = time() - t_start
        g.m.statusBar().showMessage(f"Kymograph generated ({elapsed:.2f} s)")

        return self.kymo_window

    def get_roi_coordinates(self, roi, width):
        """
        Extract coordinates from ROI with specified width.
        Returns array of (x, y) coordinates along the ROI.
        """
        try:
            # Get ROI path points using FLIKA's getPoints() method
            if hasattr(roi, 'getPoints'):
                # For line ROIs - getPoints() returns [N, 2] array in [x, y] format
                pts = roi.getPoints()

                if len(pts) < 2:
                    return None

                # Get start and end points (first and last points)
                p1 = np.array(pts[0])
                p2 = np.array(pts[-1])

                # Generate line coordinates
                length = int(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
                x = np.linspace(p1[0], p2[0], length)
                y = np.linspace(p1[1], p2[1], length)

                coords = np.column_stack([x, y])

                # Add width by creating perpendicular offsets
                if width > 1:
                    # Calculate perpendicular direction
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    perp_x = -dy / length
                    perp_y = dx / length

                    all_coords = []
                    for offset in range(-width//2, width//2 + 1):
                        offset_coords = coords + np.array([perp_x * offset, perp_y * offset])
                        all_coords.append(offset_coords)

                    return all_coords

                return [coords]

            else:
                return None

        except Exception as e:
            print(f"Error extracting ROI coordinates: {e}")
            return None

    def generate_kymograph(self, image_stack, coords_list, temporal_binning=1):
        """
        Generate kymograph from image stack along coordinates.

        Parameters:
        -----------
        image_stack : ndarray
            3D array (time, x, y)
        coords_list : list of ndarrays
            List of coordinate arrays for width
        temporal_binning : int
            Number of frames to bin together

        Returns:
        --------
        ndarray : Kymograph (time, space)
        """
        n_frames = image_stack.shape[0]

        # FLIKA stores images as (t, x, y) not standard numpy (t, y, x)
        # Use window dimensions to get correct width and height
        if hasattr(self, 'source_window'):
            w = self.source_window.mx  # width (x dimension)
            h = self.source_window.my  # height (y dimension)

            # Verify the shape matches FLIKA's (t, x, y) format
            if image_stack.shape != (n_frames, w, h):
                print(f"  Note: Image shape {image_stack.shape} doesn't match expected (t,x,y)={(n_frames, w, h)}")
        else:
            # Fallback: assume (t, x, y) format based on FLIKA's convention
            w, h = image_stack.shape[1], image_stack.shape[2]

        # Get spatial length from first coordinate set
        spatial_length = len(coords_list[0])

        # Bin frames if requested
        if temporal_binning > 1:
            n_binned = n_frames // temporal_binning
            binned_stack = np.zeros((n_binned, w, h))  # FLIKA format: (t, x, y)
            for i in range(n_binned):
                start_idx = i * temporal_binning
                end_idx = start_idx + temporal_binning
                binned_stack[i] = np.mean(image_stack[start_idx:end_idx], axis=0)
            image_stack = binned_stack
            n_frames = n_binned

        # Initialize kymograph
        kymo = np.zeros((n_frames, spatial_length))

        # Extract intensity values along line for each frame
        for t in range(n_frames):
            frame = image_stack[t]

            # Average over width
            line_intensity = np.zeros(spatial_length)
            for coords in coords_list:
                # Interpolate intensity values
                for i, (x, y) in enumerate(coords):
                    if 0 <= int(x) < w and 0 <= int(y) < h:
                        # Bilinear interpolation
                        x0, y0 = int(x), int(y)
                        x1, y1 = min(x0 + 1, w-1), min(y0 + 1, h-1)
                        dx, dy = x - x0, y - y0

                        # FLIKA stores images as (t, x, y) so we use frame[x, y] not frame[y, x]
                        val = (frame[x0, y0] * (1-dx) * (1-dy) +
                              frame[x1, y0] * dx * (1-dy) +
                              frame[x0, y1] * (1-dx) * dy +
                              frame[x1, y1] * dx * dy)

                        line_intensity[i] += val

            line_intensity /= len(coords_list)
            kymo[t] = line_intensity

        return kymo

    def normalize_kymograph(self, kymo):
        """
        Normalize kymograph to [0, 1] range.
        """
        min_val = np.min(kymo)
        max_val = np.max(kymo)
        if max_val > min_val:
            return (kymo - min_val) / (max_val - min_val)
        return kymo

    def detrend_kymograph(self, kymo):
        """
        Remove linear trend from kymograph.
        """
        from scipy.signal import detrend as scipy_detrend
        return scipy_detrend(kymo, axis=0)

    def detect_events(self, kymo=None, method='peaks', threshold=2.0,
                     min_distance=5, min_duration=3):
        """
        Automatically detect events in kymograph.

        Parameters:
        -----------
        kymo : ndarray
            Kymograph to analyze (uses self.kymograph if None)
        method : str
            Detection method ('peaks', 'threshold', 'zscore')
        threshold : float
            Detection threshold
        min_distance : int
            Minimum distance between events (frames)
        min_duration : int
            Minimum event duration (frames)

        Returns:
        --------
        dict : Dictionary containing detected events with properties
        """
        if kymo is None:
            if self.kymograph is None:
                return {}
            kymo = self.kymograph

        events = []

        for spatial_idx in range(kymo.shape[1]):
            trace = kymo[:, spatial_idx]

            if method == 'peaks':
                # Find peaks
                peaks, properties = find_peaks(
                    trace,
                    height=threshold * np.std(trace) + np.mean(trace),
                    distance=min_distance,
                    width=min_duration
                )

                for i, peak_idx in enumerate(peaks):
                    events.append({
                        'time': peak_idx,
                        'position': spatial_idx,
                        'amplitude': properties['peak_heights'][i],
                        'width': properties['widths'][i] if 'widths' in properties else np.nan
                    })

            elif method == 'threshold':
                # Simple threshold crossing
                above_threshold = trace > (threshold * np.std(trace) + np.mean(trace))
                crossings = np.diff(above_threshold.astype(int))
                starts = np.where(crossings == 1)[0]
                ends = np.where(crossings == -1)[0]

                for start, end in zip(starts, ends):
                    if end - start >= min_duration:
                        events.append({
                            'time': start,
                            'position': spatial_idx,
                            'duration': end - start,
                            'amplitude': np.max(trace[start:end])
                        })

            elif method == 'zscore':
                # Z-score based detection
                z_scores = zscore(trace)
                above_threshold = z_scores > threshold
                crossings = np.diff(above_threshold.astype(int))
                starts = np.where(crossings == 1)[0]
                ends = np.where(crossings == -1)[0]

                for start, end in zip(starts, ends):
                    if end - start >= min_duration:
                        events.append({
                            'time': start,
                            'position': spatial_idx,
                            'duration': end - start,
                            'amplitude': np.max(trace[start:end]),
                            'z_score': np.max(z_scores[start:end])
                        })

        self.analysis_results['events'] = events
        return events

    def measure_velocities(self, kymo=None, events=None):
        """
        Measure velocities of events/waves in kymograph.

        Returns:
        --------
        dict : Dictionary with velocity measurements
        """
        if kymo is None:
            kymo = self.kymograph

        if kymo is None:
            return {}

        # Simple approach: detect diagonal features
        # Calculate gradient
        dt_gradient = np.gradient(kymo, axis=0)  # temporal
        dx_gradient = np.gradient(kymo, axis=1)  # spatial

        # Velocity = dx/dt
        with np.errstate(divide='ignore', invalid='ignore'):
            velocity_map = dx_gradient / (dt_gradient + 1e-10)

        # Filter out infinite and nan values
        valid_velocities = velocity_map[np.isfinite(velocity_map)]

        velocities = {
            'mean_velocity': np.mean(valid_velocities),
            'median_velocity': np.median(valid_velocities),
            'std_velocity': np.std(valid_velocities),
            'velocity_map': velocity_map
        }

        self.analysis_results['velocities'] = velocities
        return velocities

    def calculate_statistics(self, kymo=None):
        """
        Calculate comprehensive statistics for kymograph.

        Returns:
        --------
        dict : Dictionary with statistical measures
        """
        if kymo is None:
            kymo = self.kymograph

        if kymo is None:
            return {}

        stats = {
            'mean_intensity': np.mean(kymo),
            'std_intensity': np.std(kymo),
            'min_intensity': np.min(kymo),
            'max_intensity': np.max(kymo),
            'temporal_mean': np.mean(kymo, axis=0),  # Average over time
            'spatial_mean': np.mean(kymo, axis=1),   # Average over space
            'temporal_std': np.std(kymo, axis=0),
            'spatial_std': np.std(kymo, axis=1)
        }

        self.analysis_results['statistics'] = stats
        return stats

    def plot_analysis(self):
        """
        Create comprehensive analysis plots.
        """
        if self.kymograph is None:
            g.m.statusBar().showMessage("No kymograph to analyze")
            return

        # Create figure with subplots
        fig = Figure(figsize=(12, 8))

        # Kymograph
        ax1 = fig.add_subplot(3, 2, 1)
        im = ax1.imshow(self.kymograph, aspect='auto', cmap='viridis',
                       interpolation='nearest')
        ax1.set_title('Kymograph')
        ax1.set_xlabel('Space (pixels)')
        ax1.set_ylabel('Time (frames)')
        fig.colorbar(im, ax=ax1)

        # Temporal average
        ax2 = fig.add_subplot(3, 2, 2)
        temporal_avg = np.mean(self.kymograph, axis=0)
        ax2.plot(temporal_avg)
        ax2.set_title('Spatial Profile (Time Average)')
        ax2.set_xlabel('Position (pixels)')
        ax2.set_ylabel('Intensity')
        ax2.grid(True, alpha=0.3)

        # Spatial average
        ax3 = fig.add_subplot(3, 2, 3)
        spatial_avg = np.mean(self.kymograph, axis=1)
        ax3.plot(spatial_avg)
        ax3.set_title('Temporal Profile (Space Average)')
        ax3.set_xlabel('Time (frames)')
        ax3.set_ylabel('Intensity')
        ax3.grid(True, alpha=0.3)

        # Detect and plot events
        events = self.detect_events(method='peaks', threshold=2.0)
        if events:
            ax4 = fig.add_subplot(3, 2, 4)
            times = [e['time'] for e in events]
            positions = [e['position'] for e in events]
            amplitudes = [e['amplitude'] for e in events]
            scatter = ax4.scatter(positions, times, c=amplitudes, s=50,
                                cmap='hot', alpha=0.6)
            ax4.set_title(f'Detected Events (n={len(events)})')
            ax4.set_xlabel('Position (pixels)')
            ax4.set_ylabel('Time (frames)')
            fig.colorbar(scatter, ax=ax4, label='Amplitude')

        # Intensity distribution
        ax5 = fig.add_subplot(3, 2, 5)
        ax5.hist(self.kymograph.flatten(), bins=50, alpha=0.7, color='blue')
        ax5.set_title('Intensity Distribution')
        ax5.set_xlabel('Intensity')
        ax5.set_ylabel('Count')
        ax5.grid(True, alpha=0.3)

        # Statistics text
        ax6 = fig.add_subplot(3, 2, 6)
        ax6.axis('off')
        stats = self.calculate_statistics()
        stats_text = f"""
        Kymograph Statistics:

        Mean Intensity: {stats['mean_intensity']:.3f}
        Std Intensity: {stats['std_intensity']:.3f}
        Min Intensity: {stats['min_intensity']:.3f}
        Max Intensity: {stats['max_intensity']:.3f}

        Events Detected: {len(events)}
        """
        ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center')

        fig.tight_layout()

        # Create window to display plot
        self.plot_window = QWidget()  # Keep reference as instance variable!
        self.plot_window.setWindowTitle("Kymograph Analysis")
        layout = QVBoxLayout()
        canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas)
        self.plot_window.setLayout(layout)
        self.plot_window.resize(1200, 800)
        self.plot_window.show()

        g.m.statusBar().showMessage(f"Analysis plot created with {len(events)} events detected")

        return fig

    def export_data(self, filename):
        """
        Export kymograph and analysis results.
        """
        if self.kymograph is None:
            return

        import pandas as pd

        # Save kymograph as image
        np.save(filename + '_kymograph.npy', self.kymograph)

        # Save events as CSV
        if 'events' in self.analysis_results and self.analysis_results['events']:
            events_df = pd.DataFrame(self.analysis_results['events'])
            events_df.to_csv(filename + '_events.csv', index=False)

        # Save statistics
        if 'statistics' in self.analysis_results:
            stats_df = pd.DataFrame({k: [v] if np.isscalar(v) else [str(v)]
                                    for k, v in self.analysis_results['statistics'].items()})
            stats_df.to_csv(filename + '_statistics.csv', index=False)

        g.m.statusBar().showMessage(f"Data exported to {filename}_*")

    def gui(self):
        """
        Create graphical user interface.
        """
        self.gui_reset()

        # Window selector
        window_selector = WindowSelector()

        # ROI drawing button
        self.draw_roi_button = QPushButton('Draw Line ROI')
        self.draw_roi_button.clicked.connect(self.start_roi_drawing)

        # ROI type
        roi_types = ['line', 'polyline', 'freehand']
        self.roi_type_combo = ComboBox()
        self.roi_type_combo.addItems(roi_types)

        # Line width
        self.width_spin = pg.SpinBox(int=True, step=1, bounds=[1, 50], value=5)

        # Temporal binning
        self.temporal_bin_spin = pg.SpinBox(int=True, step=1, bounds=[1, 20], value=1)

        # Processing options
        self.normalize_check = CheckBox()
        self.normalize_check.setChecked(True)

        self.detrend_check = CheckBox()
        self.detrend_check.setChecked(False)

        self.gaussian_spin = pg.SpinBox(int=False, step=0.5, bounds=[0, 10],
                                       value=0, decimals=1)

        # Analysis buttons
        self.generate_button = QPushButton('Generate Kymograph')
        self.generate_button.clicked.connect(self.generate_from_gui)

        self.detect_button = QPushButton('Detect Events')
        self.detect_button.clicked.connect(self.detect_events_gui)

        self.analyze_button = QPushButton('Full Analysis & Plot')
        self.analyze_button.clicked.connect(self.plot_analysis)

        self.export_button = QPushButton('Export Data')
        self.export_button.clicked.connect(self.export_gui)

        # Event detection parameters
        self.detection_method_combo = ComboBox()
        self.detection_method_combo.addItems(['peaks', 'threshold', 'zscore'])

        self.threshold_spin = pg.SpinBox(int=False, step=0.1, bounds=[0.1, 10],
                                        value=2.0, decimals=1)

        self.min_distance_spin = pg.SpinBox(int=True, step=1, bounds=[1, 50], value=5)

        self.min_duration_spin = pg.SpinBox(int=True, step=1, bounds=[1, 50], value=3)

        # Build items list
        self.items.append({'name': 'window', 'string': 'Source Window',
                          'object': window_selector})
        self.items.append({'name': 'draw_roi', 'string': '',
                          'object': self.draw_roi_button})
        self.items.append({'name': '', 'string': '--- Kymograph Settings ---',
                          'object': None})
        self.items.append({'name': 'roi_type', 'string': 'ROI Type',
                          'object': self.roi_type_combo})
        self.items.append({'name': 'width', 'string': 'Line Width (pixels)',
                          'object': self.width_spin})
        self.items.append({'name': 'temporal_binning', 'string': 'Temporal Binning',
                          'object': self.temporal_bin_spin})
        self.items.append({'name': '', 'string': '--- Processing ---',
                          'object': None})
        self.items.append({'name': 'normalize', 'string': 'Normalize Intensity',
                          'object': self.normalize_check})
        self.items.append({'name': 'detrend', 'string': 'Detrend',
                          'object': self.detrend_check})
        self.items.append({'name': 'gaussian_sigma', 'string': 'Gaussian Smoothing (sigma)',
                          'object': self.gaussian_spin})
        self.items.append({'name': '', 'string': '--- Event Detection ---',
                          'object': None})
        self.items.append({'name': 'detection_method', 'string': 'Detection Method',
                          'object': self.detection_method_combo})
        self.items.append({'name': 'threshold', 'string': 'Threshold (std or z-score)',
                          'object': self.threshold_spin})
        self.items.append({'name': 'min_distance', 'string': 'Min Distance (frames)',
                          'object': self.min_distance_spin})
        self.items.append({'name': 'min_duration', 'string': 'Min Duration (frames)',
                          'object': self.min_duration_spin})
        self.items.append({'name': '', 'string': '--- Actions ---',
                          'object': None})
        self.items.append({'name': 'generate', 'string': '',
                          'object': self.generate_button})
        self.items.append({'name': 'detect', 'string': '',
                          'object': self.detect_button})
        self.items.append({'name': 'analyze', 'string': '',
                          'object': self.analyze_button})
        self.items.append({'name': 'export', 'string': '',
                          'object': self.export_button})

        super().gui()

    def start_roi_drawing(self):
        """Start interactive ROI drawing."""
        window = self.getValue('window')
        if window:
            # Create line ROI with default points (user can adjust interactively)
            # Start with a line from center-left to center-right
            center_y = window.my // 2
            start_x = window.mx // 4
            end_x = 3 * window.mx // 4
            pts = [[start_x, center_y], [end_x, center_y]]
            roi = makeROI('line', pts, window)
            g.m.statusBar().showMessage("Draw a line along your region of interest")

    def generate_from_gui(self):
        """Generate kymograph from GUI parameters."""
        window = self.getValue('window')
        roi_type = self.getValue('roi_type')
        width = self.getValue('width')
        temporal_binning = self.getValue('temporal_binning')
        normalize = self.getValue('normalize')
        detrend = self.getValue('detrend')
        gaussian_sigma = self.getValue('gaussian_sigma')

        self(window, roi_type, width, temporal_binning, normalize, detrend, gaussian_sigma)

    def detect_events_gui(self):
        """Detect events using GUI parameters and visualize them."""
        if self.kymograph is None:
            g.m.statusBar().showMessage("No kymograph available. Generate one first.")
            return

        method = self.getValue('detection_method')
        threshold = self.getValue('threshold')
        min_distance = self.getValue('min_distance')
        min_duration = self.getValue('min_duration')

        events = self.detect_events(method=method, threshold=threshold,
                                   min_distance=min_distance, min_duration=min_duration)

        if len(events) == 0:
            g.m.statusBar().showMessage("No events detected. Try adjusting parameters.")
            return

        # Create visualization of detected events
        fig = Figure(figsize=(10, 6))

        # Plot kymograph with events overlaid
        ax1 = fig.add_subplot(2, 1, 1)
        im = ax1.imshow(self.kymograph, aspect='auto', cmap='viridis',
                       interpolation='nearest')
        ax1.set_title(f'Kymograph with Detected Events (n={len(events)})')
        ax1.set_xlabel('Space (pixels)')
        ax1.set_ylabel('Time (frames)')

        # Overlay detected events
        times = [e['time'] for e in events]
        positions = [e['position'] for e in events]
        amplitudes = [e.get('amplitude', 1.0) for e in events]

        scatter = ax1.scatter(positions, times, c='red', s=30,
                            marker='x', alpha=0.8, linewidths=2)
        fig.colorbar(im, ax=ax1)

        # Plot event distribution
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.hist(positions, bins=min(50, len(positions)//2 + 1),
                alpha=0.7, color='red', edgecolor='black')
        ax2.set_title('Event Spatial Distribution')
        ax2.set_xlabel('Position (pixels)')
        ax2.set_ylabel('Number of Events')
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()

        # Create window to display events
        self.events_window = QWidget()  # Keep reference!
        self.events_window.setWindowTitle("Detected Events")
        layout = QVBoxLayout()
        canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas)
        self.events_window.setLayout(layout)
        self.events_window.resize(1000, 600)
        self.events_window.show()

        g.m.statusBar().showMessage(f"Detected {len(events)} events - visualization created")

    def export_gui(self):
        """Export data via file dialog."""
        filename, _ = QFileDialog.getSaveFileName(
            self.ui, "Save Kymograph Data", "",
            "All Files (*);;NumPy Files (*.npy)"
        )
        if filename:
            self.export_data(filename.replace('.npy', ''))


# Create plugin instance
kymograph_analyzer = KymographAnalyzer()
