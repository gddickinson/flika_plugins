# tirf_background_corrector/__init__.py
"""
TIRF Background Corrector Plugin for FLIKA
Corrects for uneven illumination and background artifacts typical in TIRF microscopy
"""

import numpy as np
from scipy import ndimage, signal
from scipy.optimize import curve_fit
from skimage import filters, morphology, restoration
import matplotlib.pyplot as plt

from flika import global_vars as g
from flika.window import Window
from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout, QLabel

__version__ = '1.0.0'
__author__ = 'FLIKA Plugin Suite'

class TIRFBackgroundCorrector(BaseProcess):
    """
    Comprehensive background correction for TIRF microscopy images
    """
    
    def __init__(self):
        super().__init__()
        self.background_image = None
        self.illumination_pattern = None
        
    def get_init_settings_dict(self):
        return {
            'correction_method': ['rolling_ball', 'gaussian_high_pass', 'temporal_median', 'polynomial_fit'],
            'rolling_ball_radius': 50,
            'gaussian_sigma': 30,
            'polynomial_order': 3,
            'temporal_percentile': 10,
            'flatfield_correction': True,
            'subtract_camera_offset': True,
            'camera_offset': 100,
            'normalize_intensity': True
        }
    
    def get_params_dict(self):
        params = super().get_params_dict()
        params['correction_method'] = self.correction_method.currentText()
        params['rolling_ball_radius'] = self.rolling_ball_radius.value()
        params['gaussian_sigma'] = self.gaussian_sigma.value()
        params['polynomial_order'] = int(self.polynomial_order.value())
        params['temporal_percentile'] = self.temporal_percentile.value()
        params['flatfield_correction'] = self.flatfield_correction.isChecked()
        params['subtract_camera_offset'] = self.subtract_camera_offset.isChecked()
        params['camera_offset'] = self.camera_offset.value()
        params['normalize_intensity'] = self.normalize_intensity.isChecked()
        return params
    
    def get_name(self):
        return 'TIRF Background Corrector'
    
    def get_menu_path(self):
        return 'Plugins>TIRF Analysis>Background Corrector'
    
    def setupGUI(self):
        super().setupGUI()
        self.rolling_ball_radius.setRange(5, 200)
        self.gaussian_sigma.setRange(5, 100)
        self.polynomial_order.setRange(1, 6)
        self.temporal_percentile.setRange(1, 50)
        self.camera_offset.setRange(0, 1000)
        
        # Add preview button
        self.preview_button = QPushButton("Preview Correction")
        self.preview_button.clicked.connect(self.preview_correction)
        self.layout().addWidget(self.preview_button)
        
        # Add estimate background button
        self.estimate_bg_button = QPushButton("Estimate Background")
        self.estimate_bg_button.clicked.connect(self.estimate_background)
        self.layout().addWidget(self.estimate_bg_button)
    
    def rolling_ball_background(self, image, radius):
        """Rolling ball background subtraction"""
        # Create ball-shaped kernel
        kernel_size = int(2 * radius + 1)
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        kernel = x*x + y*y <= radius*radius
        kernel = kernel.astype(float)
        
        # Morphological opening (erosion followed by dilation)
        background = ndimage.grey_opening(image, structure=kernel)
        
        return background
    
    def gaussian_high_pass_filter(self, image, sigma):
        """High-pass filter using Gaussian blur subtraction"""
        # Low-pass filter (large-scale features)
        low_pass = ndimage.gaussian_filter(image.astype(float), sigma)
        
        return low_pass
    
    def polynomial_background_fit(self, image, order=3):
        """Fit polynomial surface to estimate background"""
        height, width = image.shape
        
        # Create coordinate grids
        y, x = np.mgrid[0:height, 0:width]
        
        # Flatten for fitting
        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = image.flatten()
        
        # Create polynomial basis functions
        A = []
        for i in range(order + 1):
            for j in range(order + 1 - i):
                A.append((x_flat ** i) * (y_flat ** j))
        
        A = np.column_stack(A)
        
        # Robust fitting (exclude outliers)
        try:
            # Initial fit
            coeffs = np.linalg.lstsq(A, z_flat, rcond=None)[0]
            fitted = A @ coeffs
            
            # Exclude outliers (> 2 std from fit)
            residuals = z_flat - fitted
            std_residuals = np.std(residuals)
            good_indices = np.abs(residuals) < 2 * std_residuals
            
            # Refit without outliers
            coeffs = np.linalg.lstsq(A[good_indices], z_flat[good_indices], rcond=None)[0]
            background = (A @ coeffs).reshape(height, width)
            
        except:
            # Fallback to simple Gaussian blur
            background = ndimage.gaussian_filter(image.astype(float), sigma=20)
        
        return background
    
    def temporal_background_estimation(self, image_stack, percentile=10):
        """Estimate background using temporal statistics"""
        # Calculate percentile across time for each pixel
        background = np.percentile(image_stack, percentile, axis=0)
        
        return background
    
    def estimate_flatfield_correction(self, image_stack=None):
        """Estimate illumination pattern for flat-field correction"""
        if image_stack is None:
            if g.win is None:
                return None
            image_stack = g.win.image
        
        # Method 1: Use median projection
        median_projection = np.median(image_stack, axis=0)
        
        # Method 2: Gaussian blur of median to get smooth illumination pattern
        illumination_pattern = ndimage.gaussian_filter(median_projection, sigma=30)
        
        # Normalize to avoid division by zero
        illumination_pattern = np.maximum(illumination_pattern, 0.1 * np.mean(illumination_pattern))
        
        # Normalize to mean = 1
        illumination_pattern = illumination_pattern / np.mean(illumination_pattern)
        
        return illumination_pattern
    
    def estimate_background(self):
        """Estimate and display background"""
        if g.win is None:
            g.alert("No window open!")
            return
        
        params = self.get_params_dict()
        image_stack = g.win.image
        
        g.m.statusBar().showMessage("Estimating background...")
        
        if params['correction_method'] == 'temporal_median':
            self.background_image = self.temporal_background_estimation(
                image_stack, params['temporal_percentile']
            )
        else:
            # Use middle frame for other methods
            middle_frame = image_stack[image_stack.shape[0] // 2]
            
            if params['correction_method'] == 'rolling_ball':
                self.background_image = self.rolling_ball_background(
                    middle_frame, params['rolling_ball_radius']
                )
            elif params['correction_method'] == 'gaussian_high_pass':
                self.background_image = self.gaussian_high_pass_filter(
                    middle_frame, params['gaussian_sigma']
                )
            elif params['correction_method'] == 'polynomial_fit':
                self.background_image = self.polynomial_background_fit(
                    middle_frame, params['polynomial_order']
                )
        
        # Estimate illumination pattern if requested
        if params['flatfield_correction']:
            self.illumination_pattern = self.estimate_flatfield_correction(image_stack)
            Window(self.illumination_pattern, name=f"{g.win.name}_illumination_pattern")
        
        # Display background
        Window(self.background_image, name=f"{g.win.name}_background")
        
        g.m.statusBar().showMessage("Background estimation complete", 2000)
    
    def preview_correction(self):
        """Preview correction on a single frame"""
        if g.win is None:
            g.alert("No window open!")
            return
        
        params = self.get_params_dict()
        
        # Use middle frame for preview
        middle_frame_idx = g.win.image.shape[0] // 2
        original_frame = g.win.image[middle_frame_idx].astype(float)
        
        # Apply correction
        corrected_frame = self.apply_correction_to_frame(original_frame, params)
        
        # Create comparison
        comparison = np.hstack([original_frame, corrected_frame])
        Window(comparison, name=f"{g.win.name}_correction_preview")
    
    def apply_correction_to_frame(self, frame, params):
        """Apply background correction to a single frame"""
        corrected = frame.astype(float)
        
        # Subtract camera offset
        if params['subtract_camera_offset']:
            corrected = corrected - params['camera_offset']
        
        # Background subtraction
        if params['correction_method'] == 'rolling_ball':
            background = self.rolling_ball_background(frame, params['rolling_ball_radius'])
            corrected = corrected - background
            
        elif params['correction_method'] == 'gaussian_high_pass':
            background = self.gaussian_high_pass_filter(frame, params['gaussian_sigma'])
            corrected = corrected - background
            
        elif params['correction_method'] == 'polynomial_fit':
            background = self.polynomial_background_fit(frame, params['polynomial_order'])
            corrected = corrected - background
            
        elif params['correction_method'] == 'temporal_median' and self.background_image is not None:
            corrected = corrected - self.background_image
        
        # Flat-field correction
        if params['flatfield_correction'] and self.illumination_pattern is not None:
            corrected = corrected / self.illumination_pattern
        
        # Normalize intensity
        if params['normalize_intensity']:
            corrected = corrected - np.min(corrected)
            if np.max(corrected) > 0:
                corrected = corrected / np.max(corrected) * np.max(frame)
        
        # Ensure non-negative values
        corrected = np.maximum(corrected, 0)
        
        return corrected
    
    def process(self):
        """Apply background correction to entire stack"""
        if g.win is None:
            g.alert("No window open!")
            return
        
        params = self.get_params_dict()
        image_stack = g.win.image
        
        g.m.statusBar().showMessage("Applying background correction...")
        
        # Pre-calculate background and illumination if needed
        if params['correction_method'] == 'temporal_median':
            self.background_image = self.temporal_background_estimation(
                image_stack, params['temporal_percentile']
            )
        
        if params['flatfield_correction']:
            self.illumination_pattern = self.estimate_flatfield_correction(image_stack)
        
        # Process each frame
        corrected_stack = np.zeros_like(image_stack, dtype=np.float32)
        
        for i in range(image_stack.shape[0]):
            frame = image_stack[i]
            corrected_stack[i] = self.apply_correction_to_frame(frame, params)
            
            # Update progress
            if i % 10 == 0:
                progress = int((i / image_stack.shape[0]) * 100)
                g.m.statusBar().showMessage(f"Processing frame {i+1}/{image_stack.shape[0]} ({progress}%)")
        
        g.m.statusBar().showMessage("Background correction complete", 2000)
        
        # Create new window with corrected data
        new_window = Window(corrected_stack, name=f"{g.win.name}_bg_corrected")
        return new_window

def create_tirf_preprocessing_workflow():
    """Create a complete TIRF preprocessing workflow"""
    if g.win is None:
        g.alert("No window open!")
        return
    
    g.alert("Starting TIRF preprocessing workflow...")
    
    # Step 1: Background correction
    corrector = TIRFBackgroundCorrector()
    corrected_window = corrector.process()
    
    if corrected_window:
        corrected_window.setAsCurrentWindow()
        
        # Step 2: Optional additional filtering
        from flika.process.filters import gaussian_blur
        gaussian_blur(sigma=0.5)  # Light smoothing
        
        g.alert("TIRF preprocessing workflow complete!")

# Register the plugin and workflow
TIRFBackgroundCorrector.menu_path = 'Plugins>TIRF Analysis>Background Corrector'
create_tirf_preprocessing_workflow.menu_path = 'Plugins>TIRF Analysis>Complete Preprocessing Workflow'