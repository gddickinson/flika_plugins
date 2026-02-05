# synthetic_data_generator.py
"""
Synthetic TIRF Data Generator for Testing and Tutorials
Creates realistic synthetic TIRF microscopy datasets for plugin testing
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import filters
from flika import global_vars as g
from flika.window import Window

__version__ = '1.0.0'
__author__ = 'FLIKA Plugin Suite'

class SyntheticTIRFGenerator:
    """Generate realistic synthetic TIRF microscopy data"""
    
    def __init__(self, image_size=(100, 100), n_frames=50, pixel_size=0.1, frame_rate=10):
        self.image_size = image_size
        self.n_frames = n_frames
        self.pixel_size = pixel_size  # µm per pixel
        self.frame_rate = frame_rate  # Hz
        
        # Imaging parameters
        self.psf_sigma = 1.2  # pixels
        self.background_level = 100
        self.noise_level = 20
        
    def generate_psf(self, center, intensity=1000):
        """Generate a 2D Gaussian PSF"""
        y, x = np.ogrid[0:self.image_size[0], 0:self.image_size[1]]
        psf = intensity * np.exp(-((x - center[0])**2 + (y - center[1])**2) / (2 * self.psf_sigma**2))
        return psf
    
    def add_noise(self, image):
        """Add realistic camera noise (Poisson + Gaussian)"""
        # Poisson noise (photon shot noise)
        noisy = np.random.poisson(np.maximum(image, 0))
        
        # Gaussian noise (readout noise)
        noisy = noisy + np.random.normal(0, self.noise_level, image.shape)
        
        return noisy.astype(np.float32)
    
    def generate_single_molecule_data(self, n_molecules=20, diffusion_coeff=1.0):
        """Generate single molecule tracking test data"""
        
        print(f"Generating single molecule data: {n_molecules} molecules, D={diffusion_coeff} µm²/s")
        
        # Initialize molecules
        molecules = []
        for i in range(n_molecules):
            # Random starting position
            start_pos = [
                np.random.uniform(10, self.image_size[1] - 10),
                np.random.uniform(10, self.image_size[0] - 10)
            ]
            
            # Random intensity
            intensity = np.random.uniform(800, 1500)
            
            # Random appearance/disappearance frames
            appear_frame = np.random.randint(0, self.n_frames // 3)
            disappear_frame = np.random.randint(2 * self.n_frames // 3, self.n_frames)
            
            molecules.append({
                'start_pos': start_pos,
                'intensity': intensity,
                'appear_frame': appear_frame,
                'disappear_frame': disappear_frame
            })
        
        # Generate image stack
        image_stack = np.zeros((self.n_frames, self.image_size[0], self.image_size[1]))
        
        # Track positions for ground truth
        ground_truth_tracks = []
        
        for mol_idx, mol in enumerate(molecules):
            track = []
            current_pos = mol['start_pos'].copy()
            
            for frame in range(self.n_frames):
                if mol['appear_frame'] <= frame <= mol['disappear_frame']:
                    # Add molecule to image
                    psf = self.generate_psf(current_pos, mol['intensity'])
                    image_stack[frame] += psf
                    
                    # Record position
                    track.append({
                        'frame': frame,
                        'x': current_pos[0],
                        'y': current_pos[1],
                        'intensity': mol['intensity']
                    })
                    
                    # Random walk for next frame
                    if frame < self.n_frames - 1:
                        # Diffusion step size
                        step_size = np.sqrt(4 * diffusion_coeff * (1/self.frame_rate) / (self.pixel_size**2))
                        
                        # Random step
                        dx = np.random.normal(0, step_size)
                        dy = np.random.normal(0, step_size)
                        
                        current_pos[0] += dx
                        current_pos[1] += dy
                        
                        # Boundary conditions (reflective)
                        current_pos[0] = np.clip(current_pos[0], 5, self.image_size[1] - 5)
                        current_pos[1] = np.clip(current_pos[1], 5, self.image_size[0] - 5)
            
            if track:
                ground_truth_tracks.append({
                    'molecule_id': mol_idx,
                    'track': track
                })
        
        # Add background and noise
        for frame in range(self.n_frames):
            # Uniform background
            image_stack[frame] += self.background_level
            
            # Add noise
            image_stack[frame] = self.add_noise(image_stack[frame])
        
        # Create FLIKA window
        window = Window(image_stack, name="Synthetic_Single_Molecules")
        window.setAsCurrentWindow()
        
        return window, ground_truth_tracks
    
    def generate_photobleaching_data(self, n_spots=30, oligomer_states=[1, 2, 3, 4]):
        """Generate photobleaching step counting test data"""
        
        print(f"Generating photobleaching data: {n_spots} spots")
        
        image_stack = np.zeros((self.n_frames, self.image_size[0], self.image_size[1]))
        ground_truth_spots = []
        
        for spot_idx in range(n_spots):
            # Random position
            pos = [
                np.random.uniform(10, self.image_size[1] - 10),
                np.random.uniform(10, self.image_size[0] - 10)
            ]
            
            # Random oligomer state
            n_subunits = np.random.choice(oligomer_states)
            subunit_intensity = np.random.uniform(200, 400)
            
            # Photobleaching kinetics
            bleach_rates = np.random.exponential(0.02, n_subunits)  # per frame
            
            # Generate intensity trace
            intensity_trace = []
            remaining_subunits = n_subunits
            
            for frame in range(self.n_frames):
                # Check for bleaching events
                for subunit in range(remaining_subunits):
                    if np.random.random() < bleach_rates[subunit]:
                        remaining_subunits -= 1
                        break
                
                # Current intensity
                current_intensity = remaining_subunits * subunit_intensity
                intensity_trace.append(current_intensity)
                
                # Add to image
                if current_intensity > 0:
                    psf = self.generate_psf(pos, current_intensity)
                    image_stack[frame] += psf
            
            ground_truth_spots.append({
                'spot_id': spot_idx,
                'position': pos,
                'n_subunits': n_subunits,
                'intensity_trace': intensity_trace
            })
        
        # Add background and noise
        for frame in range(self.n_frames):
            image_stack[frame] += self.background_level
            image_stack[frame] = self.add_noise(image_stack[frame])
        
        window = Window(image_stack, name="Synthetic_Photobleaching")
        window.setAsCurrentWindow()
        
        return window, ground_truth_spots
    
    def generate_membrane_dynamics_data(self):
        """Generate membrane dynamics test data"""
        
        print("Generating membrane dynamics data")
        
        image_stack = np.zeros((self.n_frames, self.image_size[0], self.image_size[1]))
        
        # Cell parameters
        cell_center = [self.image_size[1] // 2, self.image_size[0] // 2]
        base_radius = min(self.image_size) // 3
        membrane_width = 3
        membrane_intensity = 800
        
        for frame in range(self.n_frames):
            # Create cell outline with dynamics
            y, x = np.ogrid[0:self.image_size[0], 0:self.image_size[1]]
            
            # Base circular cell
            distance_from_center = np.sqrt((x - cell_center[0])**2 + (y - cell_center[1])**2)
            
            # Add protrusions/retractions
            angle = np.arctan2(y - cell_center[1], x - cell_center[0])
            
            # Dynamic perturbations
            perturbation = 0
            for mode in range(3, 8):  # Different spatial modes
                phase = 2 * np.pi * frame / self.n_frames * 0.5  # Temporal evolution
                amplitude = np.random.uniform(2, 8)
                perturbation += amplitude * np.sin(mode * angle + phase)
            
            # Create membrane
            effective_radius = base_radius + perturbation
            membrane_mask = np.abs(distance_from_center - effective_radius) < membrane_width
            
            # Add membrane to image
            image_stack[frame][membrane_mask] = membrane_intensity
            
            # Add some internal structure
            internal_mask = distance_from_center < (effective_radius - membrane_width)
            image_stack[frame][internal_mask] += np.random.uniform(50, 150)
        
        # Smooth and add noise
        for frame in range(self.n_frames):
            image_stack[frame] = ndimage.gaussian_filter(image_stack[frame], sigma=0.8)
            image_stack[frame] += self.background_level
            image_stack[frame] = self.add_noise(image_stack[frame])
        
        window = Window(image_stack, name="Synthetic_Membrane_Dynamics")
        window.setAsCurrentWindow()
        
        return window
    
    def generate_frap_data(self, mobile_fraction=0.7, diffusion_time=10):
        """Generate FRAP recovery test data"""
        
        print(f"Generating FRAP data: mobile fraction={mobile_fraction}, τ={diffusion_time} frames")
        
        image_stack = np.zeros((self.n_frames, self.image_size[0], self.image_size[1]))
        
        # Cell and bleach region
        cell_center = [self.image_size[1] // 2, self.image_size[0] // 2]
        cell_radius = min(self.image_size) // 3
        bleach_center = [cell_center[0] + 10, cell_center[1]]
        bleach_radius = 8
        
        # Pre-bleach intensity
        prebleach_intensity = 600
        bleach_frame = 10
        
        y, x = np.ogrid[0:self.image_size[0], 0:self.image_size[1]]
        cell_mask = (x - cell_center[0])**2 + (y - cell_center[1])**2 < cell_radius**2
        bleach_mask = (x - bleach_center[0])**2 + (y - bleach_center[1])**2 < bleach_radius**2
        
        for frame in range(self.n_frames):
            # Base cell intensity
            image_stack[frame][cell_mask] = prebleach_intensity
            
            if frame >= bleach_frame:
                # FRAP recovery
                time_after_bleach = frame - bleach_frame
                
                # Recovery model (single exponential)
                recovery = mobile_fraction * (1 - np.exp(-time_after_bleach / diffusion_time))
                bleach_intensity = prebleach_intensity * (1 - mobile_fraction + recovery)
                
                image_stack[frame][bleach_mask & cell_mask] = bleach_intensity
            
            # Add background and noise
            image_stack[frame] += self.background_level
            image_stack[frame] = self.add_noise(image_stack[frame])
        
        window = Window(image_stack, name="Synthetic_FRAP")
        window.setAsCurrentWindow()
        
        return window
    
    def generate_cluster_data(self, n_clusters=15, cluster_sizes=[5, 10, 20]):
        """Generate protein cluster test data"""
        
        print(f"Generating cluster data: {n_clusters} clusters")
        
        image_stack = np.zeros((self.n_frames, self.image_size[0], self.image_size[1]))
        ground_truth_clusters = []
        
        for cluster_idx in range(n_clusters):
            # Random cluster properties
            center = [
                np.random.uniform(15, self.image_size[1] - 15),
                np.random.uniform(15, self.image_size[0] - 15)
            ]
            
            cluster_size = np.random.choice(cluster_sizes)
            cluster_intensity = np.random.uniform(300, 800)
            
            # Cluster evolution
            appear_frame = np.random.randint(0, self.n_frames // 4)
            stable_frames = np.random.randint(self.n_frames // 2, 3 * self.n_frames // 4)
            
            for frame in range(self.n_frames):
                if frame >= appear_frame and frame < appear_frame + stable_frames:
                    # Generate cluster shape (elongated Gaussian)
                    aspect_ratio = np.random.uniform(1, 3)
                    angle = np.random.uniform(0, 2 * np.pi)
                    
                    # Create rotated elliptical cluster
                    y, x = np.ogrid[0:self.image_size[0], 0:self.image_size[1]]
                    
                    # Rotate coordinates
                    x_rot = (x - center[0]) * np.cos(angle) + (y - center[1]) * np.sin(angle)
                    y_rot = -(x - center[0]) * np.sin(angle) + (y - center[1]) * np.cos(angle)
                    
                    # Elliptical Gaussian
                    sigma_x = cluster_size / 2
                    sigma_y = sigma_x / aspect_ratio
                    
                    cluster_profile = cluster_intensity * np.exp(
                        -(x_rot**2 / (2 * sigma_x**2) + y_rot**2 / (2 * sigma_y**2))
                    )
                    
                    image_stack[frame] += cluster_profile
            
            ground_truth_clusters.append({
                'cluster_id': cluster_idx,
                'center': center,
                'size': cluster_size,
                'intensity': cluster_intensity,
                'appear_frame': appear_frame,
                'duration': stable_frames
            })
        
        # Add background and noise
        for frame in range(self.n_frames):
            image_stack[frame] += self.background_level
            image_stack[frame] = self.add_noise(image_stack[frame])
        
        window = Window(image_stack, name="Synthetic_Clusters")
        window.setAsCurrentWindow()
        
        return window, ground_truth_clusters
    
    def generate_colocalization_data(self, colocalization_fraction=0.6):
        """Generate two-channel colocalization test data"""
        
        print(f"Generating colocalization data: {colocalization_fraction*100}% colocalized")
        
        n_spots_ch1 = 40
        n_spots_ch2 = 35
        n_colocalized = int(min(n_spots_ch1, n_spots_ch2) * colocalization_fraction)
        
        # Channel 1
        image_stack_ch1 = np.zeros((self.n_frames, self.image_size[0], self.image_size[1]))
        
        # Channel 2  
        image_stack_ch2 = np.zeros((self.n_frames, self.image_size[0], self.image_size[1]))
        
        spots_ch1 = []
        spots_ch2 = []
        
        # Generate colocalized spots
        for i in range(n_colocalized):
            pos = [
                np.random.uniform(10, self.image_size[1] - 10),
                np.random.uniform(10, self.image_size[0] - 10)
            ]
            
            # Small offset between channels (registration error)
            offset = np.random.normal(0, 0.3, 2)
            pos_ch2 = [pos[0] + offset[0], pos[1] + offset[1]]
            
            intensity_ch1 = np.random.uniform(800, 1200)
            intensity_ch2 = np.random.uniform(600, 1000)
            
            spots_ch1.append({'pos': pos, 'intensity': intensity_ch1, 'colocalized': True})
            spots_ch2.append({'pos': pos_ch2, 'intensity': intensity_ch2, 'colocalized': True})
        
        # Generate non-colocalized spots for channel 1
        for i in range(n_spots_ch1 - n_colocalized):
            pos = [
                np.random.uniform(10, self.image_size[1] - 10),
                np.random.uniform(10, self.image_size[0] - 10)
            ]
            intensity = np.random.uniform(800, 1200)
            spots_ch1.append({'pos': pos, 'intensity': intensity, 'colocalized': False})
        
        # Generate non-colocalized spots for channel 2
        for i in range(n_spots_ch2 - n_colocalized):
            pos = [
                np.random.uniform(10, self.image_size[1] - 10),
                np.random.uniform(10, self.image_size[0] - 10)
            ]
            intensity = np.random.uniform(600, 1000)
            spots_ch2.append({'pos': pos, 'intensity': intensity, 'colocalized': False})
        
        # Generate images
        for frame in range(self.n_frames):
            # Channel 1
            for spot in spots_ch1:
                psf = self.generate_psf(spot['pos'], spot['intensity'])
                image_stack_ch1[frame] += psf
            
            # Channel 2  
            for spot in spots_ch2:
                psf = self.generate_psf(spot['pos'], spot['intensity'])
                image_stack_ch2[frame] += psf
            
            # Add background and noise
            image_stack_ch1[frame] += self.background_level
            image_stack_ch2[frame] += self.background_level
            
            image_stack_ch1[frame] = self.add_noise(image_stack_ch1[frame])
            image_stack_ch2[frame] = self.add_noise(image_stack_ch2[frame])
        
        # Create windows
        window_ch1 = Window(image_stack_ch1, name="Synthetic_Channel1")
        window_ch2 = Window(image_stack_ch2, name="Synthetic_Channel2")
        
        window_ch1.setAsCurrentWindow()
        
        return window_ch1, window_ch2, {'spots_ch1': spots_ch1, 'spots_ch2': spots_ch2}

# Convenience functions for menu integration
def generate_single_molecule_test_data():
    """Generate single molecule tracking test data"""
    generator = SyntheticTIRFGenerator(image_size=(128, 128), n_frames=100)
    window, tracks = generator.generate_single_molecule_data(n_molecules=25, diffusion_coeff=1.5)
    g.alert(f"Generated single molecule test data with {len(tracks)} tracks")

def generate_photobleaching_test_data():
    """Generate photobleaching analysis test data"""
    generator = SyntheticTIRFGenerator(image_size=(100, 100), n_frames=80)
    window, spots = generator.generate_photobleaching_data(n_spots=40)
    g.alert(f"Generated photobleaching test data with {len(spots)} spots")

def generate_membrane_test_data():
    """Generate membrane dynamics test data"""
    generator = SyntheticTIRFGenerator(image_size=(150, 150), n_frames=60)
    window = generator.generate_membrane_dynamics_data()
    g.alert("Generated membrane dynamics test data")

def generate_frap_test_data():
    """Generate FRAP analysis test data"""
    generator = SyntheticTIRFGenerator(image_size=(80, 80), n_frames=50)
    window = generator.generate_frap_data(mobile_fraction=0.75, diffusion_time=8)
    g.alert("Generated FRAP test data")

def generate_cluster_test_data():
    """Generate cluster analysis test data"""
    generator = SyntheticTIRFGenerator(image_size=(120, 120), n_frames=40)
    window, clusters = generator.generate_cluster_data(n_clusters=20)
    g.alert(f"Generated cluster test data with {len(clusters)} clusters")

def generate_colocalization_test_data():
    """Generate colocalization analysis test data"""
    generator = SyntheticTIRFGenerator(image_size=(100, 100), n_frames=30)
    ch1, ch2, data = generator.generate_colocalization_data(colocalization_fraction=0.65)
    g.alert("Generated two-channel colocalization test data")

def generate_complete_test_suite():
    """Generate all test datasets for comprehensive testing"""
    g.alert("Generating complete synthetic test suite...")
    
    generate_single_molecule_test_data()
    generate_photobleaching_test_data() 
    generate_membrane_test_data()
    generate_frap_test_data()
    generate_cluster_test_data()
    generate_colocalization_test_data()
    
    g.alert("Complete synthetic test suite generated! Check your windows for all datasets.")

# Register functions in FLIKA menu
generate_single_molecule_test_data.menu_path = 'Plugins>TIRF Analysis>Utilities>Generate Test Data>Single Molecule Data'
generate_photobleaching_test_data.menu_path = 'Plugins>TIRF Analysis>Utilities>Generate Test Data>Photobleaching Data'
generate_membrane_test_data.menu_path = 'Plugins>TIRF Analysis>Utilities>Generate Test Data>Membrane Dynamics Data'
generate_frap_test_data.menu_path = 'Plugins>TIRF Analysis>Utilities>Generate Test Data>FRAP Data'
generate_cluster_test_data.menu_path = 'Plugins>TIRF Analysis>Utilities>Generate Test Data>Cluster Data'
generate_colocalization_test_data.menu_path = 'Plugins>TIRF Analysis>Utilities>Generate Test Data>Colocalization Data'
generate_complete_test_suite.menu_path = 'Plugins>TIRF Analysis>Utilities>Generate Test Data>Complete Test Suite'