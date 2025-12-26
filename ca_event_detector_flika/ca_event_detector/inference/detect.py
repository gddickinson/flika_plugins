"""
Inference and post-processing for calcium event detection.
Implements the detection pipeline described in the paper.
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import label, maximum_filter
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from typing import Tuple, Optional, Dict
import warnings

from ca_event_detector.models.unet3d import UNet3D
from ca_event_detector.configs.config import Config


class CalciumEventDetector:
    """
    Detector for calcium events in confocal imaging data.
    """

    def __init__(self, model_path: str, config: Optional[Config] = None, device: str = 'cuda'):
        """
        Initialize detector.

        Args:
            model_path: Path to trained model weights
            config: Configuration object (will be loaded from checkpoint if None)
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Get config from checkpoint or use provided
        if config is None and 'config' in checkpoint:
            self.config = checkpoint['config']
        elif config is None:
            self.config = Config()
        else:
            self.config = config

        # Create model
        self.model = UNet3D(
            in_channels=self.config.model.in_channels,
            num_classes=self.config.model.num_classes,
            base_channels=self.config.model.base_channels
        ).to(self.device)

        # Load weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()

        print(f"Model loaded from {model_path}")
        print(f"Device: {self.device}")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for inference.

        Args:
            image: (T, H, W) array

        Returns:
            Normalized image array
        """
        image = image.astype(np.float32)
        image = (image - self.config.data.normalize_min) / \
                (self.config.data.normalize_max - self.config.data.normalize_min)
        return image

    def predict_probabilities(self, image: np.ndarray) -> np.ndarray:
        """
        Run model inference to get probability maps.

        Args:
            image: (T, H, W) array

        Returns:
            (C, T, H, W) probability maps
        """
        # Preprocess
        image = self.preprocess(image)

        # Split into overlapping segments
        segment_length = self.config.data.segment_length
        step_size = self.config.data.step_size
        temporal_context = self.config.data.temporal_context

        T, H, W = image.shape
        num_classes = self.config.model.num_classes

        # Initialize output probability map
        prob_map = np.zeros((num_classes, T, H, W), dtype=np.float32)
        count_map = np.zeros((T, H, W), dtype=np.float32)

        with torch.no_grad():
            for start in range(0, T - segment_length + 1, step_size):
                end = start + segment_length

                # Extract segment
                segment = image[start:end]

                # Convert to tensor: (1, 1, T, H, W)
                segment_tensor = torch.from_numpy(segment[np.newaxis, np.newaxis, ...])
                segment_tensor = segment_tensor.to(self.device)

                # Forward pass
                logits = self.model(segment_tensor)
                probas = F.softmax(logits, dim=1)

                # Move to CPU and convert to numpy
                probas = probas.cpu().numpy()[0]  # (C, T, H, W)

                # Ignore temporal context frames
                if temporal_context > 0:
                    valid_start = temporal_context
                    valid_end = segment_length - temporal_context
                    probas = probas[:, valid_start:valid_end]

                    output_start = start + valid_start
                    output_end = start + valid_end
                else:
                    output_start = start
                    output_end = end

                # Accumulate predictions
                prob_map[:, output_start:output_end] += probas
                count_map[output_start:output_end] += 1

        # Average overlapping predictions
        count_map = np.maximum(count_map, 1)  # Avoid division by zero
        prob_map = prob_map / count_map[np.newaxis, ...]

        return prob_map

    def segment_events(self, prob_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment events from probability maps using Otsu thresholding.

        Args:
            prob_map: (C, T, H, W) probability maps

        Returns:
            class_mask: (T, H, W) class labels
            binary_mask: (T, H, W) binary foreground mask
        """
        num_classes, T, H, W = prob_map.shape

        # Get class predictions
        class_predictions = np.argmax(prob_map, axis=0)  # (T, H, W)

        # Compute binary mask using Otsu thresholding on max probability
        max_prob = np.max(prob_map, axis=0)  # (T, H, W)

        try:
            threshold = threshold_otsu(max_prob.flatten())
        except:
            # Fallback to fixed threshold if Otsu fails
            threshold = 0.5

        binary_mask = max_prob > threshold

        # Apply binary mask to class predictions
        class_mask = class_predictions * binary_mask

        return class_mask, binary_mask

    def detect_spark_peaks(self, image: np.ndarray, spark_mask: np.ndarray) -> np.ndarray:
        """
        Detect peaks of Ca2+ sparks for instance separation.

        Args:
            image: (T, H, W) original image
            spark_mask: (T, H, W) binary mask of Ca2+ sparks

        Returns:
            (N, 3) array of peak coordinates (t, y, x)
        """
        # Mask the image
        masked_image = image * spark_mask

        # Convert distances to pixels/frames
        min_distance_pixels = int(self.config.inference.spark_min_spatial_distance_um /
                                 self.config.inference.pixel_size_um)
        min_distance_frames = int(self.config.inference.spark_min_temporal_distance_ms /
                                self.config.inference.frame_rate_ms)

        min_distance = (min_distance_frames, min_distance_pixels, min_distance_pixels)

        # Find local maxima
        coordinates = peak_local_max(
            masked_image,
            min_distance=min_distance,
            threshold_abs=0,
            exclude_border=False
        )

        return coordinates

    def separate_spark_instances(self, spark_mask: np.ndarray, peaks: np.ndarray) -> np.ndarray:
        """
        Separate individual Ca2+ spark instances using watershed.

        Args:
            spark_mask: (T, H, W) binary mask of Ca2+ sparks
            peaks: (N, 3) peak coordinates

        Returns:
            (T, H, W) instance mask with unique labels
        """
        if len(peaks) == 0:
            return np.zeros_like(spark_mask, dtype=np.int32)

        # Create markers from peaks
        markers = np.zeros_like(spark_mask, dtype=np.int32)
        for i, (t, y, x) in enumerate(peaks):
            markers[t, y, x] = i + 1

        # Run watershed
        instance_mask = watershed(-spark_mask.astype(float), markers, mask=spark_mask)

        return instance_mask

    def separate_puff_wave_instances(self, mask: np.ndarray) -> np.ndarray:
        """
        Separate Ca2+ puff and wave instances using connected components.

        Args:
            mask: (T, H, W) binary mask

        Returns:
            (T, H, W) instance mask with unique labels
        """
        # Label connected components
        labeled_mask, num_features = label(mask)

        # Merge instances separated by small gaps
        merge_gap = self.config.inference.merge_gap_frames

        if merge_gap > 0:
            for label_id in range(1, num_features + 1):
                component_mask = (labeled_mask == label_id)

                # Find temporal extent
                t_coords = np.where(np.any(component_mask, axis=(1, 2)))[0]
                if len(t_coords) == 0:
                    continue

                t_min, t_max = t_coords[0], t_coords[-1]

                # Check for nearby instances in time
                for t in range(max(0, t_min - merge_gap), min(mask.shape[0], t_max + merge_gap + 1)):
                    if t < t_min or t > t_max:
                        # Check for spatial overlap
                        spatial_mask_current = component_mask[t_min] if t < t_min else component_mask[t_max]
                        spatial_mask_check = labeled_mask[t]

                        if np.any((spatial_mask_current > 0) & (spatial_mask_check > 0) &
                                (spatial_mask_check != label_id)):
                            # Merge
                            merge_label = labeled_mask[t][spatial_mask_current > 0][0]
                            labeled_mask[labeled_mask == merge_label] = label_id

        return labeled_mask

    def filter_events(self, class_mask: np.ndarray, instance_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter detected events based on size constraints.

        Args:
            class_mask: (T, H, W) class labels
            instance_mask: (T, H, W) instance labels

        Returns:
            Filtered class_mask and instance_mask
        """
        # Convert sizes to pixels/frames
        min_spark_duration = int(self.config.inference.spark_min_duration_ms /
                               self.config.inference.frame_rate_ms)
        min_spark_diameter = int(self.config.inference.spark_min_diameter_um /
                               self.config.inference.pixel_size_um)
        min_puff_duration = int(self.config.inference.min_puff_duration_ms /
                              self.config.inference.frame_rate_ms)
        min_wave_diameter = int(self.config.inference.min_wave_diameter_um /
                              self.config.inference.pixel_size_um)

        filtered_class_mask = class_mask.copy()
        filtered_instance_mask = instance_mask.copy()

        unique_instances = np.unique(instance_mask)
        unique_instances = unique_instances[unique_instances > 0]

        for instance_id in unique_instances:
            instance_pixels = (instance_mask == instance_id)

            # Get class of this instance
            class_id = np.median(class_mask[instance_pixels])

            # Check duration
            temporal_extent = np.sum(np.any(instance_pixels, axis=(1, 2)))

            # Check spatial extent
            spatial_coords = np.where(np.any(instance_pixels, axis=0))
            if len(spatial_coords[0]) > 0:
                spatial_extent = max(
                    np.max(spatial_coords[0]) - np.min(spatial_coords[0]),
                    np.max(spatial_coords[1]) - np.min(spatial_coords[1])
                )
            else:
                spatial_extent = 0

            # Filter based on class
            should_remove = False

            if class_id == 1:  # Ca2+ spark
                if temporal_extent < min_spark_duration or spatial_extent < min_spark_diameter:
                    should_remove = True

            elif class_id == 2:  # Ca2+ puff
                if temporal_extent < min_puff_duration:
                    should_remove = True

            elif class_id == 3:  # Ca2+ wave
                if spatial_extent < min_wave_diameter:
                    should_remove = True

            if should_remove:
                filtered_class_mask[instance_pixels] = 0
                filtered_instance_mask[instance_pixels] = 0

        return filtered_class_mask, filtered_instance_mask

    def detect(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Complete detection pipeline.

        Args:
            image: (T, H, W) input image

        Returns:
            Dictionary containing:
                - 'probabilities': (C, T, H, W) class probabilities
                - 'class_mask': (T, H, W) classified events
                - 'instance_mask': (T, H, W) individual event instances
        """
        print("Running model inference...")
        prob_map = self.predict_probabilities(image)

        print("Segmenting events...")
        class_mask, binary_mask = self.segment_events(prob_map)

        print("Detecting instances...")

        # Separate instances for each class
        spark_mask = (class_mask == 1)
        puff_mask = (class_mask == 2)
        wave_mask = (class_mask == 3)

        # Ca2+ sparks: peak detection + watershed
        if np.any(spark_mask):
            spark_peaks = self.detect_spark_peaks(image, spark_mask)
            spark_instances = self.separate_spark_instances(spark_mask, spark_peaks)
        else:
            spark_instances = np.zeros_like(spark_mask, dtype=np.int32)

        # Ca2+ puffs: connected components
        if np.any(puff_mask):
            puff_instances = self.separate_puff_wave_instances(puff_mask)
        else:
            puff_instances = np.zeros_like(puff_mask, dtype=np.int32)

        # Ca2+ waves: connected components
        if np.any(wave_mask):
            wave_instances = self.separate_puff_wave_instances(wave_mask)
        else:
            wave_instances = np.zeros_like(wave_mask, dtype=np.int32)

        # Combine instance masks with unique labels
        instance_mask = np.zeros_like(class_mask, dtype=np.int32)

        max_spark_id = spark_instances.max() if spark_instances.max() > 0 else 0
        max_puff_id = puff_instances.max() if puff_instances.max() > 0 else 0

        instance_mask[spark_mask] = spark_instances[spark_mask]
        instance_mask[puff_mask] = puff_instances[puff_mask] + max_spark_id
        instance_mask[wave_mask] = wave_instances[wave_mask] + max_spark_id + max_puff_id

        print("Filtering events...")
        class_mask, instance_mask = self.filter_events(class_mask, instance_mask)

        print(f"Detected {np.unique(instance_mask).max()} total events")
        print(f"  - Ca2+ sparks: {np.sum(class_mask == 1) > 0}")
        print(f"  - Ca2+ puffs: {np.sum(class_mask == 2) > 0}")
        print(f"  - Ca2+ waves: {np.sum(class_mask == 3) > 0}")

        return {
            'probabilities': prob_map,
            'class_mask': class_mask,
            'instance_mask': instance_mask
        }


if __name__ == '__main__':
    import argparse
    import sys
    from pathlib import Path

    # Add package to path if running as script
    script_dir = Path(__file__).parent.absolute()
    package_root = script_dir.parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

    parser = argparse.ArgumentParser(description='Detect calcium events')
    parser.add_argument('--model', required=True, help='Path to model weights')
    parser.add_argument('--input', required=True, help='Path to input TIFF file')
    parser.add_argument('--output', required=True, help='Path to output directory')
    parser.add_argument('--config', help='Path to config file')

    args = parser.parse_args()

    # Load config
    config = Config.load(args.config) if args.config else None

    # Create detector
    detector = CalciumEventDetector(args.model, config)

    # Load image
    print(f"Loading image from {args.input}")
    try:
        from tifffile import imread, imwrite
    except ImportError:
        print("Error: tifffile not installed. Install with: pip install tifffile")
        exit(1)

    image = imread(args.input)
    print(f"Image shape: {image.shape}")

    # Run detection
    results = detector.detect(image)

    # Save results
    import os
    os.makedirs(args.output, exist_ok=True)

    output_class = os.path.join(args.output, 'class_mask.tif')
    output_instance = os.path.join(args.output, 'instance_mask.tif')

    imwrite(output_class, results['class_mask'].astype(np.uint8))
    imwrite(output_instance, results['instance_mask'].astype(np.uint16))

    print(f"\nResults saved to {args.output}")
    print(f"  - Class mask: {output_class}")
    print(f"  - Instance mask: {output_instance}")
