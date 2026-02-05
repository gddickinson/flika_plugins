#!/usr/bin/env python3
"""
Examples for Calcium Event Data Labeler
========================================

Common usage patterns for the labeling plugin.

Author: George Stuyt (with Claude)
Date: 2024-12-26
"""

from pathlib import Path
import numpy as np
from tifffile import imread, imwrite


# =============================================================================
# Example 1: Automated Labeling of Single File
# =============================================================================

def example_automated_labeling():
    """Generate automated labels for a single calcium imaging file."""
    from ca_labeler_flika.automated_labeling import AutomatedLabeler
    
    # Load data
    image = imread('data/calcium_recording.tif')
    
    # Create labeler
    labeler = AutomatedLabeler()
    
    # Generate labels
    labels = labeler.generate_labels(
        image,
        intensity_threshold=0.3,        # Adjust based on SNR
        temporal_filter_size=3,         # Reduce noise
        min_event_size=10,              # Min 10 pixels
        min_event_duration=2,           # Min 2 frames
        auto_classify=True              # Auto-classify events
    )
    
    # Save labels
    imwrite('data/calcium_recording_labels.tif', labels)
    
    print(f"Generated labels: {np.unique(labels)}")
    print(f"Spark pixels: {np.sum(labels == 1)}")
    print(f"Puff pixels: {np.sum(labels == 2)}")
    print(f"Wave pixels: {np.sum(labels == 3)}")


# =============================================================================
# Example 2: Batch Automated Labeling
# =============================================================================

def example_batch_labeling():
    """Process multiple files with automated labeling."""
    from ca_labeler_flika.automated_labeling import AutomatedLabeler
    from tqdm import tqdm
    
    input_dir = Path('data/images')
    output_dir = Path('data/masks')
    output_dir.mkdir(exist_ok=True)
    
    labeler = AutomatedLabeler()
    
    image_files = list(input_dir.glob('*.tif'))
    
    for img_file in tqdm(image_files, desc="Labeling"):
        try:
            # Load image
            image = imread(img_file)
            
            # Generate labels
            labels = labeler.generate_labels(image, auto_classify=True)
            
            # Save
            output_file = output_dir / f"{img_file.stem}_class.tif"
            imwrite(output_file, labels)
            
        except Exception as e:
            print(f"Error processing {img_file.name}: {e}")


# =============================================================================
# Example 3: Quality Assessment
# =============================================================================

def example_quality_check():
    """Assess quality of labeled dataset."""
    from ca_labeler_flika.label_quality import LabelQualityChecker
    
    checker = LabelQualityChecker()
    
    # Check quality
    results = checker.assess_quality(
        Path('data'),
        check_distribution=True,
        check_coverage=True,
        check_consistency=True,
        check_completeness=True
    )
    
    # Print results
    print("\n=== Quality Assessment Results ===\n")
    
    if 'distribution' in results:
        dist = results['distribution']
        print("Class Distribution:")
        for class_id, pct in dist['class_percentages'].items():
            print(f"  {class_id}: {pct:.2f}%")
        print(f"  Imbalance ratio: {dist['imbalance_ratio']:.1f}:1")
    
    if 'coverage' in results:
        cov = results['coverage']
        print(f"\nCoverage:")
        print(f"  Spatial (% frames): {cov['mean_spatial_coverage']*100:.1f}%")
        print(f"  Temporal (% area): {cov['mean_temporal_coverage']*100:.1f}%")
    
    if 'consistency' in results:
        con = results['consistency']
        print(f"\nConsistency:")
        print(f"  Files with isolated pixels: {con['num_files_with_isolated_pixels']}")
        print(f"  Files with temporal gaps: {con['num_files_with_temporal_gaps']}")
    
    # Check warnings
    all_warnings = []
    for section in results.values():
        if isinstance(section, dict) and 'warnings' in section:
            all_warnings.extend(section['warnings'])
    
    if all_warnings:
        print(f"\n⚠️  Warnings:")
        for warning in all_warnings:
            print(f"  - {warning}")


# =============================================================================
# Example 4: Create Training Dataset
# =============================================================================

def example_create_dataset():
    """Create training dataset with patches and splits."""
    from ca_labeler_flika.dataset_creator import DatasetCreator
    
    creator = DatasetCreator()
    
    stats = creator.create_dataset(
        input_dir=Path('data'),
        output_dir=Path('training_dataset'),
        patch_size=(16, 64, 64),        # T, H, W
        stride=(8, 32, 32),             # 50% overlap
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        augment=True,                   # Apply augmentation
        balance_classes=True,           # Balance class distribution
        random_seed=42
    )
    
    print("\n=== Dataset Statistics ===")
    print(f"Train samples: {stats['train_samples']}")
    print(f"Val samples: {stats['val_samples']}")
    print(f"Test samples: {stats['test_samples']}")
    print(f"Total patches: {stats['total_patches']}")


# =============================================================================
# Example 5: Custom Augmentation
# =============================================================================

def example_custom_augmentation():
    """Apply custom augmentation to patches."""
    from ca_labeler_flika.augmentation import AugmentationEngine
    
    # Load a patch
    image_patch = imread('patches/patch_001.tif')
    mask_patch = imread('patches/patch_001_mask.tif')
    
    # Create augmentation engine
    aug_engine = AugmentationEngine(random_seed=42)
    
    # Generate multiple augmented versions
    for i in range(5):
        aug_image, aug_mask = aug_engine.augment_patch(
            image_patch,
            mask_patch,
            p_rotate=0.8,       # High probability of rotation
            p_flip=0.5,         # 50% chance of flipping
            p_elastic=0.3,      # 30% chance of elastic deformation
            p_intensity=1.0,    # Always apply intensity transform
            p_noise=0.2         # 20% chance of noise
        )
        
        # Save augmented versions
        imwrite(f'patches/patch_001_aug_{i}.tif', aug_image)
        imwrite(f'patches/patch_001_aug_{i}_mask.tif', aug_mask)


# =============================================================================
# Example 6: Merge Annotations from Multiple Annotators
# =============================================================================

def example_merge_annotations():
    """Merge annotations from multiple annotators."""
    from ca_labeler_flika.utils import merge_annotations
    
    # Load annotations from 3 annotators
    ann1 = imread('annotations/annotator1.tif')
    ann2 = imread('annotations/annotator2.tif')
    ann3 = imread('annotations/annotator3.tif')
    
    # Merge using majority vote
    merged = merge_annotations(
        [ann1, ann2, ann3],
        method='majority'
    )
    
    # Save merged annotations
    imwrite('annotations/merged.tif', merged)


# =============================================================================
# Example 7: Inter-Annotator Agreement
# =============================================================================

def example_annotator_agreement():
    """Calculate inter-annotator agreement."""
    from ca_labeler_flika.label_quality import LabelQualityChecker
    from pathlib import Path
    
    # Get mask files from two annotators
    annotator1_masks = list(Path('annotations/annotator1').glob('*.tif'))
    annotator2_masks = list(Path('annotations/annotator2').glob('*.tif'))
    
    # Check agreement
    checker = LabelQualityChecker()
    agreement = checker.compare_annotators(annotator1_masks, annotator2_masks)
    
    print("\n=== Inter-Annotator Agreement ===")
    print(f"Files compared: {agreement['num_compared']}")
    print(f"Mean IoU: {agreement['mean_iou']:.3f}")
    print(f"Mean Dice: {agreement['mean_dice']:.3f}")
    print(f"Pixel accuracy: {agreement['mean_accuracy']:.3f}")
    print(f"Cohen's Kappa: {agreement['mean_kappa']:.3f}")
    print(f"Agreement level: {agreement['agreement_level']}")


# =============================================================================
# Example 8: Extract Event Properties
# =============================================================================

def example_extract_properties():
    """Extract properties of labeled events."""
    from ca_labeler_flika.utils import convert_to_instance_mask, extract_event_properties
    import pandas as pd
    
    # Load class mask
    class_mask = imread('data/labels.tif')
    
    # Convert to instance mask
    instance_mask = convert_to_instance_mask(class_mask)
    
    # Extract properties
    properties = extract_event_properties(
        instance_mask,
        class_mask,
        pixel_size=0.2,     # μm per pixel
        frame_rate=6.79     # ms per frame
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(properties)
    
    # Save to CSV
    df.to_csv('data/event_properties.csv', index=False)
    
    # Print summary
    print("\n=== Event Properties Summary ===")
    for class_name in ['spark', 'puff', 'wave']:
        class_events = df[df['class_name'] == class_name]
        if len(class_events) > 0:
            print(f"\n{class_name.capitalize()}:")
            print(f"  Count: {len(class_events)}")
            print(f"  Mean size: {class_events['volume_voxels'].mean():.1f} voxels")
            print(f"  Mean duration: {class_events['duration_ms'].mean():.1f} ms")


# =============================================================================
# Example 9: Visualize Labels
# =============================================================================

def example_visualize_labels():
    """Create visualization with label overlay."""
    from ca_labeler_flika.utils import visualize_labels
    
    # Load data
    image = imread('data/calcium_recording.tif')
    labels = imread('data/calcium_recording_labels.tif')
    
    # Create visualization for each frame
    for t in range(image.shape[0]):
        vis = visualize_labels(
            image[t],
            labels[t],
            alpha=0.5  # 50% transparency
        )
        
        # Save visualization
        imwrite(f'visualizations/frame_{t:04d}.tif', vis)


# =============================================================================
# Example 10: Complete Workflow
# =============================================================================

def example_complete_workflow():
    """Complete workflow from raw data to training dataset."""
    from ca_labeler_flika.automated_labeling import AutomatedLabeler
    from ca_labeler_flika.label_quality import LabelQualityChecker
    from ca_labeler_flika.dataset_creator import DatasetCreator
    
    print("=== Complete Labeling Workflow ===\n")
    
    # Step 1: Automated labeling
    print("Step 1: Generating automated labels...")
    labeler = AutomatedLabeler()
    
    for img_file in Path('raw_data').glob('*.tif'):
        image = imread(img_file)
        labels = labeler.generate_labels(image, auto_classify=True)
        
        output_file = Path('labeled_data/masks') / f"{img_file.stem}_class.tif"
        imwrite(output_file, labels)
    
    # Step 2: Quality check
    print("\nStep 2: Checking label quality...")
    checker = LabelQualityChecker()
    results = checker.assess_quality(
        Path('labeled_data'),
        check_distribution=True,
        check_coverage=True,
        check_consistency=True,
        check_completeness=True
    )
    
    # Print warnings
    for section, data in results.items():
        if isinstance(data, dict) and 'warnings' in data:
            for warning in data['warnings']:
                print(f"  ⚠️  {warning}")
    
    # Step 3: Manual review and refinement
    print("\nStep 3: Manual review (use Interactive Labeler)...")
    print("  → Review automated labels in FLIKA")
    print("  → Refine boundaries and correct misclassifications")
    
    # Step 4: Create dataset
    print("\nStep 4: Creating training dataset...")
    creator = DatasetCreator()
    stats = creator.create_dataset(
        input_dir=Path('labeled_data'),
        output_dir=Path('training_dataset'),
        patch_size=(16, 64, 64),
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        augment=True,
        balance_classes=True
    )
    
    print(f"\n✅ Complete! Created dataset with {stats['total_patches']} patches")
    print(f"  Train: {stats['train_samples']}")
    print(f"  Val: {stats['val_samples']}")
    print(f"  Test: {stats['test_samples']}")


if __name__ == '__main__':
    print("Calcium Event Data Labeler - Examples")
    print("=" * 50)
    print("\nAvailable examples:")
    print("  1. Automated labeling of single file")
    print("  2. Batch automated labeling")
    print("  3. Quality assessment")
    print("  4. Create training dataset")
    print("  5. Custom augmentation")
    print("  6. Merge annotations")
    print("  7. Inter-annotator agreement")
    print("  8. Extract event properties")
    print("  9. Visualize labels")
    print(" 10. Complete workflow")
    print("\nRun individual examples by calling the functions above")
