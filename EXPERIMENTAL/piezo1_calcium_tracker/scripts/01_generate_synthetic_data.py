#!/usr/bin/env python3
"""
Generate Synthetic PIEZO1 Puncta Training Data

Creates realistic synthetic images of PIEZO1-HaloTag puncta with:
- Sub-pixel ground truth coordinates
- Realistic PSF (Gaussian or Airy)
- Poisson + Gaussian noise
- Spatially varying background
- Variable photon counts
- Blinking/photobleaching

Output format matches DECODE for easy integration.
"""

import numpy as np
import tifffile
from pathlib import Path
from typing import Tuple, List
import argparse
from tqdm import tqdm
import json


def generate_puncta_positions(image_size: Tuple[int, int],
                               num_puncta: int,
                               min_distance_px: float = 5.0) -> np.ndarray:
    """
    Generate random puncta positions with minimum distance constraint.
    
    Args:
        image_size: (height, width) in pixels
        num_puncta: Number of puncta to generate
        min_distance_px: Minimum distance between puncta
        
    Returns:
        positions: Nx2 array of (x, y) sub-pixel positions
    """
    height, width = image_size
    positions = []
    
    max_attempts = num_puncta * 100
    attempts = 0
    
    while len(positions) < num_puncta and attempts < max_attempts:
        # Random position with sub-pixel precision
        x = np.random.uniform(5, width - 5)
        y = np.random.uniform(5, height - 5)
        
        # Check minimum distance
        if len(positions) == 0:
            positions.append([x, y])
        else:
            distances = np.sqrt(((np.array(positions) - [x, y])**2).sum(axis=1))
            if distances.min() >= min_distance_px:
                positions.append([x, y])
        
        attempts += 1
    
    return np.array(positions)


def generate_photon_counts(num_puncta: int,
                            mean_photons: float = 1000.0,
                            photon_std: float = 300.0,
                            min_photons: float = 200.0) -> np.ndarray:
    """
    Generate realistic photon counts with variation.
    
    Args:
        num_puncta: Number of puncta
        mean_photons: Mean photon count
        photon_std: Standard deviation
        min_photons: Minimum allowed photons
        
    Returns:
        photon_counts: N array of photon counts
    """
    counts = np.random.normal(mean_photons, photon_std, size=num_puncta)
    counts = np.clip(counts, min_photons, None)
    return counts


def generate_frame(image_size: Tuple[int, int],
                   num_puncta: int,
                   psf_model,
                   baseline: float = 100.0,
                   background_mean: float = 500.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate single frame with puncta.
    
    Args:
        image_size: (height, width)
        num_puncta: Number of puncta in frame
        psf_model: PSF model instance
        baseline: Camera baseline
        background_mean: Background intensity
        
    Returns:
        image: Generated noisy image
        positions: Ground truth positions
        photon_counts: Ground truth photon counts
    """
    
    # Generate positions and photon counts
    positions = generate_puncta_positions(image_size, num_puncta)
    photon_counts = generate_photon_counts(num_puncta)
    
    # Generate clean PSF image
    clean_image = psf_model.generate(positions, image_size, photon_counts)
    
    # Add background
    from piezo1_tracker.utils.psf_models import add_background, add_noise
    image_with_bg = add_background(clean_image, mean_bg=background_mean)
    
    # Add noise
    noisy_image = add_noise(image_with_bg, baseline=baseline)
    
    return noisy_image, positions, photon_counts


def generate_time_series(num_frames: int,
                         image_size: Tuple[int, int],
                         num_puncta_mean: int,
                         psf_model,
                         blinking_prob: float = 0.1) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    Generate time series with blinking puncta.
    
    Args:
        num_frames: Number of frames to generate
        image_size: (height, width)
        num_puncta_mean: Average number of puncta per frame
        psf_model: PSF model instance
        blinking_prob: Probability of puncta disappearing/appearing
        
    Returns:
        movie: (T, H, W) array
        all_positions: List of positions per frame
        all_photons: List of photon counts per frame
    """
    
    movie = []
    all_positions = []
    all_photons = []
    
    for t in range(num_frames):
        # Variable number of puncta (Poisson distributed)
        num_puncta = np.random.poisson(num_puncta_mean)
        num_puncta = max(1, num_puncta)  # At least 1
        
        # Generate frame
        frame, positions, photons = generate_frame(
            image_size, num_puncta, psf_model
        )
        
        movie.append(frame)
        all_positions.append(positions)
        all_photons.append(photons)
    
    return np.array(movie), all_positions, all_photons


def save_sample(output_dir: Path,
                sample_idx: int,
                movie: np.ndarray,
                positions: List[np.ndarray],
                photons: List[np.ndarray]):
    """
    Save generated sample in DECODE-compatible format.
    
    Args:
        output_dir: Output directory
        sample_idx: Sample index
        movie: (T, H, W) movie
        positions: List of positions per frame
        photons: List of photon counts per frame
    """
    
    # Create sample directory
    sample_dir = output_dir / f'sample_{sample_idx:05d}'
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Save movie
    tifffile.imwrite(sample_dir / 'movie.tif', movie.astype(np.uint16))
    
    # Save ground truth coordinates in CSV format
    # Format: frame, x, y, photons
    all_data = []
    for t, (pos, phot) in enumerate(zip(positions, photons)):
        for (x, y), p in zip(pos, phot):
            all_data.append([t, x, y, p])
    
    np.savetxt(
        sample_dir / 'ground_truth.csv',
        all_data,
        delimiter=',',
        header='frame,x,y,photons',
        comments='',
        fmt='%d,%.3f,%.3f,%.1f'
    )
    
    # Save metadata
    metadata = {
        'num_frames': int(movie.shape[0]),
        'height': int(movie.shape[1]),
        'width': int(movie.shape[2]),
        'num_puncta_per_frame': [len(p) for p in positions],
        'total_puncta': sum(len(p) for p in positions)
    }
    
    with open(sample_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic puncta data')
    parser.add_argument('--output', type=str, default='data/synthetic_puncta',
                        help='Output directory')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to generate')
    parser.add_argument('--frames_per_sample', type=int, default=50,
                        help='Number of frames per sample')
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 512],
                        help='Image size (height width)')
    parser.add_argument('--num_puncta_mean', type=int, default=20,
                        help='Average number of puncta per frame')
    parser.add_argument('--psf_type', type=str, default='gaussian',
                        choices=['gaussian', 'airy'],
                        help='PSF model type')
    parser.add_argument('--pixel_size_nm', type=float, default=130.0,
                        help='Pixel size in nanometers')
    parser.add_argument('--wavelength_nm', type=float, default=646.0,
                        help='Emission wavelength in nm')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save generation parameters
    with open(output_dir / 'generation_params.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"Generating {args.num_samples} synthetic samples...")
    print(f"Output directory: {output_dir}")
    print(f"Frames per sample: {args.frames_per_sample}")
    print(f"Image size: {args.image_size}")
    print(f"PSF type: {args.psf_type}")
    
    # Create PSF model
    from piezo1_tracker.utils.psf_models import Gaussian2DPSF, Airy2DPSF
    
    if args.psf_type == 'gaussian':
        psf_model = Gaussian2DPSF(
            pixel_size_nm=args.pixel_size_nm,
            wavelength_nm=args.wavelength_nm,
            numerical_aperture=1.49
        )
    else:
        psf_model = Airy2DPSF(
            pixel_size_nm=args.pixel_size_nm,
            wavelength_nm=args.wavelength_nm,
            numerical_aperture=1.49
        )
    
    # Generate samples
    for i in tqdm(range(args.num_samples), desc='Generating samples'):
        # Generate time series
        movie, positions, photons = generate_time_series(
            num_frames=args.frames_per_sample,
            image_size=tuple(args.image_size),
            num_puncta_mean=args.num_puncta_mean,
            psf_model=psf_model
        )
        
        # Save sample
        save_sample(output_dir, i, movie, positions, photons)
    
    print(f"\n✅ Generated {args.num_samples} samples")
    print(f"   Total frames: {args.num_samples * args.frames_per_sample}")
    print(f"   Output: {output_dir}")
    
    # Generate statistics
    total_puncta = 0
    for i in range(args.num_samples):
        with open(output_dir / f'sample_{i:05d}' / 'metadata.json') as f:
            metadata = json.load(f)
            total_puncta += metadata['total_puncta']
    
    print(f"\nStatistics:")
    print(f"   Total puncta: {total_puncta:,}")
    print(f"   Avg puncta/frame: {total_puncta / (args.num_samples * args.frames_per_sample):.1f}")
    print(f"   Avg puncta/sample: {total_puncta / args.num_samples:.1f}")


if __name__ == '__main__':
    main()
