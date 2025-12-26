"""
Utility to download and prepare the dataset from Zenodo.
Dataset DOI: 10.5281/zenodo.10391727
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile
import argparse


ZENODO_RECORD_ID = "10391727"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"


def download_file(url: str, dest_path: Path, desc: str = "Downloading"):
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from
        dest_path: Destination file path
        desc: Description for progress bar
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=desc,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            pbar.update(size)


def get_zenodo_files():
    """
    Get list of files from Zenodo record.
    
    Returns:
        List of file dictionaries with 'filename' and 'url' keys
    """
    print(f"Fetching file list from Zenodo record {ZENODO_RECORD_ID}...")
    
    response = requests.get(ZENODO_API_URL)
    response.raise_for_status()
    
    data = response.json()
    files = data.get('files', [])
    
    return [{'filename': f['key'], 'url': f['links']['self'], 'size': f['size']} 
            for f in files]


def download_zenodo_dataset(output_dir: str = "./data/zenodo"):
    """
    Download the complete dataset from Zenodo.
    
    Args:
        output_dir: Directory to save downloaded files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get file list
    files = get_zenodo_files()
    
    if not files:
        print("No files found in Zenodo record!")
        return
    
    print(f"\nFound {len(files)} files:")
    for f in files:
        size_mb = f['size'] / (1024 * 1024)
        print(f"  - {f['filename']} ({size_mb:.1f} MB)")
    
    print(f"\nDownloading to: {output_path.absolute()}")
    
    # Download each file
    for file_info in files:
        filename = file_info['filename']
        url = file_info['url']
        dest_path = output_path / filename
        
        if dest_path.exists():
            print(f"\nSkipping {filename} (already exists)")
            continue
        
        print(f"\nDownloading {filename}...")
        try:
            download_file(url, dest_path, desc=filename)
            print(f"✓ Downloaded: {filename}")
        except Exception as e:
            print(f"✗ Error downloading {filename}: {e}")
    
    # Extract any zip files
    for file_path in output_path.glob("*.zip"):
        print(f"\nExtracting {file_path.name}...")
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(output_path)
            print(f"✓ Extracted: {file_path.name}")
        except Exception as e:
            print(f"✗ Error extracting {file_path.name}: {e}")
    
    print("\n" + "="*60)
    print("Dataset download completed!")
    print(f"Data saved to: {output_path.absolute()}")
    print("="*60)


def setup_directory_structure(data_dir: str):
    """
    Set up the expected directory structure for training.
    
    Args:
        data_dir: Root data directory
    """
    data_path = Path(data_dir)
    
    # Create subdirectories
    (data_path / "images").mkdir(parents=True, exist_ok=True)
    (data_path / "masks").mkdir(parents=True, exist_ok=True)
    (data_path / "raw").mkdir(parents=True, exist_ok=True)
    
    print(f"Directory structure created at {data_path.absolute()}")
    print("\nExpected structure:")
    print(f"{data_path}/")
    print("  ├── images/          # Place your .tif recording files here")
    print("  ├── masks/           # Place annotation masks here")
    print("  │   ├── *_class.tif    # Class masks")
    print("  │   └── *_instance.tif # Instance masks")
    print("  └── raw/             # Original/raw data from Zenodo")


def main():
    """Main function for data preparation."""
    import sys
    from pathlib import Path
    
    # Add package to path if running as script
    script_dir = Path(__file__).parent.absolute()
    package_root = script_dir.parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    
    parser = argparse.ArgumentParser(
        description='Download and prepare calcium imaging dataset from Zenodo'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data',
        help='Directory to save data (default: ./data)'
    )
    parser.add_argument(
        '--download',
        action='store_true',
        help='Download dataset from Zenodo'
    )
    parser.add_argument(
        '--setup',
        action='store_true',
        help='Set up directory structure'
    )
    
    args = parser.parse_args()
    
    if not args.download and not args.setup:
        print("Please specify --download and/or --setup")
        print("Use --help for more information")
        return
    
    if args.download:
        zenodo_dir = os.path.join(args.output_dir, 'zenodo')
        download_zenodo_dataset(zenodo_dir)
    
    if args.setup:
        setup_directory_structure(args.output_dir)


if __name__ == '__main__':
    main()
