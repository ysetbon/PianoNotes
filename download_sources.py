#!/usr/bin/env python3
"""
Direct download script for piano sample sources.
Downloads from sources that allow direct access.
"""

import os
import sys
import zipfile
import tarfile
import tempfile
import shutil
from pathlib import Path

# Optional imports
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Download URLs for various sources
DOWNLOAD_SOURCES = {
    'nsynth_test': {
        'name': 'NSynth Test Set',
        'url': 'http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz',
        'size': '~3GB',
        'description': 'Smaller test set with ~4,000 keyboard samples'
    },
    'freesound_packs': {
        'name': 'Freesound Sample Packs (VSCO)',
        'urls': [
            # VSCO Community Edition - Upright Piano
            'https://freesound.org/people/sgossner/packs/21055/download/',
        ],
        'note': 'Requires Freesound account login'
    },
}

# Curated list of direct download links (where available)
PIANOBOOK_SAMPLES = [
    {
        'name': 'Keys Upright (via archive)',
        'info': 'Download manually from pianobook.co.uk',
        'pack_url': 'https://www.pianobook.co.uk/packs/'
    }
]


def download_with_progress(url: str, dest_path: str, desc: str = None):
    """Download a file with progress bar"""
    if not HAS_REQUESTS:
        print("Error: 'requests' module not installed. Run: pip install requests")
        return None

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(dest_path, 'wb') as f:
        if HAS_TQDM:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        else:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    pct = (downloaded / total_size) * 100
                    print(f"\r{desc}: {pct:.1f}% ({downloaded}/{total_size})", end='')
            print()

    return dest_path


def extract_archive(archive_path: str, dest_dir: str):
    """Extract tar.gz or zip archive"""
    if archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(dest_dir)
    elif archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)


def print_download_links():
    """Print all available download links"""
    print("\n" + "="*70)
    print("PIANO SAMPLE DOWNLOAD LINKS")
    print("="*70)

    print("\n1. PIANOBOOK (Primary Source - ~1500+ piano samples)")
    print("-" * 50)
    print("   Main URL: https://www.pianobook.co.uk/packs/")
    print("   Filter: Pianos -> Acoustic")
    print("   Format: Download 'Decent Sampler' or 'SFZ' versions")
    print("\n   Recommended packs:")
    print("   - The Experience: https://www.pianobook.co.uk/packs/the-experience/")
    print("   - Soft Piano: https://www.pianobook.co.uk/packs/soft-piano/")
    print("   - Community Grand: https://www.pianobook.co.uk/packs/community-grand/")
    print("   - The Grand Toy Piano: https://www.pianobook.co.uk/packs/the-grand-toy-piano/")

    print("\n2. NSYNTH DATASET (Google Magenta)")
    print("-" * 50)
    print("   Info: https://magenta.withgoogle.com/datasets/nsynth")
    print("   Test Set (~4K keyboard): http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz")
    print("   Valid Set (~12K keyboard): http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz")
    print("   HuggingFace: https://huggingface.co/datasets/jg583/NSynth")

    print("\n3. UNIVERSITY OF IOWA MIS")
    print("-" * 50)
    print("   Main: http://theremin.music.uiowa.edu/MISpiano.html")
    print("   Archive: https://archive.org/details/UniversityOfIowaMIS")

    print("\n4. FREESOUND")
    print("-" * 50)
    print("   Main: https://freesound.org/")
    print("   Search: https://freesound.org/search/?q=piano+note+single")
    print("   VSCO Piano Pack: https://freesound.org/people/sgossner/packs/21055/")
    print("   API Docs: https://freesound.org/docs/api/")

    print("\n5. ADDITIONAL SOURCES")
    print("-" * 50)
    print("   MAPS Database: https://adasp.telecom-paris.fr/resources/2010-07-08-maps-database/")
    print("   Philharmonia: https://philharmonia.co.uk/resources/sound-samples/")
    print("   Dataset List: https://github.com/AMAAI-Lab/ai-audio-datasets-list")

    print("\n" + "="*70)
    print("QUICK START")
    print("="*70)
    print("""
1. Download samples from the sources above
2. Extract all ZIP/TAR files to a folder (e.g., ./downloads/pianobook)
3. Run the organizer:

   python piano_sample_downloader.py --pianobook-path ./downloads/pianobook --output ./notes

   Or scan any folder with audio files:

   python piano_sample_downloader.py --scan-path ./my_samples --output ./notes
""")


def download_nsynth(output_dir: str = "downloads"):
    """Download NSynth test set"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    url = "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz"
    archive_name = "nsynth-test.jsonwav.tar.gz"
    archive_path = output_path / archive_name

    print(f"Downloading NSynth test set (~3GB)...")
    print(f"URL: {url}")

    try:
        download_with_progress(url, str(archive_path), "NSynth Test")
        print(f"\nExtracting to {output_path}/nsynth-test/...")
        extract_archive(str(archive_path), str(output_path))
        print("Done!")
        return str(output_path / "nsynth-test")
    except Exception as e:
        print(f"Error: {e}")
        print("\nManual download instructions:")
        print(f"  1. Download from: {url}")
        print(f"  2. Extract to: {output_path}")
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Download piano sample sources')
    parser.add_argument('--links', action='store_true', help='Show all download links')
    parser.add_argument('--download-nsynth', action='store_true', help='Download NSynth test set')
    parser.add_argument('--output', '-o', default='downloads', help='Output directory')

    args = parser.parse_args()

    if args.links or (not args.download_nsynth):
        print_download_links()

    if args.download_nsynth:
        if not HAS_TQDM:
            print("Note: Install 'tqdm' for progress bars: pip install tqdm")
        download_nsynth(args.output)


if __name__ == "__main__":
    main()
