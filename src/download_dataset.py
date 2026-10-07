"""
Download PlantVillage tomato subset from GitHub (no authentication required).

Source: https://github.com/spMohanty/PlantVillage-Dataset
License: Public domain / Open access

This script downloads ONLY the 4 required tomato classes from the
PlantVillage GitHub repository using the GitHub API (no git clone needed).

Usage:
    python src/download_dataset.py
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CLASS_NAMES, RAW_DATA_DIR


# GitHub API base for the PlantVillage repo
GITHUB_API_BASE = "https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color"


def list_github_folder(class_name: str) -> list:
    """List all files in a class folder via GitHub API."""
    url = f"{GITHUB_API_BASE}/{class_name}"
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "PlantVillage-Downloader")

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())
            return [item for item in data if item["name"].lower().endswith((".jpg", ".jpeg", ".png"))]
    except urllib.error.HTTPError as e:
        if e.code == 403:
            print(f"  [WARN] GitHub API rate limit hit. Trying direct download...")
            return None
        raise


def download_file(url: str, dest: Path) -> bool:
    """Download a single file."""
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "PlantVillage-Downloader")
        with urllib.request.urlopen(req, timeout=30) as response:
            dest.write_bytes(response.read())
        return True
    except Exception as e:
        return False


def download_class_images(class_name: str, target_dir: Path, max_images: int = None) -> int:
    """Download all images for a single class."""
    class_dir = target_dir / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG")) + list(class_dir.glob("*.jpeg"))
    if len(existing) >= 100:  # Reasonable threshold
        print(f"  {class_name}: Already have {len(existing)} images, skipping.")
        return len(existing)

    print(f"  {class_name}: Listing files from GitHub...")

    # Try GitHub API first
    files = list_github_folder(class_name)

    if files is None:
        # API rate limited — fall back to known file patterns
        print(f"  {class_name}: API rate limited, cannot list files.")
        return 0

    if max_images:
        files = files[:max_images]

    total = len(files)
    print(f"  {class_name}: Downloading {total} images...")

    downloaded = 0
    failed = 0

    def download_one(file_info):
        name = file_info["name"]
        url = file_info["download_url"]
        dest = class_dir / name
        if dest.exists():
            return True
        return download_file(url, dest)

    # Download with thread pool for speed
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(download_one, f): f for f in files}
        for i, future in enumerate(as_completed(futures)):
            if future.result():
                downloaded += 1
            else:
                failed += 1

            # Progress every 50 images
            if (i + 1) % 50 == 0 or (i + 1) == total:
                print(f"    Progress: {i+1}/{total} ({downloaded} ok, {failed} failed)")

    print(f"  {class_name}: {downloaded} downloaded, {failed} failed")
    return downloaded


def verify_download(target_dir: Path) -> dict:
    """Verify downloaded data and return counts."""
    result = {}
    for class_name in CLASS_NAMES:
        class_dir = target_dir / class_name
        if class_dir.exists():
            count = len(list(class_dir.glob("*.jpg"))) + \
                    len(list(class_dir.glob("*.JPG"))) + \
                    len(list(class_dir.glob("*.jpeg"))) + \
                    len(list(class_dir.glob("*.png")))
            result[class_name] = count
        else:
            result[class_name] = 0
    return result


def main():
    """Download PlantVillage tomato subset."""
    print("=" * 70)
    print("PlantVillage Dataset Download (Tomato Subset)")
    print("=" * 70)
    print()
    print(f"Source: GitHub (spMohanty/PlantVillage-Dataset)")
    print(f"Target: {RAW_DATA_DIR}")
    print(f"Classes: {len(CLASS_NAMES)}")
    for cls in CLASS_NAMES:
        print(f"  - {cls}")
    print()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    start = time.time()

    for class_name in CLASS_NAMES:
        download_class_images(class_name, RAW_DATA_DIR)
        print()

    elapsed = time.time() - start

    # Verify
    print("=" * 70)
    print("Download Verification")
    print("=" * 70)
    counts = verify_download(RAW_DATA_DIR)
    total = 0
    all_ok = True
    for class_name, count in counts.items():
        status = "OK" if count > 0 else "MISSING"
        if count == 0:
            all_ok = False
        print(f"  {class_name}: {count} images [{status}]")
        total += count

    print(f"\n  Total: {total} images")
    print(f"  Time: {elapsed:.1f}s")

    if all_ok:
        print("\n[SUCCESS] All classes downloaded successfully!")
    else:
        print("\n[WARNING] Some classes are missing. See above for details.")

    return all_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
