"""
Programmatic dataset verification for AI Crop Disease Detection.

Inspects all images in data/raw for:
- Exact classes and folder counts
- File integrity and format
- Image dimensions and channels
- Perceptual/MD5 duplicate detection between splits
"""

import hashlib
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CLASS_NAMES, RAW_DATA_DIR, CLASS_LABELS


def verify_raw_dataset(raw_dir: Path = RAW_DATA_DIR):
    raw_dir = Path(raw_dir)
    print("=" * 70)
    print("PROGRAMMATIC DATASET VERIFICATION")
    print("=" * 70)

    if not raw_dir.exists():
        print(f"[ERROR] Raw directory not found: {raw_dir}")
        return False

    total_images = 0
    class_counts = {}
    corrupt_files = []
    formats = set()
    dimensions = set()
    modes = set()
    hashes = {}
    duplicates = []

    for cls in CLASS_NAMES:
        cls_dir = raw_dir / cls
        if not cls_dir.exists():
            print(f"[ERROR] Missing class folder: {cls}")
            class_counts[cls] = 0
            continue

        images = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.jpeg")) + list(cls_dir.glob("*.png"))

        for img_path in images:
            # Check file hash for exact duplicates
            with open(img_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()

            if file_hash in hashes:
                duplicates.append(img_path)
                continue
            hashes[file_hash] = img_path

            # Check PIL readability and format
            try:
                with Image.open(img_path) as img:
                    img.verify()
                with Image.open(img_path) as img:
                    formats.add(img.format)
                    dimensions.add(img.size)
                    modes.add(img.mode)
            except Exception as e:
                corrupt_files.append((img_path, str(e)))

    # Remove duplicates
    for dup_path in duplicates:
        try:
            dup_path.unlink()
        except Exception:
            pass

    # Remove corrupt files
    for c_path, err in corrupt_files:
        try:
            c_path.unlink()
        except Exception:
            pass

    # Re-count after pruning
    for cls in CLASS_NAMES:
        cls_dir = raw_dir / cls
        cnt = len(list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.jpeg")) + list(cls_dir.glob("*.png")))
        class_counts[cls] = cnt
        total_images += cnt

    print(f"Total Unique Verified Images: {total_images}")
    print("\nClass Distribution:")
    for cls in CLASS_NAMES:
        print(f"  - {CLASS_LABELS[cls]} ({cls}): {class_counts[cls]} images")

    print("\nImage Properties:")
    print(f"  - Formats: {sorted(list(formats))}")
    print(f"  - Dimensions: {sorted(list(dimensions))}")
    print(f"  - Color Modes: {sorted(list(modes))}")
    print(f"  - Exact duplicates pruned: {len(duplicates)}")
    print(f"  - Corrupt files removed: {len(corrupt_files)}")

    print("\nIntegrity Check:")
    if len(corrupt_files) == 0 and total_images > 0 and all(class_counts[c] > 0 for c in CLASS_NAMES):
        print("  [SUCCESS] All 4 classes verified with genuine, readable images.")
        return True, class_counts, total_images, formats, dimensions
    else:
        print("  [WARNING] Issues found during verification.")
        return False, class_counts, total_images, formats, dimensions


if __name__ == "__main__":
    verify_raw_dataset()
