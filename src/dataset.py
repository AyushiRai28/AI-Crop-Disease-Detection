"""
Dataset preparation for AI Crop Disease Detection.

This module handles:
1. Downloading the PlantVillage dataset (tomato subset)
2. Organizing images into train/validation/test splits
3. Creating TensorFlow data pipelines with augmentation

Dataset Source:
    PlantVillage dataset via tensorflow_datasets or direct download.
    We use only 4 tomato classes for the initial demonstration.
"""

import os
import shutil
import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.config import (
    CLASS_NAMES,
    IMG_SIZE,
    BATCH_SIZE,
    VALIDATION_SPLIT,
    TEST_SPLIT,
    RANDOM_SEED,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    AUGMENTATION_CONFIG,
)


def download_plantvillage_subset(
    target_dir: Optional[Path] = None,
) -> Path:
    """
    Download the PlantVillage tomato subset.

    Strategy:
    1. First check if data already exists locally.
    2. Try to use tensorflow_datasets (if installed).
    3. Fall back to downloading from a known GitHub mirror.

    Returns:
        Path to the directory containing class subdirectories.
    """
    if target_dir is None:
        target_dir = RAW_DATA_DIR

    target_dir = Path(target_dir)

    # Check if data already exists
    existing = _check_existing_data(target_dir)
    if existing:
        print(f"[INFO] Dataset already exists at: {existing}")
        return existing

    target_dir.mkdir(parents=True, exist_ok=True)

    # Strategy: Download from GitHub mirror of PlantVillage
    print("[INFO] Downloading PlantVillage tomato subset...")
    print("[INFO] Source: GitHub PlantVillage mirror (public repository)")

    _download_from_github_mirror(target_dir)

    return target_dir


def _check_existing_data(data_dir: Path) -> Optional[Path]:
    """Check if dataset already exists with expected class folders."""
    if not data_dir.exists():
        return None

    found_classes = []
    for class_name in CLASS_NAMES:
        class_dir = data_dir / class_name
        if class_dir.exists() and len(list(class_dir.glob("*.jpg"))) > 0:
            found_classes.append(class_name)
        elif class_dir.exists() and len(list(class_dir.glob("*.JPG"))) > 0:
            found_classes.append(class_name)

    if len(found_classes) == len(CLASS_NAMES):
        return data_dir

    return None


def _download_from_github_mirror(target_dir: Path):
    """
    Download PlantVillage images from a GitHub mirror.

    Uses the commonly available PlantVillage GitHub repository.
    Falls back to creating a small synthetic placeholder if download fails
    (clearly marked as synthetic for honesty).
    """
    # The PlantVillage dataset is available on Kaggle and various GitHub mirrors.
    # We provide instructions for manual download as a fallback.
    print("=" * 70)
    print("DATASET DOWNLOAD INSTRUCTIONS")
    print("=" * 70)
    print()
    print("The PlantVillage dataset needs to be placed in:")
    print(f"  {target_dir}")
    print()
    print("Option 1 (Recommended - Kaggle):")
    print("  1. Visit: https://www.kaggle.com/datasets/emmarex/plantdisease")
    print("  2. Download the dataset")
    print("  3. Extract the following folders into the data/raw/ directory:")
    for cls in CLASS_NAMES:
        print(f"     - {cls}/")
    print()
    print("Option 2 (GitHub):")
    print("  1. Visit: https://github.com/spMohanty/PlantVillage-Dataset")
    print("  2. Clone or download the 'color' directory")
    print("  3. Copy the relevant tomato folders to data/raw/")
    print()
    print("Option 3 (Automatic - requires kaggle CLI):")
    print("  Run: kaggle datasets download -d emmarex/plantdisease")
    print()
    print("=" * 70)

    # Try automatic download via kaggle CLI if available
    if _try_kaggle_download(target_dir):
        return

    # Check if user has manually placed data
    if _check_existing_data(target_dir):
        print("[INFO] Data found after instructions!")
        return

    print()
    print("[WARNING] Could not auto-download dataset.")
    print("[INFO] Creating minimal synthetic dataset for code verification...")
    print("[INFO] THIS IS SYNTHETIC DATA - NOT for real evaluation!")
    print("[INFO] Please download real data before training.")
    print()
    _create_synthetic_placeholder(target_dir)


def _try_kaggle_download(target_dir: Path) -> bool:
    """Attempt to download via kaggle CLI."""
    try:
        import subprocess
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", "emmarex/plantdisease",
             "-p", str(target_dir), "--unzip"],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        if result.returncode == 0:
            # Kaggle download may nest in subdirectories; find and move
            _reorganize_kaggle_download(target_dir)
            if _check_existing_data(target_dir):
                print("[INFO] Successfully downloaded via Kaggle CLI!")
                return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False


def _reorganize_kaggle_download(target_dir: Path):
    """Reorganize Kaggle download which may have nested directories."""
    # Kaggle often puts data in PlantVillage/ or similar subdirectory
    for root, dirs, files in os.walk(target_dir):
        root_path = Path(root)
        if root_path.name in CLASS_NAMES and root_path.parent != target_dir:
            dest = target_dir / root_path.name
            if not dest.exists():
                shutil.move(str(root_path), str(dest))


def _create_synthetic_placeholder(target_dir: Path, images_per_class: int = 50):
    """
    Create synthetic placeholder images for code verification ONLY.
    These are random noise images and CANNOT be used for real evaluation.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    for class_name in CLASS_NAMES:
        class_dir = target_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for i in range(images_per_class):
            # Create random colored images (clearly synthetic)
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img_tf = tf.image.encode_jpeg(tf.constant(img))
            filepath = class_dir / f"synthetic_{i:04d}.jpg"
            tf.io.write_file(str(filepath), img_tf)

    # Write a marker file
    marker = target_dir / "SYNTHETIC_DATA_WARNING.txt"
    marker.write_text(
        "WARNING: This directory contains SYNTHETIC placeholder data.\n"
        "These are random noise images used only for code verification.\n"
        "Do NOT use these for model evaluation or report results from them.\n"
        "Please download the real PlantVillage dataset before training.\n"
    )

    print(f"[INFO] Created {images_per_class} synthetic images per class.")
    print(f"[INFO] Total: {images_per_class * len(CLASS_NAMES)} synthetic images.")


def create_data_splits(
    raw_dir: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
) -> dict:
    """
    Split the raw dataset into train/validation/test sets.

    Uses stratified splitting to ensure balanced class representation.

    Returns:
        Dictionary with paths and counts for each split.
    """
    if raw_dir is None:
        raw_dir = RAW_DATA_DIR
    if processed_dir is None:
        processed_dir = PROCESSED_DATA_DIR

    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

    print("[INFO] Creating train/validation/test splits...")

    # Collect all image paths and labels
    all_images = []
    all_labels = []

    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = raw_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(
                f"Class directory not found: {class_dir}\n"
                f"Please ensure the dataset is downloaded first."
            )

        # Use set of resolved paths to avoid case-insensitive glob duplication on Windows
        images = sorted(list({
            p.resolve() for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        }))

        if len(images) == 0:
            raise ValueError(f"No images found in {class_dir}")

        print(f"  {class_name}: {len(images)} unique images")
        all_images.extend(images)
        all_labels.extend([label_idx] * len(images))

    all_images = np.array(all_images)
    all_labels = np.array(all_labels)

    print(f"  Total: {len(all_images)} unique images across {len(CLASS_NAMES)} classes")

    # First split: separate test set
    train_val_imgs, test_imgs, train_val_labels, test_labels = train_test_split(
        all_images, all_labels,
        test_size=TEST_SPLIT,
        stratify=all_labels,
        random_state=RANDOM_SEED,
    )

    # Second split: separate validation from training
    adjusted_val_split = VALIDATION_SPLIT / (1 - TEST_SPLIT)
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        train_val_imgs, train_val_labels,
        test_size=adjusted_val_split,
        stratify=train_val_labels,
        random_state=RANDOM_SEED,
    )

    # Clean processed directory to avoid stale files
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Copy files to processed directory structure
    splits = {
        "train": (train_imgs, train_labels),
        "validation": (val_imgs, val_labels),
        "test": (test_imgs, test_labels),
    }

    split_info = {}
    for split_name, (images, labels) in splits.items():
        split_dir = processed_dir / split_name
        for class_name in CLASS_NAMES:
            (split_dir / class_name).mkdir(parents=True, exist_ok=True)

        for img_path, label in zip(images, labels):
            class_name = CLASS_NAMES[label]
            dest = split_dir / class_name / img_path.name
            if not dest.exists():
                shutil.copy2(str(img_path), str(dest))

        split_info[split_name] = {
            "count": len(images),
            "class_distribution": {
                CLASS_NAMES[i]: int(np.sum(labels == i))
                for i in range(len(CLASS_NAMES))
            },
        }
        print(f"  {split_name}: {len(images)} images")

    # Save split metadata
    meta_path = processed_dir / "split_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"[INFO] Split metadata saved to {meta_path}")
    return split_info


def create_tf_dataset(
    split: str = "train",
    processed_dir: Optional[Path] = None,
    augment: bool = False,
    batch_size: int = BATCH_SIZE,
) -> tf.data.Dataset:
    """
    Create a tf.data.Dataset from the processed split directory.

    Args:
        split: One of 'train', 'validation', 'test'.
        processed_dir: Path to processed data root.
        augment: Whether to apply data augmentation (train only).

    Returns:
        A batched, prefetched tf.data.Dataset yielding (images, labels).
    """
    if processed_dir is None:
        processed_dir = PROCESSED_DATA_DIR

    split_dir = Path(processed_dir) / split

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Split directory not found: {split_dir}\n"
            f"Run create_data_splits() first."
        )

    # Use keras utility to load from directory
    dataset = tf.keras.utils.image_dataset_from_directory(
        str(split_dir),
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=(split == "train"),
        seed=RANDOM_SEED,
    )

    # Apply MobileNetV2 preprocessing
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

    if augment and split == "train":
        # Build augmentation layers
        augmentation_layer = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
        ], name="augmentation")

        dataset = dataset.map(
            lambda x, y: (preprocess(augmentation_layer(x, training=True)), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        dataset = dataset.map(
            lambda x, y: (preprocess(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    # Optimize pipeline
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def get_dataset_summary(processed_dir: Optional[Path] = None) -> dict:
    """Load and return the dataset split metadata."""
    if processed_dir is None:
        processed_dir = PROCESSED_DATA_DIR

    meta_path = Path(processed_dir) / "split_metadata.json"
    if not meta_path.exists():
        return {"error": "No split metadata found. Run create_data_splits() first."}

    with open(meta_path) as f:
        return json.load(f)


if __name__ == "__main__":
    print("=" * 70)
    print("AI Crop Disease Detection - Dataset Preparation")
    print("=" * 70)
    print()
    print(f"Target classes ({len(CLASS_NAMES)}):")
    for cls in CLASS_NAMES:
        print(f"  - {cls}")
    print()

    # Step 1: Download / verify dataset
    raw_path = download_plantvillage_subset()

    # Step 2: Create splits
    split_info = create_data_splits(raw_dir=raw_path)

    print()
    print("[INFO] Dataset preparation complete!")
    print()
    for split_name, info in split_info.items():
        print(f"  {split_name}: {info['count']} images")
