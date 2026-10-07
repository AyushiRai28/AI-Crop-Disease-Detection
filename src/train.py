"""
Training script for AI Crop Disease Detection.

This script orchestrates the full training pipeline:
1. Load and prepare data (with augmentation for training)
2. Build MobileNetV2 transfer learning model
3. Phase A: Feature extraction training (base frozen)
4. Phase B: Fine-tuning (top base layers unfrozen)
5. Save model and training history

Usage:
    python -m src.train
    python -m src.train --epochs-frozen 5 --epochs-finetune 5
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf

from src.config import (
    BATCH_SIZE,
    EPOCHS_FROZEN,
    EPOCHS_FINETUNE,
    LEARNING_RATE_FROZEN,
    LEARNING_RATE_FINETUNE,
    SAVED_MODEL_PATH,
    HISTORY_PATH,
    MODELS_DIR,
    RESULTS_DIR,
    PROCESSED_DATA_DIR,
)
from src.dataset import (
    download_plantvillage_subset,
    create_data_splits,
    create_tf_dataset,
    get_dataset_summary,
)
from src.model import (
    build_model,
    compile_for_feature_extraction,
    compile_for_finetuning,
    get_callbacks,
    get_model_summary,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train crop disease classification model"
    )
    parser.add_argument(
        "--epochs-frozen", type=int, default=EPOCHS_FROZEN,
        help=f"Epochs for feature extraction phase (default: {EPOCHS_FROZEN})"
    )
    parser.add_argument(
        "--epochs-finetune", type=int, default=EPOCHS_FINETUNE,
        help=f"Epochs for fine-tuning phase (default: {EPOCHS_FINETUNE})"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip dataset download (assume data exists)"
    )
    parser.add_argument(
        "--skip-finetune", action="store_true",
        help="Skip the fine-tuning phase"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Training batch size (default: {BATCH_SIZE})"
    )
    return parser.parse_args()


def train(args):
    """Run the full training pipeline."""
    start_time = time.time()

    print("=" * 70)
    print("AI Crop Disease Detection — Training Pipeline")
    print("=" * 70)
    print()

    # ─── GPU / Device Info ────────────────────────────────────────
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"[INFO] GPU detected: {gpus}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("[INFO] No GPU detected. Training on CPU (will be slower).")
    print()

    # ─── Step 1: Dataset ──────────────────────────────────────────
    if not args.skip_download:
        print("[STEP 1] Preparing dataset...")
        raw_path = download_plantvillage_subset()
        create_data_splits(raw_dir=raw_path)
    else:
        print("[STEP 1] Skipping download (--skip-download flag)")

    # Verify data exists
    summary = get_dataset_summary()
    if "error" in summary:
        print(f"[ERROR] {summary['error']}")
        sys.exit(1)

    print()
    print("[INFO] Dataset summary:")
    for split_name, info in summary.items():
        print(f"  {split_name}: {info['count']} images")
    print()

    # ─── Step 2: Create Data Pipelines ────────────────────────────
    print("[STEP 2] Creating data pipelines...")
    train_ds = create_tf_dataset("train", augment=True, batch_size=args.batch_size)
    val_ds = create_tf_dataset("validation", augment=False, batch_size=args.batch_size)
    print(f"  Train batches: {tf.data.experimental.cardinality(train_ds).numpy()}")
    print(f"  Val batches:   {tf.data.experimental.cardinality(val_ds).numpy()}")
    print()

    # ─── Step 3: Build Model ──────────────────────────────────────
    print("[STEP 3] Building MobileNetV2 model...")
    model = build_model()
    model = compile_for_feature_extraction(
        model, learning_rate=LEARNING_RATE_FROZEN
    )
    print(get_model_summary(model))
    print()

    # Ensure output directories exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ─── Step 4: Feature Extraction Training ──────────────────────
    print("[STEP 4] Phase A: Feature Extraction Training")
    print(f"  Epochs: {args.epochs_frozen}")
    print(f"  Learning rate: {LEARNING_RATE_FROZEN}")
    print(f"  Base model: FROZEN")
    print()

    callbacks_frozen = get_callbacks(
        model_path=str(SAVED_MODEL_PATH),
        patience_es=max(3, args.epochs_frozen),
    )

    history_frozen = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_frozen,
        callbacks=callbacks_frozen,
        verbose=1,
    )

    print()
    print(f"  Phase A complete!")
    print(f"  Best val_accuracy: {max(history_frozen.history['val_accuracy']):.4f}")
    print()

    # ─── Step 5: Fine-Tuning ──────────────────────────────────────
    all_history = {
        "phase_a": {k: [float(v) for v in vals] for k, vals in history_frozen.history.items()},
    }

    if not args.skip_finetune:
        print("[STEP 5] Phase B: Fine-Tuning")
        print(f"  Epochs: {args.epochs_finetune}")
        print(f"  Learning rate: {LEARNING_RATE_FINETUNE}")
        print(f"  Unfreezing top layers of base model...")
        print()

        model = compile_for_finetuning(
            model,
            learning_rate=LEARNING_RATE_FINETUNE,
            fine_tune_at=100,  # Unfreeze from layer 100 onwards
        )

        best_frozen_val_acc = max(history_frozen.history["val_accuracy"])
        callbacks_finetune = get_callbacks(
            model_path=str(SAVED_MODEL_PATH),
            patience_es=max(3, args.epochs_finetune),
            initial_value_threshold=best_frozen_val_acc,
        )

        # Continue from last epoch number for proper history
        initial_epoch = args.epochs_frozen
        total_epochs = args.epochs_frozen + args.epochs_finetune

        history_finetune = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=total_epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks_finetune,
            verbose=1,
        )

        print()
        print(f"  Phase B complete!")
        print(f"  Best val_accuracy: {max(history_finetune.history['val_accuracy']):.4f}")
        print()

        all_history["phase_b"] = {
            k: [float(v) for v in vals]
            for k, vals in history_finetune.history.items()
        }
    else:
        print("[STEP 5] Skipping fine-tuning (--skip-finetune flag)")
        print()

    # ─── Step 6: Save Results ─────────────────────────────────────
    print("[STEP 6] Saving results...")

    # Save training history
    with open(HISTORY_PATH, "w") as f:
        json.dump(all_history, f, indent=2)
    print(f"  Training history: {HISTORY_PATH}")

    # Confirm model saved
    if SAVED_MODEL_PATH.exists():
        model_size_mb = SAVED_MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"  Model saved:      {SAVED_MODEL_PATH} ({model_size_mb:.1f} MB)")
    else:
        # Save if checkpoint didn't trigger
        model.save(str(SAVED_MODEL_PATH))
        print(f"  Model saved:      {SAVED_MODEL_PATH}")

    # ─── Summary ──────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Total time:    {elapsed / 60:.1f} minutes")
    print(f"  Model path:    {SAVED_MODEL_PATH}")
    print(f"  History path:  {HISTORY_PATH}")
    print()
    print("Next steps:")
    print("  1. Run evaluation:  python -m src.evaluate")
    print("  2. Run prediction:  python -m src.predict --image <path>")
    print("  3. Launch app:      streamlit run app.py")
    print()


if __name__ == "__main__":
    args = parse_args()
    train(args)
