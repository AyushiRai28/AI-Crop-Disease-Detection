"""
Evaluation script for AI Crop Disease Detection.

This script:
1. Loads the trained model
2. Evaluates on the held-out test set
3. Generates classification report, confusion matrix
4. Saves evaluation results and plots

Usage:
    python -m src.evaluate
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)

from src.config import (
    CLASS_NAMES,
    CLASS_LABELS,
    SAVED_MODEL_PATH,
    RESULTS_DIR,
    PROCESSED_DATA_DIR,
    IMG_SIZE,
    BATCH_SIZE,
)
from src.dataset import create_tf_dataset


def evaluate_model(
    model_path: Path = SAVED_MODEL_PATH,
    results_dir: Path = RESULTS_DIR,
):
    """
    Evaluate the trained model on the test set.

    Generates:
    - Classification report (precision, recall, F1 per class)
    - Confusion matrix plot
    - Overall accuracy
    - Results JSON file
    """
    print("=" * 70)
    print("AI Crop Disease Detection — Model Evaluation")
    print("=" * 70)
    print()

    # ─── Load Model ───────────────────────────────────────────────
    if not model_path.exists():
        print(f"[ERROR] No trained model found at: {model_path}")
        print("  Run training first: python -m src.train")
        sys.exit(1)

    print(f"[INFO] Loading model from: {model_path}")
    model = tf.keras.models.load_model(str(model_path))
    print("[INFO] Model loaded successfully.")
    print()

    # ─── Check for Synthetic Data ─────────────────────────────────
    is_synthetic = (PROCESSED_DATA_DIR.parent / "raw" / "SYNTHETIC_DATA_WARNING.txt").exists()
    if is_synthetic:
        print("!" * 70)
        print("WARNING: Evaluating on SYNTHETIC placeholder data!")
        print("Results are MEANINGLESS — for code verification only.")
        print("Download real PlantVillage data before reporting results.")
        print("!" * 70)
        print()

    # ─── Load Test Data ───────────────────────────────────────────
    print("[INFO] Loading test dataset...")
    test_ds = create_tf_dataset("test", augment=False)
    test_batches = tf.data.experimental.cardinality(test_ds).numpy()
    print(f"  Test batches: {test_batches}")
    print()

    # ─── Predict ──────────────────────────────────────────────────
    print("[INFO] Running predictions on test set...")
    all_predictions = []
    all_labels = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        all_predictions.append(preds)
        all_labels.append(labels.numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Convert from one-hot to class indices
    pred_classes = np.argmax(all_predictions, axis=1)
    true_classes = np.argmax(all_labels, axis=1)
    pred_confidences = np.max(all_predictions, axis=1)

    total_samples = len(true_classes)
    print(f"  Total test samples: {total_samples}")
    print()

    # ─── Classification Report ────────────────────────────────────
    display_names = [CLASS_LABELS[c] for c in CLASS_NAMES]

    report_str = classification_report(
        true_classes, pred_classes,
        target_names=display_names,
        digits=4,
    )

    report_dict = classification_report(
        true_classes, pred_classes,
        target_names=display_names,
        output_dict=True,
    )

    overall_accuracy = accuracy_score(true_classes, pred_classes)
    macro_precision = report_dict["macro avg"]["precision"]
    macro_recall = report_dict["macro avg"]["recall"]
    macro_f1 = report_dict["macro avg"]["f1-score"]

    print("[RESULTS] Classification Report:")
    print(report_str)
    print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy * 100:.2f}%)")
    print(f"Macro Precision:  {macro_precision:.4f} ({macro_precision * 100:.2f}%)")
    print(f"Macro Recall:     {macro_recall:.4f} ({macro_recall * 100:.2f}%)")
    print(f"Macro F1-Score:   {macro_f1:.4f} ({macro_f1 * 100:.2f}%)")
    print(f"Mean Confidence:  {pred_confidences.mean():.4f}")
    print()

    # ─── Confusion Matrix ─────────────────────────────────────────
    cm = confusion_matrix(true_classes, pred_classes)

    print("[RESULTS] Confusion Matrix:")
    print(cm)
    print()

    # ─── Save Plots ───────────────────────────────────────────────
    results_dir.mkdir(parents=True, exist_ok=True)

    # Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=display_names,
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix — Tomato Disease Classification")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    cm_path = results_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix saved: {cm_path}")

    # ─── Save Training History Plots (if available) ───────────────
    history_path = results_dir / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

        _plot_training_history(history, results_dir)

    # ─── Save Results JSON ────────────────────────────────────────
    results = {
        "overall_accuracy": float(overall_accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "mean_confidence": float(pred_confidences.mean()),
        "total_test_samples": int(total_samples),
        "is_synthetic_data": is_synthetic,
        "per_class_metrics": report_dict,
        "confusion_matrix": cm.tolist(),
        "class_names": CLASS_NAMES,
        "class_labels": list(display_names),
        "model_path": str(model_path),
    }

    results_path = results_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Evaluation results saved: {results_path}")

    # ─── Summary ──────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    if is_synthetic:
        print()
        print("[WARNING] Results are based on SYNTHETIC data and are not meaningful.")
        print("   Download real PlantVillage data for valid evaluation.")
    else:
        print()
        print(f"  Accuracy:     {overall_accuracy * 100:.2f}%")
        print(f"  Confidence:   {pred_confidences.mean() * 100:.2f}%")

    print()
    print(f"  Saved to:     {results_dir}")
    print()


def _plot_training_history(history: dict, results_dir: Path):
    """Plot training history (accuracy and loss curves)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Combine phase A and phase B histories
    acc = []
    val_acc = []
    loss = []
    val_loss = []

    for phase in ["phase_a", "phase_b"]:
        if phase in history:
            acc.extend(history[phase].get("accuracy", []))
            val_acc.extend(history[phase].get("val_accuracy", []))
            loss.extend(history[phase].get("loss", []))
            val_loss.extend(history[phase].get("val_loss", []))

    epochs = range(1, len(acc) + 1)

    # Accuracy Plot
    axes[0].plot(epochs, acc, "b-", label="Training Accuracy", linewidth=2)
    axes[0].plot(epochs, val_acc, "r-", label="Validation Accuracy", linewidth=2)
    if "phase_b" in history:
        phase_a_len = len(history["phase_a"].get("accuracy", []))
        axes[0].axvline(x=phase_a_len + 0.5, color="gray", linestyle="--",
                       alpha=0.5, label="Fine-tuning starts")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss Plot
    axes[1].plot(epochs, loss, "b-", label="Training Loss", linewidth=2)
    axes[1].plot(epochs, val_loss, "r-", label="Validation Loss", linewidth=2)
    if "phase_b" in history:
        phase_a_len = len(history["phase_a"].get("loss", []))
        axes[1].axvline(x=phase_a_len + 0.5, color="gray", linestyle="--",
                       alpha=0.5, label="Fine-tuning starts")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Training History — Tomato Disease Classification", fontsize=14)
    plt.tight_layout()

    history_plot_path = results_dir / "training_history.png"
    plt.savefig(history_plot_path, dpi=150)
    plt.close()
    print(f"[INFO] Training history plot saved: {history_plot_path}")


if __name__ == "__main__":
    evaluate_model()
