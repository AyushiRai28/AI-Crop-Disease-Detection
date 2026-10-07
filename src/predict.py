"""
Production Inference and Prediction Pipeline for AI Crop Disease Detection.

Features:
- Validates input files, paths, and formats
- Gracefully handles missing, corrupted, or unsupported images
- Cached model loader to prevent redundant disk I/O in long-running sessions
- Exact training-time preprocessing pipeline (MobileNetV2 preprocess_input)
- Structured dictionary output with probabilities, top-k rankings, and crop-care advice
- Confidence thresholding (< 60% flags cautious warning)
- Scientific disclaimers emphasizing AI assistance over guaranteed diagnosis

Usage:
    # CLI
    python -m src.predict --image path/to/leaf.jpg
    python -m src.predict --image path/to/leaf.jpg --json

    # Python API
    from src.predict import predict_image
    result = predict_image("path/to/leaf.jpg")
"""

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    CLASS_NAMES,
    CLASS_LABELS,
    IMG_SIZE,
    SAVED_MODEL_PATH,
    NUM_CLASSES,
    SUPPORTED_IMAGE_EXTENSIONS,
    CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_WARNING,
    DISCLAIMER_TEXT,
    DISEASE_INFO,
)

# Global in-memory cache for the loaded Keras model
_MODEL_CACHE: Optional[tf.keras.Model] = None
_CACHED_MODEL_PATH: Optional[str] = None


def get_model(
    model_path: Union[str, Path] = SAVED_MODEL_PATH,
    force_reload: bool = False,
) -> tf.keras.Model:
    """
    Retrieve the trained MobileNetV2 model, caching it in memory.

    Args:
        model_path: Path to the .keras model artifact.
        force_reload: If True, reload from disk even if cached.

    Returns:
        Loaded tf.keras.Model.

    Raises:
        FileNotFoundError: If the model file is not found on disk.
    """
    global _MODEL_CACHE, _CACHED_MODEL_PATH

    model_path_str = str(Path(model_path).resolve())

    if not force_reload and _MODEL_CACHE is not None and _CACHED_MODEL_PATH == model_path_str:
        return _MODEL_CACHE

    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Trained model artifact not found at: {p.resolve()}\n"
            "Please ensure Phase 2 training has completed successfully."
        )

    _MODEL_CACHE = tf.keras.models.load_model(model_path_str)
    _CACHED_MODEL_PATH = model_path_str
    return _MODEL_CACHE


def load_and_preprocess_image(
    image_input: Union[str, Path, bytes, io.BytesIO, Image.Image],
    target_size: int = IMG_SIZE,
) -> np.ndarray:
    """
    Load and preprocess a single image for MobileNetV2 inference.

    Preprocesses image using the exact pipeline used during training:
    1. Reads image into RGB PIL Image
    2. Resizes to (target_size, target_size) using LANCZOS interpolation
    3. Converts to float32 NumPy array
    4. Applies tf.keras.applications.mobilenet_v2.preprocess_input (scales to [-1, 1])
    5. Adds batch dimension (1, 224, 224, 3)

    Args:
        image_input: File path, raw bytes, BytesIO buffer, or PIL Image.
        target_size: Target square dimension (224 for MobileNetV2).

    Returns:
        Preprocessed NumPy array of shape (1, 224, 224, 3).

    Raises:
        FileNotFoundError: If file path does not exist.
        ValueError: If file extension is unsupported or input is unreadable.
    """
    # 1. Resolve PIL Image
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.exists():
            raise FileNotFoundError(f"Image file not found: {p}")
        if not p.is_file():
            raise ValueError(f"Specified path is not a file: {p}")
        if p.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format '{p.suffix}'. Supported formats: {sorted(list(SUPPORTED_IMAGE_EXTENSIONS))}"
            )
        try:
            with Image.open(p) as img_file:
                img = img_file.convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise ValueError(f"Corrupt or unreadable image file '{p}': {e}") from e

    elif isinstance(image_input, bytes):
        try:
            with Image.open(io.BytesIO(image_input)) as img_file:
                img = img_file.convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise ValueError(f"Corrupt or unreadable image bytes: {e}") from e

    elif isinstance(image_input, io.BytesIO):
        try:
            with Image.open(image_input) as img_file:
                img = img_file.convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise ValueError(f"Corrupt or unreadable buffer: {e}") from e

    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")

    else:
        raise TypeError(
            f"Unsupported image input type: {type(image_input)}. "
            "Expected str, Path, bytes, BytesIO, or PIL.Image."
        )

    # 2. Resize
    img_resized = img.resize((target_size, target_size), Image.LANCZOS)

    # 3. Convert to float32 array
    img_array = np.array(img_resized, dtype=np.float32)

    # 4. MobileNetV2 preprocessing (scales pixel values between -1 and 1)
    img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # 5. Expand batch dimension
    img_batched = np.expand_dims(img_preprocessed, axis=0)

    return img_batched


def predict_image(
    image_input: Union[str, Path, bytes, io.BytesIO, Image.Image],
    model: Optional[tf.keras.Model] = None,
    model_path: Union[str, Path] = SAVED_MODEL_PATH,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Run disease prediction on a single plant leaf image.

    Args:
        image_input: Path, bytes, buffer, or PIL Image.
        model: Pre-loaded Keras model. If None, uses cached model from model_path.
        model_path: Path to model file if model is not provided.
        top_k: Number of ranked predictions to return (1-4).

    Returns:
        Structured prediction dictionary with status, predictions, probabilities,
        confidence warning, and disease guidance.
    """
    source_name = (
        str(image_input) if isinstance(image_input, (str, Path)) else "<in-memory-image>"
    )

    # Validate and preprocess image
    try:
        img_array = load_and_preprocess_image(image_input)
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "image_source": source_name,
            "predicted_class": None,
            "label": None,
            "confidence": 0.0,
            "probabilities": {},
            "top_3": [],
            "is_low_confidence": False,
            "warning": None,
            "disease_info": {},
            "disclaimer": DISCLAIMER_TEXT,
        }

    # Load model
    try:
        active_model = model if model is not None else get_model(model_path)
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to load model: {e}",
            "image_source": source_name,
            "predicted_class": None,
            "label": None,
            "confidence": 0.0,
            "probabilities": {},
            "top_3": [],
            "is_low_confidence": False,
            "warning": None,
            "disease_info": {},
            "disclaimer": DISCLAIMER_TEXT,
        }

    # Run inference
    try:
        raw_preds = active_model.predict(img_array, verbose=0)[0]
    except Exception as e:
        return {
            "success": False,
            "error": f"Inference execution failed: {e}",
            "image_source": source_name,
            "predicted_class": None,
            "label": None,
            "confidence": 0.0,
            "probabilities": {},
            "top_3": [],
            "is_low_confidence": False,
            "warning": None,
            "disease_info": {},
            "disclaimer": DISCLAIMER_TEXT,
        }

    # Sort class predictions
    top_k_clamped = min(max(1, top_k), NUM_CLASSES)
    sorted_indices = np.argsort(raw_preds)[::-1]
    best_idx = int(sorted_indices[0])

    best_class = CLASS_NAMES[best_idx]
    best_label = CLASS_LABELS[best_class]
    confidence = float(raw_preds[best_idx])

    is_low_conf = confidence < CONFIDENCE_THRESHOLD
    warning_msg = LOW_CONFIDENCE_WARNING if is_low_conf else None

    # Full probability distribution mapped to readable labels
    probabilities = {
        CLASS_LABELS[CLASS_NAMES[i]]: float(raw_preds[i])
        for i in range(NUM_CLASSES)
    }

    # Top-K list
    top_predictions = [
        {
            "rank": i + 1,
            "class_name": CLASS_NAMES[idx],
            "label": CLASS_LABELS[CLASS_NAMES[idx]],
            "confidence": float(raw_preds[idx]),
        }
        for i, idx in enumerate(sorted_indices[:top_k_clamped])
    ]

    # Associated crop care guidance
    info = DISEASE_INFO.get(best_class, {})

    return {
        "success": True,
        "image_source": source_name,
        "predicted_class": best_class,
        "label": best_label,
        "confidence": confidence,
        "probabilities": probabilities,
        "top_3": top_predictions,
        "is_low_confidence": is_low_conf,
        "warning": warning_msg,
        "disease_info": info,
        "disclaimer": DISCLAIMER_TEXT,
    }


def format_prediction(result: Dict[str, Any]) -> str:
    """Format structured prediction results into a readable text report."""
    lines = []
    lines.append("=" * 60)
    lines.append("AI CROP DISEASE PREDICTION REPORT")
    lines.append("=" * 60)

    if not result.get("success", False):
        lines.append(f"[ERROR] Inference failed: {result.get('error', 'Unknown error')}")
        lines.append(f"Source: {result.get('image_source')}")
        lines.append("=" * 60)
        return "\n".join(lines)

    lines.append(f"Image Source: {result['image_source']}")
    lines.append("")
    lines.append(f"  Likely Disease Class:  {result['label']}")
    lines.append(f"  AI Confidence:         {result['confidence'] * 100:.2f}%")

    if result.get("is_low_confidence"):
        lines.append("")
        lines.append(f"  [ATTENTION] {result['warning']}")

    lines.append("")
    lines.append("  Class Probabilities (Ranked):")
    for pred in result["top_3"]:
        bar_len = int(pred["confidence"] * 25)
        bar = "#" * bar_len + "." * (25 - bar_len)
        lines.append(
            f"    #{pred['rank']} {pred['label']:<15} {pred['confidence'] * 100:6.2f}% [{bar}]"
        )

    info = result.get("disease_info", {})
    if info:
        lines.append("")
        lines.append("  Symptom Description:")
        lines.append(f"    {info.get('description', 'N/A')}")
        lines.append("")
        lines.append("  Recommended Next Steps:")
        for rec in info.get("recommendations", []):
            lines.append(f"    - {rec}")

    lines.append("")
    lines.append(f"  Notice: {result.get('disclaimer', DISCLAIMER_TEXT)}")
    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    """CLI entrypoint for single-image prediction."""
    parser = argparse.ArgumentParser(
        description="Predict tomato crop disease from a leaf image"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to the leaf image file"
    )
    parser.add_argument(
        "--model", type=str, default=str(SAVED_MODEL_PATH),
        help=f"Path to trained model (default: {SAVED_MODEL_PATH})"
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Number of ranked predictions to display (default: 3)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output raw JSON results"
    )
    args = parser.parse_args()

    res = predict_image(args.image, model_path=args.model, top_k=args.top_k)

    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print(format_prediction(res))


if __name__ == "__main__":
    main()
