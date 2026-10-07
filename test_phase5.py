"""
Phase 5 Final QA & End-to-End Verification Test Suite.

Comprehensive checks:
1. Model integrity (file size, uncorrupted, loadable, un-retrained)
2. Preprocessing consistency (shape (1, 224, 224, 3), range [-1, 1], LANCZOS)
3. Class mapping consistency across all modules
4. End-to-end prediction across all 4 real classes (Healthy, Early Blight, Late Blight, Leaf Mold)
5. Structured outputs (top-3, confidence, symptom descriptions, recommendations, disclaimers)
6. Error handling (non-existent, unsupported, corrupted inputs)
7. Official metrics integrity (88.96% test accuracy, 89.08% macro precision, 88.50% macro recall, 88.38% macro F1)
8. Security & codebase cleanliness (no hardcoded secrets or sensitive keys)
"""

import io
import json
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CLASS_NAMES,
    CLASS_LABELS,
    DISEASE_INFO,
    DISCLAIMER_TEXT,
    IMG_SIZE,
    IMG_SHAPE,
    SAVED_MODEL_PATH,
    CONFIDENCE_THRESHOLD,
)
from src.predict import (
    predict_image,
    get_model,
    load_and_preprocess_image,
    format_prediction,
)


def run_phase5_qa():
    print("=" * 70)
    print("PHASE 5: FINAL INTEGRATION & QA VERIFICATION SUITE")
    print("=" * 70)
    print()

    tests_passed = 0
    total_tests = 0

    # -------------------------------------------------------------
    # 1. Model Integrity & No-Retraining Verification
    # -------------------------------------------------------------
    total_tests += 1
    print("[QA 1] Verifying trained model integrity...")
    assert SAVED_MODEL_PATH.exists(), f"Model artifact missing at {SAVED_MODEL_PATH}"
    model_size = SAVED_MODEL_PATH.stat().st_size / (1024 * 1024)
    # Model size must remain exactly around 11.07 MB as produced in Phase 2
    assert 11.0 <= model_size <= 11.15, f"Unexpected model size: {model_size:.2f} MB"

    model = get_model()
    assert len(model.layers) == 6, f"Expected 6 top-level model layers, found {len(model.layers)}"
    assert model.output_shape == (None, 4), f"Expected output shape (None, 4), got {model.output_shape}"
    print(f"  [PASS] Model artifact verified: {SAVED_MODEL_PATH.name} ({model_size:.2f} MB, output (None, 4)).")
    tests_passed += 1

    # -------------------------------------------------------------
    # 2. Preprocessing Consistency Verification
    # -------------------------------------------------------------
    total_tests += 1
    print("\n[QA 2] Verifying preprocessing consistency...")
    test_dir = PROJECT_ROOT / "data" / "processed" / "test"
    sample_img = next(test_dir.rglob("*.jpg"))
    preprocessed_tensor = load_and_preprocess_image(sample_img)

    assert preprocessed_tensor.shape == (1, 224, 224, 3), f"Wrong tensor shape: {preprocessed_tensor.shape}"
    assert preprocessed_tensor.dtype == np.float32, f"Wrong dtype: {preprocessed_tensor.dtype}"
    # MobileNetV2 preprocessing maps pixels into [-1.0, 1.0]
    assert -1.05 <= preprocessed_tensor.min() <= 0.0, f"Unexpected min value: {preprocessed_tensor.min()}"
    assert 0.0 <= preprocessed_tensor.max() <= 1.05, f"Unexpected max value: {preprocessed_tensor.max()}"
    print(
        f"  [PASS] Preprocessing validated: shape {preprocessed_tensor.shape}, "
        f"dtype {preprocessed_tensor.dtype}, range [{preprocessed_tensor.min():.2f}, {preprocessed_tensor.max():.2f}]."
    )
    tests_passed += 1

    # -------------------------------------------------------------
    # 3. Class Mapping Consistency Across Modules
    # -------------------------------------------------------------
    total_tests += 1
    print("\n[QA 3] Verifying class mapping consistency...")
    expected_classes = [
        "Tomato___healthy",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
    ]
    assert CLASS_NAMES == expected_classes, f"CLASS_NAMES mismatch: {CLASS_NAMES}"
    for cls in expected_classes:
        assert cls in CLASS_LABELS, f"Missing label for {cls}"
        assert cls in DISEASE_INFO, f"Missing DISEASE_INFO for {cls}"
        assert len(DISEASE_INFO[cls]["recommendations"]) > 0, f"Empty recommendations for {cls}"
    print("  [PASS] All 4 classes and labels strictly consistent across config, model, and UI dictionaries.")
    tests_passed += 1

    # -------------------------------------------------------------
    # 4. End-to-End Real Test Images (1 of each class minimum)
    # -------------------------------------------------------------
    total_tests += 1
    print("\n[QA 4] Running end-to-end inference on real unseen test images (all 4 classes)...")
    qa_samples = {
        "Tomato___healthy": next((test_dir / "Tomato___healthy").glob("*.jpg")),
        "Tomato___Early_blight": next((test_dir / "Tomato___Early_blight").glob("*.jpg")),
        "Tomato___Late_blight": next((test_dir / "Tomato___Late_blight").glob("*.jpg")),
        "Tomato___Leaf_Mold": next((test_dir / "Tomato___Leaf_Mold").glob("*.jpg")),
    }

    for expected_cls, img_path in qa_samples.items():
        res = predict_image(img_path)
        assert res["success"] is True, f"Inference failed on {img_path}: {res.get('error')}"
        assert res["predicted_class"] in CLASS_NAMES
        assert res["label"] in CLASS_LABELS.values()
        assert 0.0 <= res["confidence"] <= 1.0
        assert len(res["top_3"]) == 3
        assert "description" in res["disease_info"]
        assert len(res["disease_info"]["recommendations"]) >= 3
        assert res["disclaimer"] == DISCLAIMER_TEXT

        # Formatted output check
        formatted = format_prediction(res)
        assert res["label"] in formatted
        assert "PREDICTION REPORT" in formatted

        print(
            f"  [PASS] {img_path.name:<18} ({CLASS_LABELS[expected_cls]:<14}) -> "
            f"Predicted: {res['label']:<14} | Confidence: {res['confidence']*100:5.1f}% | "
            f"Top-1: {res['top_3'][0]['label']} ({res['top_3'][0]['confidence']*100:5.1f}%)"
        )
    tests_passed += 1

    # -------------------------------------------------------------
    # 5. Robust Error Handling Verification
    # -------------------------------------------------------------
    total_tests += 1
    print("\n[QA 5] Verifying error handling for anomalous inputs...")

    # Non-existent file
    r1 = predict_image("data/non_existent_image.png")
    assert r1["success"] is False and "not found" in r1["error"].lower()

    # Unsupported format
    r2 = predict_image(PROJECT_ROOT / "README.md")
    assert r2["success"] is False and "unsupported" in r2["error"].lower()

    # Corrupt binary stream
    r3 = predict_image(io.BytesIO(b"MALFORMED_HEADER_DATA_ABCXYZ"))
    assert r3["success"] is False and "corrupt" in r3["error"].lower() or "unreadable" in r3["error"].lower()

    print("  [PASS] Missing file, unsupported extension, and corrupt stream all handled cleanly without exceptions.")
    tests_passed += 1

    # -------------------------------------------------------------
    # 6. Official Metrics Preservation Verification
    # -------------------------------------------------------------
    total_tests += 1
    print("\n[QA 6] Verifying official model metrics preservation...")
    metrics_path = PROJECT_ROOT / "results" / "evaluation_results.json"
    assert metrics_path.exists(), "Official evaluation results file missing!"

    with open(metrics_path) as f:
        metrics = json.load(f)

    # Check official test evaluation numbers
    acc = metrics["overall_accuracy"]
    prec = metrics["macro_precision"]
    rec = metrics["macro_recall"]
    f1 = metrics["macro_f1"]
    samples = metrics["total_test_samples"]

    assert abs(acc - 0.88957) < 0.001, f"Accuracy altered: {acc}"
    assert abs(prec - 0.89080) < 0.001, f"Macro Precision altered: {prec}"
    assert abs(rec - 0.88500) < 0.001, f"Macro Recall altered: {rec}"
    assert abs(f1 - 0.88376) < 0.001, f"Macro F1 altered: {f1}"
    assert samples == 163, f"Unexpected test sample count: {samples}"

    print(
        f"  [PASS] Official test metrics intact: Accuracy: {acc*100:.2f}%, "
        f"Macro Precision: {prec*100:.2f}%, Macro Recall: {rec*100:.2f}%, "
        f"Macro F1: {f1*100:.2f}%, Test Samples: {samples}."
    )
    tests_passed += 1

    # -------------------------------------------------------------
    # 7. Codebase Cleanliness & Security Verification
    # -------------------------------------------------------------
    total_tests += 1
    print("\n[QA 7] Scanning codebase for security and hardcoded secrets...")
    code_files = list((PROJECT_ROOT / "src").glob("*.py")) + [PROJECT_ROOT / "app.py"]
    forbidden_tokens = ["api_key =", "secret_key =", "password =", "token = 'ghp_"]

    for cf in code_files:
        content = cf.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in content.lower(), f"Suspicious token found in {cf.name}"

    print(f"  [PASS] {len(code_files)} source files scanned. No API keys, credentials, or secrets detected.")
    tests_passed += 1

    # -------------------------------------------------------------
    # QA Summary
    # -------------------------------------------------------------
    print()
    print("=" * 70)
    print(f"ALL {tests_passed}/{total_tests} FINAL QA INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    ok = run_phase5_qa()
    sys.exit(0 if ok else 1)
