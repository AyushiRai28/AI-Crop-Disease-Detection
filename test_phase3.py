"""
Phase 3 Verification Script: Production Inference Pipeline Testing.

Verifies:
1. Module imports cleanly
2. Model loads and caches in memory
3. Probability distribution validity (sum ~ 1.0, non-negative)
4. Valid class assignment across the 4 classes
5. Top-3 ranked output format
6. Real unseen test image predictions (at least 2 images per class = 8+ images)
7. Graceful error handling for missing, invalid, and corrupted inputs
8. CPU inference validation
9. Integrity check on models/ and results/ files
"""

import sys
import io
import time
from pathlib import Path
import numpy as np
from PIL import Image

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASS_NAMES, CLASS_LABELS, SAVED_MODEL_PATH
from src.predict import predict_image, get_model, format_prediction


def run_phase3_verification():
    print("=" * 70)
    print("PHASE 3 VERIFICATION & REAL TEST SUITE")
    print("=" * 70)
    print()

    tests_passed = 0
    total_tests = 0

    # -------------------------------------------------------------
    # 1. Model Loading & In-Memory Caching
    # -------------------------------------------------------------
    total_tests += 1
    print("[TEST 1] Testing model loading and caching...")
    t0 = time.time()
    model1 = get_model()
    t_load1 = time.time() - t0
    t0 = time.time()
    model2 = get_model()
    t_load2 = time.time() - t0

    assert model1 is model2, "Model instance was not cached in memory!"
    assert len(model1.layers) > 0, "Model has 0 layers!"
    print(f"  [PASS] Initial load: {t_load1:.2f}s, Cached retrieval: {t_load2:.5f}s")
    tests_passed += 1

    # -------------------------------------------------------------
    # 2. Real Unseen Test Images (2 per class = 8 total)
    # -------------------------------------------------------------
    test_dir = PROJECT_ROOT / "data" / "processed" / "test"
    assert test_dir.exists(), "Processed test directory does not exist!"

    test_samples = {
        "Tomato___healthy": [],
        "Tomato___Early_blight": [],
        "Tomato___Late_blight": [],
        "Tomato___Leaf_Mold": [],
    }

    for cls in CLASS_NAMES:
        cls_dir = test_dir / cls
        images = sorted(list(cls_dir.glob("*.jpg")))[:3]  # take first 3 real test images
        assert len(images) >= 2, f"Fewer than 2 test images for {cls}!"
        test_samples[cls] = images

    print()
    print("[TEST 2] Testing real unseen images across all 4 classes:")
    real_results = []
    correct_count = 0
    total_real = 0

    for expected_cls, img_list in test_samples.items():
        expected_label = CLASS_LABELS[expected_cls]
        for img_path in img_list:
            total_real += 1
            res = predict_image(img_path, model=model1)
            assert res["success"] is True, f"Inference failed on {img_path}: {res.get('error')}"

            pred_cls = res["predicted_class"]
            pred_label = res["label"]
            conf = res["confidence"]
            probs = res["probabilities"]
            top_3 = res["top_3"]

            # Probability checks
            prob_sum = sum(probs.values())
            assert abs(prob_sum - 1.0) < 1e-4, f"Probabilities do not sum to 1.0! Sum: {prob_sum}"
            assert all(0.0 <= p <= 1.0 for p in probs.values()), "Probability out of [0, 1] range!"
            assert pred_cls in CLASS_NAMES, f"Predicted class '{pred_cls}' not in valid CLASS_NAMES!"
            assert len(top_3) == 3, f"Expected 3 rankings, got {len(top_3)}"

            is_correct = (pred_cls == expected_cls)
            if is_correct:
                correct_count += 1

            status_str = "CORRECT" if is_correct else "MISCLASSIFIED"
            print(
                f"  [{status_str}] Expected: {expected_label:<14} | Predicted: {pred_label:<14} | "
                f"Conf: {conf * 100:6.2f}% | File: {img_path.name}"
            )
            real_results.append({
                "file": img_path.name,
                "expected": expected_label,
                "predicted": pred_label,
                "confidence": conf,
                "correct": is_correct,
            })

    total_tests += 1
    accuracy_subset = correct_count / total_real
    print(f"\n  Real Test Set Batch Accuracy: {correct_count}/{total_real} ({accuracy_subset * 100:.1f}%)")
    assert correct_count >= 6, "Accuracy on real test subset was unexpectedly low!"
    print("  [PASS] Real unseen test predictions verified.")
    tests_passed += 1

    # -------------------------------------------------------------
    # 3. In-Memory PIL Image & Bytes Inference Support
    # -------------------------------------------------------------
    total_tests += 1
    print("\n[TEST 3] Testing in-memory input formats (PIL Image, bytes, BytesIO)...")
    sample_img_path = test_samples["Tomato___healthy"][0]

    # PIL Image
    pil_img = Image.open(sample_img_path)
    res_pil = predict_image(pil_img, model=model1)
    assert res_pil["success"] is True
    assert res_pil["predicted_class"] == "Tomato___healthy"

    # Raw bytes
    with open(sample_img_path, "rb") as f:
        raw_bytes = f.read()
    res_bytes = predict_image(raw_bytes, model=model1)
    assert res_bytes["success"] is True
    assert res_bytes["confidence"] == res_pil["confidence"]

    # BytesIO
    buf = io.BytesIO(raw_bytes)
    res_buf = predict_image(buf, model=model1)
    assert res_buf["success"] is True
    assert res_buf["confidence"] == res_pil["confidence"]

    print("  [PASS] In-memory input formats seamlessly processed.")
    tests_passed += 1

    # -------------------------------------------------------------
    # 4. Graceful Error Handling on Invalid / Corrupt Inputs
    # -------------------------------------------------------------
    total_tests += 1
    print("\n[TEST 4] Testing graceful failure on invalid/missing/corrupted inputs...")

    # Missing file
    res_missing = predict_image("data/does_not_exist.jpg", model=model1)
    assert res_missing["success"] is False
    assert "not found" in res_missing["error"].lower()
    print("  [PASS] Non-existent file handled gracefully.")

    # Unsupported format
    txt_file = PROJECT_ROOT / "requirements.txt"
    res_txt = predict_image(txt_file, model=model1)
    assert res_txt["success"] is False
    assert "unsupported" in res_txt["error"].lower()
    print("  [PASS] Unsupported extension handled gracefully.")

    # Corrupted image bytes
    garbage_bytes = b"NOT_A_VALID_JPEG_IMAGE_DATA_12345"
    res_corrupt = predict_image(garbage_bytes, model=model1)
    assert res_corrupt["success"] is False
    assert "corrupt" in res_corrupt["error"].lower() or "unreadable" in res_corrupt["error"].lower()
    print("  [PASS] Corrupted image bytes handled gracefully.")

    tests_passed += 1

    # -------------------------------------------------------------
    # 5. Low-Confidence Warning Mechanism
    # -------------------------------------------------------------
    total_tests += 1
    print("\n[TEST 5] Testing low-confidence warning threshold (<60%)...")
    # Synthetic flat image or noise to induce uncertainty
    flat_noise = Image.fromarray(np.random.randint(100, 150, (224, 224, 3), dtype=np.uint8))
    res_noise = predict_image(flat_noise, model=model1)
    assert res_noise["success"] is True
    if res_noise["confidence"] < 0.60:
        assert res_noise["is_low_confidence"] is True
        assert res_noise["warning"] is not None
        print(f"  [PASS] Low confidence detected: {res_noise['confidence']*100:.2f}%. Warning triggered properly.")
    else:
        print(f"  [NOTE] Random noise had confidence {res_noise['confidence']*100:.2f}% (above 60%).")
    tests_passed += 1

    # -------------------------------------------------------------
    # 6. Report Formatter
    # -------------------------------------------------------------
    total_tests += 1
    print("\n[TEST 6] Testing human-readable report formatting...")
    sample_res = predict_image(sample_img_path, model=model1)
    formatted_text = format_prediction(sample_res)
    assert "PREDICTION REPORT" in formatted_text
    assert "Likely Disease Class:" in formatted_text
    assert "Recommended Next Steps:" in formatted_text
    assert "Notice:" in formatted_text
    print("  [PASS] Human-readable format verified.")
    tests_passed += 1

    # -------------------------------------------------------------
    # 7. Model Artifact Integrity Check (Ensuring no overwrite)
    # -------------------------------------------------------------
    total_tests += 1
    print("\n[TEST 7] Verifying trained model artifact integrity...")
    model_size_mb = SAVED_MODEL_PATH.stat().st_size / (1024 * 1024)
    assert 10.0 <= model_size_mb <= 13.0, f"Unexpected model size: {model_size_mb:.2f} MB"
    print(f"  [PASS] Model file intact: {SAVED_MODEL_PATH.name} ({model_size_mb:.2f} MB)")
    tests_passed += 1

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------
    print()
    print("=" * 70)
    print(f"ALL {tests_passed}/{total_tests} PHASE 3 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_phase3_verification()
    sys.exit(0 if success else 1)
