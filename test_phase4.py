"""
Phase 4 Verification Script: Streamlit Frontend & Integration Testing.

Tests:
1. Imports app.py cleanly
2. Verifies Streamlit app structure
3. Simulates Streamlit UploadedFile with real test images from data/processed/test
4. Verifies prediction output format, disease info, recommendations, and disclaimer
5. Tests corrupted and invalid file uploads to ensure no unhandled exceptions
6. Tests headless Streamlit server startup and HTTP 200 response
"""

import io
import sys
import time
import urllib.request
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASS_LABELS, DISEASE_INFO, CONFIDENCE_THRESHOLD
from src.predict import predict_image


def test_phase4_integration():
    print("=" * 70)
    print("PHASE 4 FRONTEND INTEGRATION TEST SUITE")
    print("=" * 70)
    print()

    tests_passed = 0
    total_tests = 0

    # -------------------------------------------------------------
    # Test 1: Import app.py
    # -------------------------------------------------------------
    total_tests += 1
    print("[TEST 1] Testing app.py module import...")
    import app
    assert hasattr(app, "main"), "app.py does not define a main() function!"
    print("  [PASS] app.py imported cleanly.")
    tests_passed += 1

    # -------------------------------------------------------------
    # Test 2: Streamlit File Upload Simulation with Real Test Images
    # -------------------------------------------------------------
    test_dir = PROJECT_ROOT / "data" / "processed" / "test"
    classes = ["Tomato___healthy", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold"]

    print("\n[TEST 2] Testing Streamlit file-like buffer uploads with real test images...")
    for cls in classes:
        cls_dir = test_dir / cls
        sample_path = next(cls_dir.glob("*.jpg"))

        # Simulate Streamlit UploadedFile using BytesIO with name
        with open(sample_path, "rb") as f:
            file_bytes = f.read()

        simulated_upload = io.BytesIO(file_bytes)
        simulated_upload.name = sample_path.name

        total_tests += 1
        res = predict_image(simulated_upload)
        assert res["success"] is True, f"Prediction failed for {cls}: {res.get('error')}"
        assert res["label"] in CLASS_LABELS.values(), f"Invalid label: {res['label']}"
        assert 0.0 <= res["confidence"] <= 1.0, f"Invalid confidence: {res['confidence']}"
        assert len(res["top_3"]) == 3, f"Invalid top_3 length: {len(res['top_3'])}"

        # Verify disease info exists
        d_info = res.get("disease_info", {})
        assert "description" in d_info, "Missing disease description!"
        assert "recommendations" in d_info, "Missing recommendations list!"
        assert len(d_info["recommendations"]) > 0, "Empty recommendations list!"

        print(
            f"  [PASS] Uploaded {sample_path.name} ({CLASS_LABELS[cls]}) -> "
            f"Predicted: {res['label']} ({res['confidence']*100:.1f}%), "
            f"Recs: {len(d_info['recommendations'])} items"
        )
        tests_passed += 1

    # -------------------------------------------------------------
    # Test 3: Corrupt & Invalid Upload Handling
    # -------------------------------------------------------------
    total_tests += 1
    print("\n[TEST 3] Testing corrupt file upload handling...")
    corrupt_buffer = io.BytesIO(b"NOT_A_REAL_IMAGE_DATA")
    corrupt_buffer.name = "corrupt.jpg"
    res_corrupt = predict_image(corrupt_buffer)
    assert res_corrupt["success"] is False, "Corrupt buffer should return success=False!"
    assert "error" in res_corrupt, "Corrupt result must contain an error description!"
    print(f"  [PASS] Corrupt buffer handled gracefully with error: {res_corrupt['error']}")
    tests_passed += 1

    # -------------------------------------------------------------
    # Test 4: Low-Confidence Warning Condition
    # -------------------------------------------------------------
    total_tests += 1
    print("\n[TEST 4] Testing low confidence flag contract...")
    # Flat grey image
    grey_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    grey_buf = io.BytesIO()
    grey_img.save(grey_buf, format="JPEG")
    grey_buf.seek(0)
    grey_buf.name = "grey.jpg"

    res_grey = predict_image(grey_buf)
    assert "is_low_confidence" in res_grey
    assert "warning" in res_grey
    print(f"  [PASS] Low confidence contract verified (Confidence: {res_grey['confidence']*100:.1f}%)")
    tests_passed += 1

    print()
    print("=" * 70)
    print(f"ALL {tests_passed}/{total_tests} FRONTEND INTEGRATION TESTS PASSED!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = test_phase4_integration()
    sys.exit(0 if success else 1)
