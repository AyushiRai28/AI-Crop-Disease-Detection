"""
Extract real PlantVillage images from Hugging Face parquet dataset.

Dataset Source: wellCh4n/tomato-leaf-disease-image (PlantVillage Tomato Dataset)
Extracts ONLY the 4 required classes:
  - Label 0 -> Tomato___healthy
  - Label 4 -> Tomato___Early_blight
  - Label 3 -> Tomato___Late_blight
  - Label 1 -> Tomato___Leaf_Mold
"""

import io
import sys
from pathlib import Path
from PIL import Image
import pyarrow.parquet as pq

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CLASS_NAMES, RAW_DATA_DIR

# Mapping from HF label integer to our project class folder
LABEL_MAP = {
    0: "Tomato___healthy",
    4: "Tomato___Early_blight",
    3: "Tomato___Late_blight",
    1: "Tomato___Leaf_Mold",
}


def extract_images_from_parquet(parquet_path: Path, output_raw_dir: Path, prefix: str = "img"):
    """
    Extract images belonging to the 4 tomato classes from a parquet file.
    """
    parquet_path = Path(parquet_path)
    output_raw_dir = Path(output_raw_dir)

    print(f"[INFO] Reading parquet table: {parquet_path.name}...")
    table = pq.read_table(str(parquet_path))
    labels = table["label"].to_pylist()
    images = table["image"]

    # Ensure class directories exist
    for cls_name in CLASS_NAMES:
        (output_raw_dir / cls_name).mkdir(parents=True, exist_ok=True)

    extracted_counts = {cls: 0 for cls in CLASS_NAMES}
    total_rows = len(labels)
    print(f"[INFO] Total rows in parquet: {total_rows}")

    for idx in range(total_rows):
        lbl = labels[idx]
        if lbl in LABEL_MAP:
            target_class = LABEL_MAP[lbl]
            img_struct = images[idx].as_py()
            img_bytes = img_struct.get("bytes") if isinstance(img_struct, dict) else img_struct

            if img_bytes:
                # Verify it is a valid image with PIL
                try:
                    img = Image.open(io.BytesIO(img_bytes))
                    img_filename = f"{prefix}_{idx:06d}.jpg"
                    img_dest = output_raw_dir / target_class / img_filename
                    img.save(str(img_dest), format="JPEG")
                    extracted_counts[target_class] += 1
                except Exception as e:
                    print(f"[WARN] Error decoding image index {idx}: {e}")

    print("\n[INFO] Extraction summary for", parquet_path.name)
    for cls, count in extracted_counts.items():
        print(f"  {cls}: {count} images extracted")

    return extracted_counts


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/extract_from_parquet.py <path_to_parquet>")
        sys.exit(1)
    parquet_file = Path(sys.argv[1])
    extract_images_from_parquet(parquet_file, RAW_DATA_DIR, prefix=parquet_file.stem)
