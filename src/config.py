"""
Configuration constants for AI Crop Disease Detection.

Crop: Tomato (Solanum lycopersicum)
Dataset: PlantVillage subset
Classes: 4 (Healthy + 3 diseases)

This is a SUBSET of the full PlantVillage dataset, chosen for
fast training while still demonstrating real transfer learning.
"""

import os
from pathlib import Path

# ─── Project Paths ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
SRC_DIR = PROJECT_ROOT / "src"

# ─── Dataset Configuration ───────────────────────────────────────
# PlantVillage folder names for the 4 selected classes
CLASS_NAMES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
]

# Human-readable labels for display
CLASS_LABELS = {
    "Tomato___healthy": "Healthy",
    "Tomato___Early_blight": "Early Blight",
    "Tomato___Late_blight": "Late Blight",
    "Tomato___Leaf_Mold": "Leaf Mold",
}

NUM_CLASSES = len(CLASS_NAMES)

# ─── Image Configuration ─────────────────────────────────────────
IMG_SIZE = 224          # MobileNetV2 default input size
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# ─── Training Configuration ──────────────────────────────────────
BATCH_SIZE = 32
EPOCHS_FROZEN = 5       # Epochs with base model frozen (feature extraction)
EPOCHS_FINETUNE = 5     # Epochs with top layers unfrozen (fine-tuning)
LEARNING_RATE_FROZEN = 1e-3
LEARNING_RATE_FINETUNE = 1e-5
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.15       # Fraction of training data held out for final test
RANDOM_SEED = 42

# ─── Model Configuration ─────────────────────────────────────────
MODEL_NAME = "tomato_disease_mobilenetv2"
SAVED_MODEL_PATH = MODELS_DIR / f"{MODEL_NAME}.keras"
HISTORY_PATH = RESULTS_DIR / "training_history.json"

# ─── Inference & Disease Information ─────────────────────────────
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CONFIDENCE_THRESHOLD = 0.60
LOW_CONFIDENCE_WARNING = (
    "Low confidence — image may be unclear, poorly lit, or outside the model's reliable domain. "
    "Consider checking with an agricultural expert."
)
DISCLAIMER_TEXT = (
    "AI prediction only — not a guaranteed laboratory diagnosis. "
    "Always consult an agricultural extension specialist or certified plant pathologist for confirmation."
)

DISEASE_INFO = {
    "Tomato___healthy": {
        "label": "Healthy",
        "description": "Foliage exhibits normal green coloration and leaf structure with no visible foliar lesions.",
        "recommendations": [
            "Maintain standard scouting and regular monitoring routines.",
            "Water at the base of the plant to keep foliage dry.",
            "Maintain balanced fertilization and proper crop spacing.",
        ],
    },
    "Tomato___Early_blight": {
        "label": "Early Blight",
        "description": "Likely early blight foliar symptoms (typically caused by the fungal pathogen Alternaria solani), characterized by dark brown spots with concentric ring patterns.",
        "recommendations": [
            "Prune and dispose of infected lower leaves to reduce inoculum spread.",
            "Avoid overhead irrigation and water at the soil level.",
            "Improve plant spacing and airflow through staking and pruning.",
            "Consult local agricultural extension agents for suitable protective fungicide guidelines if necessary.",
        ],
    },
    "Tomato___Late_blight": {
        "label": "Late Blight",
        "description": "Likely late blight foliar lesions (typically caused by the oomycete Phytophthora infestans), characterized by rapidly enlarging water-soaked, dark lesions.",
        "recommendations": [
            "Act quickly: late blight can spread rapidly under cool, moist conditions.",
            "Isolate or remove severely diseased plant tissue immediately (do not compost).",
            "Keep foliage dry and ensure strong ventilation in protected environments.",
            "Consult an agricultural specialist or extension office urgently for approved control measures.",
        ],
    },
    "Tomato___Leaf_Mold": {
        "label": "Leaf Mold",
        "description": "Likely leaf mold foliar symptoms (typically caused by Passalora fulva / Cladosporium fulvum), showing pale yellow spots on upper leaf surfaces and olive-green velvety mold beneath.",
        "recommendations": [
            "Reduce relative humidity (below 85%) and enhance air circulation.",
            "Ensure proper spacing and avoid wetting leaves during irrigation.",
            "Sterilize greenhouse equipment and prune dense canopies.",
            "Consult an agricultural expert for regional organic or chemical control recommendations.",
        ],
    },
}

