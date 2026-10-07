"""
MobileNetV2 transfer learning model for crop disease classification.

Architecture:
    1. MobileNetV2 base (ImageNet pretrained, frozen initially)
    2. Global Average Pooling
    3. Dense (128, ReLU) + Dropout
    4. Dense (NUM_CLASSES, Softmax)

Training Strategy:
    Phase A: Feature extraction (base frozen) — fast convergence
    Phase B: Fine-tuning (top layers unfrozen) — improved accuracy
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import IMG_SHAPE, NUM_CLASSES, CLASS_NAMES


def build_model(
    num_classes: int = NUM_CLASSES,
    input_shape: tuple = IMG_SHAPE,
    dropout_rate: float = 0.3,
) -> keras.Model:
    """
    Build MobileNetV2 transfer learning model.

    Args:
        num_classes: Number of output classes.
        input_shape: Input image shape (H, W, C).
        dropout_rate: Dropout rate for regularization.

    Returns:
        Compiled Keras model ready for feature extraction training.
    """
    # Load MobileNetV2 with ImageNet weights, without top classifier
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )

    # Freeze the base model for feature extraction phase
    base_model.trainable = False

    # Build the classification head
    inputs = keras.Input(shape=input_shape, name="input_image")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(128, activation="relu", name="dense_hidden")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="crop_disease_mobilenetv2")

    return model


def compile_for_feature_extraction(
    model: keras.Model,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """
    Compile model for feature extraction phase (base frozen).
    Uses Adam optimizer with standard learning rate.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def compile_for_finetuning(
    model: keras.Model,
    learning_rate: float = 1e-5,
    fine_tune_at: int = 100,
) -> keras.Model:
    """
    Prepare and compile model for fine-tuning phase.

    Unfreezes the top layers of the base model (from fine_tune_at onwards)
    and recompiles with a lower learning rate.

    Args:
        model: The trained feature extraction model.
        learning_rate: Lower learning rate for fine-tuning.
        fine_tune_at: Layer index from which to unfreeze.
    """
    # Access the base model (second layer after Input)
    base_model = model.layers[1]

    # Unfreeze the base model
    base_model.trainable = True

    # Freeze all layers before fine_tune_at
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Count trainable vs frozen layers
    total_layers = len(base_model.layers)
    trainable_layers = sum(1 for l in base_model.layers if l.trainable)
    print(f"[INFO] Fine-tuning: {trainable_layers}/{total_layers} base layers unfrozen")

    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_model_summary(model: keras.Model) -> str:
    """Return model summary as a string."""
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    return "\n".join(summary_lines)


def get_callbacks(
    model_path: str,
    patience_es: int = 3,
    patience_lr: int = 2,
    initial_value_threshold: float = None,
) -> list:
    """
    Create standard training callbacks.

    Returns:
        List of Keras callbacks for training.
    """
    checkpoint_kwargs = {
        "filepath": str(model_path),
        "monitor": "val_accuracy",
        "save_best_only": True,
        "verbose": 1,
    }
    if initial_value_threshold is not None:
        checkpoint_kwargs["initial_value_threshold"] = initial_value_threshold

    callbacks = [
        # Save best model
        keras.callbacks.ModelCheckpoint(**checkpoint_kwargs),
        # Early stopping to prevent overfitting
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=patience_es,
            restore_best_weights=True,
            verbose=1,
        ),
        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-7,
            verbose=1,
        ),
    ]
    return callbacks


if __name__ == "__main__":
    print("=" * 70)
    print("Model Architecture Summary")
    print("=" * 70)
    print()
    print(f"Classes ({NUM_CLASSES}):")
    for cls in CLASS_NAMES:
        print(f"  - {cls}")
    print()

    model = build_model()
    model = compile_for_feature_extraction(model)
    print(get_model_summary(model))

    print()
    total_params = model.count_params()
    trainable_params = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters:    {total_params - trainable_params:,}")
