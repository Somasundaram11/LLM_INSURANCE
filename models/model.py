"""
model.py
--------
CNN damage classifier built on EfficientNetB0 transfer learning.
Includes:
  - build_model()        – constructs the Keras model
  - train_model()        – full training loop with callbacks
  - evaluate_model()     – accuracy, confusion matrix, classification report
  - predict_single()     – inference on one preprocessed image array
  - load_saved_model()   – reload from disk
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from sklearn.metrics import classification_report, confusion_matrix


# ── Model builder ─────────────────────────────────────────────────────────────
def build_model(num_classes: int,
                img_size: tuple = (224, 224),
                dropout: float = 0.4,
                learning_rate: float = 1e-3) -> Model:
    """
    EfficientNetB0 backbone + custom classification head.

    Architecture:
        EfficientNetB0 (imagenet, frozen initially)
        → GlobalAveragePooling2D
        → BatchNormalization
        → Dense(256, relu)
        → Dropout(dropout)
        → Dense(num_classes, softmax)
    """
    base = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(*img_size, 3),
    )
    base.trainable = False           # freeze backbone for initial training

    inputs = tf.keras.Input(shape=(*img_size, 3))
    x      = base(inputs, training=False)
    x      = layers.GlobalAveragePooling2D()(x)
    x      = layers.BatchNormalization()(x)
    x      = layers.Dense(256, activation="relu")(x)
    x      = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="CarDamageClassifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return model


def unfreeze_top_layers(model: Model,
                         n_layers: int = 20,
                         learning_rate: float = 1e-5) -> Model:
    """
    Fine-tune the top n layers of the backbone.
    Call this after initial training converges.
    """
    base = model.layers[1]           # EfficientNetB0
    base.trainable = True

    # Freeze all except the last n layers
    for layer in base.layers[:-n_layers]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    print(f"[Model] Fine-tuning: top {n_layers} backbone layers unfrozen.")
    return model


# ── Training ──────────────────────────────────────────────────────────────────
def train_model(model: Model,
                train_gen,
                val_gen,
                epochs: int = 30,
                save_dir: str = "models") -> dict:
    """
    Full training loop.
    Phase 1: frozen backbone (epochs // 2 epochs)
    Phase 2: fine-tune top layers (remaining epochs)

    Returns history dict.
    """
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "best_model.keras")

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=7,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(ckpt_path, monitor="val_accuracy",
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=3, min_lr=1e-7, verbose=1),
        TensorBoard(log_dir=os.path.join(save_dir, "logs"), histogram_freq=1),
    ]

    phase1_epochs = max(1, epochs // 2)

    print("\n[Model] Phase 1 — training head (backbone frozen) …")
    hist1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=phase1_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    print("\n[Model] Phase 2 — fine-tuning top backbone layers …")
    model = unfreeze_top_layers(model)
    hist2 = model.fit(
        train_gen,
        validation_data=val_gen,
        initial_epoch=phase1_epochs,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Merge histories
    history = {}
    for k in hist1.history:
        history[k] = hist1.history[k] + hist2.history.get(k, [])

    _plot_training(history, save_dir)
    print(f"\n[Model] Best model saved → {ckpt_path}")
    return history


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_model(model: Model,
                   test_gen,
                   class_names: list,
                   save_dir: str = "outputs") -> dict:
    """
    Full evaluation: loss/accuracy + confusion matrix + classification report.
    Saves confusion matrix figure.
    """
    os.makedirs(save_dir, exist_ok=True)

    loss, acc, prec, rec = model.evaluate(test_gen, verbose=1)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-7)
    print(f"\n[Eval] Test accuracy : {acc:.4f}")
    print(f"[Eval] Test precision: {prec:.4f}")
    print(f"[Eval] Test recall   : {rec:.4f}")
    print(f"[Eval] Test F1-score : {f1:.4f}")

    # Predictions
    y_pred_proba = model.predict(test_gen, verbose=0)
    y_pred       = np.argmax(y_pred_proba, axis=1)
    y_true       = test_gen.classes

    # Classification report
    report = classification_report(y_true, y_pred,
                                   target_names=class_names, digits=4)
    print("\n" + report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Car Damage Classifier")
    plt.tight_layout()
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[Eval] Confusion matrix saved → {cm_path}")

    return {
        "accuracy": acc, "precision": prec,
        "recall": rec,   "f1": f1,
        "report": report, "confusion_matrix": cm,
    }


# ── Inference ─────────────────────────────────────────────────────────────────
def predict_single(model: Model,
                   img_array: np.ndarray,
                   class_names: list) -> dict:
    """
    Predict damage class for a single preprocessed image array (1,224,224,3).
    Returns:
        predicted_class, confidence, severity_score, all_probs
    """
    probs      = model.predict(img_array, verbose=0)[0]   # shape (n_classes,)
    idx        = int(np.argmax(probs))
    confidence = float(probs[idx])
    cls_name   = class_names[idx]

    # Map class to severity (heuristic; refined from DAMAGE_CLASSES)
    severity_map = {
        "no_damage": 0, "minor": 1, "minor_damage": 1,
        "moderate": 2,  "moderate_damage": 2,
        "severe": 3,    "severe_damage": 3,
    }
    key      = cls_name.lower().replace(" ", "_")
    severity = severity_map.get(key, -1)

    return {
        "predicted_class": cls_name,
        "confidence":      round(confidence * 100, 2),
        "severity_score":  severity,
        "all_probabilities": {
            class_names[i]: round(float(probs[i]) * 100, 2)
            for i in range(len(class_names))
        },
    }


# ── Persistence ───────────────────────────────────────────────────────────────
def load_saved_model(path: str = "models/best_model.keras") -> Model:
    model = tf.keras.models.load_model(path)
    print(f"[Model] Loaded from {path}")
    return model


# ── Plotting helper ───────────────────────────────────────────────────────────
def _plot_training(history: dict, save_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["accuracy"],     label="Train acc")
    axes[0].plot(history["val_accuracy"], label="Val acc")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["loss"],     label="Train loss")
    axes[1].plot(history["val_loss"], label="Val loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Model] Training curves saved → {path}")
