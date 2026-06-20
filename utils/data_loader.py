"""
data_loader.py
--------------
Handles dataset download (CarDD via Kaggle), directory setup,
train/val/test splitting, and tf.data pipeline creation.
"""

import os
import shutil
import zipfile
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Constants ───────────────────────────────────────────────────────────────
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
SEED       = 42

# Map folder names → human labels and severity levels
DAMAGE_CLASSES = {
    "minor":    {"label": "Minor Damage",    "severity": 1},
    "moderate": {"label": "Moderate Damage", "severity": 2},
    "severe":   {"label": "Severe Damage",   "severity": 3},
    "no_damage":{"label": "No Damage",       "severity": 0},
}


# ── Dataset setup ────────────────────────────────────────────────────────────
def download_cardd_kaggle(dest: str = "data/raw") -> str:
    """
    Downloads the CarDD dataset from Kaggle.
    Requires ~/.kaggle/kaggle.json with your API key.

    Dataset: https://www.kaggle.com/datasets/hendrichscullen/vehidet-dataset-automatic-vehicle-damage
    Alternative: anujms/car-damage-detection

    Returns path to extracted folder.
    """
    os.makedirs(dest, exist_ok=True)
    print("[DataLoader] Downloading dataset from Kaggle ...")
    os.system(
        f"kaggle datasets download -d anujms/car-damage-detection "
        f"--unzip -p {dest}"
    )
    print(f"[DataLoader] Dataset extracted to: {dest}")
    return dest


def build_directory_structure(raw_dir: str = "data/raw",
                               out_dir: str = "data/processed",
                               split: tuple = (0.7, 0.15, 0.15)) -> dict:
    """
    Walks raw_dir, discovers class folders, then splits images into
    processed/{train,val,test}/{class_name}/ directories.

    Returns dict with class names and split counts.
    """
    raw_path  = Path(raw_dir)
    out_path  = Path(out_dir)
    splits    = {"train": split[0], "val": split[1], "test": split[2]}
    stats     = {}

    # Discover class folders
    class_dirs = [d for d in raw_path.iterdir()
                  if d.is_dir() and not d.name.startswith(".")]

    for cls_dir in class_dirs:
        cls_name = cls_dir.name.lower().replace(" ", "_").replace("-", "_")
        images   = list(cls_dir.glob("*.jpg")) + \
                   list(cls_dir.glob("*.jpeg")) + \
                   list(cls_dir.glob("*.png"))

        random.seed(SEED)
        random.shuffle(images)

        n        = len(images)
        n_train  = int(n * split[0])
        n_val    = int(n * split[1])
        buckets  = {
            "train": images[:n_train],
            "val":   images[n_train: n_train + n_val],
            "test":  images[n_train + n_val:],
        }

        stats[cls_name] = {}
        for split_name, files in buckets.items():
            dest = out_path / split_name / cls_name
            dest.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy(f, dest / f.name)
            stats[cls_name][split_name] = len(files)
            print(f"  {cls_name}/{split_name}: {len(files)} images")

    print(f"\n[DataLoader] Directory structure built at: {out_dir}")
    return stats


# ── tf.data pipelines ────────────────────────────────────────────────────────
def get_data_generators(processed_dir: str = "data/processed"):
    """
    Returns (train_gen, val_gen, test_gen, class_names) using
    ImageDataGenerator with augmentation on train set.
    """
    train_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        shear_range=0.1,
    )
    val_aug = ImageDataGenerator(rescale=1.0 / 255)

    base = Path(processed_dir)

    train_gen = train_aug.flow_from_directory(
        base / "train",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        seed=SEED,
    )
    val_gen = val_aug.flow_from_directory(
        base / "val",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        seed=SEED,
    )
    test_gen = val_aug.flow_from_directory(
        base / "test",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
        seed=SEED,
    )

    class_names = list(train_gen.class_indices.keys())
    print(f"[DataLoader] Classes: {class_names}")
    return train_gen, val_gen, test_gen, class_names


# ── Single-image loader (inference) ──────────────────────────────────────────
def load_single_image(img_path: str) -> np.ndarray:
    """Load and preprocess a single image for model inference."""
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)   # (1, 224, 224, 3)
