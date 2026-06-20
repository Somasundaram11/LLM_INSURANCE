"""
train.py
--------
Standalone training script.

Usage:
    python train.py --epochs 40 --batch_size 32 --processed_dir data/processed

Steps:
    1. Download CarDD dataset from Kaggle (if needed)
    2. Build processed directory structure (train/val/test splits)
    3. Build EfficientNetB0 model
    4. Train with two-phase strategy (frozen → fine-tune)
    5. Evaluate on test set
    6. Save model + class names
"""

import os
import json
import argparse
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Train Car Damage CNN")
    p.add_argument("--raw_dir",       default="data/raw",
                   help="Directory with raw Kaggle dataset")
    p.add_argument("--processed_dir", default="data/processed",
                   help="Output directory for train/val/test split")
    p.add_argument("--model_dir",     default="models",
                   help="Where to save trained model")
    p.add_argument("--output_dir",    default="outputs",
                   help="Where to save evaluation plots")
    p.add_argument("--epochs",        type=int, default=40)
    p.add_argument("--batch_size",    type=int, default=32)
    p.add_argument("--download",      action="store_true",
                   help="Download dataset from Kaggle first")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Imports (heavy; import after arg parsing) ─────────────────────────────
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from utils.data_loader import (
        download_cardd_kaggle,
        build_directory_structure,
        get_data_generators,
    )
    from models.model import build_model, train_model, evaluate_model

    # ── 1. Dataset ───────────────────────────────────────────────────────────
    if args.download:
        download_cardd_kaggle(args.raw_dir)

    if not Path(args.processed_dir).exists() or \
       not any(Path(args.processed_dir).iterdir()):
        print("[Train] Building processed directory structure …")
        stats = build_directory_structure(args.raw_dir, args.processed_dir)
        print(json.dumps(stats, indent=2))
    else:
        print(f"[Train] Using existing processed data at {args.processed_dir}")

    # ── 2. Data generators ───────────────────────────────────────────────────
    train_gen, val_gen, test_gen, class_names = get_data_generators(
        args.processed_dir
    )

    # Save class names alongside model
    os.makedirs(args.model_dir, exist_ok=True)
    class_names_path = os.path.join(args.model_dir, "class_names.json")
    with open(class_names_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"[Train] Class names saved → {class_names_path}")

    # ── 3. Build model ───────────────────────────────────────────────────────
    num_classes = len(class_names)
    model = build_model(num_classes=num_classes)
    model.summary()

    # ── 4. Train ─────────────────────────────────────────────────────────────
    history = train_model(
        model, train_gen, val_gen,
        epochs=args.epochs,
        save_dir=args.model_dir,
    )

    # ── 5. Evaluate ──────────────────────────────────────────────────────────
    # Reload best model for evaluation
    import tensorflow as tf
    best_model = tf.keras.models.load_model(
        os.path.join(args.model_dir, "best_model.keras")
    )
    metrics = evaluate_model(
        best_model, test_gen, class_names,
        save_dir=args.output_dir,
    )

    # Save metrics
    metrics_save = {k: v for k, v in metrics.items()
                    if k not in ("report", "confusion_matrix")}
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics_save, f, indent=2)
    print(f"\n[Train] Metrics saved → {metrics_path}")
    print("\n✅ Training complete!")
    print(f"   Model  : {args.model_dir}/best_model.keras")
    print(f"   Outputs: {args.output_dir}/")


if __name__ == "__main__":
    main()
