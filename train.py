#!/usr/bin/env python3
import argparse
import json
import os
import random
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from PIL import Image


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    val_split: float
    seed: int
    image_size: int
    max_aug_per_class: int


def count_images_per_class(root: Path) -> dict:
    counts = {}
    for class_dir in sorted([d for d in root.rglob("*") if d.is_dir()]):
        imgs = [p for p in class_dir.iterdir() if is_image(p)]
        if imgs:
            rel = class_dir.relative_to(root).as_posix()
            counts[rel] = len(imgs)
    return counts


def discover_class_dirs(root: Path) -> list[Path]:
    class_dirs = []
    for d in sorted(root.rglob("*")):
        if d.is_dir():
            imgs = [p for p in d.iterdir() if is_image(p)]
            if imgs:
                class_dirs.append(d)
    return class_dirs


def copy_originals(src_root: Path, dst_root: Path) -> None:
    for class_dir in discover_class_dirs(src_root):
        rel = class_dir.relative_to(src_root)
        out_dir = dst_root / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in class_dir.iterdir():
            if is_image(p):
                shutil.copy2(p, out_dir / p.name)


def make_augmenter(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
    ])


def augment_to_balance(src_root: Path, dst_root: Path, cfg: TrainConfig) -> dict:
    """
    Copies originals into dst_root, then augments minority classes up to max_count,
    capped by cfg.max_aug_per_class per class (so you don’t explode disk).
    Returns counts after augmentation.
    """
    copy_originals(src_root, dst_root)

    # Count per class at dst
    class_dirs = discover_class_dirs(dst_root)
    class_to_images = {}
    for d in class_dirs:
        imgs = [p for p in d.iterdir() if is_image(p)]
        rel = d.relative_to(dst_root).as_posix()
        class_to_images[rel] = imgs

    if not class_to_images:
        raise SystemExit("No images found in the provided directory.")

    max_count = max(len(v) for v in class_to_images.values())
    augmenter = make_augmenter(cfg.image_size)

    for rel, imgs in class_to_images.items():
        d = dst_root / rel
        need = max_count - len(imgs)
        need = min(need, cfg.max_aug_per_class)

        if need <= 0:
            continue

        # Augment by loading random existing images and saving new ones
        for i in range(need):
            src_img_path = random.choice(imgs)
            img = Image.open(src_img_path).convert("RGB")
            # apply augmenter by converting to tensor and back via functional pipeline
            # easiest: use the same random transforms by running through a dataset-like pipeline
            t = augmenter(img)
            # t is PIL? No, augmenter returns PIL if it is PIL ops; here it returns PIL until ToTensor used.
            # We did not include ToTensor, so t is still PIL Image.
            out_name = f"{src_img_path.stem}_aug{i}{src_img_path.suffix}"
            t.save(d / out_name)

    return count_images_per_class(dst_root)


def split_train_val(src_root: Path, work_root: Path, val_split: float, seed: int) -> tuple[Path, Path]:
    """
    Creates ImageFolder-compatible layout:
      work_root/train/<classes...>
      work_root/val/<classes...>
    using random split within each class folder.
    """
    train_root = work_root / "train"
    val_root = work_root / "val"
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    for class_dir in discover_class_dirs(src_root):
        rel = class_dir.relative_to(src_root)
        imgs = [p for p in class_dir.iterdir() if is_image(p)]
        if not imgs:
            continue

        rng.shuffle(imgs)
        n_val = max(1, int(len(imgs) * val_split))
        val_imgs = imgs[:n_val]
        train_imgs = imgs[n_val:]

        out_train = train_root / rel
        out_val = val_root / rel
        out_train.mkdir(parents=True, exist_ok=True)
        out_val.mkdir(parents=True, exist_ok=True)

        for p in train_imgs:
            shutil.copy2(p, out_train / p.name)
        for p in val_imgs:
            shutil.copy2(p, out_val / p.name)

    return train_root, val_root


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return (correct / total) if total else 0.0


def train_one(cfg: TrainConfig, train_dir: Path, val_dir: Path, out_dir: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(cfg.image_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.15, 0.15, 0.15, 0.03),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(cfg.image_size + 32),
        transforms.CenterCrop(cfg.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)

    if len(val_ds) < 100:
        print(f"Warning: validation set has {len(val_ds)} images (< 100).")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    model = build_model(num_classes=len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_acc = 0.0
    best_path = out_dir / "model.pt"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()
            running += loss.item()

        acc = evaluate(model, val_loader, device)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_path)

        print(f"Epoch {epoch}/{cfg.epochs} | loss={running/ max(1,len(train_loader)):.4f} | val_acc={acc:.4f}")

    metrics = {
        "val_accuracy_best": best_acc,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "val_split": cfg.val_split,
        "seed": cfg.seed,
        "image_size": cfg.image_size,
        "duration_sec": round(time.time() - t0, 2),
        "classes": train_ds.classes,
        "num_train": len(train_ds),
        "num_val": len(val_ds),
    }

    # Save metadata needed by predict.py
    (out_dir / "classes.json").write_text(json.dumps(train_ds.classes, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


def make_zip(bundle_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in bundle_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=p.relative_to(bundle_dir))


def main() -> None:
    ap = argparse.ArgumentParser(description="Leaffliction Part 4 trainer")
    ap.add_argument("data_dir", help="Directory containing class subdirectories with images (e.g. ./Apple/)")
    ap.add_argument("--out", default="leaffliction_bundle.zip", help="Output zip name")
    ap.add_argument("--work", default=".leaffliction_work", help="Working directory (will be overwritten)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--max-aug-per-class", type=int, default=2000)
    args = ap.parse_args()

    src = Path(args.data_dir).resolve()
    if not src.exists() or not src.is_dir():
        raise SystemExit("data_dir must be an existing directory")

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        val_split=args.val_split,
        seed=args.seed,
        image_size=args.img_size,
        max_aug_per_class=args.max_aug_per_class,
    )
    set_seed(cfg.seed)

    work = Path(args.work).resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    bundle_dir = work / "bundle"
    augmented_root = bundle_dir / "augmented_dataset"
    model_root = bundle_dir

    print("Augmenting and balancing dataset...")
    counts_after = augment_to_balance(src, augmented_root, cfg)

    print("Splitting train/val...")
    split_root = work / "split"
    train_dir, val_dir = split_train_val(augmented_root, split_root, cfg.val_split, cfg.seed)

    print("Training model...")
    metrics = train_one(cfg, train_dir, val_dir, model_root)
    (bundle_dir / "counts_after.json").write_text(json.dumps(counts_after, indent=2), encoding="utf-8")

    zip_path = Path(args.out).resolve()
    if zip_path.exists():
        zip_path.unlink()

    print(f"Creating zip: {zip_path}")
    make_zip(bundle_dir, zip_path)

    print("Done.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
