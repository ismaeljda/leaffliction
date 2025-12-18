#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="Leaffliction predictor")
    ap.add_argument("image_path", help="Path to an image")
    ap.add_argument("--bundle", default="leaffliction_bundle.zip", help="Zip produced by train.py")
    args = ap.parse_args()

    img_path = Path(args.image_path).resolve()
    if not img_path.exists():
        raise SystemExit("Image not found")

    # Extract needed files to a temp folder next to zip (simple + transparent)
    zip_path = Path(args.bundle).resolve()
    if not zip_path.exists():
        raise SystemExit("Bundle zip not found. Run train.py first.")

    extract_dir = zip_path.parent / ".leaffliction_predict_tmp"
    if extract_dir.exists():
        for p in extract_dir.rglob("*"):
            if p.is_file():
                p.unlink()
    extract_dir.mkdir(parents=True, exist_ok=True)

    import zipfile
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extract("classes.json", path=extract_dir)
        z.extract("model.pt", path=extract_dir)

    classes = json.loads((extract_dir / "classes.json").read_text(encoding="utf-8"))
    model = build_model(num_classes=len(classes))
    state = torch.load(extract_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pil = Image.open(img_path).convert("RGB")
    x = tf(pil).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1).item()

    print(f"Prediction: {classes[pred]}")

    # Show original and a "transformed" view as required by subject
    # Here: the center-cropped resized version (what the model sees).
    model_view = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])(pil)
    try:
        pil.show(title="Original")
        model_view.show(title="Transformed (model input)")
    except Exception:
        # Headless environment: just save next to the image
        out1 = img_path.parent / (img_path.stem + "_original.png")
        out2 = img_path.parent / (img_path.stem + "_transformed.png")
        pil.save(out1)
        model_view.save(out2)
        print(f"Saved: {out1}")
        print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
