"""
test.py
=======
Inference script for the Deepfake Detection Benchmark (EE656).

Supports:
    - Single image inference
    - Folder / recursive folder inference
    - All four detector architectures
    - CPU and CUDA
    - Confidence scores and probability output
    - CSV export

Usage examples:

    # Single image
    python test.py --model meso4 --checkpoint weights/meso4_best.pth \
                   --input path/to/face.jpg

    # Folder (non-recursive)
    python test.py --model xception --checkpoint weights/xception_best.pth \
                   --input path/to/faces/ --output results/predictions.csv

    # Folder (recursive)
    python test.py --model patch_resnet \
                   --checkpoint weights/patch_resnet_best.pth \
                   --input path/to/dataset/ --recursive \
                   --output results/predictions.csv

    # Force CPU
    python test.py --model multiple_attention \
                   --checkpoint weights/multiple_attention_best.pth \
                   --input face.jpg --device cpu
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# We import models lazily so this script can be run from the repo root
# without a package install, as long as models/ and data/ are on the path.
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(name: str) -> nn.Module:
    """Instantiate a detector by name (weights NOT loaded — call load_checkpoint)."""
    from models import Meso4, MultipleAttentionDetector, PatchResNet, XceptionDetector

    name = name.lower()
    if name == "meso4":
        return Meso4()
    if name == "xception":
        return XceptionDetector(pretrained=True)
    if name == "patch_resnet":
        return PatchResNet(pretrained=True)
    if name == "multiple_attention":
        return MultipleAttentionDetector(pretrained=True)
    raise ValueError(
        f"Unknown model: {name!r}. "
        "Choose from: meso4, xception, patch_resnet, multiple_attention"
    )


def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> dict:
    """
    Load model weights from a training checkpoint.

    Returns the checkpoint dict (for accessing metadata such as best_val_loss).
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return ckpt


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def build_transform(model_name: str, image_size: int = 256) -> transforms.Compose:
    """
    Build the inference-time transform for a given architecture.

    Meso4 uses no ImageNet normalisation (pixel values remain in [0, 1]).
    All other models use ImageNet normalisation.
    """
    ops = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]

    if model_name.lower() != "meso4":
        ops.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )

    return transforms.Compose(ops)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_single(
    model: nn.Module,
    image_path: Path,
    transform: transforms.Compose,
    device: torch.device,
) -> dict:
    """
    Run inference on a single image.

    Returns
    -------
    dict with keys:
        path        : str — absolute path to the image
        probability : float — P(fake) in [0, 1]
        prediction  : str — "fake" | "real"
        confidence  : float — max(P(fake), P(real)) in [0.5, 1]
        latency_ms  : float — wall-clock inference time in milliseconds
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {
            "path": str(image_path.resolve()),
            "probability": None,
            "prediction": "ERROR",
            "confidence": None,
            "latency_ms": None,
            "error": str(e),
        }

    tensor = transform(img).unsqueeze(0).to(device)  # (1, C, H, W)

    t0 = time.perf_counter()
    logit = model(tensor).squeeze()                   # scalar
    latency_ms = (time.perf_counter() - t0) * 1000

    prob_fake = torch.sigmoid(logit).item()
    is_fake = prob_fake > 0.5

    return {
        "path": str(image_path.resolve()),
        "probability": round(prob_fake, 6),
        "prediction": "fake" if is_fake else "real",
        "confidence": round(max(prob_fake, 1 - prob_fake), 6),
        "latency_ms": round(latency_ms, 3),
        "error": "",
    }


def collect_images(input_path: Path, recursive: bool) -> list[Path]:
    """
    Collect all image files from a path (file or directory).

    Parameters
    ----------
    input_path : Path
        Either a single image file or a directory.
    recursive : bool
        If True and input_path is a directory, search subdirectories too.

    Returns
    -------
    Sorted list of image Paths.
    """
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension: {input_path.suffix}. "
                f"Supported: {SUPPORTED_EXTENSIONS}"
            )
        return [input_path]

    if input_path.is_dir():
        pattern = "**/*" if recursive else "*"
        images = sorted(
            p for p in input_path.glob(pattern)
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not images:
            raise RuntimeError(
                f"No images found in {input_path} "
                f"(recursive={recursive}, extensions={SUPPORTED_EXTENSIONS})"
            )
        return images

    raise ValueError(f"Input path does not exist: {input_path}")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

CSV_FIELDS = ["path", "probability", "prediction", "confidence", "latency_ms", "error"]


def save_csv(results: list[dict], output_path: Path) -> None:
    """Write inference results to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nPredictions saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deepfake Benchmark — Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model", required=True, type=str,
        choices=["meso4", "xception", "patch_resnet", "multiple_attention"],
        help="Detector architecture.",
    )
    p.add_argument(
        "--checkpoint", required=True, type=str,
        help="Path to the model checkpoint (.pth).",
    )
    p.add_argument(
        "--input", required=True, type=str,
        help="Path to an image file or a directory of images.",
    )
    p.add_argument(
        "--output", default=None, type=str,
        help="Optional path to save predictions as a CSV file.",
    )
    p.add_argument(
        "--image_size", default=256, type=int,
        help="Image resize target (must match training image size).",
    )
    p.add_argument(
        "--recursive", action="store_true",
        help="Recursively search subdirectories for images.",
    )
    p.add_argument(
        "--device", default=None, type=str,
        help="Device override: 'cpu' or 'cuda'. Auto-detected if omitted.",
    )
    p.add_argument(
        "--threshold", default=0.5, type=float,
        help="Decision threshold for P(fake). Default 0.5.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = get_args()

    # ---------------------------------------------------------------- device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  Deepfake Benchmark — Inference")
    print(f"{'='*60}")
    print(f"  Model      : {args.model.upper()}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Device     : {device}")
    print(f"  Threshold  : {args.threshold}")
    print(f"{'='*60}\n")

    # ----------------------------------------------------------------- model
    model = build_model(args.model).to(device)
    ckpt = load_checkpoint(model, args.checkpoint, device)

    epoch = ckpt.get("epoch", "?")
    best_val_loss = ckpt.get("best_val_loss", float("nan"))
    best_val_acc = ckpt.get("best_val_accuracy", float("nan"))
    print(
        f"Checkpoint epoch   : {epoch}\n"
        f"Best val loss      : {best_val_loss:.4f}\n"
        f"Best val accuracy  : {100 * best_val_acc:.2f}%\n"
    )

    # ------------------------------------------------------------ transform
    transform = build_transform(args.model, args.image_size)

    # --------------------------------------------------------------- images
    input_path = Path(args.input)
    image_paths = collect_images(input_path, args.recursive)
    print(f"Images to process  : {len(image_paths)}\n")

    # --------------------------------------------------------------- infer
    results = []
    fake_count = 0
    real_count = 0
    error_count = 0
    total_latency = 0.0

    for i, img_path in enumerate(image_paths):
        result = predict_single(model, img_path, transform, device)

        # Apply custom threshold (predict_single uses 0.5 by default)
        if result["probability"] is not None:
            prob = result["probability"]
            result["prediction"] = "fake" if prob > args.threshold else "real"
            result["confidence"] = round(
                max(prob, 1 - prob) if args.threshold == 0.5
                else (prob if prob > args.threshold else 1 - prob),
                6,
            )

        results.append(result)

        if result["prediction"] == "fake":
            fake_count += 1
        elif result["prediction"] == "real":
            real_count += 1
        else:
            error_count += 1

        if result["latency_ms"] is not None:
            total_latency += result["latency_ms"]

        # Print single-image result or progress every 100 images
        if len(image_paths) == 1 or (i + 1) % 100 == 0 or i == len(image_paths) - 1:
            print(
                f"  [{i+1:5d}/{len(image_paths)}] "
                f"{img_path.name:<40}  "
                f"P(fake)={result['probability']!r:>8}  "
                f"{result['prediction']:>4}  "
                f"({result.get('latency_ms', '?')} ms)"
            )

    # ---------------------------------------------------------------- summary
    n = len(image_paths)
    avg_latency = total_latency / max(n - error_count, 1)

    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  Total processed : {n}")
    print(f"  Fake            : {fake_count}  ({100 * fake_count / n:.1f}%)")
    print(f"  Real            : {real_count}  ({100 * real_count / n:.1f}%)")
    if error_count:
        print(f"  Errors          : {error_count}")
    print(f"  Avg latency     : {avg_latency:.1f} ms/image")
    print(f"  Throughput      : {1000 / avg_latency:.1f} img/s")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------- csv
    if args.output is not None:
        save_csv(results, Path(args.output))


if __name__ == "__main__":
    main()
