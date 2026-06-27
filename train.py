"""
train.py
========
Training script for the Deepfake Detection Benchmark (EE656).

Validated components (do NOT modify):
    - WeightedRandomSampler        (class-balanced batches)
    - BCEWithLogitsLoss            (without pos_weight — sampler already handles imbalance)
    - AdamW                        (weight_decay=1e-4)
    - CosineAnnealingLR            (T_max=epochs)
    - AMP autocast + GradScaler    (CUDA only)
    - Gradient clipping            (max_norm=1.0)
    - Validation-loss early stopping
    - Random seed behaviour

Engineering improvements over original:
    - Refactored into helper functions: run_epoch(), evaluate(), save_checkpoint()
    - Expanded metrics: balanced accuracy, specificity, ROC-AUC, PR-AUC, MCC,
      per-class recall, confusion matrix
    - Fixed train_loss denominator (was len(dataset), now correctly total processed samples)
    - LR logged BEFORE scheduler.step() so it reflects the epoch's actual LR
    - CSV log includes all metrics + confusion matrix cells
    - Resume training support via --resume
    - Dataset sanity checks
    - Type hints and docstrings throughout
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data import DeepfakeDataset
from models import Meso4, MultipleAttentionDetector, PatchResNet, XceptionDetector


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(name: str) -> nn.Module:
    """
    Instantiate a detector by name.

    Supported names:
        meso4, xception, patch_resnet, multiple_attention
    """
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Deepfake Benchmark — Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data_root", default="dataset", type=str,
        help="Root directory containing face images.",
    )
    p.add_argument(
        "--metadata", default="metadata.csv", type=str,
        help="Path to the metadata CSV file.",
    )
    p.add_argument(
        "--model", default="meso4", type=str,
        choices=["meso4", "xception", "patch_resnet", "multiple_attention"],
        help="Detector architecture to train.",
    )
    p.add_argument("--epochs",      default=30,   type=int)
    p.add_argument("--batch_size",  default=16,   type=int)
    p.add_argument("--lr",          default=1e-3, type=float)
    p.add_argument("--weight_decay",default=1e-4, type=float)
    p.add_argument("--image_size",  default=256,  type=int)
    p.add_argument(
        "--patience", default=8, type=int,
        help="Early-stopping patience (epochs without val_loss improvement).",
    )
    p.add_argument(
        "--num_workers",
        default=min(8, os.cpu_count() or 1),
        type=int,
    )
    p.add_argument("--seed", default=42, type=int)
    p.add_argument(
        "--resume", default=None, type=str,
        help="Path to a checkpoint (.pth) to resume training from.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def create_dataloaders(
    args: argparse.Namespace,
) -> tuple[DeepfakeDataset, DeepfakeDataset, DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders.

    Class balancing
    ---------------
    The train split is typically imbalanced (~75 % fake / 25 % real).
    WeightedRandomSampler assigns each sample a weight of 1 / class_count,
    so each draw has P(fake) = P(real) = 0.5.

    BCEWithLogitsLoss is used WITHOUT pos_weight — the sampler already
    corrects the imbalance.  Adding pos_weight on top would double-penalise
    the minority class.
    """
    train_ds = DeepfakeDataset(
        dataset_root=args.data_root,
        metadata_file=args.metadata,
        split="train",
        image_size=args.image_size,
    )
    val_ds = DeepfakeDataset(
        dataset_root=args.data_root,
        metadata_file=args.metadata,
        split="val",
        image_size=args.image_size,
    )

    # Sanity checks
    if len(train_ds) == 0:
        raise RuntimeError("Training dataset is empty. Check --data_root and --metadata.")
    if len(val_ds) == 0:
        raise RuntimeError("Validation dataset is empty. Check --data_root and --metadata.")

    labels = train_ds.metadata["label"].values
    unique_labels = np.unique(labels)
    if not set(unique_labels).issubset({0, 1}):
        raise RuntimeError(f"Unexpected label values found: {unique_labels}. Expected {{0, 1}}.")

    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = torch.DoubleTensor(class_weights[labels])

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return train_ds, val_ds, train_loader, val_loader


# ---------------------------------------------------------------------------
# Metrics container
# ---------------------------------------------------------------------------

class EvalResult:
    """All validation metrics for one epoch."""

    __slots__ = (
        "loss", "accuracy", "balanced_accuracy",
        "precision", "recall", "recall_real", "specificity",
        "f1", "roc_auc", "pr_auc", "mcc", "confusion",
    )

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> str:
        tn, fp, fn, tp = self.confusion.ravel()
        return (
            f"  Val Loss          : {self.loss:.4f}\n"
            f"  Accuracy          : {100 * self.accuracy:.2f}%\n"
            f"  Balanced Accuracy : {100 * self.balanced_accuracy:.2f}%\n"
            f"  Precision (fake)  : {100 * self.precision:.2f}%\n"
            f"  Recall    (fake)  : {100 * self.recall:.2f}%\n"
            f"  Recall    (real)  : {100 * self.recall_real:.2f}%\n"
            f"  Specificity       : {100 * self.specificity:.2f}%\n"
            f"  F1 Score          : {100 * self.f1:.2f}%\n"
            f"  ROC-AUC           : {self.roc_auc:.4f}\n"
            f"  PR-AUC            : {self.pr_auc:.4f}\n"
            f"  MCC               : {self.mcc:.4f}\n"
            f"  Confusion  TN={tn:4d}  FP={fp:4d}  FN={fn:4d}  TP={tp:4d}"
        )

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    @staticmethod
    def csv_header() -> list[str]:
        return [
            "epoch", "epoch_duration_s",
            "train_loss", "train_accuracy", "lr",
            "val_loss", "val_accuracy", "val_balanced_accuracy",
            "precision", "recall_fake", "recall_real", "specificity",
            "f1_score", "roc_auc", "pr_auc", "mcc",
            "TN", "FP", "FN", "TP",
        ]

    def csv_row(
        self,
        epoch: int,
        duration: float,
        train_loss: float,
        train_acc: float,
        lr: float,
    ) -> list:
        tn, fp, fn, tp = self.confusion.ravel()
        return [
            epoch, round(duration, 2),
            round(train_loss, 6), round(train_acc, 6), f"{lr:.2e}",
            round(self.loss, 6), round(self.accuracy, 6),
            round(self.balanced_accuracy, 6),
            round(self.precision, 6), round(self.recall, 6),
            round(self.recall_real, 6), round(self.specificity, 6),
            round(self.f1, 6), round(self.roc_auc, 6),
            round(self.pr_auc, 6), round(self.mcc, 6),
            int(tn), int(fp), int(fn), int(tp),
        ]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EvalResult:
    """
    Run the model over the validation set and compute all metrics.

    Notes
    -----
    - Threshold is 0.5 applied to sigmoid(logit).
    - ROC-AUC and PR-AUC use raw probabilities (no threshold).
    - Fake is the positive class (label = 1).
    - MCC is computed because it is informative even for imbalanced datasets.
    """
    model.eval()

    running_loss = 0.0
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[float] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images).squeeze(1)           # (B,)
            loss = criterion(logits, labels.float())
            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits)               # (B,) in [0, 1]
            preds = (probs > 0.5).long()

            all_labels.extend(labels.long().cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    n = len(all_labels)
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    recall_fake = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall_real = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = float("nan")

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        pr_auc = float("nan")

    try:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except ValueError:
        mcc = float("nan")

    return EvalResult(
        loss=running_loss / n,
        accuracy=float((y_true == y_pred).mean()),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=recall_fake,
        recall_real=recall_real,
        specificity=specificity,
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        mcc=mcc,
        confusion=cm,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    """
    One full training epoch.

    Returns
    -------
    train_loss : float
        Mean per-sample loss across all batches.
    train_accuracy : float
        Fraction of correct predictions on the (balanced) training batches.

    Note: because WeightedRandomSampler produces balanced batches, an
    untrained model will score ~50 % training accuracy.  This is expected
    and is NOT comparable to validation accuracy (which is computed on the
    imbalanced val set).
    """
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch}/{total_epochs}",
        leave=False,
    )

    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=(device.type == "cuda"),
        ):
            logits = model(images).squeeze(1)           # (B,)
            loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at epoch {epoch}: {loss.item():.6f}. "
                "Check your data normalisation and model outputs."
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)
            running_loss += loss.item() * images.size(0)

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    return running_loss / total, correct / total


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_val_loss,
    best_val_accuracy,
    best_balanced_accuracy,
    best_mcc,
    args,
):
    """Save a training checkpoint."""

    tmp_path = str(path) + ".tmp"

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),

            "best_val_loss": best_val_loss,
            "best_val_accuracy": best_val_accuracy,
            "best_balanced_accuracy": best_balanced_accuracy,
            "best_mcc": best_mcc,

            "args": vars(args),
        },
        tmp_path,
    )

    os.replace(tmp_path, path)


def load_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
):
    """
    Load a checkpoint and restore training state.

    Returns
    -------
    start_epoch
    best_val_loss
    best_val_accuracy
    best_balanced_accuracy
    best_mcc
    """

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])

    start_epoch = ckpt["epoch"] + 1

    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    best_val_accuracy = ckpt.get("best_val_accuracy", 0.0)
    best_balanced_accuracy = ckpt.get("best_balanced_accuracy", 0.0)
    best_mcc = ckpt.get("best_mcc", -1.0)

    print(
        f"Resumed from {path} "
        f"(epoch {ckpt['epoch']}, "
        f"val_loss={best_val_loss:.4f}, "
        f"bal_acc={100*best_balanced_accuracy:.2f}%, "
        f"MCC={best_mcc:.4f})"
    )

    return (
        start_epoch,
        best_val_loss,
        best_val_accuracy,
        best_balanced_accuracy,
        best_mcc,
    )

# ---------------------------------------------------------------------------
# Optimizer and scheduler
# ---------------------------------------------------------------------------

def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """AdamW — validated optimizer."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """CosineAnnealingLR — validated scheduler."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = get_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  Deepfake Benchmark — Training")
    print(f"{'='*60}")
    print(f"  Device      : {device}")
    print(f"  Model       : {args.model.upper()}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  LR          : {args.lr}")
    print(f"  Seed        : {args.seed}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------ data
    train_ds, val_ds, train_loader, val_loader = create_dataloaders(args)

    train_labels = train_ds.metadata["label"].values
    class_counts = np.bincount(train_labels)
    print(f"Train images : {len(train_ds)}  (real={class_counts[0]}, fake={class_counts[1]})")
    print(f"Val images   : {len(val_ds)}")

    # ----------------------------------------------------------------- model
    model = build_model(args.model).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters   : {n_params:,}\n")

    # --------------------------------------------------------- loss / optim
    criterion = nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    scheduler = build_scheduler(optimizer, args.epochs)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    # -------------------------------------------------------- output dirs
    os.makedirs("weights", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    weights_dir = Path("weights")

    weights_loss_path = weights_dir / f"{args.model}_best.pth"
    weights_bal_path = weights_dir / f"{args.model}_best_balacc.pth"
    weights_mcc_path = weights_dir / f"{args.model}_best_mcc.pth"

    log_path = Path("results") / f"{args.model}_training_log.csv"

    # ------------------------------------------------------- optional resume
    start_epoch = 1

    best_loss = float("inf")
    best_acc = 0.0

    best_bal_acc = 0.0
    best_mcc = -1.0

    best_epoch = 0
    patience_counter = 0

    if args.resume is not None:
        (
            start_epoch,
            best_loss,
            best_acc,
            best_bal_acc,
            best_mcc,
        ) = load_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            scaler,
        )

    # -------------------------------------------------------- training loop
    with open(log_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(EvalResult.csv_header())

        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.perf_counter()

            # Train
            train_loss, train_acc = run_epoch(
                model, train_loader, criterion,
                optimizer, scaler, device,
                epoch, args.epochs,
            )

            # --- CLEAR CACHE AFTER TRAINING LOOP ---
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # Log LR *before* scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            # Validate
            result = evaluate(model, val_loader, criterion, device)

            # --- CLEAR CACHE AFTER VALIDATION METRICS ARE CALCULATED ---
            if device.type == "cuda":
                torch.cuda.empty_cache()

            duration = time.perf_counter() - t0

            # CSV
            writer.writerow(
                result.csv_row(epoch, duration, train_loss, train_acc, current_lr)
            )
            csvfile.flush()

            # Console
            print(f"\nEpoch {epoch}/{args.epochs}  [{duration:.1f}s]")
            print(f"  Train Loss : {train_loss:.4f}")
            print(f"  Train Acc  : {100 * train_acc:.2f}%  (balanced batches — ~50% at epoch 0 is normal)")
            print(f"  LR         : {current_lr:.2e}")
            print(result.summary())

            # Early stopping on val_loss (validated criterion)
            if result.loss < best_loss:
                best_loss = result.loss
                best_acc = result.accuracy
                best_epoch = epoch
                patience_counter = 0

                save_checkpoint(
                    weights_loss_path,
                    model, optimizer, scheduler, scaler,
                    epoch, best_loss, best_acc,best_bal_acc,
                    best_mcc, args,
                )
                print(
                    f"  ✓ Saved {weights_loss_path}  "
                    f"(val_loss={best_loss:.4f}, val_acc={100 * best_acc:.2f}%)"
                )

            else:
                patience_counter += 1
                print(f"  No improvement. Patience {patience_counter}/{args.patience}")

            # ------------------------------------------
            # Best Balanced Accuracy checkpoint
            # ------------------------------------------

            if result.balanced_accuracy > best_bal_acc:

                best_bal_acc = result.balanced_accuracy

                save_checkpoint(
                    weights_bal_path,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_loss,
                    best_acc,
                    best_bal_acc,
                    best_mcc,
                    args,
                )

                print(
                    f"  ✓ Saved {weights_bal_path.name} "
                    f"(balanced_acc={100 * best_bal_acc:.2f}%)"
                )     

            # ------------------------------------------
                # Best MCC checkpoint
            # ------------------------------------------

            if result.mcc > best_mcc:

                best_mcc = result.mcc

                save_checkpoint(
                    weights_mcc_path,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_loss,
                    best_acc,
                    best_bal_acc,
                    best_mcc,
                    args,
                )

                print(
                        f"  ✓ Saved {weights_mcc_path.name} "
                        f"(MCC={best_mcc:.4f})"
                    ) 

            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered at epoch {epoch}.")
                break       

    print(f"\n{'='*60}")
    print(f"  Best val loss          : {best_loss:.4f}")
    print(f"  Best val accuracy      : {100 * best_acc:.2f}%")
    print(f"  Best balanced accuracy : {100 * best_bal_acc:.2f}%")
    print(f"  Best MCC              : {best_mcc:.4f}")
    print(f"  Best epoch            : {best_epoch}")
    print(f"  Log saved         : {log_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
