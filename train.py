"""
train.py

Generic training script for the Deepfake Benchmark.
"""
import random
import numpy as np

import argparse
import csv
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import(
    DataLoader,
    WeightedRandomSampler,
)
from tqdm import tqdm

from data import DeepfakeDataset
from models import Meso4
# later:
from models import XceptionDetector
from models import PatchResNet
from models import MultipleAttentionDetector

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

# -------------------------------------------------------
# Model Factory
# -------------------------------------------------------

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_model(name):

    name = name.lower()

    if name == "meso4":
        return Meso4()
    
    elif name == "xception":
        return XceptionDetector(pretrained=False)

    elif name == "patch_resnet":
        return PatchResNet(pretrained=False)

    elif name == "multiple_attention":
        return MultipleAttentionDetector(pretrained=False)

    raise ValueError(f"Unknown model: {name}")


# -------------------------------------------------------
# Argument Parser
# -------------------------------------------------------

def get_args():

    parser = argparse.ArgumentParser(
        description="Deepfake Benchmark Training"
    )

    parser.add_argument(
        "--data_root",
        default="dataset",
        type=str,
    )

    parser.add_argument(
        "--metadata",
        default="metadata.csv",
        type=str,
    )

    parser.add_argument(
        "--model",
        default="meso4",
        type=str,
    )

    parser.add_argument(
        "--epochs",
        default=20,
        type=int,
    )

    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
    )

    parser.add_argument(
        "--lr",
        default=1e-3,
        type=float,
    )

    parser.add_argument(
        "--image_size",
        default=256,
        type=int,
    )

    parser.add_argument(
        "--patience",
        default=8,
        type=int,
    )

    parser.add_argument(
        "--num_workers",
        default=min(8, os.cpu_count()),
        type=int,
    )
    
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )
    return parser.parse_args()


# -------------------------------------------------------
# Dataloaders
# -------------------------------------------------------

def create_dataloaders(args):

    train_dataset = DeepfakeDataset(
        dataset_root=args.data_root,
        metadata_file=args.metadata,
        split="train",
        image_size=args.image_size,
    )

    val_dataset = DeepfakeDataset(
        dataset_root=args.data_root,
        metadata_file=args.metadata,
        split="val",
        image_size=args.image_size,
    )

    # --------------------------------------------
    # Balanced sampling
    # --------------------------------------------

    labels = train_dataset.metadata["label"].values

    class_counts = np.bincount(labels)

    class_weights = 1.0 / class_counts

    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return (
        train_dataset,
        val_dataset,
        train_loader,
        val_loader,
    )

# -------------------------------------------------------
# Validation
# -------------------------------------------------------

def evaluate(model, loader, criterion, device):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs.squeeze(1),
                labels,
            )

            running_loss += loss.item() * images.size(0)

            predictions = (
                torch.sigmoid(outputs.squeeze(1)) > 0.5
            ).long()

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            correct += (
                predictions == labels.long()
            ).sum().item()

            total += labels.size(0)
    
    precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0,
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    return (
    running_loss / total,
    correct / total,
    precision,
    recall,
    f1,
    )


# -------------------------------------------------------
# Training
# -------------------------------------------------------

def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nUsing device: {device}\n")

    (
        train_dataset,
        val_dataset,
        train_loader,
        val_loader,
    ) = create_dataloaders(args)
    
    print(f"Training images   : {len(train_dataset)}")
    print(f"Validation images : {len(val_dataset)}")

    if len(val_dataset) == 0:
        raise RuntimeError(
            "Validation dataset is empty."
        )
    model = build_model(args.model).to(device)
    print(f"Model: {args.model.upper()}")

    params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Trainable parameters: {params:,}")

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    scaler = torch.amp.GradScaler(
    enabled=device.type == "cuda",
    )

    os.makedirs("weights", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    history_file = Path("results") / f"{args.model}_training_log.csv"
    best_acc = 0.0
    best_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    with open(history_file, "w", newline="") as csvfile:

        writer = csv.writer(csvfile)

        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "precision",
            "recall",
            "f1_score",
        ])

        for epoch in range(args.epochs):

            model.train()

            running_loss = 0.0
            train_correct = 0
            train_total = 0
            progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{args.epochs}",
            )

            for images, labels in progress:

                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=(device.type == "cuda"),
                ):

                    outputs = model(images)

                    predictions = (
                        torch.sigmoid(outputs.squeeze(1)) > 0.5
                    ).long()

                    train_correct += (
                        predictions == labels.long()
                    ).sum().item()

                    train_total += labels.size(0)

                    loss = criterion(
                        outputs.squeeze(1),
                        labels,
                    )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += (
                    loss.item() * images.size(0)
                )

                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

            train_loss = (
                running_loss / len(train_dataset)
            )

            train_accuracy = train_correct / train_total

            (
                val_loss,
                val_acc,
                precision,
                recall,
                f1,
            ) = evaluate(
                model,
                val_loader,
                criterion,
                device,
            )

            scheduler.step()

            writer.writerow([
                epoch + 1,
                train_loss,
                train_accuracy,
                val_loss,
                val_acc,
                precision,
                recall,
                f1,
            ])
            csvfile.flush()

            print(
                f"\nEpoch {epoch+1}/{args.epochs}"
            )

            print(
                f"Train Loss : {train_loss:.4f}"
            )

            print(
                f"Train Acc  : {100*train_accuracy:.2f}%"
            )

            print(
                f"Val Loss   : {val_loss:.4f}"
            )

            print(
                f"Val Acc    : {100*val_acc:.2f}%"
            )
            
            print(
                f"LR         : {optimizer.param_groups[0]['lr']:.2e}"
            )

            print(f"Precision : {100*precision:.2f}%")
            print(f"Recall    : {100*recall:.2f}%")
            print(f"F1 Score  : {100*f1:.2f}%")

            if val_loss < best_loss:
                best_acc = val_acc
                best_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0

                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_val_loss": best_loss,
                        "best_val_accuracy": best_acc,
                    },
                    f"weights/{args.model}_best.pth",
                )

                print(
                    f"✓ Saved weights/{args.model}_best.pth "
                    f"(Val Loss={val_loss:.4f}, "
                    f"Val Acc={100*val_acc:.2f}%)"
                )

            else:

                patience_counter += 1

            if patience_counter >= args.patience:

                print("\nEarly stopping.")

                break

    print(f"\nBest validation loss : {best_loss:.4f}")
    print(f"Best validation acc  : {100*best_acc:.2f}%")
    print(f"Best epoch           : {best_epoch}")


if __name__ == "__main__":

    main()