import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils.custom_transform as T
from dataset.lombardia_dataset import LombardiaDataset
from dataset.munich_dataset import MunichDataset
from zoo.deeplabv3_3d import DeepLabV3_3D
from zoo.fpn_3d import FPN_3D
from zoo.swin_unetr import SwinUNETRTemporal
from zoo.tsvit import TSViT, TSViT_lookup
from zoo.unet_3d import UNet_3D
from zoo.vistaformer import VistaFormer


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--ckpt_path", type=Path, default=None, help="checkpoint path")
    parser.add_argument(
        "--dataset",
        type=str,
        default="munich",
        choices=["lombardia", "munich"],
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="swin_unetr",
        choices=["deeplabv3", "fpn", "swin_unetr", "tsvit", "tsvit_lookup", "unet", "vistaformer"],
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("/home/jovyan/shared/mgatti/datasets"),
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--use_class_weights",
        action="store_true",
        help="Use class weights in CrossEntropyLoss.",
    )
    parser.add_argument(
        "--class_weight_mode",
        type=str,
        default="inverse_sqrt",
        choices=[
            "inverse",
            "inverse_sqrt",
            "median_frequency",
            "effective_num",
        ],
        help="Strategy used to compute class weights from training pixel counts.",
    )
    parser.add_argument(
        "--ignore_index",
        type=int,
        default=0,
        help="Class index ignored in the loss and excluded from weight computation.",
    )
    parser.add_argument(
        "--class_weight_max",
        type=float,
        default=10.0,
        help="Optional max clipping for class weights. Set <= 0 to disable clipping.",
    )
    parser.add_argument(
        "--effective_num_beta",
        type=float,
        default=0.9999,
        help="Beta parameter for effective number weighting.",
    )
    parser.add_argument(
        "--class_weight_cache_dir",
        type=Path,
        default=Path("cache") / "class_weights",
        help="Directory where class-count/weight cache files are stored.",
    )
    parser.add_argument(
        "--recompute_class_weights",
        action="store_true",
        help="Force recomputation of class weights even if cache exists.",
    )

    return parser.parse_args()


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_train_dates_from_dataset(dataset):
    train_dates = set()

    for _, _, doys, _ in tqdm(dataset, desc="Collect train DOYs", leave=False):
        if torch.is_tensor(doys):
            doys = doys.cpu().numpy()

        train_dates.update(np.asarray(doys).reshape(-1).astype(int).tolist())

    train_dates = sorted(d for d in train_dates if 1 <= d <= 366)

    if len(train_dates) == 0:
        raise RuntimeError("No valid DOY values found in training dataset.")

    return train_dates


def build_model(
    arch,
    depth,
    in_channels,
    out_classes,
    T_max,
    class_weights=None,
    ignore_index=0,
    train_dates=None,
):
    if arch == "deeplabv3":
        model = DeepLabV3_3D(
            depth=depth,
            in_channels=in_channels,
            out_classes=out_classes,
        )
    elif arch == "fpn":
        model = FPN_3D(
            depth=depth,
            in_channels=in_channels,
            out_classes=out_classes,
        )
    elif arch == "swin_unetr":
        model = SwinUNETRTemporal(
            depth=depth,
            in_channels=in_channels,
            out_classes=out_classes,
        )
    elif arch == "tsvit":
        model = TSViT(
            max_seq_len=depth,
            num_classes=out_classes,
            num_channels=in_channels,
        )
    elif arch == "tsvit_lookup":
        if train_dates is None:
            raise ValueError("train_dates must be provided for TSViT_lookup.")

        model = TSViT_lookup(
            train_dates=train_dates,
            max_seq_len=depth,
            num_classes=out_classes,
            num_channels=in_channels,
        )
    elif arch == "unet":
        model = UNet_3D(
            depth=depth,
            in_channels=in_channels,
            out_classes=out_classes,
        )
    elif arch == "vistaformer":
        model = VistaFormer(
            seq_lens=[depth // 2, depth // 4, depth // 8],
            in_channels=in_channels,
            num_classes=out_classes,
        )
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    loss_fn = CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)

    if arch in ["tsvit", "tsvit_lookup"]:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=0.0,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=1e-2,
            weight_decay=1e-3,
            momentum=0.9,
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=T_max,
    )
    return model, loss_fn, optimizer, scheduler


def build_datasets(args):
    transform = T.Compose([T.RandomHorizontalFlip(), T.RandomVerticalFlip()])

    if args.dataset == "lombardia":
        data_dir = args.data_dir / "sentinel2-crop-mapping"

        train_dataset = LombardiaDataset(
            root_dirs=[data_dir / "lombardia", data_dir / "lombardia2"],
            years=["data2016", "data2017", "data2018"],
            classes_path=data_dir / "lombardia-classes" / "classes25pc.txt",
            seqlength=32,
            tileids=Path("tileids") / "train_fold0.tileids",
            transform=transform,
        )

        val_dataset = LombardiaDataset(
            root_dirs=[data_dir / "lombardia", data_dir / "lombardia2"],
            years=["data2016", "data2017", "data2018"],
            classes_path=data_dir / "lombardia-classes" / "classes25pc.txt",
            seqlength=32,
            tileids=Path("tileids") / "test_fold0.tileids",
        )

        classes = train_dataset.classes
        in_channels = 9
        out_classes = 8

    elif args.dataset == "munich":
        data_dir = args.data_dir / "sentinel2-munich480" / "munich480"

        train_dataset = MunichDataset(
            data_dir,
            tileids=Path("tileids") / "train_fold0.tileids",
            seqlength=32,
            transform=transform,
        )

        val_dataset = MunichDataset(
            data_dir,
            tileids=Path("tileids") / "test_fold0.tileids",
            seqlength=32,
        )

        classes = train_dataset.classes
        in_channels = 13
        out_classes = 18

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    if len(classes) < out_classes:
        raise ValueError(
            f"classes has length {len(classes)}, but out_classes={out_classes}. "
            "Expected classes to be indexable by class id."
        )

    return train_dataset, val_dataset, classes, in_channels, out_classes


def build_dataloaders(train_dataset, val_dataset, batch_size):
    cpu_count = os.cpu_count() or 1
    num_workers = min(16, cpu_count)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    return train_loader, val_loader


def save_weights(path, model, train_dates=None):
    path.parent.mkdir(parents=True, exist_ok=True)

    if train_dates is not None:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "train_dates": train_dates,
            },
            path,
        )
    else:
        torch.save(model.state_dict(), path)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_score, args, train_dates=None):
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_score": best_score,
        "args": vars(args),
    }

    if train_dates is not None:
        payload["train_dates"] = train_dates

    torch.save(payload, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_score = checkpoint.get(
        "best_score",
        checkpoint.get("best_val_loss", None),
    )
    return start_epoch, best_score


def evaluate_predictions(preds, targets, labels, ignore_class=0):
    if len(preds) == 0 or len(targets) == 0:
        raise RuntimeError("Cannot evaluate empty predictions or targets.")

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    if ignore_class is not None:
        mask = targets != ignore_class
        preds = preds[mask]
        targets = targets[mask]
        labels = [label for label in labels if label != ignore_class]

    if len(targets) == 0:
        cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
        metrics_v = {
            "labels": np.array(labels),
            "confusion matrix": cm,
            "R": np.zeros(len(labels), dtype=np.float64),
            "P": np.zeros(len(labels), dtype=np.float64),
            "F1": np.zeros(len(labels), dtype=np.float64),
            "IoU": np.zeros(len(labels), dtype=np.float64),
            "Acc": np.zeros(len(labels), dtype=np.float64),
        }
        metrics_scalar = {
            "OA": 0.0,
            "Kappa": 0.0,
            "mIoU": 0.0,
            "wR": 0.0,
            "wP": 0.0,
            "wF1": 0.0,
            "RAcc": 0.0,
        }
        return metrics_v, metrics_scalar

    cm = confusion_matrix(targets, preds, labels=labels)

    metrics_v = {
        "labels": np.array(labels),
        "confusion matrix": cm,
        "R": recall_score(
            targets, preds, labels=labels, average=None, zero_division=0
        ),
        "P": precision_score(
            targets, preds, labels=labels, average=None, zero_division=0
        ),
        "F1": f1_score(
            targets, preds, labels=labels, average=None, zero_division=0
        ),
        "IoU": jaccard_score(
            targets, preds, labels=labels, average=None, zero_division=0
        ),
    }
    metrics_v["Acc"] = metrics_v["R"]

    metrics_scalar = {
        "OA": float((preds == targets).mean()),
        "Kappa": float(cohen_kappa_score(targets, preds, labels=labels)),
        "mIoU": float(
            jaccard_score(
                targets,
                preds,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "wR": float(
            recall_score(
                targets,
                preds,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "wP": float(
            precision_score(
                targets,
                preds,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "wF1": float(
            f1_score(
                targets,
                preds,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
    }

    total = cm.sum()
    if total > 0:
        row_probs = cm.sum(axis=1) / total
        col_probs = cm.sum(axis=0) / total
        metrics_scalar["RAcc"] = float(np.inner(row_probs, col_probs))
    else:
        metrics_scalar["RAcc"] = 0.0

    return metrics_v, metrics_scalar


def save_metrics(exp_dir, filename, metrics_v, metrics_scalar, classes):
    cls_names = np.array(classes)[metrics_v["labels"]]
    out_path = exp_dir / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        f.write("classes:\n" + np.array2string(cls_names) + "\n")
        for k, v in metrics_v.items():
            if k == "labels":
                continue

            f.write(k + "\n")
            if len(v.shape) == 1:
                for ki, vi in zip(cls_names, v):
                    f.write(f"{vi:.2f}\t{ki}\n")
            elif len(v.shape) == 2:
                num_gt = np.sum(v, axis=1)
                f.write(
                    "\n".join(
                        [
                            "".join(["{:10}".format(item) for item in row])
                            + "  "
                            + lab
                            + f"({tot:d})"
                            for row, lab, tot in zip(v, cls_names, num_gt)
                        ]
                    )
                )
                f.write("\n")

        str_metrics = "".join(
            [f"{key}| {value:f} | " for (key, value) in metrics_scalar.items()]
        )
        f.write(f"\n{str_metrics}")


def append_csv(csv_path, row):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def get_dataset_identifier(args):
    return f"{args.dataset}_fold0"


def get_exp_dir(args):
    if not args.use_class_weights:
        run_name = "train"
    else:
        clip_str = "none" if args.class_weight_max <= 0 else str(args.class_weight_max)

        parts = [
            "train_weighted",
            f"mode-{args.class_weight_mode}",
            f"clip-{clip_str}",
        ]

        # include ignore_index only if not default
        if args.ignore_index != 0:
            parts.append(f"ignore-{args.ignore_index}")

        if args.class_weight_mode == "effective_num":
            parts.append(f"beta-{args.effective_num_beta:.4f}")

        run_name = "__".join(parts)

    return Path("exp") / args.arch / args.dataset / run_name


def get_class_weight_cache_path(args, out_classes):
    dataset_id = get_dataset_identifier(args)
    clip_str = "none" if args.class_weight_max <= 0 else str(args.class_weight_max)
    beta_str = f"{args.effective_num_beta:.8f}"

    filename = (
        f"{dataset_id}"
        f"__mode-{args.class_weight_mode}"
        f"__ignore-{args.ignore_index}"
        f"__classes-{out_classes}"
        f"__beta-{beta_str}"
        f"__clip-{clip_str}.pt"
    )
    return args.class_weight_cache_dir / filename


def count_pixels_per_class(dataset, out_classes):
    counts = torch.zeros(out_classes, dtype=torch.float64)

    print("Counting training pixels per class...")
    for _, y, _, _ in tqdm(dataset, desc="Count pixels", leave=False):
        if not torch.is_tensor(y):
            y = torch.as_tensor(y)

        y = y.long().view(-1)
        valid = (y >= 0) & (y < out_classes)
        y = y[valid]

        counts += torch.bincount(y, minlength=out_classes).to(torch.float64)

    return counts


def compute_weights_from_counts(
    counts,
    mode="inverse_sqrt",
    ignore_index=0,
    max_weight=None,
    effective_num_beta=0.9999,
):
    counts = counts.clone().to(torch.float64)
    out_classes = counts.numel()

    weights = torch.zeros(out_classes, dtype=torch.float64)
    valid_mask = counts > 0

    if ignore_index is not None and 0 <= ignore_index < out_classes:
        valid_mask[ignore_index] = False

    if valid_mask.sum() == 0:
        raise RuntimeError("No valid classes found to compute class weights.")

    valid_counts = counts[valid_mask]

    if mode == "inverse":
        weights[valid_mask] = 1.0 / valid_counts

    elif mode == "inverse_sqrt":
        weights[valid_mask] = 1.0 / torch.sqrt(valid_counts)

    elif mode == "median_frequency":
        freqs = valid_counts / valid_counts.sum()
        median_freq = torch.median(freqs)
        weights[valid_mask] = median_freq / freqs

    elif mode == "effective_num":
        beta = float(effective_num_beta)
        if not (0.0 <= beta < 1.0):
            raise ValueError(
                f"effective_num_beta must be in [0, 1), got {effective_num_beta}"
            )

        effective_num = 1.0 - torch.pow(torch.tensor(beta, dtype=torch.float64), valid_counts)
        weights[valid_mask] = (1.0 - beta) / torch.clamp(effective_num, min=1e-12)

    else:
        raise ValueError(f"Unsupported class weight mode: {mode}")

    weights[valid_mask] = weights[valid_mask] / weights[valid_mask].mean()

    if max_weight is not None and max_weight > 0:
        weights = torch.clamp(weights, max=max_weight)
        weights[valid_mask] = weights[valid_mask] / weights[valid_mask].mean()

    if ignore_index is not None and 0 <= ignore_index < out_classes:
        weights[ignore_index] = 0.0

    return weights.to(torch.float32)


def load_or_compute_class_weights(args, train_dataset, out_classes):
    cache_path = get_class_weight_cache_path(args, out_classes)

    if cache_path.exists() and not args.recompute_class_weights:
        print(f"Loading cached class weights from: {cache_path}")
        payload = torch.load(cache_path, map_location="cpu")
        counts = payload["counts"].to(torch.float64)
        weights = payload["weights"].to(torch.float32)
        return counts, weights, cache_path

    counts = count_pixels_per_class(train_dataset, out_classes)
    weights = compute_weights_from_counts(
        counts=counts,
        mode=args.class_weight_mode,
        ignore_index=args.ignore_index,
        max_weight=args.class_weight_max if args.class_weight_max > 0 else None,
        effective_num_beta=args.effective_num_beta,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "counts": counts.cpu(),
            "weights": weights.cpu(),
            "meta": {
                "dataset": args.dataset,
                "mode": args.class_weight_mode,
                "ignore_index": args.ignore_index,
                "out_classes": out_classes,
                "class_weight_max": args.class_weight_max,
                "effective_num_beta": args.effective_num_beta,
            },
        },
        cache_path,
    )
    print(f"Saved class weights cache to: {cache_path}")

    return counts, weights, cache_path


def save_class_weight_report(exp_dir, counts, weights, classes, ignore_index, cache_path):
    out_path = exp_dir / "class_weights.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        f.write(f"cache_path: {cache_path}\n")
        f.write(f"ignore_index: {ignore_index}\n\n")
        f.write("class_id\tclass_name\tpixel_count\tweight\n")

        for class_id, (count, weight) in enumerate(zip(counts.tolist(), weights.tolist())):
            class_name = classes[class_id] if class_id < len(classes) else str(class_id)
            f.write(f"{class_id}\t{class_name}\t{int(count)}\t{weight:.8f}\n")


def train_one_epoch(model, loader, optimizer, loss_fn, device, arch):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc="Train", leave=False)
    for x, y, doys, _ in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if arch in ["tsvit", "tsvit_lookup"]:
            doys = doys.to(device, non_blocking=True)

            if arch == "tsvit_lookup":
                y_hat = model(x, doys, inference=False)
            else:
                y_hat = model(x, doys)
        else:
            y_hat = model(x)

        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate_one_epoch(model, loader, loss_fn, out_classes, device, arch, ignore_index=0):
    model.eval()
    total_loss = 0.0

    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc="Validate", leave=False)
    for x, y, doys, _ in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if arch in ["tsvit", "tsvit_lookup"]:
            doys = doys.to(device, non_blocking=True)

            if arch == "tsvit_lookup":
                y_hat = model(x, doys, inference=True)
            else:
                y_hat = model(x, doys)
        else:
            y_hat = model(x)

        loss = loss_fn(y_hat, y)
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        y_pred = torch.argmax(y_hat, dim=1)
        all_preds.append(y_pred.cpu().numpy().reshape(-1))
        all_targets.append(y.cpu().numpy().reshape(-1))

    avg_loss = total_loss / max(len(loader), 1)

    if len(all_preds) == 0:
        raise RuntimeError(
            "Validation loader is empty. Check batch_size and dataset size."
        )

    metrics_v, metrics_scalar = evaluate_predictions(
        all_preds,
        all_targets,
        list(range(out_classes)),
        ignore_class=ignore_index,
    )

    return avg_loss, metrics_v, metrics_scalar


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = get_device()
    exp_dir = get_exp_dir(args)
    ckpt_dir = exp_dir / "checkpoints"
    metrics_csv = exp_dir / "metrics.csv"

    train_dataset, val_dataset, classes, in_channels, out_classes = build_datasets(args)
    train_loader, val_loader = build_dataloaders(
        train_dataset,
        val_dataset,
        args.batch_size,
    )

    class_weights = None
    class_counts = None
    class_weight_cache_path = None

    if args.use_class_weights:
        class_counts, class_weights, class_weight_cache_path = load_or_compute_class_weights(
            args=args,
            train_dataset=train_dataset,
            out_classes=out_classes,
        )
        class_weights = class_weights.to(device)

        print("Using class weights in CrossEntropyLoss")
        print("Pixel counts per class:", [int(v) for v in class_counts.tolist()])
        print("Class weights:", [float(v) for v in class_weights.cpu().tolist()])

        save_class_weight_report(
            exp_dir=exp_dir,
            counts=class_counts.cpu(),
            weights=class_weights.cpu(),
            classes=classes,
            ignore_index=args.ignore_index,
            cache_path=class_weight_cache_path,
        )
    else:
        print("Using unweighted CrossEntropyLoss")

    train_dates = None
    if args.arch == "tsvit_lookup":
        train_dates = get_train_dates_from_dataset(train_dataset)
        print(f"Using TSViT_lookup with {len(train_dates)} train dates.")
        print("Train dates:", train_dates)

    model, loss_fn, optimizer, scheduler = build_model(
        arch=args.arch,
        depth=32,
        in_channels=in_channels,
        out_classes=out_classes,
        class_weights=class_weights,
        ignore_index=args.ignore_index,
        T_max=args.epochs,
        train_dates=train_dates,
    )
    model = model.to(device)

    start_epoch = 0
    best_kappa = -float("inf")
    best_val_loss = float("inf")

    if args.ckpt_path is not None and args.ckpt_path.exists():
        start_epoch, best_score = load_checkpoint(
            args.ckpt_path,
            model,
            optimizer,
            scheduler,
            device,
        )
        if best_score is not None:
            best_kappa = best_score
        print(f"Resumed from {args.ckpt_path}")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            args.arch,
        )

        val_loss, metrics_v, metrics_scalar = validate_one_epoch(
            model,
            val_loader,
            loss_fn,
            out_classes,
            device,
            args.arch,
            ignore_index=args.ignore_index,
        )

        scheduler.step()

        val_acc = metrics_scalar["OA"]
        val_kappa = metrics_scalar["Kappa"]

        is_best_kappa = val_kappa > best_kappa
        if is_best_kappa:
            best_kappa = val_kappa
        
        is_best_loss = val_loss < best_val_loss
        if is_best_loss:
            best_val_loss = val_loss

        tqdm.write(f"train_loss: {train_loss:.6f}")
        tqdm.write(f"val_loss:   {val_loss:.6f} {'**new best loss**' if is_best_loss else ''}")
        tqdm.write(f"val_acc:    {val_acc:.6f}")
        tqdm.write(f"val_kappa:  {val_kappa:.6f} {'**new best kappa**' if is_best_kappa else ''}")

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_kappa": val_kappa,
            "lr": optimizer.param_groups[0]["lr"],
            "use_class_weights": args.use_class_weights,
            "class_weight_mode": args.class_weight_mode if args.use_class_weights else "none",
        }

        if args.use_class_weights:
            row["class_weight_cache"] = str(class_weight_cache_path)

        append_csv(metrics_csv, row)

        save_checkpoint(
            ckpt_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            best_kappa,
            args,
            train_dates
        )

        save_weights(exp_dir / "weights" / "last.pt", model, train_dates=train_dates)

        if is_best_kappa:
            save_checkpoint(
                ckpt_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_kappa,
                args,
                train_dates
            )
            save_weights(exp_dir / "weights" / "best.pt", model, train_dates=train_dates)

            save_metrics(
                exp_dir,
                "best_result.txt",
                metrics_v,
                metrics_scalar,
                classes,
            )
        
            if args.use_class_weights:
                best_weight_info_path = exp_dir / "best_class_weights.json"
                best_weight_info_path.parent.mkdir(parents=True, exist_ok=True)
                with open(best_weight_info_path, "w") as f:
                    json.dump(
                        {
                            "cache_path": str(class_weight_cache_path),
                            "mode": args.class_weight_mode,
                            "ignore_index": args.ignore_index,
                            "counts": [int(v) for v in class_counts.tolist()],
                            "weights": [float(v) for v in class_weights.cpu().tolist()],
                            "best_metric": "Kappa",
                            "best_kappa": float(best_kappa),
                        },
                        f,
                        indent=2,
                    )

        if is_best_loss:
            save_checkpoint(
                ckpt_dir / "best_loss.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                args,
                train_dates
            )
            save_weights(exp_dir / "weights" / "best_loss.pt", model, train_dates=train_dates)

            save_metrics(
                exp_dir,
                "best_loss_result.txt",
                metrics_v,
                metrics_scalar,
                classes,
            )


if __name__ == "__main__":
    main()