import argparse
import random
from pathlib import Path

import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    parser.add_argument("--weights_path", type=Path, default=None, help="weights path")
    parser.add_argument("--dataset", type=str, default="munich", choices=["lombardia", "munich"])
    parser.add_argument("--test_id", type=str, default="A", choices=["A", "Y"])
    parser.add_argument("--arch", type=str, default="swin_unetr", choices=["deeplabv3", "fpn", "swin_unetr", "tsvit", "tsvit_lookup", "unet", "vistaformer"])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--data_dir", type=Path, default=Path("/home/jovyan/shared/mgatti/datasets"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(arch, depth, in_channels, out_classes, train_dates=None):
    if arch == "deeplabv3":
        return DeepLabV3_3D(depth=depth, in_channels=in_channels, out_classes=out_classes)
    if arch == "fpn":
        return FPN_3D(depth=depth, in_channels=in_channels, out_classes=out_classes)
    if arch == "swin_unetr":
        return SwinUNETRTemporal(depth=depth, in_channels=in_channels, out_classes=out_classes)
    if arch == "tsvit":
        return TSViT(num_classes=out_classes, max_seq_len=depth, num_channels=in_channels)
    if arch == "tsvit_lookup":
        if train_dates is None:
            raise ValueError("train_dates must be provided for TSViT_lookup.")

        return TSViT_lookup(
            train_dates=train_dates,
            num_classes=out_classes,
            max_seq_len=depth,
            num_channels=in_channels,
        )
    if arch == "unet":
        return UNet_3D(depth=depth, in_channels=in_channels, out_classes=out_classes)
    if arch == "vistaformer":
        return VistaFormer(
            seq_lens=[depth // 2, depth // 4, depth // 8],
            in_channels=in_channels,
            num_classes=out_classes,
        )

    raise ValueError(f"Unsupported architecture: {arch}")


def print_model_stats(model, arch, loader, device):
    model.eval()

    x, _, doys, _ = next(iter(loader))
    x = x.to(device)
    doys = doys.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    size_mb = total_params * 4 / (1024 ** 2)

    if arch in ["tsvit", "tsvit_lookup"]:
        if arch == "tsvit_lookup":
            flops = FlopCountAnalysis(model, (x, doys, True)).total()
        else:
            flops = FlopCountAnalysis(model, (x, doys)).total()
    else:
        flops = FlopCountAnalysis(model, x).total()

    print("\nModel stats:")
    print(f"Params: {total_params:,}")
    print(f"Size (MB): {size_mb:.2f}")
    print(f"FLOPs: {flops:,}")
    print(f"FLOPs (GFLOPs): {flops / 1e9:.3f}")


def build_test_dataset(args):
    if args.dataset == "munich":
        data_dir = args.data_dir / "sentinel2-munich480" / "munich480"
        test_dataset = MunichDataset(
            data_dir,
            tileids=Path("tileids") / "eval.tileids",
            seqlength=32,
        )
        classes = test_dataset.classes
        in_channels = 13
        out_classes = 18

    elif args.dataset == "lombardia":
        data_dir = args.data_dir / "sentinel2-crop-mapping"

        if args.test_id == "A":
            test_dataset = LombardiaDataset(
                root_dirs=[data_dir / "lombardia3"],
                years=["data2019"],
                classes_path=data_dir / "lombardia-classes" / "classes25pc.txt",
                seqlength=32,
                tileids=Path("tileids") / "testA.tileids",
            )
        elif args.test_id == "Y":
            test_dataset = LombardiaDataset(
                root_dirs=[data_dir / "lombardia", data_dir / "lombardia2"],
                years=["data2019"],
                classes_path=data_dir / "lombardia-classes" / "classes25pc.txt",
                seqlength=32,
                tileids=Path("tileids") / "testY2019.tileids",
            )
        else:
            raise ValueError(f"Unsupported test_id: {args.test_id}")

        classes = test_dataset.classes
        in_channels = 9
        out_classes = 8

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    return test_dataset, classes, in_channels, out_classes


def build_test_loader(test_dataset, batch_size):
    return DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=0,
    )


def resolve_weights_path(args):
    if args.weights_path is not None:
        return args.weights_path

    return Path("exp") / args.arch / args.dataset / "train" / "weights" / "best.pt"


def load_weights(path, device):
    payload = torch.load(path, map_location=device)

    if not isinstance(payload, dict):
        raise TypeError(f"Expected a dict in {path}, got {type(payload)}")

    if "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        train_dates = payload.get("train_dates", None)
    else:
        state_dict = payload
        train_dates = None

    return state_dict, train_dates


def evaluate_predictions(preds, targets, labels, ignore_class=0):
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    if preds.shape != targets.shape:
        raise ValueError(
            f"Predictions and targets must have the same shape, got {preds.shape} and {targets.shape}"
        )

    if ignore_class is not None:
        mask = targets != ignore_class
        preds = preds[mask]
        targets = targets[mask]
        labels = [label for label in labels if label != ignore_class]

    cm = confusion_matrix(targets, preds, labels=labels)

    metrics_v = {
        "labels": np.array(labels),
        "confusion matrix": cm,
        "R": recall_score(targets, preds, labels=labels, average=None, zero_division=0),
        "P": precision_score(targets, preds, labels=labels, average=None, zero_division=0),
        "F1": f1_score(targets, preds, labels=labels, average=None, zero_division=0),
        "IoU": jaccard_score(targets, preds, labels=labels, average=None, zero_division=0),
    }
    metrics_v["Acc"] = metrics_v["R"]

    metrics_scalar = {
        "OA": float((preds == targets).mean()) if len(targets) > 0 else 0.0,
        "Kappa": float(cohen_kappa_score(targets, preds, labels=labels)) if len(targets) > 0 else 0.0,
        "mIoU": float(jaccard_score(targets, preds, labels=labels, average="macro", zero_division=0)) if len(targets) > 0 else 0.0,
        "wR": float(recall_score(targets, preds, labels=labels, average="weighted", zero_division=0)) if len(targets) > 0 else 0.0,
        "wP": float(precision_score(targets, preds, labels=labels, average="weighted", zero_division=0)) if len(targets) > 0 else 0.0,
        "wF1": float(f1_score(targets, preds, labels=labels, average="weighted", zero_division=0)) if len(targets) > 0 else 0.0,
    }

    total = cm.sum()
    if total > 0:
        row_probs = cm.sum(axis=1) / total
        col_probs = cm.sum(axis=0) / total
        metrics_scalar["RAcc"] = float(np.inner(row_probs, col_probs))
    else:
        metrics_scalar["RAcc"] = 0.0

    return metrics_v, metrics_scalar


def save_metrics(log_dir, filename, metrics_v, metrics_scalar, classes):
    cls_names = np.array(classes)[metrics_v["labels"]]
    out_path = log_dir / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        f.write("classes:\n" + np.array2string(cls_names) + "\n")
        for k, v in metrics_v.items():
            if k == "labels":
                continue

            f.write(k + "\n")
            if len(v.shape) == 1:
                for ki, vi in zip(cls_names, v):
                    f.write("%.2f" % vi + "\t" + ki + "\n")
            elif len(v.shape) == 2:
                num_gt = np.sum(v, axis=1)
                f.write("\n".join(
                    ["".join(["{:10}".format(item) for item in row]) + "  " + lab + "(%d)" % tot
                     for row, lab, tot in zip(v, cls_names, num_gt)]
                ))
                f.write("\n")

        str_metrics = "".join(["%s| %f | " % (key, value) for (key, value) in metrics_scalar.items()])
        f.write(f"\n{str_metrics}")


@torch.no_grad()
def evaluate(model, arch, loader, out_classes, device):
    model.eval()

    all_preds = []
    all_targets = []

    for x, y, doys, _ in tqdm(loader, desc="Testing", leave=False):
        x = x.to(device)
        y = y.to(device)

        if arch in ["tsvit", "tsvit_lookup"]:
            doys = doys.to(device)

            if arch == "tsvit_lookup":
                y_hat = model(x, doys, inference=True)
            else:
                y_hat = model(x, doys)
        else:
            y_hat = model(x)

        y_pred = torch.argmax(y_hat, dim=1)

        if y_pred.shape != y.shape:
            raise ValueError(f"Shape mismatch: y_pred {y_pred.shape} vs y {y.shape}")

        all_preds.append(y_pred.cpu().numpy().reshape(-1))
        all_targets.append(y.cpu().numpy().reshape(-1))

    metrics_v, metrics_scalar = evaluate_predictions(
        preds=all_preds,
        targets=all_targets,
        labels=list(range(out_classes)),
        ignore_class=0,
    )

    return metrics_v, metrics_scalar


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = get_device()

    test_dataset, classes, in_channels, out_classes = build_test_dataset(args)
    test_loader = build_test_loader(test_dataset, args.batch_size)

    weights_path = resolve_weights_path(args)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    state_dict, train_dates = load_weights(weights_path, device)

    model = build_model(
        arch=args.arch,
        depth=32,
        in_channels=in_channels,
        out_classes=out_classes,
        train_dates=train_dates,
    )
    model = model.to(device)
    model.load_state_dict(state_dict)

    print(f"Loaded weights from {weights_path}")
    if train_dates is not None:
        print(f"Loaded {len(train_dates)} train dates from weights.")

    print_model_stats(model, args.arch, test_loader, device)

    metrics_v, metrics_scalar = evaluate(
        model=model,
        arch=args.arch,
        loader=test_loader,
        out_classes=out_classes,
        device=device,
    )

    test_name = "test"
    if args.dataset == "lombardia":
        test_name = f"test_{args.test_id}"

    log_dir = Path("logs") / args.arch / args.dataset / test_name
    save_metrics(
        log_dir=log_dir,
        filename="result.txt",
        metrics_v=metrics_v,
        metrics_scalar=metrics_scalar,
        classes=classes,
    )

    print("Test results")
    for key, value in metrics_scalar.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()