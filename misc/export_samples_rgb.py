import argparse
import itertools
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image
from tqdm import tqdm

from utils.results_io import apply_cmap, LOMBARDIA_COLORS, MUNICH_COLORS


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["lombardia", "munich"],
    )

    parser.add_argument(
        "--lombardia_root",
        type=Path,
        default=Path("/home/jovyan/shared/mgatti/datasets/sentinel2-crop-mapping"),
    )
    parser.add_argument(
        "--munich_root",
        type=Path,
        default=Path("/home/jovyan/shared/mgatti/datasets/sentinel2-munich480/munich480/munich480"),
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("exports/rgb_pngs"),
    )

    parser.add_argument(
        "--num_patches",
        type=int,
        default=3,
        help="Number of patches to select maximizing GT class coverage",
    )

    parser.add_argument(
        "--gt_classes",
        type=int,
        nargs="+",
        default=None,
        help="Optional: restrict coverage optimization to these GT classes only",
    )

    parser.add_argument(
        "--top_candidates",
        type=int,
        default=2000,
        help="Restrict exact search to top-N candidate patches by class count",
    )

    parser.add_argument(
        "--require_classes",
        type=int,
        nargs="+",
        default=None,
        help="Optional: keep only patches containing all these GT classes",
    )

    parser.add_argument(
        "--run_selection",
        action="store_true",
        help=(
            "Run the exact patch selection step. "
            "If not set, all matching patches are exported/listed directly."
        ),
    )

    parser.add_argument(
        "--export_limit",
        type=int,
        default=None,
        help=(
            "Optional limit on how many patches to export when --run_selection is not set. "
            "Patches are sorted lexicographically."
        ),
    )

    return parser.parse_args()


def get_patch_dirs(args):
    if args.dataset == "lombardia":
        root = args.lombardia_root
        dataset_roots = ["lombardia", "lombardia2", "lombardia3"]

        patch_dirs = []
        for ds_name in dataset_roots:
            ds_root = root / ds_name
            if not ds_root.exists():
                continue

            for year_dir in ds_root.glob("data*"):
                if not year_dir.is_dir():
                    continue

                for patch_dir in year_dir.iterdir():
                    if patch_dir.is_dir() and patch_dir.name.isdigit():
                        patch_dirs.append(patch_dir)

        return patch_dirs, root

    if args.dataset == "munich":
        root = args.munich_root

        patch_dirs = []
        for year_dir in root.glob("data*"):
            if not year_dir.is_dir():
                continue

            for patch_dir in year_dir.iterdir():
                if patch_dir.is_dir() and patch_dir.name.isdigit():
                    patch_dirs.append(patch_dir)

        return patch_dirs, root

    raise ValueError(f"Unsupported dataset: {args.dataset}")


def get_temporal_tif_files(patch_dir: Path, dataset: str):
    if dataset == "munich":
        return sorted(patch_dir.glob("*_10m.tif"))

    if dataset == "lombardia":
        tif_files = []
        for file_path in patch_dir.iterdir():
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() != ".tif":
                continue
            if file_path.name == "y.tif":
                continue
            if len(file_path.stem) == 8 and file_path.stem.isdigit():
                tif_files.append(file_path)

        return sorted(tif_files)

    raise ValueError(f"Unsupported dataset: {dataset}")


def has_enough_time_steps(patch_dir: Path, dataset: str, min_steps: int = 32) -> bool:
    tif_files = get_temporal_tif_files(patch_dir, dataset)
    return len(tif_files) >= min_steps


def get_valid_class_ids(dataset: str) -> set[int]:
    if dataset == "munich":
        return {cid for cid, _, _ in MUNICH_COLORS}
    if dataset == "lombardia":
        return {cid for cid, _, _ in LOMBARDIA_COLORS}
    raise ValueError(f"Unsupported dataset: {dataset}")


def get_gt_classes(patch_dir: Path, dataset: str):
    y_path = patch_dir / "y.tif"
    if not y_path.exists():
        return None

    with rasterio.open(y_path) as src:
        y = src.read(1)

    classes = set(np.unique(y).tolist())
    classes.discard(0)
    classes = classes.intersection(get_valid_class_ids(dataset))
    return classes


def to_uint8(rgb):
    rgb = np.clip(rgb, 0, 10000) / 10000.0
    rgb = np.power(rgb, 0.5)
    rgb = rgb * 255.0
    return rgb.astype(np.uint8)


def convert_tif(tif_path, png_path, dataset):
    with rasterio.open(tif_path) as src:
        data = src.read()

    if tif_path.name == "y.tif":
        cmap_name = "munich" if dataset == "munich" else "lombardia"
        rgb = apply_cmap(data[0], cmap_name)
    else:
        if data.shape[0] < 3:
            raise ValueError(
                f"Expected at least 3 bands for RGB conversion, got {data.shape} for {tif_path}"
            )

        rgb = data[[2, 1, 0]]
        rgb = np.transpose(rgb, (1, 2, 0))
        rgb = to_uint8(rgb)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(png_path)


def save_legend_pdf(dataset, out_path):
    colors = MUNICH_COLORS if dataset == "munich" else LOMBARDIA_COLORS
    colors = sorted(colors, key=lambda x: x[0])

    n_items = len(colors)
    n_cols = n_items

    cell_w = 1.2
    cell_h = 2.2
    square_size = 0.9

    fig_w = n_cols * cell_w
    fig_h = cell_h

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    for idx, (_, class_name, rgb) in enumerate(colors):
        x_center = idx * cell_w + cell_w / 2
        y_top = fig_h - 0.4

        square = plt.Rectangle(
            (x_center - square_size / 2, y_top - square_size),
            square_size,
            square_size,
            facecolor=tuple(c / 255.0 for c in rgb),
            edgecolor="black",
            linewidth=0.8,
        )
        ax.add_patch(square)

        ax.text(
            x_center,
            y_top - square_size - 0.15,
            class_name,
            ha="center",
            va="top",
            fontsize=9,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def get_selected_time_files(patch_dir: Path, dataset: str):
    tif_files = get_temporal_tif_files(patch_dir, dataset)

    wanted_indices = [0, 7, 15, 31]
    wanted_names = ["t1", "t8", "t16", "t32"]

    return [(tif_files[i], wanted_names[j]) for j, i in enumerate(wanted_indices)]


def build_patch_class_map(
    valid_patch_dirs,
    dataset: str,
    target_classes=None,
    required_classes=None,
):
    patch_class_map = {}

    for patch_dir in tqdm(valid_patch_dirs, desc="Reading GT classes"):
        gt_classes = get_gt_classes(patch_dir, dataset)
        if gt_classes is None:
            continue

        if required_classes is not None and not required_classes.issubset(gt_classes):
            continue

        stored_classes = gt_classes
        if target_classes is not None:
            stored_classes = gt_classes.intersection(target_classes)

        if len(stored_classes) > 0:
            patch_class_map[patch_dir] = stored_classes

    return patch_class_map


def select_top_candidates(patch_class_map, top_candidates: int):
    items = sorted(
        patch_class_map.items(),
        key=lambda x: (-len(x[1]), str(x[0])),
    )
    return items[:top_candidates]


def exact_select_patches(patch_class_map, num_patches: int, top_candidates: int):
    if num_patches != 3:
        raise ValueError("This exact search version currently supports only num_patches=3")

    candidate_items = select_top_candidates(patch_class_map, top_candidates=top_candidates)

    if len(candidate_items) < num_patches:
        selected = [p for p, _ in candidate_items]
        covered = set()
        for _, classes in candidate_items:
            covered.update(classes)
        return selected, covered

    best_combo = None
    best_union = set()
    best_key = None

    combos = itertools.combinations(candidate_items, num_patches)
    total_combos = len(candidate_items) * (len(candidate_items) - 1) * (len(candidate_items) - 2) // 6

    for combo in tqdm(combos, total=total_combos, desc="Searching best patch triple"):
        paths = [item[0] for item in combo]
        class_sets = [item[1] for item in combo]

        union_classes = set().union(*class_sets)
        sorted_paths = tuple(sorted(str(p) for p in paths))

        comparable_key = (
            len(union_classes),
            sum(len(s) for s in class_sets),
            tuple(sorted([-len(s) for s in class_sets])),
        )

        if best_combo is None:
            best_combo = combo
            best_union = union_classes
            best_key = (comparable_key, sorted_paths)
            continue

        current_key = (comparable_key, sorted_paths)

        if comparable_key > best_key[0]:
            best_combo = combo
            best_union = union_classes
            best_key = current_key
        elif comparable_key == best_key[0] and sorted_paths < best_key[1]:
            best_combo = combo
            best_union = union_classes
            best_key = current_key

    selected = [item[0] for item in best_combo]
    selected = sorted(selected, key=lambda p: str(p))
    return selected, best_union


def class_id_to_name_map(dataset):
    if dataset == "munich":
        return {cid: name for cid, name, _ in MUNICH_COLORS}
    if dataset == "lombardia":
        return {cid: name for cid, name, _ in LOMBARDIA_COLORS}
    raise ValueError(f"Unsupported dataset: {dataset}")


def export_patches(selected, root, dataset, output_dir):
    dataset_out_dir = output_dir / f"{dataset}_best_{len(selected)}"
    if dataset_out_dir.exists():
        shutil.rmtree(dataset_out_dir)

    dataset_out_dir.mkdir(parents=True, exist_ok=True)
    save_legend_pdf(dataset, dataset_out_dir / "legend.pdf")

    for patch_dir in tqdm(selected, desc="Exporting selected patches"):
        rel = patch_dir.relative_to(root)
        out_dir = dataset_out_dir / rel

        for tif_path, out_name in get_selected_time_files(patch_dir, dataset):
            png_path = out_dir / f"{out_name}.png"
            convert_tif(tif_path, png_path, dataset)

        y_path = patch_dir / "y.tif"
        if y_path.exists():
            convert_tif(y_path, out_dir / "y.png", dataset)

    return dataset_out_dir


def main():
    args = parse_args()

    if args.run_selection and args.num_patches != 3:
        raise ValueError("This script currently supports only --num_patches 3 when --run_selection is used")

    patch_dirs, root = get_patch_dirs(args)
    if not patch_dirs:
        raise RuntimeError("No patch folders found")

    valid_patch_dirs = [
        p for p in patch_dirs
        if has_enough_time_steps(p, args.dataset, min_steps=32)
    ]
    skipped_time = len(patch_dirs) - len(valid_patch_dirs)

    if not valid_patch_dirs:
        raise RuntimeError("No valid patch folders found with at least 32 temporal files")

    target_classes = set(args.gt_classes) if args.gt_classes is not None else None
    if target_classes is not None:
        target_classes.discard(0)
        target_classes = target_classes.intersection(get_valid_class_ids(args.dataset))

    required_classes = set(args.require_classes) if args.require_classes is not None else None
    if required_classes is not None:
        required_classes.discard(0)
        required_classes = required_classes.intersection(get_valid_class_ids(args.dataset))

    patch_class_map = build_patch_class_map(
        valid_patch_dirs,
        dataset=args.dataset,
        target_classes=target_classes,
        required_classes=required_classes,
    )
    if not patch_class_map:
        raise RuntimeError("No valid patch folders found after GT processing")

    id_to_name = class_id_to_name_map(args.dataset)

    print(f"Dataset: {args.dataset}")
    print(f"Total patches: {len(patch_dirs)}")
    print(f"Valid patches (>=32 timestamps): {len(valid_patch_dirs)}")
    print(f"Skipped patches (<32 timestamps): {skipped_time}")
    print(f"Patches with GT classes considered: {len(patch_class_map)}")

    if required_classes is not None:
        print(f"Required GT classes in each patch: {sorted(required_classes)}")
        print(
            "Required GT class names:",
            [id_to_name.get(c, f"UNKNOWN_{c}") for c in sorted(required_classes)],
        )

    if target_classes is not None:
        print(f"Restricted GT classes for scoring/storage: {sorted(target_classes)}")
        print(
            "Restricted GT class names:",
            [id_to_name.get(c, f"UNKNOWN_{c}") for c in sorted(target_classes)],
        )

    if args.run_selection:
        selected, covered = exact_select_patches(
            patch_class_map=patch_class_map,
            num_patches=args.num_patches,
            top_candidates=args.top_candidates,
        )

        if not selected:
            raise RuntimeError("No patches selected")

        print("Mode: exact 3-patch selection")
        print(f"Requested number of patches: {args.num_patches}")
        print(f"Top candidates searched exactly: {min(args.top_candidates, len(patch_class_map))}")
        print(f"Selected: {len(selected)}")
        print(f"Covered classes: {sorted(covered)}")
        print(
            "Covered class names:",
            [id_to_name.get(c, f"UNKNOWN_{c}") for c in sorted(covered)],
        )

        print("\nSelected patches:")
        running_covered = set()
        for i, patch_dir in enumerate(selected, start=1):
            classes = patch_class_map[patch_dir]
            new_classes = classes - running_covered
            running_covered.update(classes)

            rel = patch_dir.relative_to(root)
            print(f"{i}. {rel}")
            print(f"   classes: {sorted(classes)}")
            print(
                f"   class names: {[id_to_name.get(c, f'UNKNOWN_{c}') for c in sorted(classes)]}"
            )
            print(f"   adds new: {sorted(new_classes)}")
            print(
                f"   adds names: {[id_to_name.get(c, f'UNKNOWN_{c}') for c in sorted(new_classes)]}"
            )

    else:
        selected = sorted(patch_class_map.keys(), key=lambda p: str(p))
        if args.export_limit is not None:
            selected = selected[:args.export_limit]

        covered = set()
        for patch_dir in selected:
            covered.update(patch_class_map[patch_dir])

        print("Mode: direct export/list of all matching patches")
        if args.export_limit is not None:
            print(f"Export limit: {args.export_limit}")
        print(f"Selected: {len(selected)}")
        print(f"Covered classes across exported patches: {sorted(covered)}")
        print(
            "Covered class names:",
            [id_to_name.get(c, f"UNKNOWN_{c}") for c in sorted(covered)],
        )

        print("\nMatching patches:")
        for i, patch_dir in enumerate(selected, start=1):
            classes = patch_class_map[patch_dir]
            rel = patch_dir.relative_to(root)
            print(f"{i}. {rel}")
            print(f"   classes: {sorted(classes)}")
            print(
                f"   class names: {[id_to_name.get(c, f'UNKNOWN_{c}') for c in sorted(classes)]}"
            )

    dataset_out_dir = export_patches(
        selected=selected,
        root=root,
        dataset=args.dataset,
        output_dir=args.output_dir,
    )
    print(f"\nExported to: {dataset_out_dir}")


if __name__ == "__main__":
    main()