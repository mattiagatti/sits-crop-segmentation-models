import random
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.download_datasets import ensure_kaggle_dataset


LABEL_FILENAME = "y.tif"


class MunichDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        seqlength=32,
        tileids=None,
        transform=None,
        sampling="random",  # "random" or "equal"
    ):
        root_dir = ensure_kaggle_dataset(
            "artelabsuper/sentinel2-munich480",
            Path(root_dir),
        )

        if not (root_dir / "tileids").exists() and (root_dir / "munich480").is_dir():
            nested_root = root_dir / "munich480"
            if (nested_root / "tileids").exists():
                root_dir = nested_root

        self.sampling = sampling
        self.root_dir = Path(root_dir)
        self.name = self.root_dir.name
        self.data_dirs = [
            data_dir
            for data_dir in self.root_dir.iterdir()
            if data_dir.is_dir() and data_dir.name.startswith("data")
        ]
        self.seqlength = seqlength
        self.transform = transform
        self.munich_format = None
        self.src_labels = None
        self.dst_labels = None
        self.unique_labels = np.array([], dtype=float)

        self.samples = []
        self.ndates = []

        stats = {
            "rejected_nopath": 0,
            "rejected_length": 0,
            "total_samples": 0,
        }

        dirs = []
        if tileids is None:
            for data_dir in self.data_dirs:
                dirs.extend([path for path in data_dir.iterdir()])
        else:
            tileids_path = self.root_dir / tileids
            with tileids_path.open("r") as f:
                files = [line.strip() for line in f]

            for data_dir in self.data_dirs:
                dirs.extend([data_dir / file_name for file_name in files])

        self.classids, self.classes = read_classes(self.root_dir / "classes.txt")

        for path in tqdm(dirs, desc="Scanning Munich samples"):
            if not path.exists():
                stats["rejected_nopath"] += 1
                continue

            if not (path / LABEL_FILENAME).exists():
                stats["rejected_nopath"] += 1
                continue

            ndates = len(get_dates(path))

            if ndates < self.seqlength:
                stats["rejected_length"] += 1
                continue

            stats["total_samples"] += 1
            self.samples.append(path)
            self.ndates.append(ndates)

        print(stats)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]

        label, profile = read(path / LABEL_FILENAME)
        profile["name"] = str(path)

        dates = get_dates(path, n=self.seqlength, sampling=self.sampling)
        doys = torch.tensor(
            [date_to_doy(date_str) for date_str in dates],
            dtype=torch.long,
        )

        x10 = []
        x20 = []
        x60 = []

        for date in dates:
            if self.munich_format is None:
                self.munich_format = (path / f"{date}_10m.tif").exists()

            if self.munich_format:
                x10.append(read(path / f"{date}_10m.tif")[0])
                x20.append(read(path / f"{date}_20m.tif")[0])
                x60.append(read(path / f"{date}_60m.tif")[0])
            else:
                x10.append(read(path / f"{date}.tif")[0])

        x10 = np.array(x10) * 1e-4

        if self.munich_format:
            x20 = np.array(x20) * 1e-4
            x60 = np.array(x60) * 1e-4

        label = label[0]
        self.unique_labels = np.unique(
            np.concatenate([label.flatten(), self.unique_labels])
        )

        new = np.zeros(label.shape, dtype=int)
        for class_id, index in zip(self.classids, range(len(self.classids))):
            new[label == class_id] = index
        label = new

        label = torch.from_numpy(label)
        x10 = torch.from_numpy(x10)

        if self.munich_format:
            x20 = torch.from_numpy(x20)
            x60 = torch.from_numpy(x60)

            x20 = F.interpolate(x20, size=x10.shape[2:4])
            x60 = F.interpolate(x60, size=x10.shape[2:4])

            x = torch.cat((x10, x20, x60), 1)
        else:
            x = x10

        x = x.permute(1, 0, 2, 3).float()

        if self.transform is not None:
            label, x = self.transform(label, x)

        label = label.long()
        filename = str(Path(path.parent.name) / path.name)

        return x, label, doys, filename


def get_dates(path, n=None, sampling="random"):
    path = Path(path)

    dates = []
    for file_path in path.iterdir():
        prefix = file_path.name.split("_")[0]
        if len(prefix) == 8:
            dates.append(prefix)

    dates = sorted(set(dates))

    if n is None or len(dates) <= n:
        return dates

    if sampling == "random":
        dates = random.sample(dates, n)
        dates.sort()
        return dates

    if sampling == "equal":
        indices = np.linspace(0, len(dates) - 1, n)
        indices = np.round(indices).astype(int)
        indices = np.unique(indices)

        # safety in case rounding creates duplicates
        if len(indices) < n:
            all_indices = np.arange(len(dates))
            missing = [i for i in all_indices if i not in set(indices)]
            needed = n - len(indices)
            indices = np.concatenate([indices, missing[:needed]])
            indices = np.sort(indices)

        return [dates[i] for i in indices]

    raise ValueError(f"Unsupported sampling mode: {sampling}")


def date_to_doy(date_str):
    dt = datetime.strptime(date_str, "%Y%m%d")
    return dt.timetuple().tm_yday


def read_classes(csv_path):
    csv_path = Path(csv_path)

    with csv_path.open("r") as f:
        classes = f.readlines()

    ids = []
    names = []
    for row in classes:
        row = row.strip()
        if "|" in row:
            class_id, class_name = row.split("|")
            ids.append(int(class_id))
            names.append(class_name)

    return ids, names


def read(file_path):
    file_path = Path(file_path)
    with rasterio.open(file_path) as src:
        return src.read(), src.profile