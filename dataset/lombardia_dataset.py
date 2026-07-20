import random
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import torch
from tqdm import tqdm

from utils.download_datasets import ensure_kaggle_dataset


LABEL_FILENAME = "y.tif"


class LombardiaDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dirs,
        years,
        classes_path,
        seqlength,
        tileids=None,
        transform=None,
        sampling="random",  # "random" or "equal"
    ):
        self.seqlength = seqlength
        self.transform = transform
        self.sampling = sampling
        self.unique_labels = np.array([], dtype=float)

        if not isinstance(years, list):
            years = [years]
        self.data_dirs = years

        if not isinstance(root_dirs, list):
            root_dirs = [root_dirs]

        self.root_dirs = [Path(root_dir) for root_dir in root_dirs]
        classes_path = Path(classes_path)

        dataset_root = self.root_dirs[0].parent
        ensure_kaggle_dataset("ignazio/sentinel2-crop-mapping", dataset_root)

        self.classids, self.classes = read_classes(classes_path)

        self.name = ""
        self.samples = []
        self.ndates = []

        for root_dir in self.root_dirs:
            print(f"Reading dataset info: {root_dir}")
            self.name += f"{root_dir.name}_"

            for data_dir in self.data_dirs:
                year_dir = root_dir / data_dir
                if not year_dir.is_dir():
                    raise FileNotFoundError(
                        f"[LombardiaDataset] Missing directory: {year_dir} "
                        f"(root={root_dir}, year={data_dir})"
                    )

            stats = {
                "rejected_nopath": 0,
                "rejected_length": 0,
                "total_samples": 0,
            }

            dirs = []
            if tileids is None:
                for data_dir in self.data_dirs:
                    year_dir = root_dir / data_dir
                    dirs.extend([path for path in year_dir.iterdir()])
            else:
                tileids_path = root_dir / tileids
                with tileids_path.open("r") as f:
                    files = [line.strip() for line in f]

                for data_dir in self.data_dirs:
                    year_dir = root_dir / data_dir
                    dirs.extend([year_dir / file_name for file_name in files])

            for path in tqdm(dirs, desc="Scanning Lombardia samples"):
                if not path.exists():
                    stats["rejected_nopath"] += 1
                    continue

                if not (path / LABEL_FILENAME).exists():
                    stats["rejected_nopath"] += 1
                    continue

                ndates = len(get_dates(path))
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
        for date in dates:
            x10.append(read(path / f"{date}.tif")[0])

        x10 = np.array(x10) * 1e-4

        label = label[0]
        self.unique_labels = np.unique(
            np.concatenate([label.flatten(), self.unique_labels])
        )

        new = np.zeros(label.shape, dtype=int)
        for class_ids, index in zip(self.classids, range(len(self.classids))):
            for class_id in class_ids:
                new[label == class_id] = index
        label = new

        label = torch.from_numpy(label)
        x10 = torch.from_numpy(x10)
        x = x10.permute(1, 0, 2, 3).float()

        if self.transform is not None:
            label, x = self.transform(label, x)

        label = label.long()
        filename = str(Path(*path.parts[-3:]))

        return x, label, doys, filename


def get_dates(path, n=None, sampling="equal"):
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

        if len(indices) < n:
            selected = set(indices.tolist())
            missing = [i for i in range(len(dates)) if i not in selected]
            indices = np.concatenate([indices, missing[: n - len(indices)]])
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
            class_info = row.split("|")
            id_info = [int(value) for value in class_info[0].split(",")]
            ids.append(id_info)
            names.append(class_info[1])

    return ids, names


def read(file_path):
    file_path = Path(file_path)
    with rasterio.open(file_path) as src:
        return src.read(), src.profile