import subprocess
from pathlib import Path


def ensure_kaggle_dataset(dataset_slug: str, target_dir: Path):
    target_dir = Path(target_dir)

    # If folder exists and is not empty → assume already downloaded
    if target_dir.exists() and any(target_dir.iterdir()):
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {dataset_slug} into {target_dir}...")

    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            dataset_slug,
            "-p",
            str(target_dir),
            "--unzip",
        ],
        check=True,
    )

    return target_dir