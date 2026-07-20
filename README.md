# 🌾 Satellite Time Series Crop Segmentation Models

This project provides training, evaluation, and inference pipelines for crop segmentation from Sentinel-2 satellite image time series (SITS), comparing convolutional and transformer-based architectures on the Munich and Lombardia datasets.

> 📄 **Related paper:** *A Comparative Study of Transformer and Convolutional Models for Crop Segmentation from Satellite Image Time Series.*
> Available on [arXiv](https://arxiv.org/abs/2412.01944). See [Citation](#-citation) below.

---

## ⚙️ Setup

```bash
git clone https://github.com/mattiagatti/sits-crop-segmentation-models.git
cd sits-crop-segmentation-models

sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.11 python3.11-venv

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🔑 Kaggle credentials

The datasets are downloaded through the Kaggle CLI, which authenticates using a
`kaggle.json` API token.

1. Go to https://www.kaggle.com/settings, open the **API** section, and click
   **Create New Token**. This downloads a `kaggle.json` file.
2. Make it available to the CLI in either of these ways:

```bash
# Option A: place the token file in the default location
mkdir -p ~/.kaggle
mv /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

```bash
# Option B: export the credentials as environment variables
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
```

---

## 📦 Datasets (auto-downloaded)

-   :de: Munich dataset\
    https://www.kaggle.com/datasets/artelabsuper/sentinel2-munich480

-   :it: Lombardia dataset\
    https://www.kaggle.com/datasets/ignazio/sentinel2-crop-mapping

Datasets are downloaded automatically on first use into the directory given by
`--data_dir`. **Set this to a writable path on your machine**, since it defaults to an
internal path used by the authors:

```bash
--data_dir /path/to/your/datasets
```

Add this flag to any `train.py` / `test.py` / `inference.py` command below if you are
not using the default path.

---

## 🚀 Training

### :de: Munich

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --arch deeplabv3 --dataset munich
CUDA_VISIBLE_DEVICES=1 python train.py --arch fpn --dataset munich
CUDA_VISIBLE_DEVICES=2 python train.py --arch swin_unetr --dataset munich
CUDA_VISIBLE_DEVICES=3 python train.py --arch tsvit --dataset munich
CUDA_VISIBLE_DEVICES=4 python train.py --arch tsvit_lookup --dataset munich
CUDA_VISIBLE_DEVICES=5 python train.py --arch unet --dataset munich
CUDA_VISIBLE_DEVICES=6 python train.py --arch vistaformer --dataset munich
```

### 🔁 Munich resume

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --arch deeplabv3 --dataset munich --ckpt_path exp/deeplabv3/munich/train/checkpoints/last.ckpt
CUDA_VISIBLE_DEVICES=1 python train.py --arch fpn --dataset munich --ckpt_path exp/fpn/munich/train/checkpoints/last.ckpt 
CUDA_VISIBLE_DEVICES=2 python train.py --arch swin_unetr --dataset munich --ckpt_path exp/swin_unetr/munich/train/checkpoints/last.ckpt
CUDA_VISIBLE_DEVICES=3 python train.py --arch tsvit --dataset munich --ckpt_path exp/tsvit/munich/train/checkpoints/last.ckpt
CUDA_VISIBLE_DEVICES=4 python train.py --arch tsvit_lookup --dataset munich --ckpt_path exp/tsvit_lookup/munich/train/checkpoints/last.ckpt
CUDA_VISIBLE_DEVICES=5 python train.py --arch unet --dataset munich --ckpt_path exp/unet/munich/train/checkpoints/last.ckpt
CUDA_VISIBLE_DEVICES=6 python train.py --arch vistaformer --dataset munich --ckpt_path exp/vistaformer/munich/train/checkpoints/last.ckpt
```

---

### :it: Lombardia

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --arch deeplabv3 --dataset lombardia
CUDA_VISIBLE_DEVICES=1 python train.py --arch fpn --dataset lombardia
CUDA_VISIBLE_DEVICES=2 python train.py --arch swin_unetr --dataset lombardia
CUDA_VISIBLE_DEVICES=3 python train.py --arch tsvit --dataset lombardia
CUDA_VISIBLE_DEVICES=4 python train.py --arch tsvit_lookup --dataset lombardia
CUDA_VISIBLE_DEVICES=5 python train.py --arch unet --dataset lombardia
CUDA_VISIBLE_DEVICES=6 python train.py --arch vistaformer --dataset lombardia
```

### 🔁 Lombardia resume

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --arch deeplabv3 --dataset lombardia --ckpt_path exp/deeplabv3/lombardia/train/checkpoints/last.ckpt
CUDA_VISIBLE_DEVICES=1 python train.py --arch fpn --dataset lombardia --ckpt_path exp/fpn/lombardia/train/checkpoints/last.ckpt
CUDA_VISIBLE_DEVICES=2 python train.py --arch swin_unetr --dataset lombardia --ckpt_path exp/swin_unetr/lombardia/train/checkpoints/last.ckpt
CUDA_VISIBLE_DEVICES=3 python train.py --arch tsvit --dataset lombardia --ckpt_path exp/tsvit/lombardia/train/checkpoints/last.ckpt
CUDA_VISIBLE_DEVICES=4 python train.py --arch tsvit_lookup --dataset lombardia --ckpt_path exp/tsvit_lookup/lombardia/train/checkpoints/last.ckpt
CUDA_VISIBLE_DEVICES=5 python train.py --arch unet --dataset lombardia --ckpt_path exp/unet/lombardia/train/checkpoints/last.ckpt
CUDA_VISIBLE_DEVICES=6 python train.py --arch vistaformer --dataset lombardia --ckpt_path exp/vistaformer/lombardia/train/checkpoints/last.ckpt
```

---

## 🧪 Evaluation

### :de: Munich

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --arch deeplabv3 --dataset munich --weights_path exp/deeplabv3/munich/train/weights/best.pt
CUDA_VISIBLE_DEVICES=1 python test.py --arch fpn --dataset munich --weights_path exp/fpn/munich/train/weights/best.pt
CUDA_VISIBLE_DEVICES=2 python test.py --arch swin_unetr --dataset munich --weights_path exp/swin_unetr/munich/train/weights/best.pt
CUDA_VISIBLE_DEVICES=3 python test.py --arch tsvit --dataset munich --weights_path exp/tsvit/munich/train/weights/best.pt
CUDA_VISIBLE_DEVICES=4 python test.py --arch tsvit_lookup --dataset munich --weights_path exp/tsvit_lookup/munich/train/weights/best.pt
CUDA_VISIBLE_DEVICES=5 python test.py --arch unet --dataset munich --weights_path exp/unet/munich/train/weights/best.pt
CUDA_VISIBLE_DEVICES=6 python test.py --arch vistaformer --dataset munich --weights_path exp/vistaformer/munich/train/weights/best.pt
```

### :it: Lombardia A

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --arch deeplabv3 --dataset lombardia --test_id A --weights_path exp/deeplabv3/lombardia/train/weights/best.pt
CUDA_VISIBLE_DEVICES=1 python test.py --arch fpn --dataset lombardia --test_id A --weights_path exp/fpn/lombardia/train/weights/best.pt
CUDA_VISIBLE_DEVICES=2 python test.py --arch swin_unetr --dataset lombardia --test_id A --weights_path exp/swin_unetr/lombardia/train/weights/best.pt
CUDA_VISIBLE_DEVICES=3 python test.py --arch tsvit --dataset lombardia --test_id A --weights_path exp/tsvit/lombardia/train/weights/best.pt
CUDA_VISIBLE_DEVICES=4 python test.py --arch tsvit_lookup --dataset lombardia --test_id A --weights_path exp/tsvit_lookup/lombardia/train/weights/best.pt
CUDA_VISIBLE_DEVICES=5 python test.py --arch unet --dataset lombardia --test_id A --weights_path exp/unet/lombardia/train/weights/best.pt
CUDA_VISIBLE_DEVICES=6 python test.py --arch vistaformer --dataset lombardia --test_id A --weights_path exp/vistaformer/lombardia/train/weights/best.pt
```

### :it: Lombardia Y

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --arch deeplabv3 --dataset lombardia --test_id Y --weights_path exp/deeplabv3/lombardia/train/weights/best.pt
CUDA_VISIBLE_DEVICES=1 python test.py --arch fpn --dataset lombardia --test_id Y --weights_path exp/fpn/lombardia/train/weights/best.pt
CUDA_VISIBLE_DEVICES=2 python test.py --arch swin_unetr --dataset lombardia --test_id Y --weights_path exp/swin_unetr/lombardia/train/weights/best.pt
CUDA_VISIBLE_DEVICES=3 python test.py --arch tsvit --dataset lombardia --test_id Y --weights_path exp/tsvit/lombardia/train/weights/best.pt
CUDA_VISIBLE_DEVICES=4 python test.py --arch tsvit_lookup --dataset lombardia --test_id Y --weights_path exp/tsvit_lookup/lombardia/train/weights/best.pt
CUDA_VISIBLE_DEVICES=5 python test.py --arch unet --dataset lombardia --test_id Y --weights_path exp/unet/lombardia/train/weights/best.pt
CUDA_VISIBLE_DEVICES=6 python test.py --arch vistaformer --dataset lombardia --test_id Y --weights_path exp/vistaformer/lombardia/train/weights/best.pt
```

### Export sample RGB time series
```bash
python -m misc.export_samples_rgb --dataset munich
```

### Run inference and merge patches into a full segmentation map
```bash
CUDA_VISIBLE_DEVICES=0 python inference.py --arch tsvit --dataset munich --save_tifs --weights_path exp/tsvit/munich/train/weights/best.pt
CUDA_VISIBLE_DEVICES=1 python inference.py --arch tsvit --dataset lombardia --test_id Y --save_tifs --merge_patches --weights_path exp/tsvit/lombardia/train/weights/best.pt
```

---

## 📝 Notes

-   📥 Datasets are downloaded automatically via the Kaggle CLI
-   🔐 Make sure Kaggle credentials are set before running
-   📂 Choose a writable `--data_dir` (datasets are stored/read there)
-   🎯 `CUDA_VISIBLE_DEVICES` controls GPU selection
-   📁 Logs and checkpoints are saved under
    `exp/<arch>/<dataset>/train/`

---

## 📖 Citation

If you use this repository, please cite the related paper.

Published version (SPIE, ICMV 2023):

``` bibtex
@inproceedings{Gatti_2026,
  title     = {A comparative study of transformer and convolutional models for crop segmentation from satellite image time series},
  url       = {http://dx.doi.org/10.1117/12.3120038},
  DOI       = {10.1117/12.3120038},
  booktitle = {Sixteenth International Conference on Machine Vision (ICMV 2023)},
  publisher = {SPIE},
  author    = {Gatti, M. and Gallo, I. and Landro, N. and Loschiavo, C. and Rehman, A. U. and Boschetti, M. and La Grassa, R.},
  editor    = {Osten, Wolfgang},
  year      = {2026},
  month     = May,
  pages     = {100}
}
```

Preprint (arXiv):

``` bibtex
@article{gatti2024sits,
  title   = {A Comparative Study of Transformer and Convolutional Models for Crop Segmentation from Satellite Image Time Series},
  author  = {Gatti, Mattia and Gallo, Ignazio and Landro, Nicola and Loschiavo, Christian and Rehman, Anwar Ur and Boschetti, Mirco and La Grassa, Riccardo},
  journal = {arXiv preprint arXiv:2412.01944},
  year    = {2024}
}
```
