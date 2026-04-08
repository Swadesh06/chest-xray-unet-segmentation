# Chest X-Ray Lung Segmentation with U-Net

U-Net based lung segmentation on chest X-ray images. Achieves **Dice: 0.9648, IoU: 0.9330** on the test set.

## Dataset

[Kaggle - Chest X-ray Masks and Labels](https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels) — 704 chest X-ray images with corresponding lung masks.

## Setup

```bash
# clone
git clone https://github.com/Swadesh06/chest-xray-unet-segmentation.git
cd chest-xray-unet-segmentation

# conda env
source /path/to/miniconda3/etc/profile.d/conda.sh
conda create --prefix ./env python=3.10 -y
conda activate ./env

# install deps
export PIP_CACHE_DIR=./.cache/pip
pip install torch torchvision numpy matplotlib pillow scikit-learn jupyter kaggle tqdm

# download dataset
export KAGGLE_API_TOKEN=<your-token>
kaggle datasets download -d nikhilpandey360/chest-xray-masks-and-labels -p ./data/
cd data && unzip chest-xray-masks-and-labels.zip && cd ..
```

## Run

Open and run the notebook:

```bash
jupyter notebook unet_segmentation.ipynb
```

## Results

| Metric | Score |
|--------|-------|
| Dice | 0.9648 |
| IoU | 0.9330 |

### Sample Predictions

![Sample Predictions](results/sample_predictions.png)

### Loss Curve

![Loss Curve](results/training_loss.png)

## Project Structure

```
├── README.md                  # this file
├── unet_segmentation.ipynb    # main notebook
├── report.md                  # detailed report
├── results/                   # saved figures and metrics
│   ├── sample_predictions.png
│   ├── training_loss.png
│   └── metrics.txt
├── data/                      # dataset (not in repo)
└── env/                       # conda env (not in repo)
```
