Colab Quick Start
=================

Steps to push this repo to GitHub and run training in Google Colab using the MosMed dataset from Kaggle.

1. Push to GitHub

```bash
# from your local machine
cd /path/to/fyp-dtcr-unet
git init   # if needed
git add .
git commit -m "Add Colab training helpers and dataset loader"
git remote add origin <your-github-repo-url>
git push -u origin main
```

2. Open a new Colab notebook and run these setup steps (cells):

Install dependencies and the Kaggle CLI:

```bash
!pip install -r /content/fyp-dtcr-unet/dtcr-unet/requirements.txt
```

Upload your `kaggle.json` (Kaggle API token) to Colab (use the left Files pane or `files.upload()`), then move it:

```bash
from google.colab import files
files.upload()  # choose kaggle.json
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

3. Download the MosMed dataset (example):

```bash
!python /content/fyp-dtcr-unet/dtcr-unet/scripts/download_mosmed.py --dest /content/mosmeddata
```

4. Run a quick training test (one epoch):

```bash
!python /content/fyp-dtcr-unet/dtcr-unet/train_colab.py --data /content/mosmeddata --epochs 1 --batch-size 2
```

Optional: enable activation checkpointing to reduce memory usage (slower backward pass):

```bash
!python /content/fyp-dtcr-unet/dtcr-unet/train_colab.py --data /content/mosmeddata --epochs 1 --batch-size 2 --checkpoint
```

Notes
- The repository includes `data/mosmed_dataset.py` as a simple `Dataset` implementation. You may need to adapt file paths depending on how the Kaggle dataset is structured after extraction.
- If you want GPU/CUDA, in Colab enable `Runtime > Change runtime type > GPU`.
- Adjust `train_colab.py` for augmentations, proper train/val splits, and metrics.
