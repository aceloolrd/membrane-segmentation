"""
Download ISBI 2012 EM Segmentation dataset from Kaggle.

Prerequisites:
    pip install kaggle
    Set up Kaggle API credentials: https://www.kaggle.com/docs/api
    (place kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME / KAGGLE_KEY env vars)

Usage:
    python download_data.py
"""

import os
import zipfile

DATASET = "kmader/isbi-challenges-em-segmentation"
DEST    = "data"


def main():
    try:
        import kaggle  # noqa: F401
    except ImportError:
        raise SystemExit("Install kaggle package: pip install kaggle")

    os.makedirs(DEST, exist_ok=True)
    print(f"Downloading {DATASET} …")
    os.system(f'kaggle datasets download -d {DATASET} -p {DEST} --unzip')

    # Expected structure after unzip:
    #   data/membrane/train/image/*.png
    #   data/membrane/train/mask/*.png
    #   data/membrane/test/image/*.png

    train_img = os.path.join(DEST, "membrane", "train", "image")
    if os.path.isdir(train_img):
        n = len(os.listdir(train_img))
        print(f"Done — {n} training images in {train_img}")
    else:
        print("Download complete. Check data/membrane/ structure.")


if __name__ == "__main__":
    main()
