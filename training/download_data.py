"""
Utility script for downloading the StackOverflow dataset from Kaggle.

This dataset is used only for fine‑tuning the embedding model.
It is not part of the production document pipeline.

Dataset source:
https://www.kaggle.com/datasets/stackoverflow/stacksample/data
"""

import kagglehub
import shutil
import os


def download_stacksample() -> str:
    """
    Download the StackSample dataset and copy it into data/kaggle/.
    Returns the local path where the dataset is stored.
    """
    print("Downloading StackSample dataset using kagglehub...")

    # kagglehub handles all authentication internally
    source_path = kagglehub.dataset_download("stackoverflow/stacksample")

    target_dir = "data/kaggle"
    os.makedirs(target_dir, exist_ok=True)

    # Copy downloaded files into your project directory
    for filename in os.listdir(source_path):
        src = os.path.join(source_path, filename)
        dst = os.path.join(target_dir, filename)
        shutil.copy(src, dst)

    print(f"Dataset copied to {target_dir}")
    return target_dir


if __name__ == "__main__":
    download_stacksample()
