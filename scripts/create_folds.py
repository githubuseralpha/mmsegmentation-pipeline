import os
import shutil
import json

import numpy as np

def main():
    image_dir = "/workspace/data/pipeline/images/train"
    label_dir = "/workspace/data/pipeline/annotations/train"
    target_dir = "/workspace/folds"

    # Create target directory if not exists
    os.makedirs(target_dir, exist_ok=True)
    files = os.listdir(image_dir)

    # Split the data into 5 folds
    files = np.array(files)
    np.random.shuffle(files)
    num_files = len(files)
    folds = np.array_split(files, 5)

    # Save the folds
    for i, fold in enumerate(folds):
        fold_dir = os.path.join(target_dir, f"fold_{i}")
        os.makedirs(fold_dir, exist_ok=True)
        os.makedirs(os.path.join(fold_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, "annotations"), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, "images", "train"), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, "annotations", "train"), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, "images", "val"), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, "annotations", "val"), exist_ok=True)

        for file in fold:
            image_path = os.path.join(image_dir, file)
            label_path = os.path.join(label_dir, file)
            shutil.copy(image_path, os.path.join(fold_dir, "images", "val", file))
            shutil.copy(label_path, os.path.join(fold_dir, "annotations", "val", file))

        for j, other_fold in enumerate(folds):
            if i != j:
                for file in other_fold:
                    image_path = os.path.join(image_dir, file)
                    label_path = os.path.join(label_dir, file)
                    shutil.copy(image_path, os.path.join(fold_dir, "images", "train", file))
                    shutil.copy(label_path, os.path.join(fold_dir, "annotations", "train", file))


if __name__ == "__main__":
    main()
    