import os
import cv2
import numpy as np
import yaml
from torch.utils.data import Dataset


class RellisDataset(Dataset):
    """RELLIS-3D semantic segmentation dataset.

    Reads image/mask pairs listed in the official split files (train.lst/val.lst/test.lst).
    See docs/decisions/0003-rellis-split-strategy.md for why we use the official split.
    """

    def __init__(self, dataset_config_path, split, transform=None):
        assert split in ("train", "val", "test"), f"invalid split: {split}"
        with open(dataset_config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.base = self.cfg["base_path"]
        self.images_dir = self.cfg["images_dir"]
        self.labels_dir = self.cfg["labels_dir"]
        split_file = self.cfg["split_files"][split]
        split_path = os.path.join(self.base, self.cfg["split_dir"], split_file)

        self.pairs = []
        with open(split_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_rel, mask_rel = line.split(" ")
                self.pairs.append((img_rel, mask_rel))

        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_rel, mask_rel = self.pairs[idx]
        img_path = os.path.join(self.base, self.images_dir,
 img_rel)
        mask_path = os.path.join(self.base, self.labels_dir, mask_rel)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            raise FileNotFoundError(f"Missing image: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"Missing mask: {mask_path}")

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        return image, mask
