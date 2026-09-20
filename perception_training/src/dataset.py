import os
import cv2
import numpy as np
import yaml
import torch
from torch.utils.data import Dataset

from preprocessing import build_class_remap, remap_mask, resize_image_and_mask, normalize_image, random_hflip


class RellisDataset(Dataset):
    """RELLIS-3D semantic segmentation dataset.

    Reads image/mask pairs listed in the official split files (train.lst/val.lst/test.lst).
    See docs/decisions/0003-rellis-split-strategy.md for why we use the official split.

    When training_config_path is provided, __getitem__ returns preprocessed torch tensors
    (resized, normalized image; resized, contiguous-remapped mask). When it is None,
    __getitem__ returns raw numpy arrays (useful for visualization/debugging, as in earlier steps).
    """

    def __init__(self, dataset_config_path, split, classes_config_path=None, training_config_path=None):
        assert split in ("train", "val", "test"), f"invalid split: {split}"
        self.split = split
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

        self.classes_config_path = classes_config_path
        self.training_config_path = training_config_path
        self.remap = None
        self.train_cfg = None

        if classes_config_path is not None:
            self.remap, self.inverse_remap = build_class_remap(classes_config_path)

        if training_config_path is not None:
            with open(training_config_path) as f:
                self.train_cfg = yaml.safe_load(f)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_rel, mask_rel = self.pairs[idx]
        img_path = os.path.join(self.base, self.images_dir, img_rel)
        mask_path = os.path.join(self.base, self.labels_dir, mask_rel)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            raise FileNotFoundError(f"Missing image: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"Missing mask: {mask_path}")

        if self.remap is not None:
            mask = remap_mask(mask, self.remap)

        if self.train_cfg is not None:
            size = tuple(self.train_cfg["input_size"])
            image, mask = resize_image_and_mask(image, mask, size=size)

            if self.split == "train":
                image, mask = random_hflip(image, mask, p=0.5)

            image = normalize_image(
                image,
                mean=tuple(self.train_cfg["normalize_mean"]),
                std=tuple(self.train_cfg["normalize_std"]),
            )
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
            mask_tensor = torch.from_numpy(mask).long()
            return image_tensor, mask_tensor

        return image, mask
