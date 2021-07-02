"""
Created by edwardli on 6/30/21
"""

import glob
import os

import cv2
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class _DogsDatasetInternal(Dataset):
    def __init__(self, path, pattern, trans=None):
        self.files = glob.glob(os.path.join(path, pattern))
        self.len = len(self.files)
        # you can use torch transforms here, or write your own.
        self.transforms = trans

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = self.files[idx]
        img = cv2.imread(img)

        ## can add augmentations or any things here or use torch.transform to do it,
        if self.transforms:
            img = self.transforms(img)

        return img


class DogsDataset:
    def __init__(self, code_dir, cfg):
        # do some augmentations
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(size=cfg.img_size),
            transforms.ColorJitter(brightness=cfg.brightness_jitter, contrast=cfg.contrast_jitter,
                                   saturation=cfg.saturation_jitter, hue=cfg.hue_jitter),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])

        # for validation keep it simple.
        validation_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(cfg.img_size,cfg.img_size))
        ])

        train_path = os.path.join(code_dir, cfg.dataset_path, "train")
        val_path = os.path.join(code_dir, cfg.dataset_path, "val")
        self.training_data = _DogsDatasetInternal(train_path, cfg.file_pattern, trans=train_transforms)
        self.validation_data = _DogsDatasetInternal(val_path, cfg.file_pattern, trans=validation_transforms)
