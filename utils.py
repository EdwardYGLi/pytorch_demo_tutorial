"""
Created by edwardli on 6/30/21
"""
import os
import random

import cv2
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf


def init_wandb(cfg, wandb_dir):
    # initialize weights and biases.
    # wandb.tensorboard.patch(save=False, pytorch=True)
    wandb.init(project=cfg.project_name, dir=wandb_dir, tags=cfg.tags,
               name=cfg.experiment_name, reinit=True, sync_tensorboard=True)
    wandb.config.update(OmegaConf.to_object(cfg))


def seed_random(seed: int):
    """
    Seeds random number generators with the same seed for reproducibility.
    cudnn.deterministed = True changes cudnn background algorithm selection to be deterministic.
        (https://stackoverflow.com/questions/56354461/reproducibility-and-performance-in-pytorch)
    cudnn.benchmark = False (https://discuss.pytorch.org/t/how-should-i-disable-using-cudnn-in-my-code/38053/3)
    reference: https://pytorch.org/docs/stable/notes/randomness.html
    """
    torch.random.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def psnr(pred, target):
    # implement PSNR (peak to peak signal to noise ratio)
    # between prediction and target here
    mse = torch.mean((pred - target) ** 2)
    return 20 * torch.log10(1 / torch.sqrt(mse))


def ssim(pred, target):
    # implement SSIM (structured similarity index),
    # between prediction and target here\

    # original dimension (b, c, h, w)
    pred = pred.detach().cpu().permute(0, 2, 3, 1).numpy()
    target = target.detach().cpu().permute(0, 2, 3, 1).numpy()
    # dimension (b, h,w,c)
    # we should implement this in pytorch native tensor operations for speed,
    # but for readability we will do this the slow way (looping)
    # for now.
    ssim = 0
    for i in range(pred.shape[0]):
        p = pred[i]
        t = target[i]
        # constants for numerical stability
        C1 = (0.01) ** 2
        C2 = (0.03) ** 2

        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())
        mu1 = cv2.filter2D(p, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(t, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(p ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(t ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(p * t, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
                (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        ssim += ssim_map.mean()
    # return batch avg ssim
    return ssim / pred.shape[0]
