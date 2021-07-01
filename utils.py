"""
Created by edwardli on 6/30/21
"""
import cv2
import numpy as np
import torch


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
