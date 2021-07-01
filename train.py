"""
Created by edwardli on 6/30/21
"""

import hydra
import torch
import datetime
import os

from datasets.dataset import DogsDataset
from models.net import ConvolutionalAutoEncoder
from utils import init_wandb, seed_random

# prefer use Gpu for everything
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    pass


def eval():
    pass


@hydra.main(config_name='config', config_path='config')
def main(cfg):

    # init experiment in wandb
    init_wandb(cfg)

    # reproducibility
    seed_random(cfg.seed)

    # create model
    model = ConvolutionalAutoEncoder()

    # create output directory
    now = datetime.datetime.now()
    out_dir = os.path.join(cfg.paths.output_dir, now.strftime("%Y-%m-%d_%H_%M") + "_{}".format(model.__class__.__name__))
    # make output directory
    os.makedirs(out_dir, exist_ok=True)

    # we can do reflection/importlib etc here as well, but being explicit is better for readability and understanding the code.
    if cfg.optimizer.type == "adam":
        optimizer = torch.optim.Adam(model.parameters, cfg.optimizer.lr, cfg.optimizer.betas, cfg.optimizer.eps)
    elif cfg.optimizer.type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters, cfg.optimizer.lr, cfg.optimizer.betas, cfg.optimizer.eps,
                                      cfg.optimizer.weight_decay)
    elif cfg.optimizer.type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), cfg.optimizer.lr, cfg.optimizer.momentum,
                                    cfg.optimizer.dampening, cfg.optmizer.weight_decay, cfg.optimizer.nesterov)
    else:
        raise AssertionError("optimizer type not correct")

    # create dataset
    dataset = DogsDataset(cfg.dataset)


if __name__ == '__main__':
    main()
