"""
Created by edwardli on 6/30/21
"""

import datetime
import os

import hydra
import os
import torch
from torch.utils.data import DataLoader

from datasets.dataset import DogsDataset
from models.net import ConvolutionalAutoEncoder
from utils import init_wandb, seed_random
from torch.utils.tensorboard import SummaryWriter

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

    # hydra creates a directory for us so we can use current working directory as output directory.
    output_dir = os.getcwd()

    tboard_writter = SummaryWriter(os.path.join(output_dir,"tensorboard/"))

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

    train_loader = DataLoader(dataset.training_data, cfg.params.batch_size, shuffle=True)
    validation_loader = DataLoader(dataset.validation_data, cfg.params.batch_size, shuffle=False)









if __name__ == '__main__':
    main()
