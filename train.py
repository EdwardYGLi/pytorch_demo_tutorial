"""
Created by edwardli on 6/30/21
"""

import os

import hydra
import torch
from hydra.utils import get_original_cwd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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

    # hydra creates a directory for us so we can use current working directory as output directory.
    output_dir = os.getcwd()
    code_dir = get_original_cwd()

    tboard_writter = SummaryWriter(os.path.join(output_dir, "tensorboard/"))

    # we can do reflection/importlib etc here as well, but being explicit is better for readability and understanding the code.
    if cfg.optimizer.name == "adam":
        optimizer = torch.optim.Adam(model.parameters, cfg.optimizer.lr, (cfg.optimizer.beta1, cfg.optimizer.beta2),
                                     cfg.optimizer.eps)
    elif cfg.optimizer.name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters, cfg.optimizer.lr, (cfg.optimizer.beta1, cfg.optimizer.beta2),
                                      cfg.optimizer.eps,
                                      cfg.optimizer.weight_decay)
    elif cfg.optimizer.name == "sgd":
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
