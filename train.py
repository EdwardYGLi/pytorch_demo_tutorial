"""
Created by edwardli on 6/30/21
"""
import os

import hydra
import torch
import wandb
from hydra.utils import get_original_cwd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.dataset import DogsDataset
from models.net import ConvolutionalAutoEncoder
from perceptual_loss import perceptual_loss
from utils import init_wandb, seed_random

# prefer use Gpu for everything
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gen_validation_samples(writer, model, batch):
    pass


def train(model, validation_data, loss_fn, optimizer):
    pass


def eval(model, validation_data, loss_fn):
    pass


def get_optimizer(model, cfg):
    # we can do reflection/importlib etc here as well, but being explicit is better for readability and understanding the code.
    if cfg.optimizer.name == "adam":
        return torch.optim.Adam(model.parameters, cfg.optimizer.lr, (cfg.optimizer.beta1, cfg.optimizer.beta2),
                                cfg.optimizer.eps)
    elif cfg.optimizer.name == "adamw":
        return torch.optim.AdamW(model.parameters, cfg.optimizer.lr, (cfg.optimizer.beta1, cfg.optimizer.beta2),
                                 cfg.optimizer.eps,
                                 cfg.optimizer.weight_decay)
    elif cfg.optimizer.name == "sgd":
        return torch.optim.SGD(model.parameters(), cfg.optimizer.lr, cfg.optimizer.momentum,
                               cfg.optimizer.dampening, cfg.optmizer.weight_decay, cfg.optimizer.nesterov)
    else:
        raise AssertionError("optimizer type ({}) not correct".format(cfg.optimizer.name))


def get_loss_fn(cfg):
    # we can do reflection/importlib etc here as well, but being explicit is more pythonic
    if cfg.loss_fn == "mae":
        return {"mae": torch.nn.L1Loss()}
    elif cfg.loss_fn == "mse":
        return {"mse": torch.nn.MSELoss()}
    elif cfg.loss_fn == "perceptual":
        return {"perceptual_loss": perceptual_loss,
                "l1_loss": torch.nn.L1Loss()}
    else:
        raise AssertionError("loss type ({}) not available".format(cfg.loss_fn))


@hydra.main(config_name='config', config_path='config')
def main(cfg):
    # hydra creates a directory for us so we can use current working directory as output directory.
    output_dir = os.getcwd()
    project_out_dir = cfg.env.output_dir
    code_dir = get_original_cwd()

    # init experiment in wandb
    wandb_dir = os.path.join(project_out_dir, ".wandb")
    os.makedirs(wandb_dir, exist_ok=True)
    init_wandb(cfg, wandb_dir)

    # reproducibility
    seed_random(cfg.seed)

    # create model
    model = ConvolutionalAutoEncoder()
    # weight and biases can watch the model and track gradients.
    wandb.watch(model)

    # create a tensorboard writer that writes to our directory
    tboard_writer = SummaryWriter(output_dir)

    # get optimizer
    optimizer = get_optimizer(model, cfg)

    # create dataset
    dataset = DogsDataset(cfg.dataset)

    train_loader = DataLoader(dataset.training_data, cfg.batch_size, shuffle=True)
    validation_loader = DataLoader(dataset.validation_data, cfg.batch_size, shuffle=False)

    visualize_batch = next(iter(validation_loader))

    # we can do reflection here as wel
    loss_fn = get_loss_fn(cfg)

    curr_best = 1.0E10

    for epoch in range(cfg.epochs):
        # pause autograd for speed for validation
        torch.set_grad_enabled(False)
        # do validation first
        model.eval()
        # loss some validation samples first
        gen_validation_samples(model, visualize_batch, tboard_writer)

        epoch_val_loss = eval(model, validation_loader, loss_fn)

        if epoch_val_loss["total"] < curr_best:
            torch.save(model.state_dict(),os.path.join(output_dir,"_best.pt"))

        # log losses to tensorboard
        for key, value in epoch_val_loss.items():
            tboard_writer.add_scalar(key,value,epoch)

        # resume autograd
        torch.set_grad_enabled(True)
        model.train()

        epoch_train_loss = train(model, validation_loader, loss_fn, optimizer)

        # log losses to tensorboard on epoch basis, can also log every few steps within the train function.
        for key, value in epoch_train_loss.items():
            tboard_writer.add_scalar(key, value, epoch)





if __name__ == '__main__':
    main()
