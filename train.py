"""
Created by edwardli on 6/30/21
"""

import hydra
import wandb


def init_wandb(cfg):
    # initialize weights and biases.
    wandb.init(project="torch tutorial", dir=cfg.paths.wandb_path, tags=cfg.params.tags)
    wandb.tensorboard.patch(save=True, tensorboardX=False)
    wandb.config.update(cfg)


def train():
    pass


def eval():
    pass


@hydra.main(config_name='config', config_path='config')
def main():
    pass


if __name__ == '__main__':
    main()
