"""
Created by edwardli on 7/1/21
"""
import hydra
from hydra.utils import get_original_cwd
import os
from omegaconf import OmegaConf

@hydra.main(config_name='config', config_path='configs')
def main(cfg):
    print(OmegaConf.to_object(cfg))
    print(cfg.experiment_name)
    print(os.getcwd())


if __name__ == "__main__":
    main()