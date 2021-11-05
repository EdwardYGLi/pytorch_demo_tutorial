"""
Created by edwardli on 6/30/21
"""
import os

import hydra
import numpy as np
import torch
import torch.nn as nn
import wandb
from hydra.utils import get_original_cwd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.dataset import DogsDataset
from models.net import ConvolutionalAutoEncoder
from perceptual_loss import perceptual_loss
from utils import init_wandb, seed_random, ssim, psnr

# prefer use Gpu for everything
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def activation_hook(model, layers):
    """
    add a forward hook to aid in visualizations
    @param model: input model
    @param layers: layers to include in the hook
    @return: a dictionary of visualizers, a dictionary of handlers used to remove hooks later on
    """
    # this is more as an example for hooks, but you could also just output the layer
    # you want in your forward() function and plot.
    visualisation = {}
    name_dicts = {}
    handles = {}

    def hook_fn(m, i, o):
        visualisation[name_dicts[m]] = o

    def hook_layers(net, layers=None):
        for name, layer in net.named_modules():
            # If it is a sequential, don't register a hook on it
            if not isinstance(layer, nn.Sequential):
                # check if we want to hook everything
                if layers is None or name in layers:
                    name_dicts[layer] = name
                    handle = layer.register_forward_hook(hook_fn)
                    handles[name] = handle

    hook_layers(model, layers)

    return visualisation, handles


def grad_hook(model, layers):
    """
    Demo backward hook function
    @param model: input model
    @param layers: layers to watch
    @return: handles
    """
    name_dicts = {}
    handles = {}

    def back_hook(m, i, o):
        print("dl/dw: {}".format(m.weight.grad.mean()))
        for grad in i:
            print("dl/di: {}".format(grad.mean()))
        for grad in o:
            print("dl/do: {}".format(grad.mean()))

    def back_hooks(net, layers=["decoder.6"]):
        for name, layer in net.named_modules():
            # If it is a sequential, don't register a hook on it
            # but recursively register hook on all it's module children
            if not isinstance(layer, nn.Sequential):
                if name in layers:
                    name_dicts[layer] = name
                    handle = layer.register_full_backward_hook(back_hook)
                    handles[name] = handle

    back_hooks(model, layers)
    return handles


def gen_validation_samples(model, vis_batches, tboard, epoch):
    """
    Generate some validation plots to help us visualize our training
    @param model: model
    @param vis_batches: validation batch images
    @param tboard: summary writer
    @param epoch: current epoch
    @return: None
    """
    # we will hook to our activation hooks and unhook after.
    vis_layers = ["encoder.0"]

    # hook our pre hook to model
    activations, handles = activation_hook(model, vis_layers)
    plot_act = False
    for batch_idx, vis_batch in enumerate(vis_batches):
        b, c, h, w = vis_batch.shape
        vis_batch = vis_batch.to(_device)
        outputs, _ = model(vis_batch)

        if not plot_act:
            for vis_layer in vis_layers:
                # for now just visualize first level activations for the first image in first batch
                curr_act = activations[vis_layer][0]
                c, h, w = curr_act.shape
                for k in range(c):
                    act = curr_act[k:k + 1]  # slicing keep dim
                    tboard.add_image("img_0_{}_act_{}".format(vis_layer,k), img_tensor=act, global_step=epoch)
            plot_act = True

        for i in range(b):
            ind = batch_idx * b + i
            img = outputs[i].cpu().numpy()[::-1, :, :]
            orig = vis_batch[i].cpu().numpy()[::-1, :, :]
            out_img = np.concatenate([orig, img], axis=1)
            tboard.add_image("recon_{}".format(ind), img_tensor=out_img, global_step=epoch)

    # unhooking will free some memory here.
    for vis_layer in vis_layers:
        handles[vis_layer].remove()


def run_epoch(model, data_loader, loss_fns, val_criteria=None, optimizer=None, debug=False):
    """
    Run model for an epoch
    @param model: model
    @param data_loader: data loader
    @param loss_fns: loss functions
    @param val_criteria: additional validation critiers
    @param optimizer: optimizer for train step
    @param debug: print grads or any debug stuff
    @return: loss summary for current epoch.
    """
    loss_dict = {}
    for key in loss_fns.keys():
        loss_dict[key] = 0
    # add validation criteria
    if val_criteria:
        for key in val_criteria.keys():
            loss_dict[key] = 0

    loss_dict["total"] = 0

    num_batches = len(data_loader)

    for batch in data_loader:
        batch = batch.to(_device)

        if optimizer:
            optimizer.zero_grad()

        output, latents = model(batch)

        losses = []
        for key, criterion in loss_fns.items():
            curr_loss = criterion(output, batch)
            losses.append(curr_loss)
            loss_dict[key] += float(curr_loss)

        # validation criteria that aren't used in optimization.
        if val_criteria:
            for key, criterion in val_criteria.items():
                curr_loss = criterion(output, batch)
                loss_dict[key] += float(curr_loss)

        total_loss = sum(losses)
        loss_dict["total"] += float(total_loss)

        if optimizer:
            total_loss.backward(retain_graph=debug)
            if debug:
                print("dl/dw (loop):{}".format(model.decoder[6].weight.grad.mean()))
            optimizer.step()

    # divide loss by number of batches for per batch loss as losses are batch
    # mean errors
    for key, value in loss_dict.items():
        loss_dict[key] = value / num_batches

    return loss_dict


def get_optimizer(model, cfg):
    """
    based on config create the optimizer
    @param model: network model
    @param cfg: config file
    @return: optimizer
    """
    # we can do reflection/importlib etc here as well, but being explicit is better for readability and understanding the code.
    if cfg.optimizer.name == "adam":
        return torch.optim.Adam(model.parameters(), cfg.optimizer.lr, (cfg.optimizer.beta1, cfg.optimizer.beta2),
                                cfg.optimizer.eps)
    elif cfg.optimizer.name == "adamw":
        return torch.optim.AdamW(model.parameters(), cfg.optimizer.lr, (cfg.optimizer.beta1, cfg.optimizer.beta2),
                                 cfg.optimizer.eps,
                                 cfg.optimizer.weight_decay)
    elif cfg.optimizer.name == "sgd":
        return torch.optim.SGD(model.parameters(), cfg.optimizer.lr, cfg.optimizer.momentum,
                               cfg.optimizer.dampening, cfg.optimizer.weight_decay, cfg.optimizer.nesterov)
    else:
        raise AssertionError("optimizer type ({}) not correct".format(cfg.optimizer.name))


def get_loss_fn(cfg):
    """
    get loss function based on cfg object
    @param cfg: config file
    @return: dictionary of loss functions with name and function
    """
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


@hydra.main(config_name='config', config_path='configs')
def main(cfg):
    """
    main training function
    @param cfg: config file
    @return:
    """
    # hydra creates a directory for us so we can use current working directory as output directory.
    output_dir = os.getcwd()
    project_out_dir = cfg.paths.output_dir
    code_dir = get_original_cwd()

    # init experiment in wandb
    os.makedirs(os.path.join(project_out_dir, "wandb"), exist_ok=True)
    init_wandb(cfg, project_out_dir)

    # reproducibility
    seed_random(cfg.seed)

    # create model
    model = ConvolutionalAutoEncoder()
    model = model.to(_device)

    # load pretrained checkpoint to resume training.
    if cfg.pretrained != "":
        print("loading ckpt: {}".format(cfg.pretrained))
        # if you ckpt has other caveats or you want loose loading you can use non strict. It will load whatever matches
        state_dict = torch.load(cfg.pretrained)
        model.load_state_dict(state_dict)

    # create a tensorboard writer that writes to our directory
    tboard_writer = SummaryWriter(output_dir)

    # get optimizer
    optimizer = get_optimizer(model, cfg)

    # get scheduler
    scheduler = None
    # scheduler = get_scheduler(optimizer,cfg)

    # create dataset
    dataset = DogsDataset(code_dir, cfg.dataset)

    train_loader = DataLoader(dataset.training_data, cfg.batch_size, shuffle=True)
    validation_loader = DataLoader(dataset.validation_data, cfg.batch_size, shuffle=False)

    # weight and biases can watch the model and track gradients (twice every epoch).
    wandb.watch(model, log_freq=100)

    visualize_batches = []
    val_iter = iter(validation_loader)
    # create some validation visualization batches.
    for i in range(cfg.vis_batches):
        visualize_batches.append(next(val_iter))

    # we can do reflection here as well
    loss_fns = get_loss_fn(cfg)

    validation_criteria = {"ssim": ssim,
                           "psnr": psnr
                           }

    curr_best = 1.0E10

    for epoch in tqdm(range(cfg.epochs)):
        print("epoch {}".format(epoch))
        # pause autograd for speed for validation
        torch.set_grad_enabled(False)
        # do validation first
        model.eval()
        # loss some validation samples first
        gen_validation_samples(model, visualize_batches, tboard_writer, epoch)

        # run an validation epoch
        epoch_val_loss = run_epoch(model=model, data_loader=validation_loader, loss_fns=loss_fns,
                                   val_criteria=validation_criteria, debug=cfg.debug)

        if epoch_val_loss["total"] < curr_best:
            torch.save(model.state_dict(), os.path.join(output_dir, cfg.experiment_name + "_best.pt"))

        # log losses to tensorboard
        for key, value in epoch_val_loss.items():
            print("val " + key, value)
            tboard_writer.add_scalar("val_" + key, value, epoch)

        # resume autograd
        torch.set_grad_enabled(True)
        model.train()

        # quick demo of backward hooks in network.
        if cfg.debug:
            handles = grad_hook(model, ["decoder.6"])

        # run an training epoch
        epoch_train_loss = run_epoch(model=model, data_loader=train_loader, loss_fns=loss_fns, optimizer=optimizer,
                                     debug=cfg.debug)

        # clear handles
        if cfg.debug:
            for key, handle in handles.items():
                handle.remove()
        # log losses to tensorboard on epoch basis, can also log every few steps within the train function.
        for key, value in epoch_train_loss.items():
            print("train " + key, value)
            tboard_writer.add_scalar("train_" + key, value, epoch)

        # step your scheduler
        if scheduler:
            scheduler.step()
            tboard_writer.add_scalar("learning_rate", scheduler.get_lr()[0], epoch)

    torch.save(model.state_dict(), os.path.join(output_dir, cfg.experiment_name + "_final.pt"))
    wandb.finish()


if __name__ == '__main__':
    main()
