#!/usr/bin/env python3
# Continuous diffusion version of the phoneme classifier training script.
# This merges the existing phoneme classifier training logic with the
# continuous SDE noise injection path used by the mel generation models.

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import warnings
warnings.filterwarnings("ignore")
from functools import partial
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from dataloaders import dataloader, CollateFn
from utils import (
    find_max_epoch,
    print_size,
    # calc_diffusion_hyperparams,
    get_diffusion_hyperparams,
    local_directory,
    plot_melspec,
)

from distributed_util import init_distributed, apply_gradient_allreduce, reduce_tensor
from models.model_builder import ModelBuilder
from models.audiovisual_model import AudioVisualModel
from SDE import VPSDE, VESDE


def distributed_train(rank, num_gpus, group_name, cfg):
    dist_cfg = cfg.pop("distributed")
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_cfg)

    train(
        rank=rank,
        num_gpus=num_gpus,
        diffusion_cfg=cfg.diffusion,
        model_cfg=cfg.phoneme_classifier,
        dataset_cfg=cfg.dataset,
        **cfg.train,
        cfg=cfg,
    )


def build_sde(diffusion_cfg):
    if not hasattr(diffusion_cfg, "name"):
        return None, None

    if diffusion_cfg.name == "VPSDE":
        dh = get_diffusion_hyperparams(diffusion_cfg)
        sde = VPSDE(dh["beta_min"], dh["beta_max"], dh["N"])
        return dh, sde

    if diffusion_cfg.name == "VESDE":
        dh = get_diffusion_hyperparams(diffusion_cfg)
        sde = VESDE(dh["sigma_min"], dh["sigma_max"], dh["N"])
        return dh, sde

    return None, None


def train(
    rank,
    num_gpus,
    save_dir,
    diffusion_cfg,
    model_cfg,
    dataset_cfg,
    ckpt_iter,
    n_iters,
    iters_per_ckpt,
    iters_per_logging,
    learning_rate,
    batch_size_per_gpu,
    w_masked_pix,
    on_noisy_masked_melspec,
    name=None,
    cfg=None,
):
    local_path, checkpoint_directory = local_directory(name, model_cfg, diffusion_cfg, save_dir, "checkpoint")

    if rank == 0:
        if not (name is None or name == ""):
            path_log = os.path.join(save_dir, "exp", name, local_path, "logs")
            path_config = os.path.join(save_dir, "exp", name, local_path, "config")
        else:
            path_log = os.path.join(save_dir, "exp", local_path, "logs")
            path_config = os.path.join(save_dir, "exp", local_path, "config")
        Path(path_config).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=path_log)

    if rank == 0:
        config_path = os.path.join(path_config, "config.yaml")
        with open(config_path, "w") as f:
            OmegaConf.save(cfg, f)
        print("Configuration saved")

    diffusion_hyperparams, sde = build_sde(diffusion_cfg)
    if diffusion_hyperparams is None:
        diffusion_hyperparams = get_diffusion_hyperparams(diffusion_cfg)

    collate_fn = CollateFn(
        inputs_params=[
            {"axis": 1, "end_number": "min", "max_length": 1701},
            {"axis": 2, "end_number": "min", "max_length": 1701},
            {"axis": 3, "end_number": 0, "max_length": 16000 * 17},
            {"axis": 4, "end_number": 1, "max_length": 1701},
        ],
        targets_params=[{"axis": 0, "end_number": 1, "max_length": 1701}],
    )
    trainloader = dataloader(dataset_cfg, batch_size=batch_size_per_gpu, num_gpus=num_gpus, collate_fn=collate_fn, split="Train")
    trainloader_test = dataloader(dataset_cfg, batch_size=batch_size_per_gpu, num_gpus=num_gpus, collate_fn=collate_fn, split="Test")
    print("Data loaded")

    builder = ModelBuilder()
    net_diffwave = builder.build_diffwave_model(model_cfg)
    net = AudioVisualModel(net_diffwave).cuda()
    print_size(net, verbose=False)

    criterion = nn.CrossEntropyLoss(reduction="none")

    if num_gpus > 1:
        net = apply_gradient_allreduce(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    if ckpt_iter == "max":
        ckpt_iter = find_max_epoch(checkpoint_directory)
    if ckpt_iter >= 0:
        try:
            model_path = os.path.join(checkpoint_directory, f"{ckpt_iter}.pkl")
            checkpoint = torch.load(model_path, map_location="cpu")
            net.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                optimizer.param_groups[0]["lr"] = learning_rate
            print(f"Successfully loaded model at iteration {ckpt_iter}")
        except Exception as e:
            print(f"Model checkpoint found at iteration {ckpt_iter}, but was not successfully loaded - training from scratch. {e}")
            ckpt_iter = -1
    else:
        print("No valid checkpoint model found - training from scratch.")
        ckpt_iter = -1

    dataset_type = dataset_cfg.dataset_type
    n_iter = ckpt_iter + 1
    while n_iter < n_iters + 1:
        epoch_loss = 0.0
        net.train()
        for data in tqdm(trainloader, desc=f"Train Epoch {n_iter // len(trainloader)}") if rank == 0 else trainloader:
            phoneme_target = None
            phoneme_target_mask = None
            masked_audio_time_mask = None
            if dataset_type == "explosion_speech_inpainting":
                raise NotImplementedError("Explosion phoneme classifier training is not supported in this script.")
            elif dataset_type == "speech_inpainting":
                melspec, *masked_cond, mask = data
                masked_cond = [masked_cond[i].cuda() for i in range(len(masked_cond))]
                melspec, mask = melspec.cuda(), mask.cuda()
                masked_audio_time_mask = None
            elif dataset_type == "speech_inpainting_phoneme_classifier":
                phoneme_target, phoneme_target_mask = data["targets"]
                phoneme_target, phoneme_target_mask = phoneme_target.cuda(), phoneme_target_mask.cuda()
                inputs, inputs_masks = data["inputs"]
                melspec, masked_melspec, masked_audio_time, mask = (
                    inputs[0].cuda(),
                    inputs[1].cuda(),
                    inputs[2].cuda(),
                    inputs[3].cuda(),
                )
                melspec_mask, masked_melspec_mask, masked_audio_time_mask, mask_mask = (
                    inputs_masks[0].cuda(),
                    inputs_masks[1].cuda(),
                    inputs_masks[2].cuda(),
                    inputs_masks[3].cuda(),
                )
                masked_cond = [masked_melspec, masked_audio_time]
            else:
                raise ValueError(f"Unsupported dataset type: {dataset_type}")

            optimizer.zero_grad()
            loss = training_loss(
                net,
                criterion,
                melspec,
                masked_cond,
                mask,
                mask_mask,
                phoneme_target,
                phoneme_target_mask,
                diffusion_hyperparams,
                masked_audio_time_mask=masked_audio_time_mask,
                on_noisy_masked_melspec=on_noisy_masked_melspec,
                w_masked_pix=w_masked_pix,
                sde=sde,
            )
            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.item()
            loss.backward()
            optimizer.step()

            epoch_loss += reduced_loss

            if n_iter % iters_per_logging == 0 and rank == 0:
                print(f"iteration: {n_iter} \tloss: {reduced_loss}")
                writer.add_scalar("train_loss", reduced_loss, n_iter)

            if n_iter % iters_per_ckpt == 0 and rank == 0:
                torch.save(
                    {
                        "model_state_dict": net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    os.path.join(checkpoint_directory, f"{n_iter}.pkl"),
                )
                print(f"model at iteration {n_iter} is saved")

            n_iter += 1

        if rank == 0:
            epoch_loss /= len(trainloader)
            writer.add_scalar("train_loss", epoch_loss, n_iter)

        epoch_loss = 0.0
        net.eval()
        with torch.no_grad():
            for data in tqdm(trainloader_test, desc=f"Test Epoch {n_iter // len(trainloader_test)}") if rank == 0 else trainloader_test:
                if dataset_type == "explosion_speech_inpainting":
                    raise NotImplementedError("Explosion phoneme classifier test is not supported in this script.")
                elif dataset_type == "speech_inpainting":
                    melspec, *masked_cond, mask = data
                    masked_cond = [masked_cond[i].cuda() for i in range(len(masked_cond))]
                    melspec, mask = melspec.cuda(), mask.cuda()
                    masked_audio_time_mask = None
                elif dataset_type == "speech_inpainting_phoneme_classifier":
                    phoneme_target, phoneme_target_mask = data["targets"]
                    phoneme_target, phoneme_target_mask = phoneme_target.cuda(), phoneme_target_mask.cuda()
                    inputs, inputs_masks = data["inputs"]
                    melspec, masked_melspec, masked_audio_time, mask = (
                        inputs[0].cuda(),
                        inputs[1].cuda(),
                        inputs[2].cuda(),
                        inputs[3].cuda(),
                    )
                    melspec_mask, masked_melspec_mask, masked_audio_time_mask, mask_mask = (
                        inputs_masks[0].cuda(),
                        inputs_masks[1].cuda(),
                        inputs_masks[2].cuda(),
                        inputs_masks[3].cuda(),
                    )
                    masked_cond = [masked_melspec, masked_audio_time]
                else:
                    raise ValueError(f"Unsupported dataset type: {dataset_type}")

                loss = test_loss(
                    net,
                    criterion,
                    melspec,
                    masked_cond,
                    mask,
                    mask_mask,
                    phoneme_target,
                    phoneme_target_mask,
                    diffusion_hyperparams,
                    masked_audio_time_mask=masked_audio_time_mask,
                    on_noisy_masked_melspec=on_noisy_masked_melspec,
                    w_masked_pix=w_masked_pix,
                    sde=sde,
                )
                if num_gpus > 1:
                    reduced_loss = reduce_tensor(loss.data, num_gpus).item()
                else:
                    reduced_loss = loss.item()
                epoch_loss += reduced_loss

            if rank == 0:
                epoch_loss /= len(trainloader_test)
                print(f"Test loss: {epoch_loss}")
                writer.add_scalar("test_loss", epoch_loss, n_iter)

    if rank == 0:
        writer.close()


def training_loss(
    net,
    loss_fn,
    melspec,
    masked_cond,
    mask,
    mask_mask,
    phoneme_target,
    phoneme_target_mask,
    diffusion_hyperparams,
    masked_audio_time_mask,
    on_noisy_masked_melspec,
    w_masked_pix=0.7,
    sde=None,
):
    if sde is None:
        _dh = diffusion_hyperparams
        T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]
        B, C, L = melspec.shape
        diffusion_steps = torch.randint(T, size=(B, 1, 1)).cuda()
        z = torch.normal(0, 1, size=melspec.shape).cuda()
        if on_noisy_masked_melspec:
            transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * melspec + torch.sqrt(1 - Alpha_bar[diffusion_steps]) * z
            transformed_X = melspec * torch.unsqueeze(mask, dim=1) + transformed_X * (1 - torch.unsqueeze(mask, dim=1))
        else:
            transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * melspec + torch.sqrt(1 - Alpha_bar[diffusion_steps]) * z
        phoneme_estimated = net(transformed_X, masked_cond, diffusion_steps.view(B, 1), cond_drop_prob=0, mask_padding=masked_audio_time_mask)
    else:
        B = melspec.shape[0]
        device = melspec.device
        eps = 1e-5
        t = torch.rand(B, device=device) * (sde.T - eps) + eps
        z = torch.randn_like(melspec)
        mean, std = sde.marginal_prob(melspec, None, t)
        std_expanded = std[:, None, None]
        x_t = mean + std_expanded * z
        if on_noisy_masked_melspec:
            x_t = melspec * torch.unsqueeze(mask, dim=1) + x_t * (1 - torch.unsqueeze(mask, dim=1))
        phoneme_estimated = net(x_t, masked_cond, t.view(B, 1), cond_drop_prob=0, mask_padding=masked_audio_time_mask)

    loss = loss_fn(phoneme_estimated, phoneme_target)
    loss = loss * phoneme_target_mask
    unmaksed_loss = torch.sum(mask * loss) / torch.sum(mask * mask_mask)
    masked_loss = torch.sum((1 - mask) * loss) / torch.sum((1 - mask) * mask_mask)
    return (1 - w_masked_pix) * unmaksed_loss + w_masked_pix * masked_loss


def test_loss(*args, **kwargs):
    return training_loss(*args, **kwargs)


@hydra.main(version_base=None, config_path="configs/", config_name="phoneme_classifier_config_original")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)

    if not os.path.isdir("exp/"):
        os.makedirs("exp/")
        os.chmod("exp/", 0o775)

    num_gpus = torch.cuda.device_count()
    print(f"there are {num_gpus} gpus")
    train_fn = partial(
        distributed_train,
        num_gpus=num_gpus,
        group_name=time.strftime("%Y%m%d-%H%M%S"),
        cfg=cfg,
    )

    if num_gpus <= 1:
        train_fn(0)
    else:
        mp.set_start_method("spawn")
        processes = []
        for i in range(num_gpus):
            p = mp.Process(target=train_fn, args=(i,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


if __name__ == "__main__":
    main()
