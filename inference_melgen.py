# this file is an adapated version https://github.com/albertfgu/diffwave-sashimi, licensed
# under https://github.com/albertfgu/diffwave-sashimi/blob/master/LICENSE


import os
import time
import warnings
warnings.filterwarnings("ignore")

from functools import partial
import multiprocessing as mp

import matplotlib.image
import numpy as np
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from models.model_builder import ModelBuilder
from models.audiovisual_model import AudioVisualModel
from dataloaders.dataset_lipvoicer import get_dataset
from dataloaders.stft import denormalise_mel
from tqdm import tqdm
from utils import find_max_epoch, print_size, calc_diffusion_hyperparams, local_directory

def sampling(net, diffusion_hyperparams, w_mel_cond, on_masked_melspec, mask, conditions=None):
    """
    Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

    Parameters:
    net (torch network):            the model
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors

    Returns:
    the generated melspec(s) in torch.tensor, shape=size
    """

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T

    # print('begin sampling, total number of reverse steps = %s' % T)
    #This is Algorithm 2 in the paper of classifier-free but with the regular sampler(the one shown in the paper ddpm)
    masked_melspec, masked_audio_time = conditions
    x = torch.normal(0, 1, size=masked_melspec.shape).cuda()
    with torch.no_grad():
        for t in tqdm(range(T-1, -1, -1)):
            diffusion_steps = (t * torch.ones((x.shape[0], 1))).cuda()  # use the corresponding reverse step
            if on_masked_melspec is not None:
                z = torch.normal(0, 1, size=masked_melspec.shape).cuda()
                noisy_masked_melspec = torch.sqrt(Alpha_bar[diffusion_steps.int()]) * masked_melspec + torch.sqrt(1-Alpha_bar[diffusion_steps.int()]) * z
                x = noisy_masked_melspec * mask + x * (1 - mask)
            
            epsilon_theta = net(x, conditions, diffusion_steps, cond_drop_prob=0)   # predict \epsilon according to \epsilon_\theta
            epsilon_theta_uncond = net(x, conditions, diffusion_steps, cond_drop_prob=1)
            epsilon_theta = (1+w_mel_cond) * epsilon_theta - w_mel_cond * epsilon_theta_uncond

            x = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])  # update x_{t-1} to \mu_\theta(x_t)
            if t > 0:
                x = x + Sigma[t] * torch.normal(0, 1, size=x.shape).cuda()  # add the variance term to x_{t-1}
    if on_masked_melspec is not None:
        x = masked_melspec * mask + x * (1 - mask)
    return x


@torch.no_grad()
def generate(
        rank,
        diffusion_cfg,
        model_cfg,
        dataset_cfg,
        save_dir,
        ckpt_iter="max",
        name=None,
        n_samples=None,
        w_mel_cond=0,
        on_masked_melspec=False
    ):
    """
    Generate melspectrograms based on lips movement
    """

    if rank is not None:
        print(f"rank {rank} {torch.cuda.device_count()} GPUs")
        torch.cuda.set_device(rank % torch.cuda.device_count())

    local_path, checkpoint_directory = local_directory(name, model_cfg, diffusion_cfg, save_dir, 'checkpoint')

    # map diffusion hyperparameters to gpu
    diffusion_hyperparams  = calc_diffusion_hyperparams(**diffusion_cfg, fast=True)  # dictionary of all diffusion hyperparameters

    # predefine model
    builder = ModelBuilder()
    # net_lipreading = builder.build_lipreadingnet()
    # net_facial = builder.build_facial(fc_out=128, with_fc=True)
    net_diffwave = builder.build_diffwave_model(model_cfg)
    net = AudioVisualModel(net_diffwave).cuda()
    # net = torch.compile(net)
    # print_size(net)
    net.eval()

    # load checkpoint
    print('ckpt_iter', ckpt_iter)
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(checkpoint_directory)
    ckpt_iter = int(ckpt_iter)

    try:
        model_path = os.path.join(checkpoint_directory, '{}.pkl'.format(ckpt_iter))
        checkpoint = torch.load(model_path, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
        print('Successfully loaded model at iteration {}'.format(ckpt_iter))
    except:
        raise Exception('No valid model found')
    
    dataset = get_dataset(dataset_cfg, split='test', return_mask_properties=False)
    dataset_type = dataset_cfg.dataset_type
    dataset_indices = list(range(n_samples))
    groundtruth_melspec, masked_cond, masks = [], [], []
    for i in dataset_indices:
        
        if dataset_type == 'explosion_speech_inpainting':
                speech_melspec, mix_melspec, mix_time, _, masked_speech_time, explosions_activity, start_explosions, explosions_length = dataset[i]
                mask = 1 - explosions_activity # zero = explosion, one = no explosion
                _mask = mask.cuda()
                _gt_melspec = speech_melspec.cuda()
                mix_melspec, mix_time = mix_melspec.cuda(), mix_time.cuda()
                _masked_cond = [mix_melspec, mix_time]
        elif dataset_type == 'speech_inpainting':
            _gt_melspec, *_masked_cond, _mask = dataset[i]
            _masked_cond = [_masked_cond[i].unsqueeze(0).cuda() for i in range(len(_masked_cond))]
        elif dataset_type == 'plc_task':
            _gt_melspec, masked_melspec, masked_audio_time, frame_mask, sample_mask = dataset[i]
            _mask = frame_mask
            _masked_cond = [masked_melspec.cuda(), masked_audio_time.cuda()]
            _masked_cond = [_masked_cond[i].unsqueeze(0).cuda() for i in range(len(_masked_cond))]
        elif dataset_cfg.dataset_type == 'speech_inpainting_anechoic':
        #melspec, masked_melspec, mask, masked_audio_time
            _gt_melspec, masked_melspec, masked_audio_time, _mask = dataset[i]
            _mask = _mask.unsqueeze(0).cuda()
            _masked_cond = [masked_melspec.cuda(), masked_audio_time.cuda()]
            _masked_cond = [_masked_cond[i].unsqueeze(0) for i in range(len(_masked_cond))]

        # for i in range(len(_masked_cond)):
        #     _masked_cond[i] = _masked_cond[i].unsqueeze(0).cuda()
        _gt_melspec = denormalise_mel(_gt_melspec)
        groundtruth_melspec.append(_gt_melspec.unsqueeze(0))
        masked_cond.append(_masked_cond)
        masks.append(_mask)

    print(f'begin generating melspectrograms | {n_samples} samples')

    # inference
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    generated_melspec = []

    for i in tqdm(range(n_samples)):
        _melspec = sampling(
            net,
            diffusion_hyperparams,
            w_mel_cond,
            conditions=masked_cond[i],
            mask=masks[i],
            on_masked_melspec=on_masked_melspec
        )
        generated_melspec.append(denormalise_mel(_melspec))

    end.record()
    torch.cuda.synchronize()
    print('generated {} samples at iteration {} in {} seconds'.format(n_samples,
        ckpt_iter,
        int(start.elapsed_time(end)/1000)))

    return generated_melspec, groundtruth_melspec, masked_cond


@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    generate(0,
        name=cfg.generate['name'],
        diffusion_cfg=cfg.diffusion,
        model_cfg=cfg.model,
        dataset_cfg=cfg.dataset,
        **cfg.generate,
    )


if __name__ == "__main__":
    main()
