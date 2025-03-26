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
from utils import find_max_epoch, print_size, get_diffusion_hyperparams, local_directory, fix_len_compatibility, pad_last_dim

def sampling(net, diffusion_hyperparams, w_mel_cond, on_masked_melspec, mask, mask_frames=None, masked_audio_time_mask=None, conditions=None, text=None, input_text=None):
    """
    Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

    Parameters:
    net (torch network):            the model
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by get_diffusion_hyperparams
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
            # if t< T-10:
            #     break
            diffusion_steps = (t * torch.ones((x.shape[0], 1))).cuda()  # use the corresponding reverse step
            if on_masked_melspec is not None:
                x = masked_melspec * mask + x * (1 - mask)
            else:
                z = torch.normal(0, 1, size=masked_melspec.shape).cuda()
                noisy_masked_melspec = torch.sqrt(Alpha_bar[diffusion_steps.int()]) * masked_melspec + torch.sqrt(1-Alpha_bar[diffusion_steps.int()]) * z
                x = noisy_masked_melspec * mask + x * (1 - mask)
            epsilon_theta = net(x, conditions, diffusion_steps, cond_drop_prob=0, text=text, input_text=input_text,  mask_padding_time=masked_audio_time_mask, mask_padding_frames=mask_frames)   # predict \epsilon according to \epsilon_\theta
            if net.g_model_cfg.predict_type =='speech':
                epsilon_theta = (x - torch.sqrt(Alpha_bar[diffusion_steps.int()]) * epsilon_theta) / torch.sqrt(1-Alpha_bar[diffusion_steps.int()])
            epsilon_theta_uncond = net(x, conditions, diffusion_steps, cond_drop_prob=1, text=text, input_text=input_text,  mask_padding_time=masked_audio_time_mask, mask_padding_frames=mask_frames)   # predict \epsilon according to \epsilon_\theta
            if net.g_model_cfg.predict_type =='speech':
                epsilon_theta_uncond = (x - torch.sqrt(Alpha_bar[diffusion_steps.int()]) * epsilon_theta_uncond) / torch.sqrt(1-Alpha_bar[diffusion_steps.int()])
            epsilon_theta = (1+w_mel_cond) * epsilon_theta - w_mel_cond * epsilon_theta_uncond

            x = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])  # update x_{t-1} to \mu_\theta(x_t)
            if t > 0:
                x = x + Sigma[t] * torch.normal(0, 1, size=x.shape).cuda()  # add the variance term to x_{t-1}
    if on_masked_melspec is not None:
        x = masked_melspec * mask + x * (1 - mask)
    if mask_frames is not None:
        x = x[..., :int(torch.sum(mask_frames, dim=-1).item())]
    return x


@torch.no_grad()
def generate(
        rank,
        diffusion_cfg,
        model_cfg,
        g_model_cfg,
        dataset_cfg,
        save_dir,
        ckpt_iter="max",
        name=None,
        n_samples=None,
        w_mel_cond=0,
        on_masked_melspec=False,
    ):
    """
    Generate melspectrograms based on lips movement
    """

    if rank is not None:
        print(f"rank {rank} {torch.cuda.device_count()} GPUs")
        torch.cuda.set_device(rank % torch.cuda.device_count())

    local_path, checkpoint_directory = local_directory(name, model_cfg, diffusion_cfg, save_dir, 'checkpoint')

    # map diffusion hyperparameters to gpu
    # diffusion_hyperparams  = calc_diffusion_hyperparams(**diffusion_cfg, fast=True)  # dictionary of all diffusion hyperparameters
    diffusion_hyperparams = get_diffusion_hyperparams(diffusion_cfg, fast=True)
    # predefine model
    builder = ModelBuilder()
    # net_lipreading = builder.build_lipreadingnet()
    # net_facial = builder.build_facial(fc_out=128, with_fc=True)
    net_diffwave = builder.build_model(model_cfg)
    net = AudioVisualModel(g_model_cfg, net_diffwave).cuda()
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
        model_weights = checkpoint['model_state_dict']
        model_weights = {k: v for k, v in model_weights.items() if 'wavlm_model' not in k}
        missing_keys , _ = net.load_state_dict(model_weights, strict=False)
        filtered_missing_keys = [key for key in missing_keys if 'wavlm_model' not in key]
        if not filtered_missing_keys:
            print('All keys loaded successfully')
            print('Successfully loaded model at iteration {}'.format(ckpt_iter))
        else:
            raise Exception(f'The following keys were not loaded: {filtered_missing_keys}')

        
    except:
        raise Exception('No valid model found')
    
    dataset = get_dataset(dataset_cfg, split='test', return_mask_properties=False, return_true_text=True)
    dataset_type = dataset_cfg.dataset_type
    dataset_indices = list(range(n_samples))
    groundtruth_melspec, masked_cond, masks, mask_frames_list, text_list, input_text_list, masked_audio_time_mask_list = [], [], [], [], [], [], []
    for i in dataset_indices:
        text = None
        input_text = None
        if dataset_type == 'explosion_speech_inpainting':
                speech_melspec, mix_melspec, mix_time, _, masked_speech_time, explosions_activity, start_explosions, explosions_length = dataset[i]
                mask = 1 - explosions_activity # zero = explosion, one = no explosion
                _mask = mask.cuda()
                _gt_melspec = speech_melspec.cuda()
                mix_melspec, mix_time = mix_melspec.cuda(), mix_time.cuda()
                _masked_cond = [mix_melspec, mix_time]
        elif dataset_type == 'speech_inpainting':
            _gt_melspec, *_masked_cond, _mask = dataset[i]
            mask = _mask.unsqueeze(0).cuda()
            _masked_cond = [_masked_cond[i].unsqueeze(0).cuda() for i in range(len(_masked_cond))]
        elif dataset_type == 'plc_task':
            _gt_melspec, masked_melspec, masked_audio_time, frame_mask, sample_mask = dataset[i]
            _mask = frame_mask
            _masked_cond = [masked_melspec.cuda(), masked_audio_time.cuda()]
            _masked_cond = [_masked_cond[i].unsqueeze(0).cuda() for i in range(len(_masked_cond))]
        elif dataset_cfg.dataset_type == 'speech_inpainting_anechoic':
        #melspec, masked_melspec, mask, masked_audio_time
            if model_cfg.text_embed_prop.use_text_embed_rep or model_cfg.tts_kw.use_tts:
                _gt_melspec, masked_melspec, masked_audio_time, _mask, text, input_text = dataset[i]
                input_text = [input_text]
            else:
                _gt_melspec, masked_melspec, masked_audio_time, _mask, text = dataset[i]
            _mask = _mask.unsqueeze(0).cuda()
            _masked_cond = [masked_melspec.cuda(), masked_audio_time.cuda()]
            _masked_cond = [_masked_cond[i].unsqueeze(0) for i in range(len(_masked_cond))]

        
        #For Unet we need to fix the length of the input to be divided with 4 (2**2)
        if model_cfg._name_ == 'unet':
            freq_siganl, time_signal = _masked_cond
            desired_num_frames = fix_len_compatibility(_gt_melspec.shape[-1])
            masked_audio_time_mask = torch.ones_like(time_signal)
            masked_audio_time_mask = pad_last_dim(masked_audio_time_mask, (desired_num_frames - freq_siganl.shape[-1]) * dataset_cfg[dataset_type]["audio_stft_hop"])
            _gt_melspec = pad_last_dim(_gt_melspec, desired_num_frames - _gt_melspec.shape[-1])
            time_signal = pad_last_dim(time_signal, (desired_num_frames - freq_siganl.shape[-1]) * dataset_cfg[dataset_type]["audio_stft_hop"])
            freq_siganl = pad_last_dim(freq_siganl, desired_num_frames - freq_siganl.shape[-1]).cuda()
            mask_frames = torch.zeros((list(_mask.shape[:-1]) + [desired_num_frames]))
            mask_frames[..., :_mask.shape[-1]] = 1
            mask_frames = mask_frames.cuda()
            _mask =pad_last_dim(_mask, desired_num_frames - _mask.shape[-1], pad_value=1)
            _masked_cond = [freq_siganl, time_signal]
            
        else:
            mask_frames = None
            masked_audio_time_mask = None

        # for i in range(len(_masked_cond)):
        #     _masked_cond[i] = _masked_cond[i].unsqueeze(0).cuda()
        _gt_melspec = denormalise_mel(_gt_melspec)
        groundtruth_melspec.append(_gt_melspec.unsqueeze(0))
        masked_cond.append(_masked_cond)
        masks.append(_mask)
        mask_frames_list.append(mask_frames)
        text_list.append(text)
        input_text_list.append(input_text)
        masked_audio_time_mask_list.append(masked_audio_time_mask)
    
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
            on_masked_melspec=on_masked_melspec,
            mask_frames=mask_frames_list[i],
            masked_audio_time_mask=masked_audio_time_mask_list[i],
            text=text_list[i],
            input_text=input_text_list[i]
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
        g_model_cfg = cfg.g_model,
        dataset_cfg=cfg.dataset,
        **cfg.generate,
    )


if __name__ == "__main__":
    main()
