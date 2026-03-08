# this file is an adapated version https://github.com/albertfgu/diffwave-sashimi, licensed
# under https://github.com/albertfgu/diffwave-sashimi/blob/master/LICENSE


import os
import time
import warnings
warnings.filterwarnings("ignore")

from functools import partial
import multiprocessing as mp

import matplotlib.image
import torchaudio
import numpy as np
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from models.model_builder import ModelBuilder
from models.audiovisual_model import AudioVisualModel
from dataloaders.dataset_lipvoicer import get_dataset
from dataloaders.stft import denormalise_mel, normalise_mel
from tqdm import tqdm
from utils import find_max_epoch, print_size, calc_diffusion_hyperparams, local_directory
from dataloaders.wav2mel import STFT, load_wav_to_torch
import soundfile as sf
from hifi_gan.generator import Generator as Vocoder
from hifi_gan.env import AttrDict
from hifi_gan import utils as vocoder_utils
import json
from collections import Counter
from utils import zero_regions_mask, samples2frames, insert_random_values



def sampling(net, diffusion_hyperparams, w_mel_cond, conditions=None, output_directory=None):
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
            epsilon_theta = net(x, conditions, diffusion_steps, cond_drop_prob=0)   # predict \epsilon according to \epsilon_\theta
            epsilon_theta_uncond = net(x, conditions, diffusion_steps, cond_drop_prob=1)
            epsilon_theta = (1+w_mel_cond) * epsilon_theta - w_mel_cond * epsilon_theta_uncond

            x = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])  # update x_{t-1} to \mu_\theta(x_t)
            if t > 0:
                x = x + Sigma[t] * torch.normal(0, 1, size=x.shape).cuda()  # add the variance term to x_{t-1}
            if t % 50 == 0:
                matplotlib.image.imsave(os.path.join(output_directory, f'generated_melspec_{t}.png'), x[0].cpu().numpy()[::-1])
            # if t < T -5:
            #     break
    return x


@torch.no_grad()
def generate(
        rank,
        diffusion_cfg,
        model_cfg,
        audio_cfg,
        ckpt_path,
        w_mel_cond=0,
        w_asr=1.1,
        asr_start=250,
        save_dir=None,
        n_samples_test = 20,
        inference_mel_only_name_dir='generated_mels',
        without_condtion=False,
        config_filename_asr_cond=None,
        apply_asr_guidance=False,
        lipread_text_dir=None,
        on_noisy_masked_melspec=False,
        **kwargs
    ):
    """
    Generate melspectrograms based on lips movement
    """

    if rank is not None:
        print(f"rank {rank} {torch.cuda.device_count()} GPUs")
        torch.cuda.set_device(rank % torch.cuda.device_count())



    # map diffusion hyperparameters to gpu
    diffusion_hyperparams  = calc_diffusion_hyperparams(**diffusion_cfg, fast=True)  # dictionary of all diffusion hyperparameters

    stft = STFT(**audio_cfg)
    

    print('Load HiFi-GAN')
    config_file = 'hifi_gan/config.json'
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    vocoder = Vocoder(h).cuda()
    checkpoint_file = '/dsi/gannot-lab1/users/mordehay/hifi_gan/g_02400000'
    state_dict_g = vocoder_utils.load_checkpoint(checkpoint_file, 'cuda')
    vocoder.load_state_dict(state_dict_g['generator'])
    vocoder.eval()
    vocoder.remove_weight_norm()
    print('Finish Loading HiFi-GAN')
    
    
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
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
        print('Successfully loaded MelGen checkpoint')
    except Exception as e:
        print(e)
        raise Exception('No valid model found')
    
    

    groundtruth_melspec, masked_cond = [], []
    output_directory = os.path.join(save_dir, inference_mel_only_name_dir)
    guidance_dir_name = f'w1={w_mel_cond}'
    guidance_dir_name += f'_w2={w_asr}_asr_start={asr_start}'
    guidance_dir_name += f'_mask={on_noisy_masked_melspec}'
    _output_directory = os.path.join(output_directory, guidance_dir_name)
    path_audio = ["/dsi/gannot-lab1/datasets/lossy_audio/sample_0.wav"]
    n_samples = len(path_audio)
    dataset_indices = torch.arange(n_samples)
    for i in dataset_indices:
        audio, sr = load_wav_to_torch(path_audio[i])
        
        # Define resampling parameters
        target_sr = 16000
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            audio = resampler(audio)
            sr = target_sr  # Update the sample rate
        audio = audio / torch.abs(audio).max()
        melspectrogram = stft.get_mel(audio)
        melspectrogram = normalise_mel(melspectrogram)
        mask_time = zero_regions_mask(audio.numpy(), min_length=1000)
        mask_time_int = mask_time.astype(int)
        frame_mask = samples2frames(mask_time_int, audio_cfg.filter_length, audio_cfg.hop_length)
        masked_melspectrogram = insert_random_values(melspectrogram.numpy(), frame_mask)
        matplotlib.image.imsave(os.path.join(output_directory, 'masked_melspectrogram_with_noise.png'), masked_melspectrogram[::-1])
        masked_melspectrogram = torch.from_numpy(masked_melspectrogram)
        _masked_cond = [masked_melspectrogram, audio]
        _masked_cond = [_masked_cond[i].unsqueeze(0).cuda() for i in range(len(_masked_cond))]


        masked_cond.append(_masked_cond)


    print(f'begin generating melspectrograms | {n_samples} samples')

    # inference
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    generated_melspec = []

    
    
    
    for i in tqdm(range(n_samples)):
        os.makedirs(os.path.join(_output_directory, f'sample_{i}'), exist_ok=True)
        _melspec = sampling(
            net,
            diffusion_hyperparams,
            w_mel_cond,
            conditions=masked_cond[i],
            output_directory=os.path.join(_output_directory, f'sample_{i}')
        )
        
        
        print("saving to output directory", _output_directory)
        est_X = denormalise_mel(_melspec)
        est_audio = vocoder(est_X)
        est_audio = est_audio.squeeze()
        est_audio = est_audio / 1.1 / est_audio.abs().max()
        est_audio = est_audio.cpu().numpy()
        sf.write(os.path.join(_output_directory, f'sample_{i}', f'generated_audio.wav'), est_audio, 16000)
        sf.write(os.path.join(_output_directory, f'sample_{i}', f'masked_audio.wav'), masked_cond[i][1][0].cpu().numpy(), 16000)
        
        est_X = denormalise_mel(melspectrogram)
        est_audio = vocoder(est_X)
        est_audio = est_audio.squeeze()
        est_audio = est_audio / 1.1 / est_audio.abs().max()
        est_audio = est_audio.cpu().numpy()
        sf.write(os.path.join(_output_directory, f'sample_{i}', f'masked_audio_hifigan.wav'), est_audio, 16000)
        generated_melspec.append(denormalise_mel(_melspec))

    end.record()
    torch.cuda.synchronize()
    print('generated {} samples at iteration {} in {} seconds'.format(n_samples,
        i,
        int(start.elapsed_time(end)/1000)))

    return generated_melspec, groundtruth_melspec, masked_cond


@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    generate(0,
        diffusion_cfg=cfg.diffusion,
        model_cfg=cfg.melgen,
        audio_cfg=cfg.audio,
        **cfg.generate,
    )


if __name__ == "__main__":
    main()
