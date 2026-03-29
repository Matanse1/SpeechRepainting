# this file is an adapated version https://github.com/albertfgu/diffwave-sashimi, licensed
# under https://github.com/albertfgu/diffwave-sashimi/blob/master/LICENSE

# this is the full test without asr: mel condition, free-classifier and vocoder(mel2audio)
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import subprocess
import time
import warnings
from itertools import product
import random
warnings.filterwarnings("ignore")

# from functools import partialks
# import multiprocessing as mp

import soundfile as sf
import matplotlib.image
import numpy as np
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from models.model_builder import ModelBuilder
from models.audiovisual_model import AudioVisualModel
# import ASR.asr_models as asr_models
from dataloaders.dataset_lipvoicer import get_dataset
from dataloaders.stft import denormalise_mel
from hifi_gan.generator import Generator as Vocoder
from BigVGAN.bigvgan import BigVGAN as Generator
from BigVGAN.inference_e2e import load_checkpoint as load_checkpoint_vgan
from hifi_gan import utils as vocoder_utils
from hifi_gan.env import AttrDict
from utils import find_max_epoch, print_size, get_diffusion_hyperparams,  local_directory, preprocess_text, fix_len_compatibility, pad_last_dim # ,linear_t_given_cosine_t
import csv
from mouthroi_processing.pipelines.pipeline import InferencePipeline
from scipy.io.wavfile import write
import tempfile
# from my_utils.compute_metrics import Metrics
import ASR.asr_models as asr_models

from utils import mask_time_all_frequencies_mask, mask_time_specific_frequencies_mask, mask_specific_frequencies_all_time_mask, mask_combined_mask, mask_with_shape_mask
print("finished imports")


def get_g2p_pipeline(g2p_model, with_space=False):
    p2n = '/home/dsi/moradim/SpeechRepainting/phoneme_to_number.json'
    with open(p2n, 'r') as f:
        valid_chars = json.load(f)
        valid_chars = list(valid_chars.keys())
    def g2p(text):
        phonemes = g2p_model(text)
        processed_list = []
        for item in phonemes:
            if item == ' ':
                if with_space:
                    processed_list.append('space')
                # else: don't append, effectively removing it
            elif item in valid_chars:
                processed_list.append(item)
            # else: don't append, removing invalid char.
        return processed_list
    return g2p


def get_phones_dict(file_path):
    phoneme_dict_p2d = {}
    phoneme_dict_d2p = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split()
            phoneme_dict_p2d[key] = int(value) #phone to digit
            phoneme_dict_d2p[int(value)] = key #digit to phone
    return phoneme_dict_p2d, phoneme_dict_d2p







# ------ the training loss DDPM ---------------------------------------------

# def training_loss(net, loss_fn, melspec, masked_melspec, mask, diffusion_hyperparams, w_masked_pix=0.8, mask_frames=None, masked_audio_time_mask=None, text=None, input_text=None,  mask_padding_time=None):
#     """
#     Compute the training loss of epsilon and epsilon_theta

#     Parameters:
#     net (torch network):            the wavenet model
#     loss_fn (torch loss function):  the loss function, default is nn.MSELoss()
#     X (torch.tensor):               training data, shape=(batchsize, 1, length of audio)
#     diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
#                                     note, the tensors need to be cuda tensors

#     Returns:
#     training loss
#     """
#     # Predict melspectrogram from visual features using diffusion model
    
#     _dh = diffusion_hyperparams
#     T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]
#     # This is Algorithm 1 in the paper of classifier-free
#     B, C, L = melspec.shape  # B is batchsize, C=80, L is number of melspec frames
#     diffusion_steps = torch.randint(T, size=(B,1,1)).cuda()  # randomly sample diffusion steps from 1~T
#     z = torch.normal(0, 1, size=melspec.shape).cuda()
#     transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * melspec + torch.sqrt(1-Alpha_bar[diffusion_steps]) * z  # training from Denoising Diffusion Probabilistic Models paper compute x_t from q(x_t|x_0)
#     cond_drop_prob = 0.2
#     epsilon_theta = net(transformed_X, masked_melspec, diffusion_steps.view(B,1), cond_drop_prob, mask_padding_frames=mask_frames, text=text, input_text=input_text,  mask_padding_time=masked_audio_time_mask)
#     if net.g_model_cfg.predict_type =='speech':
#         epsilon_theta = (transformed_X - torch.sqrt(Alpha_bar[diffusion_steps]) * epsilon_theta) / torch.sqrt(1-Alpha_bar[diffusion_steps])
#     loss = loss_fn(epsilon_theta, z) #[B, F, T]
#     mean_loss = round(torch.mean(loss).item(), 2)
#     unmaksed_loss = torch.sum(mask * loss) / torch.sum(mask)
#     masked_loss = torch.sum((1-mask) * loss) / torch.sum(1-mask)
#     weighted_loss = (1 - w_masked_pix) * unmaksed_loss + w_masked_pix * masked_loss
#     est_X = (transformed_X - torch.sqrt(1-Alpha_bar[diffusion_steps]) * epsilon_theta) / torch.sqrt(Alpha_bar[diffusion_steps])
#     return weighted_loss, est_X, transformed_X, diffusion_steps, mean_loss



def sampling(net, diffusion_cfg, diffusion_hyperparams,
            w_mel_cond, condition=None, 
            asr_guidance_net=None,
            w_asr=None,
            asr_start=None,
            guidance_text=None,
            tokenizer=None,
            decoder=None,
            without_condtion=False,
            mask=None,
            on_noisy_masked_melspec=False,
            mask_frames=None,
            masked_audio_time_mask=None,
            text=None, 
            input_text=None,
            phoneme4guidance=None,
            per_frame_phoneme4guidance=None,
            type_input_guidance='text',
            skip_step=1,
            ):
    """
    Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

    Parameters:
    net (torch network):            the model
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors

    Returns:
    the generated melspec(s) in torch.tensor, shape=size
    """
    # Part 1: calculating hyperparameters, tokenizing text, sampling noise (X_T)
    # _dh = diffusion_hyperparams
    # T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    # assert len(Alpha) == T
    # assert len(Alpha_bar) == T
    # assert len(Sigma) == T
    preds_ao = 'None'

    # torch.set_anomaly_enabled(True)
    # tokenize text
    if asr_guidance_net is not None:
        if type_input_guidance == 'text':
            tokens = torch.LongTensor(tokenizer.encode(guidance_text))
            tokens = tokens.unsqueeze(0).cuda()
        elif type_input_guidance == 'phoneme':
            tokens = torch.tensor([tokenizer[t] for t in phoneme4guidance[0]]).unsqueeze(0).cuda()
        elif type_input_guidance == 'frame_level_phoneme':
            tokens = torch.tensor(per_frame_phoneme4guidance[0], dtype=torch.int64).unsqueeze(0).cuda()
            loss_ce = nn.CrossEntropyLoss(reduction='none')

    masked_melspec, _ = condition
    # This is Algorithm 1 in the paper of classifier-free
    B, C, L = masked_melspec.shape  # B is batchsize, C=80, L is number of melspec frames
    mask = mask.cuda()  
    generator = torch.Generator().manual_seed(42)
    preds_ao = 'None'

    # # ── DDPM path (original, unchanged) ──────────────────────────
    # if _dh["name"] in ["linear", "cosine"]:
    #     x = torch.normal(0, 1, size=masked_melspec.shape, generator=generator).cuda()
    #     repeat_asr = 1
    #     asr_finish = -1
    #     add_noise = True
    #     start_noise_time = 50 
    #     tho = 1

    #     # part 2: statrting the sampling loop and stiching the melspec together according to the mask
    #     with torch.no_grad():
    #         for t in tqdm(range(T - 1, -1, -skip_step)):
    #             if t < skip_step:  # Ensure the last step is exactly zero
    #                 t = 0
    #             if _dh["name"] == "cosine":
    #                 t_linear_asr = linear_t_given_cosine_t(t)
    #             else:
    #                 t_linear_asr = t
                    
    #             diffusion_steps_asr_linear = (t_linear_asr * torch.ones((x.shape[0], 1))).cuda()  # use the corresponding reverse step
    #             diffusion_steps = (t * torch.ones((x.shape[0], 1))).cuda()  # use the corresponding reverse step
    #             # if asr_guidance_net is not None and (t <= asr_start and t > asr_finish):
    #                 # repeat_asr = 1
    #             # for _ in range(repeat_asr):
    #             if on_noisy_masked_melspec:
    #                 z = torch.normal(0, 1, size=masked_melspec.shape, generator=generator).cuda()
    #                 noisy_masked_melspec = torch.sqrt(Alpha_bar[diffusion_steps.int()]) * masked_melspec + torch.sqrt(1-Alpha_bar[diffusion_steps.int()]) * z
    #                 x = noisy_masked_melspec * mask + x * (1 - mask)
                    
    #             # part 3: get the noise from the model, with or without condition, and with or without asr guidance.
    #             ''' what is the fucking condition?'''
    #             # what is the fucking condition?  
    #             if without_condtion:
    #                 epsilon_theta_mel = net(x, condition, diffusion_steps, cond_drop_prob=1, mask_padding_frames=mask_frames, text=text, input_text=input_text,  mask_padding_time=masked_audio_time_mask)
    #                 if net.g_model_cfg.predict_type =='speech':
    #                     epsilon_theta_mel = (x - torch.sqrt(Alpha_bar[diffusion_steps.int()]) * epsilon_theta_mel) / torch.sqrt(1-Alpha_bar[diffusion_steps.int()])
    #             else:
    #                 epsilon_theta_cond = net(x, condition, diffusion_steps, cond_drop_prob=0, mask_padding_frames=mask_frames, text=text, input_text=input_text,  mask_padding_time=masked_audio_time_mask)   # predict \epsilon according to \epsilon_\theta
    #                 if net.g_model_cfg.predict_type =='speech':
    #                     epsilon_theta_cond = (x - torch.sqrt(Alpha_bar[diffusion_steps.int()]) * epsilon_theta_cond) / torch.sqrt(1-Alpha_bar[diffusion_steps.int()])
    #                 epsilon_theta_uncond = net(x, condition, diffusion_steps, cond_drop_prob=1, mask_padding_frames=mask_frames, text=text, input_text=input_text,  mask_padding_time=masked_audio_time_mask)
    #                 if net.g_model_cfg.predict_type =='speech':
    #                     epsilon_theta_uncond = (x - torch.sqrt(Alpha_bar[diffusion_steps.int()]) * epsilon_theta_uncond) / torch.sqrt(1-Alpha_bar[diffusion_steps.int()])
    #                 epsilon_theta_mel = (1+w_mel_cond) * epsilon_theta_cond - w_mel_cond * epsilon_theta_uncond
                
    #             # part 4: apply ASR guidance or not
    #             if (asr_guidance_net is not None) and (t <= asr_start and t > asr_finish):
    #                 for r in range(repeat_asr):
    #                     if r > 1:
    #                         if on_noisy_masked_melspec:
    #                             z = torch.normal(0, 1, size=masked_melspec.shape, generator=generator).cuda()
    #                             noisy_masked_melspec = torch.sqrt(Alpha_bar[diffusion_steps_asr_linear.int()]) * masked_melspec + torch.sqrt(1-Alpha_bar[diffusion_steps_asr_linear.int()]) * z
    #                             x = noisy_masked_melspec * mask + x * (1 - mask)
    #                     with torch.enable_grad():
    #                         # torch.set_anomaly_enabled(True)
    #                         if type_input_guidance == 'text' or type_input_guidance == 'phoneme':
    #                             length_input = torch.tensor([x.shape[2]]).cuda()
    #                             inputs = x.detach().requires_grad_(True), length_input
    #                             targets = tokens, torch.tensor([tokens.shape[1]]).cuda()
    #                             asr_guidance_net.device = torch.device("cuda")
    #                             batch_losses = asr_guidance_net.forward_model(inputs, diffusion_steps_asr_linear, targets, compute_metrics=False, verbose=0)[0] #batch_losses, batch_metrics, batch_truths, batch_preds
    #                             asr_grad = torch.autograd.grad(batch_losses["loss"], inputs[0])[0]
    #                         elif type_input_guidance == 'frame_level_phoneme':
    #                             masked_melspec, audio_time_masked = condition
    #                             masked_melspec_noisy = masked_melspec.clone()
    #                             z = torch.normal(0, 1, size=masked_melspec.shape, generator=generator).cuda()
    #                             masked_melspec_noisy = masked_melspec * mask + z * (1 - mask)
    #                             condition2 = masked_melspec_noisy, audio_time_masked
    #                             inputs = x.detach().requires_grad_(True)
    #                             outputs = asr_guidance_net(inputs, condition2, diffusion_steps_asr_linear, cond_drop_prob=0, mask_padding_time=masked_audio_time_mask)
    #                             l_ce = loss_ce(outputs, tokens)
    #                             l_ce = torch.sum(l_ce * (1-mask)) / torch.sum(1-mask)
    #                             asr_grad = torch.autograd.grad(l_ce, inputs)[0]
                                
    #                         if torch.sum(asr_grad) == 0 and type_input_guidance == 'phoneme':
    #                             print("Zero grad")
    #                             grad_normaliser = 0
    #                         else:
    #                             grad_normaliser = torch.norm(epsilon_theta_mel / torch.sqrt(1 - Alpha_bar[t])) / torch.norm(asr_grad)
    #                         # if torch.isnan(asr_grad).any():
    #                         #     print("NAN in asr_grad")
    #                         asr_guidance_net.device = torch.device("cpu")
    #                     num_tokens = tokens.shape[1]
    #                     epsilon_theta = epsilon_theta_mel + torch.sqrt(1 - Alpha_bar[t]) * w_asr * grad_normaliser * asr_grad #the asr_grad include the minus
    #                     # print(f"grad_normaliser: {grad_normaliser} \t w_asr: {w_asr} \t grad: {torch.norm(asr_grad)} \t 1-alpha_bar: {torch.sqrt(1 - Alpha_bar[t])}")
    #                     # part 5: update x according to the diffusion formula.
    #                     x = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])  # update x_{t-1} to \mu_\theta(x_t)
    #                     if t > 0 and add_noise:
    #                         x = x + tho * Sigma[t] * torch.normal(0, 1, size=x.shape, generator=generator).cuda()  # add the variance term to x_{t-1}
    #             else:
    #             # part 5: update x according to the diffusion formula without ASR guidance.
    #             # THIS IS THE INTERESTING PART ________________________________________________________________________________________________________________________
    #                 x = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * epsilon_theta_mel) / torch.sqrt(Alpha[t])  # update x_{t-1} to \mu_\theta(x_t)
    #                 if t > 0 and add_noise:
    #                     x = x + tho *  Sigma[t] * torch.normal(0, 1, size=x.shape, generator=generator).cuda()  # add the variance term to x_{t-1}
    #             # x = x.clip(-1, 1.5)
    #             if t < start_noise_time:
    #                 add_noise = True
                
    #             if t % 10 == 0:
    #                 if (asr_guidance_net is not None) and (t <= asr_start) and (type_input_guidance != 'frame_level_phoneme'):
    #                     inputs = x, length_input
    #                     outputs_ao = asr_guidance_net(inputs, diffusion_steps_asr_linear)["outputs"]
    #                     preds_ao = decoder(outputs_ao)[0]
    #                     print(preds_ao)

    # ── SDE path (new) ────────────────────────────────────────────
    _dh = diffusion_hyperparams
    if _dh["name"] in ["VPSDE", "VESDE"]:
        from SDE import VPSDE, VESDE
        from sampling import get_pc_sampler
        loss_ce = nn.CrossEntropyLoss(reduction='none')

        if _dh["name"] == "VPSDE":
            sde = VPSDE(_dh["beta_min"], _dh["beta_max"], _dh["N"])
        else:
            sde = VESDE(_dh["sigma_min"], _dh["sigma_max"], _dh["N"])

        # ── score_fn: CFG  ────────────────────
        def score_fn_without_asr(x, t):
            # Ensure input is 3D [Batch, Mel, Time] if SDE/Guidance added a channel dim
            if x.ndim == 4:
                x = x.squeeze(1)

            B = x.shape[0]
            t_input = t.view(B, 1)

            if without_condtion:
                score = net(
                    x, condition, t_input, cond_drop_prob=1,
                    mask_padding_frames=mask_frames, text=text,
                    input_text=input_text,
                    mask_padding_time=masked_audio_time_mask
                )
            else:
                score_cond = net(
                    x, condition, t_input, cond_drop_prob=0,
                    mask_padding_frames=mask_frames, text=text,
                    input_text=input_text,
                    mask_padding_time=masked_audio_time_mask
                )
                score_uncond = net(
                    x, condition, t_input, cond_drop_prob=1,
                    mask_padding_frames=mask_frames, text=text,
                    input_text=input_text,
                    mask_padding_time=masked_audio_time_mask
                )
                score = (1 + w_mel_cond) * score_cond - w_mel_cond * score_uncond

            return score

        # ── asr_guidance_fn ───────────────────────────────────────
        def asr_guidance_fn(x, y, t):
            # Ensure input is 3D [Batch, Mel, Time] to avoid dimension mismatch in ASR
            if x.ndim == 4:
                x = x.squeeze(1)
            if asr_guidance_net is None:
                return torch.zeros_like(x)
            asr_start_val = asr_start[0] if isinstance(asr_start, list) else asr_start
            if not (t[0].item() <= asr_start_val):
                return torch.zeros_like(x)

            with torch.no_grad():
                score = score_fn_without_asr(x, t)
                grad_norm = torch.norm(score.reshape(score.shape[0], -1), dim=-1).mean()

            with torch.enable_grad():
                if type_input_guidance in ['text', 'phoneme']:
                    length_input = torch.tensor([x.shape[2]]).cuda()
                    inputs = x.detach().requires_grad_(True), length_input
                    targets = tokens, torch.tensor([tokens.shape[1]]).cuda()
                    asr_guidance_net.device = torch.device("cuda")
                    batch_losses = asr_guidance_net.forward_model(inputs, t.view(x.shape[0], 1), targets, compute_metrics=False, verbose=0)[0]  # batch_losses, batch_metrics, batch_truths, batch_preds
                    asr_grad = torch.autograd.grad(batch_losses["loss"], inputs[0])[0]
                elif type_input_guidance == 'frame_level_phoneme':
                    masked_melspec, audio_time_masked = condition
                    masked_melspec_noisy = masked_melspec.clone()
                    z = torch.normal(0, 1, size=masked_melspec.shape).cuda()
                    masked_melspec_noisy = masked_melspec * mask + z * (1 - mask)
                    condition2 = masked_melspec_noisy, audio_time_masked
                    inputs = x.detach().requires_grad_(True)
                    outputs = asr_guidance_net(inputs, condition2, t.view(x.shape[0], 1), cond_drop_prob=0, mask_padding_time=masked_audio_time_mask)
                    l_ce = loss_ce(outputs, tokens)
                    l_ce = torch.sum(l_ce * (1 - mask)) / torch.sum(1 - mask)
                    asr_grad = torch.autograd.grad(l_ce, inputs)[0]

                 
                asr_grad_norm = torch.norm(asr_grad.reshape(asr_grad.shape[0], -1), dim=-1).mean()
                grad_normaliser = grad_norm / (asr_grad_norm + 1e-8)
                asr_guidance_net.device = torch.device("cpu")

            return grad_normaliser * asr_grad

        def score_fn(x, y, t):
            score = score_fn_without_asr(x, t)
            guidance = asr_guidance_fn(x, y, t)
            return score + w_asr * guidance

        # --- call PC sampler with dynamic config parameters ---
        pc_sampler = get_pc_sampler(
            # Fetch predictor and corrector names from diffusion_cfg, default to standard if missing
            predictor_name=diffusion_cfg.get('predictor', "reverse_diffusion"), # Options: "reverse_diffusion", "none"
            corrector_name=diffusion_cfg.get('corrector', "langevin"),          # Options: "langevin", "ald", "none"
            sde=sde,
            score_fn=score_fn,
            y=masked_melspec,
            # Set SNR (Signal-to-Noise Ratio) for Langevin or ALD steps
            snr=diffusion_cfg.get('snr', 0.1),
            # Number of corrector iterations to run per diffusion time-step
            corrector_steps=diffusion_cfg.get('corrector_steps', 1),
            w_mel_cond=w_mel_cond,
            mask=mask,
            mask_noise=on_noisy_masked_melspec
            )

        x, nfe = pc_sampler()
        print(f"PC sampler finished in {nfe} function evaluations")

        # ── final masking (same as original) ─────────────────────
        x = masked_melspec * mask + x * (1 - mask)
        if mask_frames is not None:
            x = x[..., :int(torch.sum(mask_frames, dim=-1).item())]
        return x, preds_ao


@torch.no_grad()
def generate(
        rank,
        diffusion_cfg,
        model_cfg,
        g_model_cfg,
        dataset_cfg,
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
        type_input_guidance = 'text',
        lipread_text_dir=None,
        on_noisy_masked_melspec=False,
        mask_info=None,
        mel_text=None,
        with_space=False,
        skip_step=1,
        **kwargs
    ):

    if rank is not None:
        print(f"rank {rank} {torch.cuda.device_count()} GPUs")
        torch.cuda.set_device(rank % torch.cuda.device_count())

    # map diffusion hyperparameters to gpu
    # diffusion_hyperparams  = calc_diffusion_hyperparams(**diffusion_cfg, fast=True)  # dictionary of all diffusion hyperparameters
    diffusion_hyperparams = get_diffusion_hyperparams(diffusion_cfg, fast=True)
    # predefine MelGen model
    builder = ModelBuilder()
    net_diffwave = builder.build_model(model_cfg)
    net = AudioVisualModel(g_model_cfg, net_diffwave).cuda()
    print_size(net)
    net.eval()

    # load checkpoint
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model_weights = checkpoint['model_state_dict']
        model_weights = {k: v for k, v in model_weights.items() if 'wavlm_model' not in k}
        missing_keys , _ = net.load_state_dict(model_weights, strict=False)
        filtered_missing_keys = [key for key in missing_keys if 'wavlm_model' not in key]
        if not filtered_missing_keys:
            print('All keys loaded successfully')
            print('Successfully loaded MelGen checkpoint')
        else:
            raise Exception(f'The following keys were not loaded: {filtered_missing_keys}')
    except Exception as e:
        print(e)
        raise Exception('No valid model found')
        

    if save_dir is None:
        save_dir = os.getcwd()
    output_directory = os.path.join(save_dir, inference_mel_only_name_dir)
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        print("saving to output directory", output_directory)

    # print('Loading ASR, tokenizer and decoder')
    ds_name = 'LRS3' # 'LRS2'
    if apply_asr_guidance:
        if type_input_guidance == 'frame_level_phoneme':
            phoneme_dict_path = "/home/dsi/moradim/SpeechRepainting/phones.txt"
            phoneme_dict_p2d, phoneme_dict_d2p = get_phones_dict(phoneme_dict_path)

            # predefine MelGen model
            cfg = OmegaConf.load("/dsi/gannot-lab/gannot-lab1/users/mordehay/phoneme_classifier/exp/LibSp_wavlm-base-plus-rep_w_masked_pix=0.8_two_branch=True_all_hidden_states_randn-filled/wnet_h512_d12_T400_betaT0.02/config/config.yaml")
            ckpt_path = "/dsi/gannot-lab/gannot-lab1/users/mordehay/phoneme_classifier/exp/LibSp_wavlm-base-plus-rep_w_masked_pix=0.8_two_branch=True_all_hidden_states_randn-filled/wnet_h512_d12_T400_betaT0.02/checkpoint/8000.pkl"
            model_cfg = cfg.phoneme_classifier
            g_model = cfg.g_model
            builder = ModelBuilder()
            net_diffwave = builder.build_model(model_cfg)
            asr_guidance_net = AudioVisualModel(g_model, net_diffwave).cuda()
            print_size(net)
            asr_guidance_net.eval()
            tokenizer, decoder = None, None

            # load checkpoint
            try:
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                asr_guidance_net.load_state_dict(checkpoint['model_state_dict'])
                print('Successfully loaded MelGen checkpoint')
            except Exception as e:
                print(e)
                raise Exception('No valid model found')
        else:
            if type_input_guidance == "text":
                print(f'Apply {type_input_guidance} guidance')
            elif type_input_guidance == "phoneme":
                from g2p_en import G2p
                g2p_model = G2p()
                g2p = get_g2p_pipeline(g2p_model, with_space=with_space)
                print(f'Apply {type_input_guidance} guidance with space={with_space}')    
            asr_guidance_net, tokenizer, decoder = asr_models.get_models(ds_name, type_input_guidance=type_input_guidance, with_space=with_space)
            print('ASR, tokenizer and decoder loaded')
    else:
        asr_guidance_net, tokenizer, decoder, text = None, None, None, None
    
    vocoders = {}
    # HiFi-GAN
    print('Load HiFi-GAN')
    config_file = 'hifi_gan/config.json'
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    vocoder = Vocoder(h).cuda()
    checkpoint_file = '/dsi/gannot-lab/gannot-lab1/users/mordehay/hifi_gan/g_02400000'
    state_dict_g = vocoder_utils.load_checkpoint(checkpoint_file, 'cuda')
    vocoder.load_state_dict(state_dict_g['generator'])
    vocoder.eval()
    vocoder.remove_weight_norm()
    print('Finish Loading HiFi-GAN')
    vocoders['hifi_gan'] = vocoder
    # BigVGAN
    checkpoint_file = '/dsi/gannot-lab/gannot-lab1/users/mordehay/bigvgan/g_00550000'
    config_file = '/dsi/gannot-lab/gannot-lab1/users/mordehay/bigvgan/config.json'
    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    device_bigvgan = torch.device("cuda")
    generator = Generator(h, use_cuda_kernel=False).to(device_bigvgan)
    state_dict_g = load_checkpoint_vgan(checkpoint_file, device_bigvgan)
    generator.load_state_dict(state_dict_g["generator"])
    generator.eval()
    generator.remove_weight_norm()
    vocoders['bigvgan'] = generator
    
    dataset_type = dataset_cfg['dataset_type']
    criterion = nn.L1Loss(reduction='none')
    w_asr_list = w_asr
    asr_start_list = asr_start
    w_mel_cond_list = w_mel_cond if OmegaConf.is_list(w_mel_cond) else [w_mel_cond]
    for w_asr, asr_start, w_mel_cond in product(w_asr_list, asr_start_list, w_mel_cond_list):
        # if w_mel_cond ==2 and w_asr == 0.8 and asr_start == 320:
        #     continue
        dataset = get_dataset(dataset_cfg, split='test', return_mask_properties=True, return_true_text=True, return_target_time=True)
        guidance_dir_name = f'w1={w_mel_cond}'
        guidance_dir_name += f'_w2={w_asr}_asr_start={asr_start}' #_asr_finish=80'
        guidance_dir_name += f'_mask={on_noisy_masked_melspec}' #_repeat=5_same-theta_-mel'
        _output_directory = os.path.join(output_directory, guidance_dir_name)
        os.makedirs(_output_directory, exist_ok=True)
        print("saving to output directory", _output_directory)

        # Create a CSV file
        csv_file = open(os.path.join(_output_directory, 'samples_info.csv'), 'w', newline='')
        csv_writer = csv.writer(csv_file, delimiter='|')
        
        # compute_metrics = Metrics()

        # Write the header row
        if dataset_type == 'explosion_speech_inpainting':
            csv_writer.writerow(['Sample', 'start_explosions', 'explosions_length'])
        elif dataset_type == 'speech_inpainting' or dataset_type == 'speech_inpainting_anechoic':
            titles = ['Sample', 'block_size_list', 'num_blocks']#, 'plcmos_masked_init'] #+  \
                # [met + '_' + voc for met in ['WER_init', 'plcmos_target_init', 'LSD_init', 'STOI_init', 'PESQ_init'] for voc in vocoders.keys()] + \
                #     [met + '_' + voc for met in ['WER', 'plcmos_pred' 'LSD', 'STOI', 'PESQ'] for voc in vocoders.keys()]
            csv_writer.writerow(titles)
            # csv_writer.writerow(['Sample', 'block_size_list', 'num_blocks'])

        # ASR based on audio-only model, this is used for getting transcription for guidance, so the input is the masked audio in time domain
        pipeline_asr = InferencePipeline(config_filename_asr_cond, device='cuda')

        rng = random.Random(131)  # Create an independent random number generator with a specific seed
        length_data = len(dataset)
        used_indexes = []
        print("length_data", length_data, " But only ", n_samples_test, " will be generated")
        progress = tqdm(total=n_samples_test)
        i = 0
        while i < n_samples_test:
            if mask_info['mask_type'] != 'none':
                indx_data = rng.randint(0, length_data)  # Generate a random integer between 0 and length_data (inclusive)
                if indx_data in used_indexes: #or indx_data != 603:
                    print("Index already used")
                    continue
                used_indexes.append(indx_data)
            else:
                indx_data = i
            input_text = None
            if dataset_type == 'explosion_speech_inpainting':
                speech_melspec, mix_melspec, mix_time, masked_speech, masked_speech_time, explosions_activity, start_explosions, explosions_length = dataset[indx_data]
                mask = 1 - explosions_activity # zero = explosion, one = no explosion
                # for j in range(len(masked_cond)):
                #     masked_cond[j] = masked_cond[j].unsqueeze(0).cuda()
                # row_dict = {'Sample': indx_data, 'start_explosions': start_explosions, 'explosions_length': explosions_length}
                csv_writer.writerow([indx_data, start_explosions, explosions_length]) # in samples
                gt_melspec = speech_melspec.unsqueeze(0)
                
                masked_melspec, masked_audio_time = mix_melspec.unsqueeze(0).cuda(), mix_time.unsqueeze(0).cuda()
                masked_audio_time4text = masked_speech_time
                masked_cond = [masked_melspec, masked_audio_time]
                
            
            elif dataset_type == 'speech_inpainting':
                gt_melspec, *masked_cond, mask, block_size_list, num_blocks = dataset[indx_data]
                # row_dict = {'Sample': indx_data, 'block_size_list': block_size_list, 'num_blocks': num_blocks}
                masked_cond = [masked_cond[j].unsqueeze(0).cuda() for j in range(len(masked_cond))]
                # for j in range(len(masked_cond)):
                #     masked_cond[j] = masked_cond[j].unsqueeze(0).cuda()
                csv_writer.writerow([indx_data, block_size_list, num_blocks])
                gt_melspec = gt_melspec.unsqueeze(0)
                masked_melspec, masked_audio_time = masked_cond
                masked_audio_time4text = masked_audio_time.squeeze().cpu()
            
            elif dataset_type == 'plc_task':
                gt_melspec, masked_melspec, masked_audio_time, frame_mask, sample_mask = dataset[indx_data]
                mask = frame_mask
                gt_melspec = gt_melspec.unsqueeze(0)
                masked_cond = [masked_melspec.cuda(), masked_audio_time.cuda()]
                masked_cond = [masked_cond[j].unsqueeze(0) for j in range(len(masked_cond))]
                masked_melspec, masked_audio_time = masked_cond
                masked_audio_time4text = masked_audio_time.squeeze().cpu()
            elif dataset_cfg.dataset_type == 'speech_inpainting_anechoic':
                #melspec, masked_melspec, masked_audio_time, mask, block_size_list, num_blocks
                if dataset_cfg.speech_inpainting_anechoic.use_input_text != 'none':
                    audio_time, gt_melspec, masked_melspec, masked_audio_time, mask, block_size_list, num_blocks, true_text, input_text = dataset[indx_data]
                    input_text = [input_text]
                else:
                    audio_time, gt_melspec, masked_melspec, masked_audio_time, mask, block_size_list, num_blocks, true_text = dataset[indx_data]
                
                gt_melspec = gt_melspec.unsqueeze(0)
                mask = mask.unsqueeze(0).cuda()
                masked_cond = [masked_melspec.cuda(), masked_audio_time.cuda()]
                masked_cond = [masked_cond[j].unsqueeze(0) for j in range(len(masked_cond))]
                masked_melspec, masked_audio_time = masked_cond

                 # dubging purpose print(masked_melspec.shape)
                print(f"masked_melspec shape: {masked_melspec.shape}, masked_audio_time shape: {masked_audio_time.shape}")
                
                masked_audio_time4text = masked_audio_time.squeeze().cpu()
            
            if mask_info['mask_type'] != 'none':
                if mask_info['minimum_length'] > (masked_audio_time.shape[-1] / 16000):
                    print("The audio is too short, the minimum length is ", mask_info['minimum_length'], "[sec], The current audio is ", masked_audio_time.shape[-1] / 16000, "[sec]")
                    continue
                    #For Unet we need to fix the length of the input to be divided with 4 (2**2)

            os.makedirs(os.path.join(_output_directory, f'sample_{indx_data}'), exist_ok=True)
            if (os.path.exists(os.path.join(_output_directory, f'sample_{indx_data}', 'generated_audio_hifi_gan.wav'))) and (os.path.exists(os.path.join(_output_directory, f'sample_{indx_data}', 'generated_audio_bigvgan.wav'))):
                print(f"{os.path.join(_output_directory, f'sample_{indx_data}')} already exists")
                progress.update(1)  # Manually updating tqdm
                i += 1 
                continue
            else:
                print(f"proccessing {os.path.join(_output_directory, f'sample_{indx_data}')}")
                csv_writer.writerow([indx_data, block_size_list, num_blocks])
            
            if mask_info['mask_type'] == 'repeat_all_freq':
                masked_melspec, masked_audio_time, mask, _ = mask_time_all_frequencies_mask(gt_melspec[0], audio_time, mask_info['repeat_all_freq']['length'], mask_info['repeat_all_freq']['skip'], noise_type=mask_info['noise_type'], hop_length=dataset_cfg[dataset_type]["audio_stft_hop"])
                masked_melspec = masked_melspec.unsqueeze(0).cuda()
                mask = mask.unsqueeze(0).cuda()
                masked_audio_time = masked_audio_time.unsqueeze(0).cuda()
                masked_cond = [masked_melspec, masked_audio_time]
                masked_audio_time4text = masked_audio_time.squeeze().cpu()
            elif mask_info['mask_type'] == 'repeat_specific_freq':
                masked_melspec, masked_audio_time, mask, _ = mask_time_specific_frequencies_mask(gt_melspec[0], mask_info['repeat_specific_freq']['length'], mask_info['repeat_specific_freq']['skip'], mask_info['repeat_specific_freq']['freq'], noise_type=mask_info['noise_type'])
                masked_melspec = gt_melspec[0] * mask
                masked_melspec = masked_melspec.unsqueeze(0).cuda()
                mask = mask.unsqueeze(0).cuda()
                masked_cond = [masked_melspec, masked_audio_time]
                masked_audio_time4text = masked_audio_time.squeeze().cpu()
            elif mask_info['mask_type'] == 'by_number':
                masked_melspec, mask = mask_with_shape_mask(gt_melspec[0], mask_info['by_number']['number'], noise_type=mask_info['noise_type'])
                masked_melspec = masked_melspec.unsqueeze(0).cuda()
                mask = mask.unsqueeze(0).cuda()
                masked_cond = [masked_melspec, masked_audio_time]
                masked_audio_time4text = masked_audio_time.squeeze().cpu()
            elif mask_info['mask_type'] == 'all_time_specific_freq':
                masked_melspec, masked_audio_time, mask, _ = mask_specific_frequencies_all_time_mask(gt_melspec[0], mask_info['all_time_specific_freq']['freq'], noise_type=mask_info['noise_type'])
                masked_melspec = gt_melspec[0] * mask
                masked_melspec = masked_melspec.unsqueeze(0).cuda()
                mask = mask.unsqueeze(0).cuda()
                masked_audio_time4text = masked_audio_time.squeeze().cpu()


            if model_cfg._name_ == 'unet':
                # mask should be [1, 1, T] but  mask type option output it as [1, F, T], so we need to collapse this
                mask = mask.mean(dim=1, keepdim=True)
                freq_siganl, time_signal = masked_cond
                desired_num_frames = fix_len_compatibility(gt_melspec.shape[-1])
                masked_audio_time_mask = torch.ones_like(time_signal)
                masked_audio_time_mask = pad_last_dim(masked_audio_time_mask, (desired_num_frames - freq_siganl.shape[-1]) * dataset_cfg[dataset_type]["audio_stft_hop"])
                gt_melspec = pad_last_dim(gt_melspec, desired_num_frames - gt_melspec.shape[-1])
                time_signal = pad_last_dim(time_signal, (desired_num_frames - freq_siganl.shape[-1]) * dataset_cfg[dataset_type]["audio_stft_hop"])
                freq_siganl = pad_last_dim(freq_siganl, desired_num_frames - freq_siganl.shape[-1]).cuda()
                mask_frames = torch.zeros((list(mask.shape[:-1]) + [desired_num_frames]))
                mask_frames[..., :mask.shape[-1]] = 1
                mask_frames = mask_frames.cuda()
                mask = pad_last_dim(mask, desired_num_frames - mask.shape[-1], pad_value=1)
                masked_cond = [freq_siganl, time_signal]
            else:
                mask_frames = None
                masked_audio_time_mask=None
                
            text = true_text[0]
            true_text_str = true_text[0]
            phoneme4guidance=['None']
            per_frame_phoneme4guidance = ['None']
            if apply_asr_guidance:
                if type_input_guidance == 'text':
                    if mel_text: # use the true text of the sentence
                        print(f"The transcript is: {text}")
                        text = preprocess_text(text)
                        print(f"The normalized transcript is: {text}")
                    else: # predict the text from the masked audio
                        # Create a temporary file
                        audio4text = masked_audio_time4text
                        sample_rate = 16000  # Example value
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
                            # Save the masked audio array as a WAV file in the temporary file
                            write(temp_wav.name, sample_rate, audio4text.numpy().astype(np.float32)) # TODO maybe we need to do something more clever here
                            # Send the temporary WAV file to the pipeline
                            transcript_from_condition = pipeline_asr(temp_wav.name)
                            text = transcript_from_condition
                            print(f"The transcript is: {text}")
                            text = preprocess_text(text)
                            print(f"The normalized transcript is: {text}")
                        
                elif type_input_guidance == 'phoneme':
                    if mel_text: # use the true text of the sentence
                        phoneme4guidance = [input_text[0]]
                        if not with_space:
                            phoneme4guidance[0] = [item for item in input_text[0] if item != "space"]
                    else: # predict the text from the masked audio
                        # Create a temporary file
                        audio4text = masked_audio_time4text
                        sample_rate = 16000  # Example value
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
                            # Save the masked audio array as a WAV file in the temporary file
                            write(temp_wav.name, sample_rate, audio4text.numpy().astype(np.float32)) # TODO maybe we need to do something more clever here
                            # Send the temporary WAV file to the pipeline
                            transcript_from_condition = pipeline_asr(temp_wav.name)
                            text = transcript_from_condition
                            print(f"The transcript is: {text}")
                            text = preprocess_text(text)
                            print(f"The normalized transcript is: {text}")
                            phoneme4guidance = [g2p(text)]
                            
                    print(f"The Ground thruth phoneme is: {' '.join(phoneme4guidance[0])}")

                elif type_input_guidance == 'frame_level_phoneme':
                    per_frame_phoneme4guidance = [input_text[0]]
                    # print(f"The Ground thruth phoneme is: {per_frame_phoneme[0]}")
        
            
            if dataset_type == 'explosion_speech_inpainting':
                ## save the masked audio in time domain
                masked_audio_time4saveing = masked_speech_time.squeeze().cpu().numpy()
                sf.write(os.path.join(_output_directory, f'sample_{indx_data}', 'speech_time_masking_audio.wav'), masked_audio_time4saveing, 16000)
                ## save the masked audio in time domain
                masked_audio_time4saveing = masked_audio_time.squeeze().cpu().numpy()
                sf.write(os.path.join(_output_directory, f'sample_{indx_data}', 'mix_explsions.wav'), masked_audio_time4saveing, 16000)
            elif dataset_type == 'speech_inpainting' or dataset_type == 'plc_task' or dataset_type == 'speech_inpainting_anechoic':
                ## save the masked audio in time domain
                masked_audio_time4saveing = masked_audio_time.squeeze().cpu().numpy()
                sf.write(os.path.join(_output_directory, f'sample_{indx_data}', 'masked_audio_time.wav'), masked_audio_time4saveing, 16000)
           
        # ------------ dont care about visualization of the denoising process. -------------------------------------------------------   
        
            ## get the the clean version of the noisy melspec and the noisy melspec
            # weighted_loss, est_X, transformed_X, diffusion_steps, mean_loss = training_loss(net, criterion, gt_melspec.cuda(), masked_cond,  mask.cuda(), diffusion_hyperparams, w_masked_pix=0.8, mask_frames=mask_frames,
            #                                                                                 text=true_text, input_text=input_text,  mask_padding_time=masked_audio_time_mask)
            # # save the est audio
            # est_X = denormalise_mel(est_X)
            # est_audio = vocoder(est_X)
            # est_audio = est_audio.squeeze()
            # est_audio = est_audio / 1.1 / est_audio.abs().max()
            # est_audio = est_audio.cpu().numpy()
            # sf.write(os.path.join(_output_directory, f'sample_{indx_data}', f'est_audio_after_clean_loss={mean_loss}.wav'), est_audio, 16000)
            
            # est_X = est_X.squeeze(0).cpu().numpy()
            # matplotlib.image.imsave(os.path.join(_output_directory, f'sample_{indx_data}', 'est_melspec_after_clean_image.png'), est_X[::-1])
            
            # #save the noisy audio
            # transformed_X = denormalise_mel(transformed_X)
            # transformed_X_audio = vocoder(transformed_X)
            # transformed_X_audio = transformed_X_audio.squeeze()
            # transformed_X_audio = transformed_X_audio / 1.1 / transformed_X_audio.abs().max()
            # transformed_X_audio = transformed_X_audio.cpu().numpy()
            # sf.write(os.path.join(_output_directory, f'sample_{indx_data}', f'noisy_audio={diffusion_steps.item()}.wav'), transformed_X_audio, 16000)
            
            # transformed_X = transformed_X.squeeze(0).cpu().numpy()
            # matplotlib.image.imsave(os.path.join(_output_directory, f'sample_{indx_data}', 'noisy_melspec_image.png'), transformed_X[::-1])
            
            # # inference
            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)
            # start.record()

            melspec, preds_ao = sampling(net, diffusion_cfg,
                            diffusion_hyperparams,
                            w_mel_cond,
                            condition=masked_cond,
                            asr_guidance_net=asr_guidance_net,
                            w_asr=w_asr,
                            asr_start=asr_start,
                            guidance_text=text,
                            tokenizer=tokenizer,
                            decoder=decoder,
                            without_condtion=without_condtion,
                            mask=mask,
                            on_noisy_masked_melspec=on_noisy_masked_melspec,
                            mask_frames=mask_frames,
                            masked_audio_time_mask=masked_audio_time_mask,
                            text=true_text, 
                            input_text=input_text,
                            phoneme4guidance=phoneme4guidance,
                            per_frame_phoneme4guidance=per_frame_phoneme4guidance,
                            type_input_guidance=type_input_guidance,
                            skip_step=skip_step
                            )
            melspec = denormalise_mel(melspec)
            #end.record()
            # torch.cuda.synchronize()
            # print('generated sample_{} in {} seconds'.format(indx_data, int(start.elapsed_time(end)/1000)))

            # save text
            text_filename = os.path.join(_output_directory, f'sample_{indx_data}', 'asr_text.txt')
            with open(text_filename, 'w') as f:
                if type_input_guidance == 'text':
                    f.write("True text:  " + true_text_str + "\n")
                    f.write("asr_condition       :  " +text+"\n")
                    f.write("asr_generated_signal:  " + preds_ao)
                elif type_input_guidance == 'phoneme':
                    f.write("True text:  " + true_text_str + "\n")
                    f.write("text4phoneme:  " + text + "\n")
                    f.write("asr_condition       :  " +" ".join(phoneme4guidance[0])+"\n")
                    f.write("asr_generated_signal:  " + " ".join(preds_ao))
                elif type_input_guidance == 'frame_level_phoneme':
                    f.write("True text:  " + true_text_str + "\n")
                    f.write("text4phoneme:  " + text + "\n")

            
            # plcmos_masked_init = compute_metrics.compute_plcmos(masked_audio_time.squeeze().cpu().numpy())
            # row_dict.update({'plcmos_masked_init': plcmos_masked_init})
            
            # generate audio from masked melspec
            masked_melspec = denormalise_mel(masked_melspec)
            for vocoder_name, vocoder in vocoders.items():
                masked_audio = vocoder(masked_melspec.cuda())
                masked_audio = masked_audio.squeeze()
                masked_audio = masked_audio / 1.1 / masked_audio.abs().max()
                masked_audio = masked_audio.cpu().numpy()
                sf.write(os.path.join(_output_directory, f'sample_{indx_data}', f'spec_masking_audio_{vocoder_name}.wav'), masked_audio, 16000)

            # Fix dimension mismatch: HiFi-GAN expects [B, C, T], not [B, 1, C, T]
            if melspec.ndim == 4:
                melspec = melspec.squeeze(1)
            # generate audio from generated melspec
            for vocoder_name, vocoder in vocoders.items():
                audio = vocoder(melspec)
                audio = audio.squeeze()
                audio = audio / 1.1 / audio.abs().max()
                audio = audio.cpu().numpy()
                sf.write(os.path.join(_output_directory, f'sample_{indx_data}', f'generated_audio_{vocoder_name}.wav'), audio, 16000)

            
        # generate audio from gt melspec
            gt_melspec = denormalise_mel(gt_melspec)
            for vocoder_name, vocoder in vocoders.items():
                gt_audio = vocoder(gt_melspec.cuda())
                gt_audio = gt_audio.squeeze()
                gt_audio = gt_audio / 1.1 / gt_audio.abs().max()
                gt_audio = gt_audio.cpu().numpy()
                sf.write(os.path.join(_output_directory, f'sample_{indx_data}', f'gt_audio_{vocoder_name}.wav'), gt_audio, 16000)

            # save as file
            melspec = melspec.squeeze(0).cpu()
            torch.save(melspec, os.path.join(_output_directory, f'sample_{indx_data}', 'generated_spec.npz'))
            
            mask_cpu = mask.squeeze(0).cpu()
            torch.save(mask_cpu, os.path.join(_output_directory, f'sample_{indx_data}', 'mask.npz'))
            # save as image
            melspec = melspec.numpy()
            masked_melspec = masked_melspec.squeeze(0).cpu().numpy()
            gt_melspec = gt_melspec.squeeze(0).numpy()
            matplotlib.image.imsave(os.path.join(_output_directory, f'sample_{indx_data}', 'generated_spec_image.png'), melspec[::-1])
            matplotlib.image.imsave(os.path.join(_output_directory, f'sample_{indx_data}', 'gt_spec_image.png'), gt_melspec[::-1])
            matplotlib.image.imsave(os.path.join(_output_directory, f'sample_{indx_data}', 'masked_spec_image.png'), masked_melspec[::-1])
            
            
            progress.update(1)  # Manually updating tqdm
            i += 1    
        
        # Close the CSV file
        csv_file.close()
            
    return

# config_dit_without-space-phoneme
# tts-dit_without-space

@hydra.main(version_base=None, config_path="configs_Alon_Matan/", config_name="config_dit_without-space-phoneme_on-masked-mel_for_inference")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    generate(0,
        diffusion_cfg=cfg.diffusion,
        model_cfg=cfg[cfg.melgen],
        g_model_cfg=cfg.g_model,
        dataset_cfg=cfg.dataset,
        **cfg.generate,
    )



if __name__ == "__main__":
    main()
