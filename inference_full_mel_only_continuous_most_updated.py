# this file is an adapated version https://github.com/albertfgu/diffwave-sashimi, licensed
# under https://github.com/albertfgu/diffwave-sashimi/blob/master/LICENSE

# this is the full test without asr: mel condition, free-classifier and vocoder(mel2audio)
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import subprocess
import time
import warnings
from itertools import product
import random
warnings.filterwarnings("ignore")
from SDE import VPSDE, VESDE

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
from utils import find_max_epoch, print_size, get_diffusion_hyperparams, local_directory, preprocess_text, fix_len_compatibility, pad_last_dim
import csv
from mouthroi_processing.pipelines.pipeline import InferencePipeline
from scipy.io.wavfile import write
import tempfile
# from my_utils.compute_metrics import Metrics
import ASR.asr_models as asr_models

from utils import mask_time_all_frequencies_mask, mask_time_specific_frequencies_mask, mask_specific_frequencies_all_time_mask, mask_combined_mask, mask_with_shape_mask
print("finished imports")


def get_g2p_pipeline(g2p_model, with_space=False):
    p2n = '/home/dsi/sellama/SpeechRepainting/phoneme_to_number.json'
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

def training_loss(
    net,
    loss_fn,
    melspec,
    masked_melspec,
    mask,
    diffusion_hyperparams,
    w_masked_pix=0.8,
    mask_frames=None,
    masked_audio_time_mask=None,
    text=None,
    input_text=None,
    mask_padding_time=None,
    on_noisy_masked_melspec=False,
):
    """
    Continuous SDE diagnostic loss.

    This replaces the original DDPM diagnostic training_loss.
    It follows the same noising/score-matching path used in continuous SDE training:

        x_t = mean(x_0, t) + std(t) * z
        loss = || predicted_score * std + z ||

    Returns the same tuple format as the old function:
        weighted_loss, est_X, transformed_X, diffusion_steps, mean_loss

    where:
        transformed_X = x_t
        est_X         = one-step x0 estimate from the predicted score
    """

    _dh = diffusion_hyperparams
    device = melspec.device

    if _dh["name"] == "VPSDE":
        sde = VPSDE(_dh["beta_min"], _dh["beta_max"], _dh["N"])
    elif _dh["name"] == "VESDE":
        sde = VESDE(_dh["sigma_min"], _dh["sigma_max"], _dh["N"])
    else:
        raise ValueError(
            f"continuous training_loss expected VPSDE/VESDE, got {_dh['name']}"
        )

    B, C, L = melspec.shape

    # Make sure mask has a broadcastable shape.
    mask = mask.to(device)
    if mask.ndim == 2:
        mask = mask.unsqueeze(1)

    # Valid-frame mask for padded samples.
    if mask_frames is None:
        mask_mask = torch.ones((B, 1, L), device=device, dtype=melspec.dtype)
    else:
        mask_mask = mask_frames.to(device).to(melspec.dtype)
        if mask_mask.ndim == 2:
            mask_mask = mask_mask.unsqueeze(1)
        if mask_mask.shape[-1] != L:
            mask_mask = mask_mask[..., :L]

    # ── Continuous SDE noising ─────────────────────────────────────────────
    eps = 1e-5
    t = torch.rand(B, device=device) * (sde.T - eps) + eps  # [B]
    z = torch.randn_like(melspec)                           # [B, C, L]

    mean, std = sde.marginal_prob(melspec, None, t)
    std_expanded = std[:, None, None]                       # [B, 1, 1]
    x_t = mean + std_expanded * z

    # Follow the same convention as your training code:
    # mask == 1 is known/unmasked, mask == 0 is missing/masked.
    if on_noisy_masked_melspec:
        x_t = melspec * mask + x_t * (1 - mask)

    cond_drop_prob = 0.2

    predicted_score = net(
        x_t,
        masked_melspec,
        t.view(B, 1),
        cond_drop_prob,
        text=text,
        input_text=input_text,
        mask_padding_time=masked_audio_time_mask,
        mask_padding_frames=mask_frames,
    )

    # Denoising score matching target:
    # score target = -z / std
    # therefore predicted_score * std should match -z.
    loss = loss_fn(predicted_score * std_expanded, -z)  # [B, C, L]

    # Apply valid-frame mask.
    loss = loss * mask_mask

    # Robust denominators depending on whether mask is [B,1,L] or [B,C,L].
    if mask.shape[1] == 1 and loss.shape[1] != 1:
        unmasked_denom = torch.sum(mask * mask_mask) * loss.shape[1]
        masked_denom = torch.sum((1 - mask) * mask_mask) * loss.shape[1]
    else:
        unmasked_denom = torch.sum(mask * mask_mask)
        masked_denom = torch.sum((1 - mask) * mask_mask)

    unmasked_loss = torch.sum(mask * loss) / (unmasked_denom + 1e-8)
    masked_loss = torch.sum((1 - mask) * loss) / (masked_denom + 1e-8)

    weighted_loss = (1 - w_masked_pix) * unmasked_loss + w_masked_pix * masked_loss
    mean_loss = round(torch.mean(loss).item(), 2)

    # Debug one-step x0 estimate.
    # Since score ≈ -(x_t - mean) / std^2:
    # mean_hat ≈ x_t + std^2 * score.
    # For VP-SDE, mean = alpha(t) * x0, so divide by alpha(t).
    mean_hat = x_t + (std_expanded ** 2) * predicted_score

    if _dh["name"] == "VPSDE":
        ones = torch.ones_like(melspec)
        mean_ones, _ = sde.marginal_prob(ones, None, t)
        mean_coeff = mean_ones.mean(dim=(1, 2), keepdim=True)
        est_X = mean_hat / (mean_coeff + 1e-8)
    else:
        # VE-SDE usually has mean = x0.
        est_X = mean_hat

    # For file naming/debug: convert continuous t to approximate discrete step.
    diffusion_steps = (t / sde.T * (sde.N - 1)).view(B, 1, 1)

    return weighted_loss, est_X, x_t, diffusion_steps, mean_loss



def sampling(net, diffusion_cfg, diffusion_hyperparams,
            w_mel_cond, condition=None, 
            asr_guidance_net=None,
            w_asr=None,
            asr_start=None,
            guidance_text=None,
            tokenizer=None,
            decoder=None,
            without_condition=False,
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
            phoneme_guidance_debug=False,
            ):
    """
    Minimal continuous-SDE replacement for the original DDPM sampling().

    This keeps the original generate()/dataset/mask/vocoder flow unchanged.
    The only difference is that when diffusion_hyperparams['name'] is VPSDE/VESDE,
    sampling is done with the project PC sampler, and phoneme/text guidance still
    goes through asr_models.get_models(...).forward_model(...).
    """
    preds_ao = 'None'

    masked_melspec, _ = condition
    if masked_melspec.ndim == 4 and masked_melspec.shape[1] == 1:
        masked_melspec = masked_melspec.squeeze(1)
    if mask is not None and mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask.squeeze(1)
    mask = mask.cuda()
    condition = [masked_melspec, condition[1]]

    _dh = diffusion_hyperparams

    # Build ASR/phoneme targets exactly like the original code.
    tokens = None
    loss_ce = None
    if asr_guidance_net is not None:
        if type_input_guidance == 'text':
            tokens = torch.LongTensor(tokenizer.encode(guidance_text)).unsqueeze(0).cuda()
        elif type_input_guidance == 'phoneme':
            token_ids = asr_models.phonemes_to_ctc_ids(
                phoneme4guidance[0],
                tokenizer,
                asr_guidance_net,
            )
            tokens = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).cuda()
        elif type_input_guidance == 'frame_level_phoneme':
            tokens = torch.tensor(per_frame_phoneme4guidance[0], dtype=torch.int64).unsqueeze(0).cuda()
            loss_ce = nn.CrossEntropyLoss(reduction='none')

    # ---------------------------------------------------------------------
    # Continuous SDE path: this is the intended path for VPSDE/VESDE.
    # ---------------------------------------------------------------------
    if _dh["name"] in ["VPSDE", "VESDE"]:
        from SDE import VPSDE, VESDE
        from sampling import get_ode_sampler, get_pc_sampler

        if _dh["name"] == "VPSDE":
            sde = VPSDE(_dh["beta_min"], _dh["beta_max"], _dh["N"])
        else:
            sde = VESDE(_dh["sigma_min"], _dh["sigma_max"], _dh["N"])

        sampler_type = str(diffusion_cfg.get('sampler_type', 'sde')).lower()

        def score_fn_without_asr(x, t):
            if x.ndim == 4 and x.shape[1] == 1:
                x = x.squeeze(1)

            B = x.shape[0]
            t_input = t.view(B, 1)

            if without_condition:
                score = net(
                    x, condition, t_input,
                    cond_drop_prob=1,
                    mask_padding_frames=mask_frames,
                    text=text,
                    input_text=input_text,
                    mask_padding_time=masked_audio_time_mask,
                )
            else:
                score_cond = net(
                    x, condition, t_input,
                    cond_drop_prob=0,
                    mask_padding_frames=mask_frames,
                    text=text,
                    input_text=input_text,
                    mask_padding_time=masked_audio_time_mask,
                )
                score_uncond = net(
                    x, condition, t_input,
                    cond_drop_prob=1,
                    mask_padding_frames=mask_frames,
                    text=text,
                    input_text=input_text,
                    mask_padding_time=masked_audio_time_mask,
                )
                score = (1 + w_mel_cond) * score_cond - w_mel_cond * score_uncond

            return score

        def _asr_start_to_continuous_t(asr_start_value):
            """
            Accept both conventions:
            - asr_start <= sde.T: already continuous time, e.g. 0.05
            - asr_start >  sde.T: discrete step, e.g. 100 or 250
            """
            if asr_start_value is None:
                return None
            if isinstance(asr_start_value, (list, tuple)):
                asr_start_value = asr_start_value[0]
            asr_start_value = float(asr_start_value)
            if asr_start_value > float(sde.T):
                return asr_start_value / float(sde.N - 1) * float(sde.T)
            return asr_start_value

        def _continuous_t_to_asr_step(t):
            # The diffusion model receives continuous t, but the ASR/phoneme
            # classifier was trained with a diffusion-step embedding scale.
            classifier_sde = asr_guidance_net.sde
            return (t / float(classifier_sde.T) * float(classifier_sde.N - 1)).view(t.shape[0], 1)

        ode_guidance_calls = 0

        def asr_guidance_fn(x, y, t, mask_aware=False, ode_debug=False):
            """Return ASR guidance with independently controlled ODE logging.

            ``mask_aware`` controls whether normalization uses only the gap.
            The current SDE and ODE paths both use complete-mel normalization.
            """
            nonlocal ode_guidance_calls

            if x.ndim == 4 and x.shape[1] == 1:
                x = x.squeeze(1)

            if asr_guidance_net is None or tokens is None:
                return torch.zeros_like(x)

            asr_start_t = _asr_start_to_continuous_t(asr_start)
            if asr_start_t is not None and t[0].item() > asr_start_t:
                return torch.zeros_like(x)

            with torch.no_grad():
                score = score_fn_without_asr(x, t)

                if mask_aware:
                    unknown = (1.0 - mask.to(device=x.device, dtype=x.dtype)).clamp(0.0, 1.0)
                    if unknown.ndim == 4 and unknown.shape[1] == 1:
                        unknown = unknown.squeeze(1)
                    if mask_frames is not None:
                        valid_frames = mask_frames.to(device=x.device, dtype=x.dtype)
                        if valid_frames.ndim == 4 and valid_frames.shape[1] == 1:
                            valid_frames = valid_frames.squeeze(1)
                        unknown = unknown * valid_frames
                    score_for_norm = score * unknown
                else:
                    unknown = None
                    score_for_norm = score

                score_norm = torch.norm(
                    score_for_norm.reshape(score_for_norm.shape[0], -1),
                    dim=-1,
                )
                if not mask_aware:
                    # Preserve the original PC/SDE batch normalization exactly.
                    score_norm = score_norm.mean()

            with torch.enable_grad():
                if type_input_guidance in ['text', 'phoneme']:
                    length_input = torch.full((x.shape[0],), x.shape[2], dtype=torch.long, device=x.device)
                    x_for_asr = x.detach().requires_grad_(True)
                    inputs = x_for_asr, length_input
                    targets = tokens, torch.tensor([tokens.shape[1]], dtype=torch.long, device=x.device)
                    diffusion_steps_asr = _continuous_t_to_asr_step(t.to(x.device))

                    asr_guidance_net.device = torch.device("cuda")
                    batch_losses = asr_guidance_net.forward_model(
                        inputs,
                        diffusion_steps_asr,
                        targets,
                        compute_metrics=False,
                        verbose=0,
                    )[0]
                    loss = batch_losses["loss"]
                    loss_grad = torch.autograd.grad(loss, x_for_asr)[0]
                    guidance_grad = -loss_grad  # CTC loss = -log p(target | x_t)

                elif type_input_guidance == 'frame_level_phoneme':
                    x_for_asr = x.detach().requires_grad_(True)
                    outputs = asr_guidance_net(
                        x_for_asr,
                        condition,
                        _continuous_t_to_asr_step(t.to(x.device)),
                        cond_drop_prob=0,
                        mask_padding_time=masked_audio_time_mask,
                    )
                    l_ce = loss_ce(outputs, tokens)
                    loss = torch.sum(l_ce * (1 - mask)) / torch.sum(1 - mask)
                    loss_grad = torch.autograd.grad(loss, x_for_asr)[0]
                    guidance_grad = -loss_grad

                else:
                    return torch.zeros_like(x)

                # guidance_norm = torch.norm(guidance_grad.reshape(guidance_grad.shape[0], -1), dim=-1).mean()
                # grad_normaliser = score_norm / (guidance_norm + 1e-8)

            # return (grad_normaliser * guidance_grad).detach()
                if mask_aware:
                    # The ODE projects every update onto the missing region. Do
                    # the same before normalization, otherwise gradients on the
                    # observed region inflate the norm and are then discarded.
                    guidance_grad = guidance_grad * unknown

                guidance_norm = torch.norm(
                    guidance_grad.reshape(guidance_grad.shape[0], -1),
                    dim=-1
                )
                if not mask_aware:
                    # Preserve the original PC/SDE batch normalization exactly.
                    guidance_norm = guidance_norm.mean()
                grad_normaliser = score_norm / guidance_norm.clamp_min(1e-8)

            if mask_aware:
                normaliser_shape = (
                    guidance_grad.shape[0],
                    *([1] * (guidance_grad.ndim - 1)),
                )
                normalised_guidance = (
                    grad_normaliser.reshape(normaliser_shape) * guidance_grad
                ).detach()
            else:
                normalised_guidance = (grad_normaliser * guidance_grad).detach()

            # # ====================== ASR DEBUG BLOCK - DELETE LATER ======================
            # if ode_debug:
            #     ode_guidance_calls += 1
            #     if phoneme_guidance_debug and (
            #         ode_guidance_calls == 1 or ode_guidance_calls % 10 == 0
            #     ):
            #         weighted_guidance_norm = torch.norm(
            #             (
            #                 (0.0 if w_asr is None else float(w_asr))
            #                 * normalised_guidance
            #             ).reshape(normalised_guidance.shape[0], -1),
            #             dim=-1,
            #         )
            #         effective_ratio = weighted_guidance_norm / score_norm.clamp_min(1e-8)
            #         print(
            #             "[ODE PHONEME GUIDANCE]",
            #             f"call={ode_guidance_calls}",
            #             f"t={float(t.detach().mean().cpu()):.4f}",
            #             f"ctc_loss={float(loss.detach().mean().cpu()):.5f}",
            #             f"score_full_norm={float(score_norm.mean().detach().cpu()):.5f}",
            #             f"raw_guidance_full_norm={float(guidance_norm.mean().detach().cpu()):.5f}",
            #             f"normaliser={float(grad_normaliser.mean().detach().cpu()):.5f}",
            #             f"weighted_ratio={float(effective_ratio.mean().detach().cpu()):.5f}",
            #         )

            # normalised_guidance_norm = torch.norm(
            #     normalised_guidance.reshape(normalised_guidance.shape[0], -1),
            #     dim=-1
            # ).mean()

            # asr_weight = 0.0 if w_asr is None else float(w_asr)

            # weighted_ratio = (
            #     asr_weight * normalised_guidance_norm / (score_norm + 1e-8)
            # ).detach()

            # print(
            #     "[ASR DEBUG ACTIVE]",
            #     "t=", float(t.detach().mean().cpu()),
            #     "asr_start_t=", float(asr_start_t) if asr_start_t is not None else None,
            #     "ctc_loss=", float(loss.detach().mean().cpu()),
            #     "raw_asr_grad_norm=", float(guidance_norm.detach().cpu()),
            #     "score_norm=", float(score_norm.detach().cpu()),
            #     "grad_normaliser=", float(grad_normaliser.detach().cpu()),
            #     "normalised_guidance_norm=", float(normalised_guidance_norm.detach().cpu()),
            #     "w_asr=", asr_weight,
            #     "weighted_asr_to_score_ratio=", float(weighted_ratio.cpu()),
            # )
            # ==================== END ASR DEBUG BLOCK - DELETE LATER ====================

            return normalised_guidance

        def score_fn(x, y, t):
            score = score_fn_without_asr(x, t)
            guidance = asr_guidance_fn(x, y, t)
            asr_weight = 0.0 if w_asr is None else float(w_asr)
            return score + asr_weight * guidance

        def score_fn_ode(x, y, t):
            score = score_fn_without_asr(x, t)
            # Match the SDE guidance calculation: estimate and normalize both
            # the diffusion score and the ASR gradient over the complete mel.
            # The ODE sampler masks the resulting drift afterwards, so only
            # the generated/missing region is actually updated.
            guidance = asr_guidance_fn(
                x,
                y,
                t,
                mask_aware=False,
                ode_debug=True,
            )
            asr_weight = 0.0 if w_asr is None else float(w_asr)
            return score + asr_weight * guidance

        if sampler_type == 'sde':
            sampler = get_pc_sampler(
                predictor_name=diffusion_cfg.get('predictor', "reverse_diffusion"),
                corrector_name=diffusion_cfg.get('corrector', "langevin"),
                sde=sde,
                score_fn=score_fn,
                y=masked_melspec,
                snr=diffusion_cfg.get('snr', 0.1),
                corrector_steps=diffusion_cfg.get('corrector_steps', 1),
                eps=diffusion_cfg.get('eps', 3e-2),
                w_mel_cond=w_mel_cond,
                mask=mask,
                mask_noise=on_noisy_masked_melspec,
            )
        elif sampler_type == 'ode':
            sampler = get_ode_sampler(
                sde=sde,
                score_fn=score_fn_ode,
                y=masked_melspec,
                mask=mask,
                on_noisy_masked_melspec=on_noisy_masked_melspec,
                method=diffusion_cfg.get('ode_method', 'heun'),
                steps=diffusion_cfg.get('ode_steps', 50),
                denoise=diffusion_cfg.get('ode_denoise', True),
                eps=diffusion_cfg.get('eps', 3e-2),
            )
        else:
            raise ValueError(
                "diffusion.sampler_type must be 'sde' or 'ode', "
                f"got {sampler_type!r}."
            )

        print(f"{sampler_type.upper()} input mel:", masked_melspec.shape)
        print(f"{sampler_type.upper()} input mask:", mask.shape)
        x, nfe = sampler()
        print(
            f"{sampler_type.upper()} sampler finished in "
            f"{nfe} function evaluations"
        )

        # A sampler may return [B, 1, 80, T]. HiFi-GAN expects [B, 80, T].
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.squeeze(1)

        if masked_melspec.ndim == 4 and masked_melspec.shape[1] == 1:
            masked_melspec = masked_melspec.squeeze(1)

        if mask.ndim == 4 and mask.shape[1] == 1:
            mask = mask.squeeze(1)
            
        x = masked_melspec * mask + x * (1 - mask)
        if mask_frames is not None:
            x = x[..., :int(torch.sum(mask_frames, dim=-1).item())]

        # Decode the final, clean generated mel for logging in asr_text.txt.
        # This is separate from the CTC loss used above to guide sampling.
        if (
            asr_guidance_net is not None
            and decoder is not None
            and type_input_guidance in ['text', 'phoneme']
        ):
            length_input = torch.full(
                (x.shape[0],),
                x.shape[-1],
                dtype=torch.long,
                device=x.device,
            )
            diffusion_steps_asr = torch.zeros(
                (x.shape[0], 1),
                dtype=x.dtype,
                device=x.device,
            )
            outputs_ao = asr_guidance_net(
                (x, length_input),
                diffusion_steps_asr,
            )["outputs"]
            preds_ao = decoder(outputs_ao)[0]

        return x, preds_ao

    # ---------------------------------------------------------------------
    # Safety: this minimal version is meant for continuous SDE configs only.
    # Keep the original file if you still need linear/cosine DDPM inference.
    # ---------------------------------------------------------------------
    raise ValueError(
        f"This minimal continuous version expects VPSDE/VESDE, got diffusion name: {_dh['name']}"
    )


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
        without_condition=False,
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
            cfg = OmegaConf.load("/dsi/gannot-lab1/users/mordehay/phoneme_classifier/exp/LibSp_wavlm-base-plus-rep_w_masked_pix=0.8_two_branch=True_all_hidden_states_randn-filled/wnet_h512_d12_T400_betaT0.02/config/config.yaml")
            ckpt_path = "/dsi/gannot-lab1/users/mordehay/phoneme_classifier/exp/LibSp_wavlm-base-plus-rep_w_masked_pix=0.8_two_branch=True_all_hidden_states_randn-filled/wnet_h512_d12_T400_betaT0.02/checkpoint/8000.pkl"
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
            asr_guidance_net, tokenizer, decoder = asr_models.get_models(ds_name, type_input_guidance=type_input_guidance, with_space=with_space, checkpoint_ao=kwargs.get("asr_checkpoint_ao", None))
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
    w_asr_list = w_asr if OmegaConf.is_list(w_asr) or isinstance(w_asr, (list, tuple)) else [w_asr]
    asr_start_list = asr_start if OmegaConf.is_list(asr_start) or isinstance(asr_start, (list, tuple)) else [asr_start]
    w_mel_cond_list = w_mel_cond if OmegaConf.is_list(w_mel_cond) or isinstance(w_mel_cond, (list, tuple)) else [w_mel_cond]
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
                ## save the clean/GT audio in time domain when available. This is for debugging only.
                if 'audio_time' in locals():
                    sf.write(os.path.join(_output_directory, f'sample_{indx_data}', 'gt_audio_time.wav'), audio_time.squeeze().cpu().numpy(), 16000)
            ## get the the clean version of the noisy melspec and the noisy melspec
            weighted_loss, est_X, transformed_X, diffusion_steps, mean_loss = training_loss(net, criterion, gt_melspec.cuda(), masked_cond,  mask.cuda(), diffusion_hyperparams, w_masked_pix=0.8, mask_frames=mask_frames,
                                                                                            text=true_text, input_text=input_text,  mask_padding_time=masked_audio_time_mask, masked_audio_time_mask=masked_audio_time_mask,on_noisy_masked_melspec=on_noisy_masked_melspec,)
            # save the est audio
            est_X = denormalise_mel(est_X)
            est_audio = vocoder(est_X)
            est_audio = est_audio.squeeze()
            est_audio = est_audio / 1.1 / est_audio.abs().max()
            est_audio = est_audio.cpu().numpy()
            sf.write(os.path.join(_output_directory, f'sample_{indx_data}', f'est_audio_after_clean_loss={mean_loss}.wav'), est_audio, 16000)
            
            est_X = est_X.squeeze(0).cpu().numpy()
            matplotlib.image.imsave(os.path.join(_output_directory, f'sample_{indx_data}', 'est_melspec_after_clean_image.png'), est_X[::-1])
            
            #save the noisy audio
            transformed_X = denormalise_mel(transformed_X)
            transformed_X_audio = vocoder(transformed_X)
            transformed_X_audio = transformed_X_audio.squeeze()
            transformed_X_audio = transformed_X_audio / 1.1 / transformed_X_audio.abs().max()
            transformed_X_audio = transformed_X_audio.cpu().numpy()
            sf.write(os.path.join(_output_directory, f'sample_{indx_data}', f'noisy_audio={diffusion_steps.item()}.wav'), transformed_X_audio, 16000)
            
            transformed_X = transformed_X.squeeze(0).cpu().numpy()
            matplotlib.image.imsave(os.path.join(_output_directory, f'sample_{indx_data}', 'noisy_melspec_image.png'), transformed_X[::-1])
            
            # ODE-only classifier baseline: decode the clean GT mel with the
            # same t=0 phoneme model used to decode the generated mel. This
            # separates generator errors from the classifier's own PER floor.
            preds_gt_mel = 'None'
            if (
                str(diffusion_cfg.get('sampler_type', 'sde')).lower() == 'ode'
                and asr_guidance_net is not None
                and decoder is not None
                and type_input_guidance in ['text', 'phoneme']
            ):
                gt_for_asr = gt_melspec.cuda()
                if gt_for_asr.ndim == 4 and gt_for_asr.shape[1] == 1:
                    gt_for_asr = gt_for_asr.squeeze(1)
                if mask_frames is None:
                    gt_length = gt_for_asr.shape[-1]
                else:
                    valid_time = mask_frames[0].reshape(
                        -1, mask_frames.shape[-1]
                    ).amax(dim=0)
                    gt_length = int((valid_time > 0).sum().item())
                gt_lengths = torch.full(
                    (gt_for_asr.shape[0],),
                    gt_length,
                    dtype=torch.long,
                    device=gt_for_asr.device,
                )
                gt_diffusion_steps = torch.zeros(
                    (gt_for_asr.shape[0], 1),
                    dtype=gt_for_asr.dtype,
                    device=gt_for_asr.device,
                )
                gt_outputs = asr_guidance_net(
                    (gt_for_asr, gt_lengths),
                    gt_diffusion_steps,
                )["outputs"]
                preds_gt_mel = decoder(gt_outputs)[0]

            # inference
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            melspec, preds_ao = sampling(net, 
                            diffusion_cfg,
                            diffusion_hyperparams,
                            w_mel_cond,
                            condition=masked_cond,
                            asr_guidance_net=asr_guidance_net,
                            w_asr=w_asr,
                            asr_start=asr_start,
                            guidance_text=text,
                            tokenizer=tokenizer,
                            decoder=decoder,
                            without_condition=without_condition,
                            mask=mask,
                            on_noisy_masked_melspec=on_noisy_masked_melspec,
                            mask_frames=mask_frames,
                            masked_audio_time_mask=masked_audio_time_mask,
                            text=true_text, 
                            input_text=input_text,
                            phoneme4guidance=phoneme4guidance,
                            per_frame_phoneme4guidance=per_frame_phoneme4guidance,
                            type_input_guidance=type_input_guidance,
                            skip_step=skip_step,
                            phoneme_guidance_debug=bool(
                                kwargs.get("phoneme_guidance_debug", False)
                            ),
                            )
            
            # -------------------------------------------------------------------------
            # Debug alternative final blend to test mask convention.
            # This must be done BEFORE denormalise_mel, while all tensors are still
            # in the normalized mel domain.
            # Current convention:
            #   mask == 1 -> known/unmasked
            #   mask == 0 -> missing/masked
            # Alternative convention tests the opposite.
            # -------------------------------------------------------------------------
            # melspec_for_alt = melspec
            # masked_for_alt = masked_melspec
            # mask_for_alt = mask

            # if melspec_for_alt.ndim == 4 and melspec_for_alt.shape[1] == 1:
            #     melspec_for_alt = melspec_for_alt.squeeze(1)

            # if masked_for_alt.ndim == 4 and masked_for_alt.shape[1] == 1:
            #     masked_for_alt = masked_for_alt.squeeze(1)

            # if mask_for_alt.ndim == 4 and mask_for_alt.shape[1] == 1:
            #     mask_for_alt = mask_for_alt.squeeze(1)

            # # ALT blend:
            # # If this sounds better, our mask convention is probably reversed somewhere.
            # melspec_alt = masked_for_alt * (1 - mask_for_alt) + melspec_for_alt * mask_for_alt
            
            # -------------------------------------------------------------------------
            
            melspec = denormalise_mel(melspec)
            end.record()
            torch.cuda.synchronize()
            print('generated sample_{} in {} seconds'.format(indx_data, int(start.elapsed_time(end)/1000)))

            # save text
            text_filename = os.path.join(_output_directory, f'sample_{indx_data}', 'asr_text.txt')
            with open(text_filename, 'w') as f:
                if type_input_guidance == 'text':
                    f.write("True text:  " + true_text_str + "\n")
                    f.write("asr_condition       :  " +text+"\n")
                    f.write("asr_generated_signal:  " + preds_ao + "\n")
                    if preds_gt_mel != 'None':
                        f.write("asr_ground_truth_mel:  " + preds_gt_mel)
                elif type_input_guidance == 'phoneme':
                    f.write("True text:  " + true_text_str + "\n")
                    f.write("text4phoneme:  " + text + "\n")
                    f.write("asr_condition       :  " +" ".join(phoneme4guidance[0])+"\n")
                    f.write("asr_generated_signal:  " + " ".join(preds_ao) + "\n")
                    if preds_gt_mel != 'None':
                        f.write("asr_ground_truth_mel:  " + " ".join(preds_gt_mel))
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

            # generate audio from generated melspec
            for vocoder_name, vocoder in vocoders.items():
                audio = vocoder(melspec)
                audio = audio.squeeze()
                audio = audio / 1.1 / audio.abs().max()
                audio = audio.cpu().numpy()
                sf.write(os.path.join(_output_directory, f'sample_{indx_data}', f'generated_audio_{vocoder_name}.wav'), audio, 16000)

            # # -------------------------------------------------------------------------
            # # Save alternative mask-convention generated audio.
            # # This is only for debugging whether the final mask blending is reversed.
            # # -------------------------------------------------------------------------
            # melspec_alt_denorm = denormalise_mel(melspec_alt)
            # if melspec_alt_denorm.ndim == 4 and melspec_alt_denorm.shape[1] == 1:
            #     melspec_alt_denorm = melspec_alt_denorm.squeeze(1)

            # if melspec_alt_denorm.ndim == 2:
            #     melspec_alt_denorm = melspec_alt_denorm.unsqueeze(0)

            # melspec_alt_denorm = melspec_alt_denorm.float().contiguous()

            # for vocoder_name, vocoder in vocoders.items():
            #     audio_alt = vocoder(melspec_alt_denorm.cuda())
            #     audio_alt = audio_alt.squeeze()
            #     audio_alt = audio_alt / 1.1 / (audio_alt.abs().max() + 1e-8)
            #     audio_alt = audio_alt.cpu().numpy()
            #     sf.write(os.path.join(_output_directory, f'sample_{indx_data}', f'generated_audio_ALT_MASK_{vocoder_name}.wav'), audio_alt, 16000)
            # # ------------------------------------------------------------------------
                        
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

@hydra.main(version_base=None, config_path="/home/dsi/sellama/SpeechRepainting/configs_Alon_Matan", config_name="config_dit_without-space-phoneme_on-masked-mel_for_inference")
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
