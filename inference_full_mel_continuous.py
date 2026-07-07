# Continuous SDE inference with trained phoneme-classifier guidance.
#
# This file is based on your continuous inference structure:
#   MelGen checkpoint -> continuous VPSDE/VESDE PC sampler -> generated mel
# and replaces the old ASR-guidance loader with the phoneme classifier that was
# trained through ASR.nnet.AudioEfficientConformerInterCTC.
#
# Expected usage:
#   python inference_continuous_phoneme_guidance.py
#
# Expected Hydra config additions under cfg.generate, for example:
#   apply_asr_guidance: true
#   type_input_guidance: phoneme
#   w_asr: 0.05          # start small: 0.02, 0.05, 0.1
#   asr_start: 250       # if > SDE.T, interpreted as discrete step and converted to continuous t
#   phoneme_classifier_ckpt: null  # if null, latest checkpoint is auto-found from phoneme_classifier_dir
#   phoneme_classifier_dir: /dsi/gannot-lab/gannot-lab1/users/Alon_Matan/phoneme_classifier/phoneme_guidance_EffConfCTC_without-space
#   phoneme_map_path: phoneme_to_number.json
#   phoneme_remove_space: true
#   phoneme_guidance_debug: false

import glob
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

# Make local repo imports robust when this script is launched from the repo root.
sys.path.insert(0, os.getcwd())

import matplotlib.image
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from hifi_gan.generator import Generator as Vocoder
from hifi_gan import utils as vocoder_utils
from hifi_gan.env import AttrDict

from models.model_builder import ModelBuilder
from models.audiovisual_model import AudioVisualModel
from dataloaders.dataset_lipvoicer import get_dataset
from dataloaders.stft import denormalise_mel
from utils import (
    find_max_epoch,
    print_size,
    get_diffusion_hyperparams,
    local_directory,
    fix_len_compatibility,
    pad_last_dim,
    preprocess_text,
)
from SDE import VPSDE, VESDE
from sampling import get_pc_sampler

# This import is required for the trained phoneme classifier.
import ASR.nnet as nnet

DEFAULT_HIFIGAN_CONFIG = "hifi_gan/config.json"
DEFAULT_HIFIGAN_CHECKPOINT = "/dsi/gannot-lab/gannot-lab1/users/mordehay/hifi_gan/g_02400000"


def load_hifigan_vocoder(config_path, checkpoint_path, device):
    with open(config_path) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    vocoder = Vocoder(h).to(device)
    state_dict_g = vocoder_utils.load_checkpoint(checkpoint_path, str(device))
    vocoder.load_state_dict(state_dict_g["generator"])

    vocoder.eval()
    vocoder.remove_weight_norm()
    return vocoder


def save_audio_from_mel(mel, vocoder, output_path, sample_rate=16000):
    device = next(vocoder.parameters()).device

    if mel.ndim == 2:
        mel = mel.unsqueeze(0)

    if mel.ndim == 4:
        mel = mel.squeeze(1)

    mel = mel.to(device)

    with torch.no_grad():
        audio = vocoder(mel)

    audio = audio.squeeze()
    audio = audio / 1.1 / (audio.abs().max() + 1e-8)
    audio = audio.detach().cpu().numpy()

    sf.write(output_path, audio, sample_rate)

# Phoneme helpers are provided by a separate module to keep this file focused
# on mel-generation and sampling. Import the helpers from the phoneme-only
# inference module so the full pipeline can use the same code.
from inference_phoneme_continuous import (
    DEFAULT_PHONEME_CLASSIFIER_DIR,
    resolve_existing_path,
    find_latest_phoneme_checkpoint,
    load_phoneme_map,
    unwrap_singleton,
    text_to_string,
    maybe_g2p_text,
    extract_phoneme_sequence,
    phoneme_sequence_to_ctc_targets,
    load_trained_phoneme_classifier,
    continuous_t_to_classifier_diffusion_steps,
    asr_start_to_continuous_t,
)


# -----------------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------------


def sampling(
    net,
    diffusion_cfg,
    diffusion_hyperparams,
    w_mel_cond,
    conditions=None,
    condition=None,
    mask=None,
    on_noisy_masked_melspec=False,
    mask_frames=None,
    masked_audio_time_mask=None,
    text=None,
    input_text=None,
    # Guidance-specific parameters
    asr_guidance_net=None,
    w_asr=None,
    asr_start=None,
    guidance_text=None,
    tokenizer=None,  # here this means phoneme_to_number dict
    decoder=None,
    without_condtion=False,
    without_condition=False,
    phoneme4guidance=None,
    per_frame_phoneme4guidance=None,
    type_input_guidance="phoneme",
    skip_step=1,
    tokens=None,
    token_lengths=None,
    with_space=False,
    phoneme_remove_space=True,
    phoneme_guidance_debug=False,
):
    """
    Sampling step supporting:
      - DDPM path for linear/cosine configs.
      - Continuous SDE path for VPSDE/VESDE with predictor-corrector sampler.
      - Trained phoneme classifier guidance in the continuous SDE path.
    """
    if condition is None:
        condition = conditions
    if conditions is None:
        conditions = condition
    if condition is None:
        raise RuntimeError("sampling expected condition/conditions=(masked_melspec, masked_audio_time)")

    without_cond_val = without_condtion or without_condition
    preds_ao = "None"

    masked_melspec, masked_audio_time = condition
    if masked_melspec.ndim == 4 and masked_melspec.shape[1] == 1:
        masked_melspec = masked_melspec.squeeze(1)
    if mask is not None and mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask.squeeze(1)
    condition = (masked_melspec, masked_audio_time)

    _dh = diffusion_hyperparams

    # ------------------------------------------------------------------
    # DDPM path. Kept mostly for compatibility; phoneme guidance below is
    # intended for the continuous SDE path.
    # ------------------------------------------------------------------
    if _dh["name"] in ["linear", "cosine"]:
        T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
        assert len(Alpha_bar) == T
        assert len(Sigma) == T

        x = torch.normal(0, 1, size=masked_melspec.shape, device=masked_melspec.device)
        with torch.no_grad():
            for t_step in tqdm(range(T - 1, -1, -skip_step), desc="DDPM Sampling"):
                if t_step < skip_step:
                    t_step = 0
                diffusion_steps = (t_step * torch.ones((x.shape[0], 1), device=x.device))

                if on_noisy_masked_melspec:
                    x = masked_melspec * mask + x * (1 - mask)
                else:
                    z = torch.normal(0, 1, size=masked_melspec.shape, device=x.device)
                    noisy_masked_melspec = (
                        torch.sqrt(Alpha_bar[diffusion_steps.int()]) * masked_melspec
                        + torch.sqrt(1 - Alpha_bar[diffusion_steps.int()]) * z
                    )
                    x = noisy_masked_melspec * mask + x * (1 - mask)

                epsilon_theta = net(
                    x,
                    condition,
                    diffusion_steps,
                    cond_drop_prob=0,
                    text=text,
                    input_text=input_text,
                    mask_padding_time=masked_audio_time_mask,
                    mask_padding_frames=mask_frames,
                )
                if net.g_model_cfg.predict_type == "speech":
                    epsilon_theta = (x - torch.sqrt(Alpha_bar[diffusion_steps.int()]) * epsilon_theta) / torch.sqrt(
                        1 - Alpha_bar[diffusion_steps.int()]
                    )

                epsilon_theta_uncond = net(
                    x,
                    condition,
                    diffusion_steps,
                    cond_drop_prob=1,
                    text=text,
                    input_text=input_text,
                    mask_padding_time=masked_audio_time_mask,
                    mask_padding_frames=mask_frames,
                )
                if net.g_model_cfg.predict_type == "speech":
                    epsilon_theta_uncond = (x - torch.sqrt(Alpha_bar[diffusion_steps.int()]) * epsilon_theta_uncond) / torch.sqrt(
                        1 - Alpha_bar[diffusion_steps.int()]
                    )

                epsilon_theta = (1 + w_mel_cond) * epsilon_theta - w_mel_cond * epsilon_theta_uncond
                x = (x - (1 - Alpha[t_step]) / torch.sqrt(1 - Alpha_bar[t_step]) * epsilon_theta) / torch.sqrt(Alpha[t_step])
                if t_step > 0:
                    x = x + Sigma[t_step] * torch.normal(0, 1, size=x.shape, device=x.device)

        if on_noisy_masked_melspec:
            x = masked_melspec * mask + x * (1 - mask)
        if mask_frames is not None:
            x = x[..., : int(torch.sum(mask_frames, dim=-1).item())]
        return x, preds_ao

    # ------------------------------------------------------------------
    # Continuous SDE path.
    # ------------------------------------------------------------------
    if _dh["name"] not in ["VPSDE", "VESDE"]:
        raise ValueError(f"Unsupported diffusion name: {_dh['name']}")

    if _dh["name"] == "VPSDE":
        sde = VPSDE(_dh["beta_min"], _dh["beta_max"], _dh["N"])
    else:
        sde = VESDE(_dh["sigma_min"], _dh["sigma_max"], _dh["N"])

    device = masked_melspec.device

    # Build CTC targets once per sample if a trained phoneme classifier is active.
    if asr_guidance_net is not None and tokens is None:
        if tokenizer is None:
            raise RuntimeError("Phoneme guidance requires tokenizer=phoneme_to_number dict.")
        phoneme_seq = extract_phoneme_sequence(
            tokenizer,
            guidance_text=guidance_text,
            input_text=input_text,
            phoneme4guidance=phoneme4guidance,
            type_input_guidance=type_input_guidance,
            with_space=with_space,
            remove_space=phoneme_remove_space,
        )
        tokens, token_lengths = phoneme_sequence_to_ctc_targets(phoneme_seq, tokenizer, device)
        if phoneme_guidance_debug:
            print("phoneme guidance target:", " ".join(phoneme_seq))
            print("phoneme target ids:", tokens.detach().cpu().tolist())
    elif tokens is not None and token_lengths is None:
        token_lengths = torch.tensor([tokens.shape[1]], dtype=torch.long, device=device)

    def score_fn_without_asr(x, t):
        if x.ndim == 4:
            x = x.squeeze(1)

        B = x.shape[0]
        t_input = t.view(B, 1)

        if without_cond_val:
            score = net(
                x,
                condition,
                t_input,
                cond_drop_prob=1,
                mask_padding_frames=mask_frames,
                text=text,
                input_text=input_text,
                mask_padding_time=masked_audio_time_mask,
            )
        else:
            score_cond = net(
                x,
                condition,
                t_input,
                cond_drop_prob=0,
                mask_padding_frames=mask_frames,
                text=text,
                input_text=input_text,
                mask_padding_time=masked_audio_time_mask,
            )
            score_uncond = net(
                x,
                condition,
                t_input,
                cond_drop_prob=1,
                mask_padding_frames=mask_frames,
                text=text,
                input_text=input_text,
                mask_padding_time=masked_audio_time_mask,
            )
            score = (1 + w_mel_cond) * score_cond - w_mel_cond * score_uncond

        return score

    def phoneme_guidance_fn(x, y, t):
        if x.ndim == 4:
            x = x.squeeze(1)
        if asr_guidance_net is None:
            return torch.zeros_like(x)
        if tokens is None:
            return torch.zeros_like(x)

        asr_start_t = asr_start_to_continuous_t(asr_start, sde, asr_guidance_net)
        if asr_start_t is not None and not (t[0].item() <= asr_start_t):
            return torch.zeros_like(x)

        # Normalize classifier gradient magnitude to the MelGen score magnitude.
        with torch.no_grad():
            score = score_fn_without_asr(x, t)
            score_norm = torch.norm(score.reshape(score.shape[0], -1), dim=-1).mean()

        with torch.enable_grad():
            x_for_cls = x.detach().requires_grad_(True)
            B = x_for_cls.shape[0]
            lengths = torch.full((B,), x_for_cls.shape[2], dtype=torch.long, device=x_for_cls.device)
            targets = (tokens, token_lengths)

            # Critical: convert continuous SDE time t to the classifier's diffusion-step embedding scale.
            diffusion_steps_cls = continuous_t_to_classifier_diffusion_steps(t.to(x_for_cls.device), asr_guidance_net)

            batch_losses = asr_guidance_net.forward_model(
                inputs=(x_for_cls, lengths),
                diffusion_steps=diffusion_steps_cls,
                targets=targets,
                compute_metrics=False,
                verbose=0,
            )[0]

            ctc_loss = batch_losses["loss"]
            loss_grad = torch.autograd.grad(
                ctc_loss,
                x_for_cls,
                retain_graph=False,
                create_graph=False,
            )[0]

            # CTC loss = -log p(target phonemes | x_t), so guidance uses -grad(loss).
            guidance_grad = -loss_grad
            guidance_norm = torch.norm(guidance_grad.reshape(guidance_grad.shape[0], -1), dim=-1).mean()
            grad_normaliser = score_norm / (guidance_norm + 1e-8)

        if phoneme_guidance_debug:
            print(
                "phoneme_loss=", float(ctc_loss.detach().cpu()),
                "score_norm=", float(score_norm.detach().cpu()),
                "guidance_norm=", float(guidance_norm.detach().cpu()),
                "normaliser=", float(grad_normaliser.detach().cpu()),
                "t=", float(t[0].detach().cpu()),
            )

        return (grad_normaliser * guidance_grad).detach()

    def score_fn(x, y, t):
        score = score_fn_without_asr(x, t)
        guidance = phoneme_guidance_fn(x, y, t)
        asr_scale = 0.0 if w_asr is None else float(w_asr)
        return score + asr_scale * guidance

    predictor_opt = diffusion_cfg.get("predictor", "reverse_diffusion") if hasattr(diffusion_cfg, "get") else "reverse_diffusion"
    corrector_opt = diffusion_cfg.get("corrector", "langevin") if hasattr(diffusion_cfg, "get") else "langevin"
    snr_opt = diffusion_cfg.get("snr", 0.1) if hasattr(diffusion_cfg, "get") else 0.1
    corr_steps_opt = diffusion_cfg.get("corrector_steps", 1) if hasattr(diffusion_cfg, "get") else 1

    pc_sampler = get_pc_sampler(
        predictor_name=predictor_opt,
        corrector_name=corrector_opt,
        sde=sde,
        score_fn=score_fn,
        y=masked_melspec,
        snr=snr_opt,
        corrector_steps=corr_steps_opt,
        w_mel_cond=w_mel_cond,
        mask=mask,
        mask_noise=on_noisy_masked_melspec,
    )

    x, nfe = pc_sampler()
    print(f"PC sampler finished in {nfe} function evaluations")

    # Final masking.
    x = masked_melspec * mask + x * (1 - mask)
    if mask_frames is not None:
        x = x[..., : int(torch.sum(mask_frames, dim=-1).item())]
    return x, preds_ao


# -----------------------------------------------------------------------------
# Generate loop
# -----------------------------------------------------------------------------


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
    on_noisy_masked_melspec=False,
    # Guidance parameters
    apply_asr_guidance=False,
    w_asr=0.05,
    asr_start=None,
    type_input_guidance="phoneme",
    without_condtion=False,
    skip_step=1,
    with_space=False,
    phoneme_classifier_ckpt=None,
    phoneme_classifier_dir=DEFAULT_PHONEME_CLASSIFIER_DIR,
    phoneme_map_path="phoneme_to_number.json",
    phoneme_remove_space=True,
    phoneme_classifier_beta_min=0.1,
    phoneme_classifier_beta_max=20.0,
    phoneme_classifier_N=1000,
    phoneme_classifier_att_type="patch",
    phoneme_classifier_strides_subsampling=1,
    phoneme_guidance_debug=False,
    export_audio=False,
    vocoder_config_path=None,
    vocoder_checkpoint_path=None,
    **kwargs,
):
    """Generate melspectrograms using continuous SDE inference and optional phoneme guidance."""
    if rank is not None:
        print(f"rank {rank} {torch.cuda.device_count()} GPUs")
        torch.cuda.set_device(rank % torch.cuda.device_count())

    device = torch.device("cuda")

    local_path, checkpoint_directory = local_directory(name, model_cfg, diffusion_cfg, save_dir, "checkpoint")

    # Map diffusion hyperparameters.
    diffusion_hyperparams = get_diffusion_hyperparams(diffusion_cfg, fast=True)

    # Build MelGen model.
    builder = ModelBuilder()
    net_diffwave = builder.build_model(model_cfg)
    net = AudioVisualModel(g_model_cfg, net_diffwave).cuda()
    net.eval()
    print_size(net)

    # Load MelGen checkpoint.
    ckpt_path = kwargs.get("ckpt_path", None)
    if ckpt_path is not None:
        model_path = ckpt_path
        ckpt_iter = "specified"
    else:
        print("ckpt_iter", ckpt_iter)
        if ckpt_iter == "max":
            ckpt_iter = find_max_epoch(checkpoint_directory)
        ckpt_iter = int(ckpt_iter)
        model_path = os.path.join(checkpoint_directory, f"{ckpt_iter}.pkl")

    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        model_weights = checkpoint["model_state_dict"]
        model_weights = {k: v for k, v in model_weights.items() if "wavlm_model" not in k}
        missing_keys, _ = net.load_state_dict(model_weights, strict=False)
        filtered_missing_keys = [key for key in missing_keys if "wavlm_model" not in key]
        if not filtered_missing_keys:
            print("All keys loaded successfully")
            print(f"Successfully loaded MelGen model at iteration {ckpt_iter}")
        else:
            raise Exception(f"The following keys were not loaded: {filtered_missing_keys}")
    except Exception as e:
        print(e)
        raise Exception("No valid MelGen model found")

    # Load phoneme map and trained phoneme classifier if requested.
    asr_guidance_net = None
    phoneme_to_number = None
    decoder = None

    if apply_asr_guidance:
        phoneme_to_number, num_to_phoneme, resolved_map_path = load_phoneme_map(
            phoneme_map_path=phoneme_map_path,
            remove_space=phoneme_remove_space,
        )

        # Force without-space vocabulary and compact ids.
        # blank remains 0, phonemes become 1..K.
        if phoneme_remove_space:
            phoneme_to_number.pop("space", None)
            phoneme_to_number.pop(" ", None)

        phoneme_to_number = {k: int(v) for k, v in phoneme_to_number.items()}

        # Compact the ids after removing space.
        # This removes gaps such as max_id=70 with only 69 entries.
        sorted_phonemes = sorted(phoneme_to_number.items(), key=lambda kv: kv[1])
        phoneme_to_number = {
            phoneme: idx + 1
            for idx, (phoneme, old_id) in enumerate(sorted_phonemes)
        }

        num_to_phoneme = {v: k for k, v in phoneme_to_number.items()}
        num_to_phoneme[0] = "blank"

        phoneme_vocab_size = len(phoneme_to_number) + 1

        print(f"Loaded phoneme map from: {resolved_map_path}")
        print("space in phoneme map:", "space" in phoneme_to_number)
        print("num phoneme entries:", len(phoneme_to_number))
        print("max phoneme id:", max(phoneme_to_number.values()))
        print(f"Phoneme vocab size including blank: {phoneme_vocab_size}")

        if phoneme_classifier_ckpt is None:
            phoneme_classifier_dir = resolve_existing_path(
                phoneme_classifier_dir,
                fallback_paths=[DEFAULT_PHONEME_CLASSIFIER_DIR],
                must_exist=True,
            )
            phoneme_classifier_ckpt = find_latest_phoneme_checkpoint(phoneme_classifier_dir)

        asr_guidance_net = load_trained_phoneme_classifier(
            checkpoint_path=phoneme_classifier_ckpt,
            device=device,
            vocab_size=phoneme_vocab_size,
            beta_min=phoneme_classifier_beta_min,
            beta_max=phoneme_classifier_beta_max,
            sde_N=phoneme_classifier_N,
            att_type=phoneme_classifier_att_type,
            interctc_blocks=[],
            strides_subsampling=phoneme_classifier_strides_subsampling,
        )
        print("Trained phoneme classifier guidance is active")
    else:
        print("Phoneme classifier guidance is disabled")

    # Dataset loading.
    if n_samples is None:
        raise ValueError("n_samples must be set in cfg.generate")

    dataset = get_dataset(dataset_cfg, split="test", return_mask_properties=False, return_true_text=True)
    dataset_type = dataset_cfg.dataset_type
    dataset_indices = list(range(n_samples))

    groundtruth_melspec = []
    masked_cond = []
    masks = []
    mask_frames_list = []
    text_list = []
    input_text_list = []
    masked_audio_time_mask_list = []

    for i in dataset_indices:
        text = None
        input_text = None

        if dataset_type == "explosion_speech_inpainting":
            (
                speech_melspec,
                mix_melspec,
                mix_time,
                _,
                masked_speech_time,
                explosions_activity,
                start_explosions,
                explosions_length,
            ) = dataset[i]
            _mask = (1 - explosions_activity).cuda()
            _gt_melspec = speech_melspec.cuda()
            mix_melspec, mix_time = mix_melspec.cuda(), mix_time.cuda()
            _masked_cond = [mix_melspec, mix_time]

        elif dataset_type == "speech_inpainting":
            _gt_melspec, *_masked_cond, _mask = dataset[i]
            _gt_melspec = _gt_melspec.cuda()
            _mask = _mask.unsqueeze(0).cuda()
            _masked_cond = [_masked_cond[j].unsqueeze(0).cuda() for j in range(len(_masked_cond))]

        elif dataset_type == "plc_task":
            _gt_melspec, masked_melspec, masked_audio_time, frame_mask, sample_mask = dataset[i]
            _gt_melspec = _gt_melspec.cuda()
            _mask = frame_mask.cuda() if hasattr(frame_mask, "cuda") else torch.tensor(frame_mask).cuda()
            _masked_cond = [masked_melspec.cuda(), masked_audio_time.cuda()]
            _masked_cond = [_masked_cond[j].unsqueeze(0) for j in range(len(_masked_cond))]

        elif dataset_cfg.dataset_type == "speech_inpainting_anechoic":
            # Possible dataset outputs depend on whether text/TTS conditioning is active.
            use_text_or_tts = False
            try:
                use_text_or_tts = bool(model_cfg.text_embed_prop.use_text_embed_rep or model_cfg.tts_kw.use_tts)
            except Exception:
                use_text_or_tts = False

            if use_text_or_tts:
                _gt_melspec, masked_melspec, masked_audio_time, _mask, text, input_text = dataset[i]
                input_text = [input_text]
            else:
                _gt_melspec, masked_melspec, masked_audio_time, _mask, text = dataset[i]

            _gt_melspec = _gt_melspec.cuda()
            _mask = _mask.unsqueeze(0).cuda()
            _masked_cond = [masked_melspec.cuda(), masked_audio_time.cuda()]
            _masked_cond = [_masked_cond[j].unsqueeze(0) for j in range(len(_masked_cond))]

        else:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")

        # U-Net length compatibility.
        if model_cfg._name_ == "unet":
            freq_signal, time_signal = _masked_cond
            desired_num_frames = fix_len_compatibility(_gt_melspec.shape[-1])
            masked_audio_time_mask = torch.ones_like(time_signal)
            masked_audio_time_mask = pad_last_dim(
                masked_audio_time_mask,
                (desired_num_frames - freq_signal.shape[-1]) * dataset_cfg[dataset_type]["audio_stft_hop"],
            )
            _gt_melspec = pad_last_dim(_gt_melspec, desired_num_frames - _gt_melspec.shape[-1])
            time_signal = pad_last_dim(
                time_signal,
                (desired_num_frames - freq_signal.shape[-1]) * dataset_cfg[dataset_type]["audio_stft_hop"],
            )
            freq_signal = pad_last_dim(freq_signal, desired_num_frames - freq_signal.shape[-1]).cuda()
            mask_frames = torch.zeros((list(_mask.shape[:-1]) + [desired_num_frames]), device=_mask.device)
            mask_frames[..., : _mask.shape[-1]] = 1
            _mask = pad_last_dim(_mask, desired_num_frames - _mask.shape[-1], pad_value=1)
            _masked_cond = [freq_signal, time_signal]
        else:
            mask_frames = None
            masked_audio_time_mask = None

        # Store for generation loop.
        groundtruth_melspec.append(denormalise_mel(_gt_melspec).unsqueeze(0))
        masked_cond.append(_masked_cond)
        masks.append(_mask)
        mask_frames_list.append(mask_frames)
        text_list.append(text)
        input_text_list.append(input_text)
        masked_audio_time_mask_list.append(masked_audio_time_mask)

    print(f"begin generating melspectrograms | {n_samples} samples")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    vocoder = None
    if export_audio:
        try:
            vocoder = load_hifigan_vocoder(
                config_path=vocoder_config_path,
                checkpoint_path=vocoder_checkpoint_path,
                device=device,
            )
            print(f"Loaded HiFi-GAN vocoder from {vocoder_checkpoint_path}")
        except Exception as exc:
            print(f"Could not load HiFi-GAN vocoder: {exc}")
            vocoder = None

    generated_melspec = []

    for i in tqdm(range(n_samples)):
        res = sampling(
            net=net,
            diffusion_cfg=diffusion_cfg,
            diffusion_hyperparams=diffusion_hyperparams,
            w_mel_cond=w_mel_cond,
            condition=masked_cond[i],
            mask=masks[i],
            on_noisy_masked_melspec=on_noisy_masked_melspec,
            mask_frames=mask_frames_list[i],
            masked_audio_time_mask=masked_audio_time_mask_list[i],
            text=text_list[i],
            input_text=input_text_list[i],
            # Phoneme guidance parameters.
            asr_guidance_net=asr_guidance_net,
            w_asr=w_asr,
            asr_start=asr_start,
            guidance_text=text_list[i],
            tokenizer=phoneme_to_number,
            decoder=decoder,
            without_condtion=without_condtion,
            type_input_guidance=type_input_guidance,
            skip_step=skip_step,
            with_space=with_space,
            phoneme_remove_space=phoneme_remove_space,
            phoneme_guidance_debug=phoneme_guidance_debug,
        )
        if isinstance(res, tuple):
            _melspec, preds_ao = res
        else:
            _melspec = res

        generated_melspec.append(denormalise_mel(_melspec))

    end.record()
    torch.cuda.synchronize()
    print(
        "generated {} samples at iteration {} in {} seconds".format(
            n_samples,
            ckpt_iter,
            int(start.elapsed_time(end) / 1000),
        )
    )

    # Save generated mels.
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_id += f"_asr{asr_start}_w{w_asr}"
    output_dir = os.path.join(save_dir, "generated_mels_continuous_phoneme_guidance", run_id)
    os.makedirs(output_dir, exist_ok=True)

    for idx, mel in enumerate(generated_melspec):
        sample_name = f"sample_{idx}"
        torch.save(mel.squeeze(0).cpu(), os.path.join(output_dir, f"{sample_name}_generated_spec.npz"))
        try:
            mel_np = mel.squeeze(0).cpu().numpy()
            # remove batch/channel singleton dimensions safely
            mel_np = np.squeeze(mel_np)
            # if still 3D, try to reduce common [1, F, T] / [F, T, 1] cases
            if mel_np.ndim == 3:
                if mel_np.shape[0] == 1:
                    mel_np = mel_np[0]
                elif mel_np.shape[-1] == 1:
                    mel_np = mel_np[..., 0]
                else:
                    print(f"Skipping mel image for sample {idx}, unexpected shape: {mel_np.shape}")
                    continue
            
            # now expected [F, T]
            matplotlib.image.imsave(os.path.join(output_dir, f"{sample_name}_spec_image.png"), mel_np[::-1])
        except Exception as exc:
            print(f"Could not save mel image for sample {idx}: {exc}")

        if vocoder is not None:
             # 1. Generated audio
            try:
                gen_audio_path = os.path.join(output_dir, f"{sample_name}_generated_audio_hifi_gan.wav")
                save_audio_from_mel(
                    mel.squeeze(0),
                    vocoder,
                    gen_audio_path,
                )
                print(f"Saved generated audio to: {gen_audio_path}")
            except Exception as exc:
                print(f"Could not save generated audio for sample {idx}: {exc}")

            # 2. Ground-truth audio through the same vocoder
            try:
                gt_audio_path = os.path.join(output_dir, f"{sample_name}_gt_audio_hifi_gan.wav")
                save_audio_from_mel(
                    groundtruth_melspec[idx].squeeze(0),
                    vocoder,
                    gt_audio_path,
                )
                print(f"Saved GT audio to: {gt_audio_path}")
            except Exception as exc:
                print(f"Could not save GT audio for sample {idx}: {exc}")

            # 3. Masked / condition audio through the same vocoder
            try:
                masked_mel = masked_cond[idx][0]  # masked melspec, still normalized
                masked_mel = denormalise_mel(masked_mel)

                masked_audio_path = os.path.join(output_dir, f"{sample_name}_masked_audio_hifi_gan.wav")
                save_audio_from_mel(
                    masked_mel.squeeze(0),
                    vocoder,
                    masked_audio_path,
                )
                print(f"Saved masked audio to: {masked_audio_path}")
            except Exception as exc:
                print(f"Could not save masked audio for sample {idx}: {exc}")

    generated_melspec = [mel.cpu() for mel in generated_melspec]
    groundtruth_melspec = [gt.cpu() for gt in groundtruth_melspec]
    masked_cond = [
        [tensor.cpu() if hasattr(tensor, "cpu") else tensor for tensor in cond_list]
        for cond_list in masked_cond
    ]

    return generated_melspec, groundtruth_melspec, masked_cond


@hydra.main(
    version_base=None,
    config_path="configs_Alon_Matan",
    config_name="config_dit_without-space-phoneme_on-masked-mel_for_inference",
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)

    generate(
        0,
        diffusion_cfg=cfg.diffusion,
        model_cfg=cfg[cfg.melgen],
        g_model_cfg=cfg.g_model,
        dataset_cfg=cfg.dataset,
        **cfg.generate,
    )


if __name__ == "__main__":
    main()
