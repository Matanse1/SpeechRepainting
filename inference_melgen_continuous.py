# this file is a combined version that performs continuous SDE diffusion inference
# using the dataset looping/generation architecture of inference_melgen.py
# and the sampling/SDE predictor-corrector logic of inference_full_mel_only_new.py.

import os
import time
import warnings
warnings.filterwarnings("ignore")
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
from utils import (
    find_max_epoch, print_size, get_diffusion_hyperparams,
    local_directory, fix_len_compatibility, pad_last_dim, preprocess_text
)
from SDE import VPSDE, VESDE
from sampling import get_pc_sampler


def sampling(
    net,
    diffusion_cfg,
    diffusion_hyperparams,
    w_mel_cond,
    conditions=None,
    mask=None,
    on_noisy_masked_melspec=False,
    mask_frames=None,
    masked_audio_time_mask=None,
    text=None,
    input_text=None,
    # SDE and ASR specific parameters from inference_full_mel_only_new
    condition=None,
    asr_guidance_net=None,
    w_asr=None,
    asr_start=None,
    guidance_text=None,
    tokenizer=None,
    decoder=None,
    without_condtion=False,
    without_condition=False,  # Support both spellings
    phoneme4guidance=None,
    per_frame_phoneme4guidance=None,
    type_input_guidance='text',
    skip_step=1,
    tokens=None,
):
    """
    Perform sampling step supporting both continuous SDE (VPSDE/VESDE) with Predictor-Corrector
    and classic DDPM (linear/cosine) reverse diffusion paths.
    """
    if condition is None:
        condition = conditions
    if conditions is None:
        conditions = condition

    # Resolve spelling discrepancy
    without_cond_val = without_condtion or without_condition

    preds_ao = 'None'

    # Handle ASR tokenization if guidance net is active
    if asr_guidance_net is not None and tokens is None:
        if type_input_guidance == 'text' and guidance_text is not None:
            tokens = torch.LongTensor(tokenizer.encode(guidance_text))
            tokens = tokens.unsqueeze(0).cuda()
        elif type_input_guidance == 'phoneme' and phoneme4guidance is not None:
            tokens = torch.tensor([tokenizer[t] for t in phoneme4guidance[0]]).unsqueeze(0).cuda()
        elif type_input_guidance == 'frame_level_phoneme' and per_frame_phoneme4guidance is not None:
            tokens = torch.tensor(per_frame_phoneme4guidance[0], dtype=torch.int64).unsqueeze(0).cuda()

    masked_melspec, masked_audio_time = condition
    if masked_melspec.ndim == 4 and masked_melspec.shape[1] == 1:
        masked_melspec = masked_melspec.squeeze(1)

    if mask is not None and mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask.squeeze(1)

    # Re-wrap condition with updated dimensions
    condition = (masked_melspec, masked_audio_time)

    _dh = diffusion_hyperparams

    # ── DDPM path (original, linear/cosine) ──
    if _dh["name"] in ["linear", "cosine"]:
        T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
        assert len(Alpha_bar) == T
        assert len(Sigma) == T

        x = torch.normal(0, 1, size=masked_melspec.shape).cuda()
        with torch.no_grad():
            for t in tqdm(range(T-1, -1, -1), desc="DDPM Sampling"):
                diffusion_steps = (t * torch.ones((x.shape[0], 1))).cuda()
                if on_noisy_masked_melspec:
                    x = masked_melspec * mask + x * (1 - mask)
                else:
                    z = torch.normal(0, 1, size=masked_melspec.shape).cuda()
                    noisy_masked_melspec = torch.sqrt(Alpha_bar[diffusion_steps.int()]) * masked_melspec + torch.sqrt(1-Alpha_bar[diffusion_steps.int()]) * z
                    x = noisy_masked_melspec * mask + x * (1 - mask)

                epsilon_theta = net(x, condition, diffusion_steps, cond_drop_prob=0, text=text, input_text=input_text, mask_padding_time=masked_audio_time_mask, mask_padding_frames=mask_frames)
                if net.g_model_cfg.predict_type == 'speech':
                    epsilon_theta = (x - torch.sqrt(Alpha_bar[diffusion_steps.int()]) * epsilon_theta) / torch.sqrt(1-Alpha_bar[diffusion_steps.int()])

                epsilon_theta_uncond = net(x, condition, diffusion_steps, cond_drop_prob=1, text=text, input_text=input_text, mask_padding_time=masked_audio_time_mask, mask_padding_frames=mask_frames)
                if net.g_model_cfg.predict_type == 'speech':
                    epsilon_theta_uncond = (x - torch.sqrt(Alpha_bar[diffusion_steps.int()]) * epsilon_theta_uncond) / torch.sqrt(1-Alpha_bar[diffusion_steps.int()])

                epsilon_theta = (1 + w_mel_cond) * epsilon_theta - w_mel_cond * epsilon_theta_uncond

                x = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])
                if t > 0:
                    x = x + Sigma[t] * torch.normal(0, 1, size=x.shape).cuda()

        if on_noisy_masked_melspec:
            x = masked_melspec * mask + x * (1 - mask)
        if mask_frames is not None:
            x = x[..., :int(torch.sum(mask_frames, dim=-1).item())]
        return x, preds_ao

    # ── SDE path (new continuous diffusion) ──
    elif _dh["name"] in ["VPSDE", "VESDE"]:
        loss_ce = nn.CrossEntropyLoss(reduction='none')

        if _dh["name"] == "VPSDE":
            sde = VPSDE(_dh["beta_min"], _dh["beta_max"], _dh["N"])
        else:
            sde = VESDE(_dh["sigma_min"], _dh["sigma_max"], _dh["N"])

        # ── score_fn: CFG ──
        def score_fn_without_asr(x, t):
            if x.ndim == 4:
                x = x.squeeze(1)

            B = x.shape[0]
            t_input = t.view(B, 1)

            if without_cond_val:
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

        # ── asr_guidance_fn ──
        def asr_guidance_fn(x, y, t):
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
                    batch_losses = asr_guidance_net.forward_model(inputs, t.view(x.shape[0], 1), targets, compute_metrics=False, verbose=0)[0]
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
            asr_scale = 0.0 if w_asr is None else w_asr
            return score + asr_scale * guidance

        # ── call PC sampler ──
        predictor_opt = diffusion_cfg.get('predictor', "reverse_diffusion") if hasattr(diffusion_cfg, 'get') else "reverse_diffusion"
        corrector_opt = diffusion_cfg.get('corrector', "langevin") if hasattr(diffusion_cfg, 'get') else "langevin"
        snr_opt = diffusion_cfg.get('snr', 0.1) if hasattr(diffusion_cfg, 'get') else 0.1
        corr_steps_opt = diffusion_cfg.get('corrector_steps', 1) if hasattr(diffusion_cfg, 'get') else 1

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
            mask_noise=on_noisy_masked_melspec
        )

        x, nfe = pc_sampler()
        print(f"PC sampler finished in {nfe} function evaluations")

        # ── final masking ──
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
        save_dir,
        ckpt_iter="max",
        name=None,
        n_samples=None,
        w_mel_cond=0,
        on_noisy_masked_melspec=False,
        # Additional guidance / SDE arguments from inference_full_mel_only_new
        apply_asr_guidance=False,
        w_asr=1.1,
        asr_start=250,
        type_input_guidance='text',
        without_condtion=False,
        skip_step=1,
        with_space=False,
        **kwargs
    ):
    """
    Generate melspectrograms based on lips movement using the simple generator logic.
    """
    if rank is not None:
        print(f"rank {rank} {torch.cuda.device_count()} GPUs")
        torch.cuda.set_device(rank % torch.cuda.device_count())

    local_path, checkpoint_directory = local_directory(name, model_cfg, diffusion_cfg, save_dir, 'checkpoint')

    # Map diffusion hyperparameters
    diffusion_hyperparams = get_diffusion_hyperparams(diffusion_cfg, fast=True)

    # Predefine model
    builder = ModelBuilder()
    net_diffwave = builder.build_model(model_cfg)
    net = AudioVisualModel(g_model_cfg, net_diffwave).cuda()
    net.eval()

    # Load checkpoint
    ckpt_path = kwargs.get('ckpt_path', None)
    if ckpt_path is not None:
        model_path = ckpt_path
        ckpt_iter = 'specified'
    else:
        print('ckpt_iter', ckpt_iter)
        if ckpt_iter == 'max':
            ckpt_iter = find_max_epoch(checkpoint_directory)
        ckpt_iter = int(ckpt_iter)
        model_path = os.path.join(checkpoint_directory, '{}.pkl'.format(ckpt_iter))

    try:
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
    except Exception as e:
        print(e)
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
            mask = 1 - explosions_activity
            _mask = mask.cuda()
            _gt_melspec = speech_melspec.cuda()
            mix_melspec, mix_time = mix_melspec.cuda(), mix_time.cuda()
            _masked_cond = [mix_melspec, mix_time]
        elif dataset_type == 'speech_inpainting':
            _gt_melspec, *_masked_cond, _mask = dataset[i]
            mask = _mask.unsqueeze(0).cuda()
            _masked_cond = [_masked_cond[j].unsqueeze(0).cuda() for j in range(len(_masked_cond))]
        elif dataset_type == 'plc_task':
            _gt_melspec, masked_melspec, masked_audio_time, frame_mask, sample_mask = dataset[i]
            _mask = frame_mask
            _masked_cond = [masked_melspec.cuda(), masked_audio_time.cuda()]
            _masked_cond = [_masked_cond[j].unsqueeze(0).cuda() for j in range(len(_masked_cond))]
        elif dataset_cfg.dataset_type == 'speech_inpainting_anechoic':
            if model_cfg.text_embed_prop.use_text_embed_rep or model_cfg.tts_kw.use_tts:
                _gt_melspec, masked_melspec, masked_audio_time, _mask, text, input_text = dataset[i]
                input_text = [input_text]
            else:
                _gt_melspec, masked_melspec, masked_audio_time, _mask, text = dataset[i]
            _mask = _mask.unsqueeze(0).cuda()
            _masked_cond = [masked_melspec.cuda(), masked_audio_time.cuda()]
            _masked_cond = [_masked_cond[j].unsqueeze(0) for j in range(len(_masked_cond))]

        # For Unet fix length compatibility
        if model_cfg._name_ == 'unet':
            freq_signal, time_signal = _masked_cond
            desired_num_frames = fix_len_compatibility(_gt_melspec.shape[-1])
            masked_audio_time_mask = torch.ones_like(time_signal)
            masked_audio_time_mask = pad_last_dim(masked_audio_time_mask, (desired_num_frames - freq_signal.shape[-1]) * dataset_cfg[dataset_type]["audio_stft_hop"])
            _gt_melspec = pad_last_dim(_gt_melspec, desired_num_frames - _gt_melspec.shape[-1])
            time_signal = pad_last_dim(time_signal, (desired_num_frames - freq_signal.shape[-1]) * dataset_cfg[dataset_type]["audio_stft_hop"])
            freq_signal = pad_last_dim(freq_signal, desired_num_frames - freq_signal.shape[-1]).cuda()
            mask_frames = torch.zeros((list(_mask.shape[:-1]) + [desired_num_frames]))
            mask_frames[..., :_mask.shape[-1]] = 1
            mask_frames = mask_frames.cuda()
            _mask = pad_last_dim(_mask, desired_num_frames - _mask.shape[-1], pad_value=1)
            _masked_cond = [freq_signal, time_signal]
        else:
            mask_frames = None
            masked_audio_time_mask = None

        _gt_melspec = denormalise_mel(_gt_melspec)
        groundtruth_melspec.append(_gt_melspec.unsqueeze(0))
        masked_cond.append(_masked_cond)
        masks.append(_mask)
        mask_frames_list.append(mask_frames)
        text_list.append(text)
        input_text_list.append(input_text)
        masked_audio_time_mask_list.append(masked_audio_time_mask)

    # Dynamic loading of ASR guidance models if requested
    asr_guidance_net, tokenizer, decoder = None, None, None
    if apply_asr_guidance:
        try:
            import ASR.asr_models as asr_models
            ds_name = 'LRS3'
            asr_guidance_net, tokenizer, decoder = asr_models.get_models(ds_name, type_input_guidance=type_input_guidance, with_space=with_space)
            print('ASR Guidance Network, Tokenizer and Decoder successfully loaded')
        except Exception as e:
            print(f"Could not load ASR guidance library components: {e}")

    print(f'begin generating melspectrograms | {n_samples} samples')

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

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
            # SDE & guidance parameters
            asr_guidance_net=asr_guidance_net,
            w_asr=w_asr,
            asr_start=asr_start,
            guidance_text=text_list[i],
            tokenizer=tokenizer,
            decoder=decoder,
            without_condtion=without_condtion,
            type_input_guidance=type_input_guidance,
            skip_step=skip_step,
        )
        if isinstance(res, tuple):
            _melspec, preds_ao = res
        else:
            _melspec = res

        generated_melspec.append(denormalise_mel(_melspec))

    end.record()
    torch.cuda.synchronize()
    print('generated {} samples at iteration {} in {} seconds'.format(n_samples, ckpt_iter, int(start.elapsed_time(end)/1000)))

    # Save to directory
    output_dir = os.path.join(save_dir, 'generated_mels_continuous')
    os.makedirs(output_dir, exist_ok=True)
    for idx, mel in enumerate(generated_melspec):
        torch.save(mel.squeeze(0).cpu(), os.path.join(output_dir, f'sample_{idx}_generated_spec.npz'))
        try:
            mel_np = mel.squeeze(0).cpu().numpy()
            matplotlib.image.imsave(os.path.join(output_dir, f'sample_{idx}_spec_image.png'), mel_np[::-1])
        except Exception:
            pass

    # Move all tensors to CPU for consistency with inference_melgen.py
    generated_melspec = [mel.cpu() for mel in generated_melspec]
    groundtruth_melspec = [gt.cpu() for gt in groundtruth_melspec]
    # masked_cond contains lists of tensors, move each to CPU
    masked_cond = [[tensor.cpu() if hasattr(tensor, 'cpu') else tensor for tensor in cond_list] for cond_list in masked_cond]

    return generated_melspec, groundtruth_melspec, masked_cond


@hydra.main(version_base=None, config_path="configs_Alon_Matan", config_name="config_dit_without-space-phoneme_on-masked-mel_for_inference")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

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