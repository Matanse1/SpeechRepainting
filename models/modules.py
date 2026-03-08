from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from .sma import StepwiseMonotonicMultiheadAttention
"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""


import math
from typing import Optional

import torch
import torch.nn.functional as F
import torchaudio
from librosa.filters import mel as librosa_mel_fn
from torch import nn
from x_transformers.x_transformers import apply_rotary_pos_emb


# raw wav to mel spec


mel_basis_cache = {}
hann_window_cache = {}


def get_bigvgan_mel_spectrogram(
    waveform,
    n_fft=1024,
    n_mel_channels=100,
    target_sample_rate=24000,
    hop_length=256,
    win_length=1024,
    fmin=0,
    fmax=None,
    center=False,
):  # Copy from https://github.com/NVIDIA/BigVGAN/tree/main
    device = waveform.device
    key = f"{n_fft}_{n_mel_channels}_{target_sample_rate}_{hop_length}_{win_length}_{fmin}_{fmax}_{device}"

    if key not in mel_basis_cache:
        mel = librosa_mel_fn(sr=target_sample_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=fmin, fmax=fmax)
        mel_basis_cache[key] = torch.from_numpy(mel).float().to(device)  # TODO: why they need .float()?
        hann_window_cache[key] = torch.hann_window(win_length).to(device)

    mel_basis = mel_basis_cache[key]
    hann_window = hann_window_cache[key]

    padding = (n_fft - hop_length) // 2
    waveform = torch.nn.functional.pad(waveform.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)

    spec = torch.stft(
        waveform,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)

    mel_spec = torch.matmul(mel_basis, spec)
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))

    return mel_spec


def get_vocos_mel_spectrogram(
    waveform,
    n_fft=1024,
    n_mel_channels=100,
    target_sample_rate=24000,
    hop_length=256,
    win_length=1024,
):
    mel_stft = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mel_channels,
        power=1,
        center=True,
        normalized=False,
        norm=None,
    ).to(waveform.device)
    if len(waveform.shape) == 3:
        waveform = waveform.squeeze(1)  # 'b 1 nw -> b nw'

    assert len(waveform.shape) == 2

    mel = mel_stft(waveform)
    mel = mel.clamp(min=1e-5).log()
    return mel


class MelSpec(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24_000,
        mel_spec_type="vocos",
    ):
        super().__init__()
        assert mel_spec_type in ["vocos", "bigvgan"], print("We only support two extract mel backend: vocos or bigvgan")

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.target_sample_rate = target_sample_rate

        if mel_spec_type == "vocos":
            self.extractor = get_vocos_mel_spectrogram
        elif mel_spec_type == "bigvgan":
            self.extractor = get_bigvgan_mel_spectrogram

        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    def forward(self, wav):
        if self.dummy.device != wav.device:
            self.to(wav.device)

        mel = self.extractor(
            waveform=wav,
            n_fft=self.n_fft,
            n_mel_channels=self.n_mel_channels,
            target_sample_rate=self.target_sample_rate,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

        return mel


# sinusoidal position embedding


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# convolutional position embedding



class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim, kernel_size=31, groups=16):
        super().__init__()
        assert kernel_size % 2 != 0
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )

    def forward(self, x: float["b n d"], mask: bool["b n"] | None = None, true_length=None):  # noqa: F722
        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.0)
        if true_length is not None:
            mask = self.create_padding_mask(x.size(1), true_length)
            x = x.masked_fill(~mask, 0.0)
            
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        out = x.permute(0, 2, 1)

        if mask is not None:
            out = out.masked_fill(~mask, 0.0)

        return out
    
    def create_padding_mask(self, max_len, actual_lengths):
        """
        Creates attention padding mask
        Args:
            max_len: Maximum length in the batch
            actual_lengths: List/tensor of actual sequence lengths
        Returns:
            mask: Boolean mask with False at valid positions and True at padding positions
        """
        batch_size = len(actual_lengths)
        mask = torch.arange(max_len)[None, :].cuda() >= actual_lengths[:, None]  # [B, max_len]
        mask = mask.unsqueeze(-1)  # [B, max_len, 1]
        return ~mask


# rotary positional embedding related


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, theta_rescale_factor=1.0):
    # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
    # has some connection to NTK literature
    # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
    # https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return torch.cat([freqs_cos, freqs_sin], dim=-1)


def get_pos_embed_indices(start, length, max_pos, scale=1.0):
    # length = length if isinstance(length, int) else length.max()
    scale = scale * torch.ones_like(start, dtype=torch.float32)  # in case scale is a scalar
    pos = (
        start.unsqueeze(1)
        + (torch.arange(length, device=start.device, dtype=torch.float32).unsqueeze(0) * scale.unsqueeze(1)).long()
    )
    # avoid extra long error.
    pos = torch.where(pos < max_pos, pos, max_pos - 1)
    return pos


# Global Response Normalization layer (Instance Normalization ?)


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


# ConvNeXt-V2 Block https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py
# ref: https://github.com/bfs18/e2_tts/blob/main/rfwave/modules.py#L108


class ConvNeXtV2Block(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
    ):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)  # b n d -> b d n
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # b d n -> b n d
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


# AdaLayerNormZero
# return with modulated x for attn input, and params for later mlp modulation


class AdaLayerNormZero(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)

        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


# AdaLayerNormZero for final layer
# return only with modulated x for attn input, cuz no more mlp modulation


class AdaLayerNormZero_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)

        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


# FeedForward


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0, approximate: str = "none"):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        activation = nn.GELU(approximate=approximate)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), activation)
        self.ff = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.ff(x)


# Attention with possible joint part
# modified from diffusers/src/diffusers/models/attention_processor.py


class Attention(nn.Module):
    def __init__(
        self,
        processor: JointAttnProcessor | AttnProcessor,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,  # if not None -> joint attention
        context_pre_only=None,
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Attention equires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.processor = processor

        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout

        self.context_dim = context_dim
        self.context_pre_only = context_pre_only

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)

        if self.context_dim is not None:
            self.to_k_c = nn.Linear(context_dim, self.inner_dim)
            self.to_v_c = nn.Linear(context_dim, self.inner_dim)
            if self.context_pre_only is not None:
                self.to_q_c = nn.Linear(context_dim, self.inner_dim)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, dim))
        self.to_out.append(nn.Dropout(dropout))

        if self.context_pre_only is not None and not self.context_pre_only:
            self.to_out_c = nn.Linear(self.inner_dim, dim)

    def forward(
        self,
        x: float["b n d"],  # noised input x  # noqa: F722
        c: float["b n d"] = None,  # context c  # noqa: F722
        mask: bool["b n"] | None = None,  # noqa: F722
        rope=None,  # rotary position embedding for x
        c_rope=None,  # rotary position embedding for c
    ) -> torch.Tensor:
        if c is not None:
            return self.processor(self, x, c=c, mask=mask, rope=rope, c_rope=c_rope)
        else:
            return self.processor(self, x, mask=mask, rope=rope)


# Attention processor


class AttnProcessor:
    def __init__(self):
        pass

    def __call__(
        self,
        attn: Attention,
        x: float["b n d"],  # noised input x  # noqa: F722
        mask: bool["b n"] | None = None,  # noqa: F722
        rope=None,  # rotary position embedding
    ) -> torch.FloatTensor:
        batch_size = x.shape[0]

        # `sample` projections.
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        # apply rotary position embedding
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)

            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        # attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # mask. e.g. inference got a batch with different target durations, mask out the padding
        if mask is not None:
            attn_mask = mask
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # 'b n -> b 1 1 n'
            attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
        else:
            attn_mask = None

        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        x = x.to(query.dtype)

        # linear proj
        x = attn.to_out[0](x)
        # dropout
        x = attn.to_out[1](x)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)

        return x


# Joint Attention processor for MM-DiT
# modified from diffusers/src/diffusers/models/attention_processor.py


class JointAttnProcessor:
    def __init__(self):
        pass

    def __call__(
        self,
        attn: Attention,
        x: float["b n d"],  # noised input x  # noqa: F722
        c: float["b nt d"] = None,  # context c, here text # noqa: F722
        mask: bool["b n"] | None = None,  # noqa: F722
        rope=None,  # rotary position embedding for x
        c_rope=None,  # rotary position embedding for c
    ) -> torch.FloatTensor:
        residual = x

        batch_size = c.shape[0]

        # `sample` projections.
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        # `context` projections.
        c_query = attn.to_q_c(c)
        c_key = attn.to_k_c(c)
        c_value = attn.to_v_c(c)

        # apply rope for context and noised input independently
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)
        if c_rope is not None:
            freqs, xpos_scale = c_rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            c_query = apply_rotary_pos_emb(c_query, freqs, q_xpos_scale)
            c_key = apply_rotary_pos_emb(c_key, freqs, k_xpos_scale)

        # attention
        query = torch.cat([query, c_query], dim=1)
        key = torch.cat([key, c_key], dim=1)
        value = torch.cat([value, c_value], dim=1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # mask. e.g. inference got a batch with different target durations, mask out the padding
        if mask is not None:
            attn_mask = F.pad(mask, (0, c.shape[1]), value=True)  # no mask for c (text)
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # 'b n -> b 1 1 n'
            attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
        else:
            attn_mask = None

        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        x = x.to(query.dtype)

        # Split the attention outputs.
        x, c = (
            x[:, : residual.shape[1]],
            x[:, residual.shape[1] :],
        )

        # linear proj
        x = attn.to_out[0](x)
        # dropout
        x = attn.to_out[1](x)
        if not attn.context_pre_only:
            c = attn.to_out_c(c)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)
            # c = c.masked_fill(~mask, 0.)  # no mask for c (text)

        return x, c
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * x.sigmoid()
  
  

class TF_Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1d_t_1 = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv1d_t_2 = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=2, dilation=2)
        self.sigmoid_t = nn.Sigmoid()
        self.prelu_t = nn.PReLU()
        self.adapt_avrg_pooling_t = nn.AdaptiveAvgPool2d((1, None))
        
        self.conv1d_f_1 = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv1d_f_2 = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=2, dilation=2)
        self.sigmoid_f = nn.Sigmoid()
        self.prelu_f = nn.PReLU()
        self.adapt_avrg_pooling_f = nn.AdaptiveAvgPool2d((None, 1))
        
    def forward(self, input):
        output_t = self.adapt_avrg_pooling_t(input) #[B, 1, T]
        output_t = self.sigmoid_t(self.prelu_t(self.conv1d_t_2(self.conv1d_t_1(output_t))))
        
        output_f = self.adapt_avrg_pooling_f(input) #[B, F, 1]
        output_f = torch.transpose(output_f, 1, 2) #[B, 1, F]
        output_f = self.sigmoid_f(self.prelu_f(self.conv1d_f_2(self.conv1d_f_1(output_f))))
        output_f = torch.transpose(output_f, 1, 2) #[B, F, 1]
        
        attention_w = output_f @ output_t #[B, F, T]
        output = input * attention_w 
        return output
    
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, dropout):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
                        # Ensure no row is entirely masked (NaN issue)
            all_inf_rows = attn == -np.inf
            all_inf_mask = all_inf_rows.all(dim=-1, keepdim=True)  # (B, T1, 1)

            # Replace fully masked rows with zeros before softmax
            
            attn = self.softmax(attn)
            # attn_weights_np = attn.detach().cpu().numpy()
            # plt.close()
            # plt.imshow(attn_weights_np[0], cmap='viridis')
            # plt.colorbar()
            # plt.title('Attention Weights')
            # plt.xlabel('Key Positions')
            # plt.ylabel('Query Positions')
            # plt.savefig("/home/dsi/moradim/SpeechRepainting/attn_mask_plot_attn.png")
            attn = attn.masked_fill(all_inf_mask, 0)
        else:
            attn = self.softmax(attn)
            
        p_attn = self.dropout(attn)

        output = torch.bmm(p_attn, v)
        return output, attn

class CrossAttentionCustom(nn.Module):
    ''' Cross Attention module with Conv layers for k, q, v '''
    def __init__(self, n_head, d_model, d_k, d_v, kernel_size=13, dropout=0., spectral_norm=False, tts_output='mel'): #TODO add here the rope, in the init and remove from other locations
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        
        if tts_output == 'mel' or tts_output == 'phoneme' or tts_output == 'phoneme_with_energy_pitch': #TODO dont forget to remove this
            self.w_qs = nn.Conv1d(d_model, n_head * d_k, kernel_size, padding=kernel_size // 2, groups=n_head)
            self.w_ks = nn.Conv1d(d_model, n_head * d_k, kernel_size, padding=kernel_size // 2, groups=n_head)
            self.w_vs = nn.Conv1d(d_model, n_head * d_v, kernel_size, padding=kernel_size // 2, groups=n_head)
        elif tts_output == 'phoneme':
            self.w_qs = nn.Linear(d_model, n_head * d_k)
            self.w_ks = nn.Linear(d_model, n_head * d_k)
            self.w_vs = nn.Linear(d_model, n_head * d_v)
        # Normalization & Activation after Conv1d
        # self.norm_q = nn.LayerNorm(n_head * d_k)
        # self.norm_k = nn.LayerNorm(n_head * d_k)
        # self.norm_v = nn.LayerNorm(n_head * d_v)

        # self.activation = Swish()
        
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), dropout=dropout)

        self.fc = nn.Linear(n_head * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        if spectral_norm:
            self.w_qs = nn.utils.spectral_norm(self.w_qs)
            self.w_ks = nn.utils.spectral_norm(self.w_ks)
            self.w_vs = nn.utils.spectral_norm(self.w_vs)
            self.fc = nn.utils.spectral_norm(self.fc)
            
        # Initialize weights
        self._init_weights()
    def _init_weights(self):
        for layer in [self.w_qs, self.w_ks, self.w_vs, self.fc]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, query, context, query_mask=None, context_mask=None, rope=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = query.size()
        sz_b, len_k, _ = context.size()
        
        # Transpose to match Conv1d input format (batch, channels, sequence_length)
        query = query.permute(0, 2, 1)  # (batch, d_model, len_q)
        context = context.permute(0, 2, 1)  # (batch, d_model, len_k)
        
        
        q = self.w_qs(query).permute(0, 2, 1)
        k = self.w_ks(context).permute(0, 2, 1)
        v = self.w_vs(context).permute(0, 2, 1)
        
        # apply rotary position embedding
        if rope is not None:
            rope_noisy_mel, rope_tts = rope
            freqs_noisy_mel, xpos_scale_noisy_mel = rope_noisy_mel
            freqs_tts, xpos_scale_tts = rope_tts
            
            q_xpos_scale, k_xpos_scale = (1.0, 1.0)

            q = apply_rotary_pos_emb(q, freqs_noisy_mel, q_xpos_scale)
            k = apply_rotary_pos_emb(k, freqs_tts, k_xpos_scale)
        
        q = q.view(sz_b, len_q, n_head, d_k)
        k = k.view(sz_b, len_k, n_head, d_k)
        v = v.view(sz_b, len_k, n_head, d_v)
        
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_v)  # (n*b) x lv x dv

        if query_mask is not None and context_mask is not None:
            attn_mask = context_mask.unsqueeze(1).unsqueeze(1)  # '[B T2] -> [B 1 1 T2]'
            attn_mask = attn_mask.expand(sz_b, n_head, len_q, len_k)
            attn_mask = attn_mask.permute(1, 0, 2, 3).contiguous().view(-1, len_q, len_k)
            
            # attn_mask = query_mask.unsqueeze(2) | context_mask.unsqueeze(1)  # [B, T1, T2]
            # attn_mask = attn_mask.unsqueeze(1).expand(-1, n_head, -1, -1)  # [B, n_head, T1, T2]
            # attn_mask = attn_mask.permute(1, 0, 2, 3).contiguous().view(-1, len_q, len_k)  # Flatten batch & heads: [(B*n_head), T1, T2]
        else:
            attn_mask = None
        output, attn = self.attention(q, k, v, mask=attn_mask) # output = [B*n_head,lq, d_v]

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)
        output = output.to(query.dtype)
        output = self.fc(output)
        output = self.dropout(output)
        return output, attn


class CrossAttention_noise(nn.Module):
    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.0, norm='LayerNorm', custom_multi_att='custom_multi_att', tts_output='mel'):
        super().__init__()
        self.norm = norm
        self.custom_multi_att = custom_multi_att
        if custom_multi_att == 'custom_multi_att':
            self.attn = CrossAttentionCustom(heads, dim_head, d_k=int(dim/heads), d_v=int(dim/heads), dropout=dropout, tts_output=tts_output) #n_head, d_model, d_k, d_v
        elif custom_multi_att == 'monotonic_custom_multi_att':
            self.attn = StepwiseMonotonicMultiheadAttention(dim_head, d_k=int(dim/heads), d_v=int(dim/heads), n_head=heads, dropout=dropout, is_tunable=True)
        else:
            self.attn = nn.MultiheadAttention(dim_head, heads, batch_first=True)
        if norm == 'LayerNorm':
            self.norm_clean = nn.LayerNorm(dim_head)
            self.norm_noisy = nn.LayerNorm(dim_head)
            
        self.attn_norm = AdaLayerNormZero(dim)


        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, noisy_feats, clean_feats, noisy_lengths, clean_lengths, t, rope=None):  # x: noised input, t: time embedding
        """
        Args:
            noisy_feats: (B, T1, D) - Noisy Mel embeddings (Queries)
            clean_feats: (B, T2, D) - Clean Mel embeddings (Keys, Values)
            noisy_lengths: List[int] - Actual lengths of noisy sequences
            clean_lengths: List[int] - Actual lengths of clean sequences
        """
        if self.norm == 'LayerNorm':
            noisy_feats = self.norm_noisy(noisy_feats)
            clean_feats = self.norm_clean(clean_feats)
        # pre-norm & modulation for attention input
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(noisy_feats, emb=t)
        # Create masks for both sequences
        noisy_key_padding_mask = self.create_padding_mask(noisy_feats.size(1), noisy_lengths)  # [B, T1]
        clean_key_padding_mask = self.create_padding_mask(clean_feats.size(1), clean_lengths)  # [B, T2]
        
        # The attn_mask should be of shape (T1, T2)
        # We don't need to create it as we only want to mask padding tokens
        # for infernce

        # attn_output, attn_weights = self.attn(
        #     norm,  # queries [B, T1, D] should be norm
        #     clean_feats,  # keys [B, T2, F]
        #     clean_feats,  # values [B, T2, F]
        #     key_padding_mask=clean_key_padding_mask,  # mask for keys/values (clean)
        #     need_weights=True
        # )
        # attn_weights_np = attn_weights.detach().cpu().numpy()
        # plt.close()
        # plt.imshow(attn_weights_np[0], cmap='viridis')
        # plt.colorbar()
        # plt.title('Attention Weights')
        # plt.xlabel('Key Positions')
        # plt.ylabel('Query Positions')
        # plt.savefig("att_weights.png")
        if self.custom_multi_att == 'custom_multi_att':
            attn_output, attn_weights = self.attn(query=norm, context=clean_feats, query_mask=noisy_key_padding_mask, context_mask=clean_key_padding_mask, rope=rope)
        elif self.custom_multi_att == 'monotonic_custom_multi_att':
            attn_output, attn_weights, _ = self.attn(q=norm, k=clean_feats, v=clean_feats, q_mask=noisy_key_padding_mask, k_mask=clean_key_padding_mask, mel_len=clean_lengths)
        else:
            attn_output = self.attn(
                norm,  # queries
                clean_feats,  # keys
                clean_feats,  # values
                key_padding_mask=clean_key_padding_mask,  # mask for keys/values (clean)
                need_weights=False
            )[0]
        
            # Apply query mask after attention (zero out padding positions in output)
            noisy_mask = ~noisy_key_padding_mask.unsqueeze(-1)  # [B, T1, 1]
            attn_output = attn_output * noisy_mask

        # process attention output for input x
        noisy_feats = noisy_feats + gate_msa.unsqueeze(1) * attn_output

        norm = self.ff_norm(noisy_feats) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        noisy_feats = noisy_feats + gate_mlp.unsqueeze(1) * ff_output

        return noisy_feats
    
    def create_padding_mask(self, max_len, actual_lengths):
        """
        Creates attention padding mask
        Args:
            max_len: Maximum length in the batch
            actual_lengths: List/tensor of actual sequence lengths
        Returns:
            mask: Boolean mask with False at valid positions and True at padding positions
        """
        batch_size = len(actual_lengths)
        mask = torch.arange(max_len)[None, :].cuda() >= actual_lengths[:, None]  # [B, max_len]
        return mask
    

# DiT Block


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1, inner_attention=False, tts_output='phoneme',
                 inner_embed_dim=768, inner_num_heads=3, inner_dim_head=768):
        super().__init__()

        self.attn_norm = AdaLayerNormZero(dim)
        self.attn = Attention(
            processor=AttnProcessor(),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )
        self.inner_attention = inner_attention
        if inner_attention:
            self.inner_attn_c = CrossAttentionCustom(inner_num_heads, inner_dim_head, d_k=int(inner_embed_dim/inner_num_heads), d_v=int(inner_embed_dim/inner_num_heads), dropout=dropout, spectral_norm=False, tts_output=tts_output)
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, t, mask=None, rope=None, rope2=None, context_output=None, context_len=None, noisy_mel_len=None):  # x: noised input, t: time embedding
        # pre-norm & modulation for attention input
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        # attention
        attn_output = self.attn(x=norm, mask=mask, rope=rope)
        if self.inner_attention:
            noisy_key_padding_mask = self.create_padding_mask(norm.size(1), noisy_mel_len)  # [B, T1]
            context_key_padding_mask = self.create_padding_mask(context_output.size(1), context_len)  # [B, T2]
            attn_output_inner, attn_weights = self.inner_attn_c(query=norm, context=context_output, query_mask=noisy_key_padding_mask, context_mask=context_key_padding_mask, rope=rope2)
            attn_output = attn_output + attn_output_inner
        # process attention output for input x
        
        x = x + gate_msa.unsqueeze(1) * attn_output

        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output

        return x
    
    def create_padding_mask(self, max_len, actual_lengths):
        """
        Creates attention padding mask
        Args:
            max_len: Maximum length in the batch
            actual_lengths: List/tensor of actual sequence lengths
        Returns:
            mask: Boolean mask with False at valid positions and True at padding positions
        """
        batch_size = len(actual_lengths)
        mask = torch.arange(max_len)[None, :].cuda() >= actual_lengths[:, None]  # [B, max_len]
        return mask


# MMDiT Block https://arxiv.org/abs/2403.03206


class MMDiTBlock(nn.Module):
    r"""
    modified from diffusers/src/diffusers/models/attention.py

    notes.
    _c: context related. text, cond, etc. (left part in sd3 fig2.b)
    _x: noised input related. (right part)
    context_pre_only: last layer only do prenorm + modulation cuz no more ffn
    """

    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1, context_pre_only=False):
        super().__init__()

        self.context_pre_only = context_pre_only

        self.attn_norm_c = AdaLayerNormZero_Final(dim) if context_pre_only else AdaLayerNormZero(dim)
        self.attn_norm_x = AdaLayerNormZero(dim)
        self.attn = Attention(
            processor=JointAttnProcessor(),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            context_dim=dim,
            context_pre_only=context_pre_only,
        )

        if not context_pre_only:
            self.ff_norm_c = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_c = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")
        else:
            self.ff_norm_c = None
            self.ff_c = None
        self.ff_norm_x = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_x = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, c, t, mask=None, rope=None, c_rope=None):  # x: noised input, c: context, t: time embedding
        # pre-norm & modulation for attention input
        if self.context_pre_only:
            norm_c = self.attn_norm_c(c, t)
        else:
            norm_c, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.attn_norm_c(c, emb=t)
        norm_x, x_gate_msa, x_shift_mlp, x_scale_mlp, x_gate_mlp = self.attn_norm_x(x, emb=t)

        # attention
        x_attn_output, c_attn_output = self.attn(x=norm_x, c=norm_c, mask=mask, rope=rope, c_rope=c_rope)

        # process attention output for context c
        if self.context_pre_only:
            c = None
        else:  # if not last layer
            c = c + c_gate_msa.unsqueeze(1) * c_attn_output

            norm_c = self.ff_norm_c(c) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            c_ff_output = self.ff_c(norm_c)
            c = c + c_gate_mlp.unsqueeze(1) * c_ff_output

        # process attention output for input x
        x = x + x_gate_msa.unsqueeze(1) * x_attn_output

        norm_x = self.ff_norm_x(x) * (1 + x_scale_mlp[:, None]) + x_shift_mlp[:, None]
        x_ff_output = self.ff_x(norm_x)
        x = x + x_gate_mlp.unsqueeze(1) * x_ff_output

        return c, x


# time step conditioning embedding


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep: float["b"]):  # noqa: F821
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        time = self.time_mlp(time_hidden)  # b d
        return time



class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, conv_layers=0, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, text: int["b nt"], seq_len, drop_text=False):  # noqa: F722
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        text = text[:, :seq_len]  # curtail if character tokens are more than the mel spec tokens
        batch, text_len = text.shape[0], text.shape[1]
        text = F.pad(text, (0, seq_len - text_len), value=0)

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # b n -> b n d

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed

            # convnextv2 blocks
            text = self.text_blocks(text)

        return text