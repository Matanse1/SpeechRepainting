"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations
# import sys
# sys.path.append("/home/dsi/moradim/SpeechRepainting/")
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataloaders.stft import normalise_mel
from StyleSpeech.audio.stft import TacotronSTFT
# from dataloaders.stft import TacotronSTFT
from x_transformers.x_transformers import RotaryEmbedding
from models import utils
from models.modules import (
    TimestepEmbedding,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    AdaLayerNormZero_Final,
    precompute_freqs_cis,
    get_pos_embed_indices,
    CrossAttention_noise
)
from models.utils import print_modle_size, WeightedSum, get_tokenizer, list_str_to_tensor, list_str_to_idx, list_str_to_idx_tts, get_StyleSpeech




# Text embedding


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


# noised input audio and context mixing embedding


class InputEmbedding3(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: float["b n d"], cond: float["b n d"], text_embed: float["b n d"]):  # noqa: F722
        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x

class InputEmbedding2(nn.Module):
    def __init__(self, mel_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: float["b n d"], cond: float["b n d"]):  # noqa: F722
        x = self.proj(torch.cat((x, cond), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x

class InputEmbedding1(nn.Module):
    def __init__(self, mel_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: float["b f t"], mask=None, true_length=None):  # noqa: F722
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = self.conv_pos_embed(x, mask, true_length) + x
        return x
    
class InputEmbeddingConv(nn.Module):
    def __init__(self, mel_dim, out_dim):
        super().__init__()
        kernel_size = 13
        self.proj = nn.Conv1d(mel_dim, out_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: float["b f t"], mask=None, true_length=None):  # noqa: F722
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.conv_pos_embed(x, mask, true_length) + x
        return x

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, norm='none'):
        super().__init__()
        self.norm = norm
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        if norm == 'LayerNorm':
            self.norm_clean = nn.LayerNorm(embed_dim)
            self.norm_noisy = nn.LayerNorm(embed_dim)
    
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
    
    def forward(self, noisy_feats, clean_feats, noisy_lengths, clean_lengths):
        """
        Args:
            noisy_feats: (B, T1, D) - Noisy Mel embeddings (Queries)
            clean_feats: (B, T2, D) - Clean Mel embeddings (Keys, Values)
            noisy_lengths: List[int] - Actual lengths of noisy sequences
            clean_lengths: List[int] - Actual lengths of clean sequences
        """
        # Create masks for both sequences
        noisy_key_padding_mask = self.create_padding_mask(noisy_feats.size(1), noisy_lengths)  # [B, T1]
        clean_key_padding_mask = self.create_padding_mask(clean_feats.size(1), clean_lengths)  # [B, T2]
        
        # The attn_mask should be of shape (T1, T2)
        # We don't need to create it as we only want to mask padding tokens
        # for infernce
        if self.norm == 'LayerNorm':
            noisy_feats = self.norm_noisy(noisy_feats)
            clean_feats = self.norm_clean(clean_feats)
        attn_output, attn_weights = self.attn(
            noisy_feats,  # queries [B, T1, D]
            clean_feats,  # keys [B, T2, F]
            clean_feats,  # values [B, T2, F]
            key_padding_mask=clean_key_padding_mask,  # mask for keys/values (clean)
            need_weights=True
        )
        attn_weights_np = attn_weights.detach().cpu().numpy()
        
        plt.imshow(attn_weights_np[0], cmap='viridis')
        plt.colorbar()
        plt.title('Attention Weights')
        plt.xlabel('Key Positions')
        plt.ylabel('Query Positions')
        plt.savefig("att_weights.png")
        # attn_output = self.attn(
        #     noisy_feats,  # queries
        #     clean_feats,  # keys
        #     clean_feats,  # values
        #     key_padding_mask=clean_key_padding_mask,  # mask for keys/values (clean)
        #     need_weights=False
        # )[0]
        
        # Apply query mask after attention (zero out padding positions in output)
        noisy_mask = ~noisy_key_padding_mask.unsqueeze(-1)  # [B, T1, 1]
        attn_output = attn_output * noisy_mask
        
        attn_output_np = attn_output.detach().cpu().numpy()
        
        plt.imshow(attn_output_np[0], cmap='viridis')
        plt.colorbar()
        plt.title('Attention Weights')
        plt.xlabel('Key Positions')
        plt.ylabel('Query Positions')
        plt.savefig("attn_output.png")
        return attn_output#, attn_weights


class StyleSpeecEncoderPhonemehWrapper(nn.Module):
    def __init__(self, tts_kw):
        super().__init__()

        self.positional_emd = tts_kw.positional_emd
        self.cross_attn_type = tts_kw.cross_attn_type
        self.inner_attention = tts_kw.inner_attention
        style_speech_ch_path = tts_kw["style_speech_ch_path"] # checkpoint path
        style_speech_config_path = tts_kw["style_speech_config_path"]
        self.style_speech_model, config = get_StyleSpeech(style_speech_config_path, style_speech_ch_path)
        self.stft = TacotronSTFT(
            config.filter_length,
            config.hop_length,
            config.win_length,
            config.n_mel_channels,
            config.sampling_rate,
            config.mel_fmin,
            config.mel_fmax,
            log=tts_kw.log)
        
        self.filter_length = config.filter_length
        self.hop_length = config.hop_length
        self.use_rope = tts_kw.use_rope
        self.tts_output = tts_kw.tts_output
        if self.use_rope:
            self.rotary_embed = RotaryEmbedding(tts_kw.embed_dim // tts_kw.num_heads)
        
        self.encoder_phoneme = nn.Sequential(nn.Linear(256, tts_kw.embed_dim), nn.LayerNorm(tts_kw.embed_dim)) # 256 is the dim of the phoneme embedding
        if tts_kw.positional_emd == 'none':
            self.encoder_noisy = nn.Conv1d(config.n_mel_channels, tts_kw.embed_dim, kernel_size=3, padding=1)
        elif tts_kw.positional_emd == 'InputEmbedding1':
            self.encoder_noisy = InputEmbedding1(config.n_mel_channels, tts_kw.embed_dim)
        elif tts_kw.positional_emd == 'InputEmbeddingConv':
            self.encoder_noisy = InputEmbeddingConv(config.n_mel_channels, tts_kw.embed_dim)
        if tts_kw.cross_attn_type == 'CrossAttention_noise':
            self.cross_attention = CrossAttention_noise(tts_kw.embed_dim, tts_kw.num_heads, tts_kw.embed_dim, norm=tts_kw.norm, custom_multi_att=tts_kw.custom_multi_att, tts_output=tts_kw.tts_output)
        else:
            self.cross_attention = CrossAttention(tts_kw.embed_dim, tts_kw.num_heads, tts_kw.norm)

    def forward(self, masked_audio_time, input_text, noisy_mel, mask_padding_time, mask_padding_frames, t):
        """_summary_

        Args:
            masked_audio_time (_type_): _description_
            input_text (_type_): _description_
            noisy_mel (_type_): [B, T, F]
            mask_padding_time (_type_): _description_

        Returns:
            _type_: _description_
        """
        B, T, F = noisy_mel.shape
        if mask_padding_time is None:
            mask_padding_time = torch.ones_like(masked_audio_time, dtype=torch.bool).cuda()
        if mask_padding_frames is None:
            mask_padding_frames = torch.ones((B, 1, T), dtype=torch.bool).cuda()
        length_masked_time = torch.sum(mask_padding_time, dim=-1, dtype=torch.int32)
        num_frames = (length_masked_time) // self.hop_length + 1 #[B, R] where is the number of frames in the ref_masked_mel
        device = masked_audio_time.device
        input_text, phoneme_length = list_str_to_idx_tts(input_text) # already padded with zeros (zero = '_')
        input_text = input_text.to(device)
        phoneme_length = phoneme_length.to(device)
        ref_masked_mel, _ = self.stft.mel_spectrogram(masked_audio_time) #[B, F, T1]
        with torch.no_grad():
            # Extract style vector
            style_vector = self.style_speech_model.get_style_vector(ref_masked_mel.transpose(1, 2), mel_len=num_frames) # the input ref_masked_mel is [B, T, F], style_vector shape is [B, D=128]
            if self.tts_output == 'phoneme_with_energy_pitch':
                outputs = self.style_speech_model.inference_only_encoder_with_pitch_energy(style_vector, input_text, phoneme_length, masked_frame_number=None)
            elif self.tts_output == 'phoneme':
                outputs = self.style_speech_model.inference_only_encoder(style_vector, input_text, phoneme_length, masked_frame_number=None) # [B, T, F]
        phoneme_output = outputs.detach() #[B, T2, 256]

        

        
        noisy_mel_len = torch.sum(mask_padding_frames, dim=-1, keepdim=False, dtype=torch.int32).squeeze(1)
        # from StyleSpeech import utils
        # utils.plot_data([phoneme_output[0].transpose(1,0).cpu().numpy()], 
        #     ['Ref Spectrogram'], filename='/home/dsi/moradim/SpeechRepainting/plot.png')
        phoneme_output = self.encoder_phoneme(phoneme_output) # (B, T2, D)
        if self.positional_emd == 'none':
            noisy_feats = self.encoder_noisy(noisy_mel.transpose(1, 2)).transpose(1, 2)  # (B, T1, D)

        if self.positional_emd == 'InputEmbedding1' or self.positional_emd == 'InputEmbeddingConv':
            noisy_feats = self.encoder_noisy(noisy_mel.transpose(1, 2), true_length=noisy_mel_len)  # (B, T1, D)


        if self.use_rope:
            # len_rope = max(T, max_generated_mel_len)
            rope_noisy_mel = self.rotary_embed.forward_from_seq_len(T)
            rope_tts = self.rotary_embed.forward_from_seq_len(max(phoneme_length))
            rope = (rope_noisy_mel, rope_tts)
        else:
            rope = None
        # Apply Cross-Attention once at the input
        if self.cross_attn_type == 'CrossAttention_noise':
            combined_feats = self.cross_attention(noisy_feats, phoneme_output, noisy_mel_len, phoneme_length, t, rope=rope)  # (B, T1, D)
        else:
            combined_feats = self.cross_attention(noisy_feats, phoneme_output, noisy_mel_len, phoneme_length)  # (B, T1, D)
        if self.inner_attention:
            return combined_feats, rope, phoneme_output, phoneme_length, noisy_mel_len
        return combined_feats, rope

class StyleSpeechWrapper(nn.Module):
    def __init__(self, tts_kw):
        super().__init__()

        self.positional_emd = tts_kw.positional_emd
        self.cross_attn_type = tts_kw.cross_attn_type
        style_speech_ch_path = tts_kw["style_speech_ch_path"] # checkpoint path
        style_speech_config_path = tts_kw["style_speech_config_path"]
        self.style_speech_model, config = get_StyleSpeech(style_speech_config_path, style_speech_ch_path)
        self.stft = TacotronSTFT(
            config.filter_length,
            config.hop_length,
            config.win_length,
            config.n_mel_channels,
            config.sampling_rate,
            config.mel_fmin,
            config.mel_fmax,
            log=tts_kw.log)
        
        self.filter_length = config.filter_length
        self.hop_length = config.hop_length
        self.use_rope = tts_kw.use_rope
        if self.use_rope:
            self.rotary_embed = RotaryEmbedding(tts_kw.embed_dim // tts_kw.num_heads)
        
        if tts_kw.positional_emd == 'none':
            self.encoder_noisy = nn.Conv1d(config.n_mel_channels, tts_kw.embed_dim, kernel_size=3, padding=1)
            self.encoder_clean = nn.Conv1d(config.n_mel_channels, tts_kw.embed_dim, kernel_size=3, padding=1)
        elif tts_kw.positional_emd == 'InputEmbedding1':
            self.encoder_noisy = InputEmbedding1(config.n_mel_channels, tts_kw.embed_dim)
            self.encoder_clean = InputEmbedding1(config.n_mel_channels, tts_kw.embed_dim)
        elif tts_kw.positional_emd == 'InputEmbeddingConv':
            self.encoder_noisy = InputEmbeddingConv(config.n_mel_channels, tts_kw.embed_dim)
            self.encoder_clean = InputEmbeddingConv(config.n_mel_channels, tts_kw.embed_dim)
        if tts_kw.cross_attn_type == 'CrossAttention_noise':
            self.cross_attention = CrossAttention_noise(tts_kw.embed_dim, tts_kw.num_heads, tts_kw.embed_dim, norm=tts_kw.norm, custom_multi_att=tts_kw.custom_multi_att)
        else:
            self.cross_attention = CrossAttention(tts_kw.embed_dim, tts_kw.num_heads, tts_kw.norm)

    def forward(self, masked_audio_time, input_text, noisy_mel, mask_padding_time, mask_padding_frames, t):
        """_summary_

        Args:
            masked_audio_time (_type_): _description_
            input_text (_type_): _description_
            noisy_mel (_type_): [B, T, F]
            mask_padding_time (_type_): _description_

        Returns:
            _type_: _description_
        """
        B, T, F = noisy_mel.shape
        if mask_padding_time is None:
            mask_padding_time = torch.ones_like(masked_audio_time, dtype=torch.bool).cuda()
        if mask_padding_frames is None:
            mask_padding_frames = torch.ones((B, 1, T), dtype=torch.bool).cuda()
        length_masked_time = torch.sum(mask_padding_time, dim=-1, dtype=torch.int32)
        num_frames = (length_masked_time) // self.hop_length + 1 #[B, R] where is the number of frames in the ref_masked_mel
        device = masked_audio_time.device
        input_text, phoneme_length = list_str_to_idx_tts(input_text) # already padded with zeros (zero = '_')
        input_text = input_text.to(device)
        phoneme_length = phoneme_length.to(device)
        ref_masked_mel, _ = self.stft.mel_spectrogram(masked_audio_time) #[B, F, T]
        with torch.no_grad():
            # Extract style vector
            style_vector = self.style_speech_model.get_style_vector(ref_masked_mel.transpose(1, 2), mel_len=num_frames) # the input ref_masked_mel is [B, T, F], style_vector shape is [B, D=128]
            outputs = self.style_speech_model.inference(style_vector, input_text, phoneme_length, masked_frame_number=None) # [B, T, F]
        mel_output = outputs[0].detach()
        mel_output = normalise_mel(mel_output)
        generated_mel_len = outputs[-1].detach()
        
        zero_frame = torch.zeros((B, 1, F), device=mel_output.device, dtype=mel_output.dtype) - 10.
        # Concatenate zero frames at the beginning and end
        mel_output = torch.cat([zero_frame, mel_output, zero_frame], dim=1)
        # Update generated_mel_len by adding 2
        generated_mel_len = generated_mel_len + 2
        max_generated_mel_len = torch.max(generated_mel_len).item()
        
        noisy_mel_len = torch.sum(mask_padding_frames, dim=-1, keepdim=False, dtype=torch.int32).squeeze(1)
        # from StyleSpeech import utils
        # utils.plot_data([mel_output[0].transpose(1,0).cpu().numpy()], 
        #     ['Ref Spectrogram'], filename='/home/dsi/moradim/SpeechRepainting/plot.png')
        if self.positional_emd == 'none':
            noisy_feats = self.encoder_noisy(noisy_mel.transpose(1, 2)).transpose(1, 2)  # (B, T1, D)
            mel_output = self.encoder_clean(mel_output.transpose(1, 2)).transpose(1, 2)  # (B, T2, D)
        if self.positional_emd == 'InputEmbedding1' or self.positional_emd == 'InputEmbeddingConv':
            noisy_feats = self.encoder_noisy(noisy_mel.transpose(1, 2), true_length=noisy_mel_len)  # (B, T1, D)
            mel_output = self.encoder_clean(mel_output.transpose(1, 2), true_length=generated_mel_len)  # (B, T2, D)

        if self.use_rope:
            # len_rope = max(T, max_generated_mel_len)
            rope_noisy_mel = self.rotary_embed.forward_from_seq_len(T)
            rope_tts = self.rotary_embed.forward_from_seq_len(max_generated_mel_len)
            rope = (rope_noisy_mel, rope_tts)
        else:
            rope = None
        # Apply Cross-Attention once at the input
        if self.cross_attn_type == 'CrossAttention_noise':
            combined_feats = self.cross_attention(noisy_feats, mel_output, noisy_mel_len, generated_mel_len, t, rope=rope)  # (B, T1, D)
        else:
            combined_feats = self.cross_attention(noisy_feats, mel_output, noisy_mel_len, generated_mel_len)  # (B, T1, D)
        return combined_feats, rope

# Transformer backbone using DiT blocks

@utils.register_model(name='dit')
class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=80,
        long_skip_connection=False,
        checkpoint_activations=False,
        text_embed_prop=None,
        unconditional=False,
        tts_kw=None,
    ):
        super().__init__()
        self.unconditional = unconditional
        self.text_embed_prop = text_embed_prop
        self.use_tts = tts_kw["use_tts"]
        self.tts_output = tts_kw["tts_output"]
        self.use_rope = tts_kw["use_rope"]
        self.inner_attention = tts_kw["inner_attention"]
        self.use_text_embed_rep = self.text_embed_prop["use_text_embed_rep"]
        self.rotary_embed = RotaryEmbedding(dim_head)
        if self.unconditional is False:
            if self.use_tts:
                if self.tts_output == 'mel':
                    self.sp_tts = StyleSpeechWrapper(tts_kw)
                elif self.tts_output == 'phoneme' or self.tts_output == 'phoneme_with_energy_pitch':
                    self.sp_tts = StyleSpeecEncoderPhonemehWrapper(tts_kw)
            elif self.use_text_embed_rep:
                if self.text_embed_prop["text_dim"] is None:
                    text_dim = mel_dim
                else:
                    text_dim = self.text_embed_prop["text_dim"]
                vocab_char_map, vocab_size = get_tokenizer(tokenizer_path=self.text_embed_prop["tokenizer_path"], tokenizer=self.text_embed_prop["tokenizer"])
                self.vocab_char_map = vocab_char_map
                self.text_embed = TextEmbedding(vocab_size, text_dim, conv_layers=self.text_embed_prop["conv_layers"])

                self.input_embed = InputEmbedding3(mel_dim, text_dim, dim)
            else:
                self.input_embed = InputEmbedding2(mel_dim, dim)
        else:
            self.input_embed = InputEmbedding1(mel_dim, dim)

        self.time_embed = TimestepEmbedding(dim)



        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout, inner_attention=self.inner_attention) for _ in range(depth)]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNormZero_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

        self.checkpoint_activations = checkpoint_activations
            
        

    def ckpt_wrapper(self, module):
        # https://github.com/chuanyangjin/fast-DiT/blob/main/models.py
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def forward(
        self,
        input_data: float["b n d"],  # nosied input audio  # noqa: F722
        cond: float["b n d"],  # masked cond audio  # noqa: F722
        text: int["b nt"],  # text  # noqa: F722
        # time: float["b"] | float[""],  # time step  # noqa: F821 F722
        drop_text,  # cfg for text
        spk=None,
        input_text=None,
        mask_padding_time=None,
        mask_padding_frames=None
    ):
        mask = mask_padding_frames
        if mask is not None:
            mask = mask_padding_frames.bool()
            mask = mask.squeeze(-2)
        x, time = input_data
        time = time.squeeze(-1)
        time = time.to(torch.float32)
        x = x.transpose(-1, -2) # [B, T, F]
        device = x.device
        masked_melspec, masked_audio_time = cond
        masked_melspec = masked_melspec.transpose(-1, -2) # [B, T, F]
        batch, seq_len = x.shape[0], x.shape[1]
        rope = self.rotary_embed.forward_from_seq_len(seq_len)
        if time.ndim == 0:
            time = time.repeat(batch)
        
        t = self.time_embed(time)
        # t: conditioning time, c: context (text + masked cond audio), x: noised input audio
        if self.unconditional is False:
            if self.use_tts:
                x, rope2tts = self.sp_tts(masked_audio_time, input_text, x, mask_padding_time, mask_padding_frames, t)
                # if self.use_rope:
                #     rope = rope2tts

            elif self.use_text_embed_rep:
                if self.vocab_char_map is not None:
                    input_text = list_str_to_idx(input_text, self.vocab_char_map).to(device)
                else:
                    input_text = list_str_to_tensor(input_text).to(device)
                text_embed = self.text_embed(input_text, seq_len, drop_text=drop_text)
                x = self.input_embed(x, masked_melspec, text_embed)
            else:
                x = self.input_embed(x, masked_melspec)
        else:
            x = self.input_embed(x.transpose(1, 2), mask=mask)

        

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, t, mask, rope)
            else:
                x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output.transpose(-1, -2)
    
    @classmethod
    def name(cls, cfg):
        print("!!!The model is DiT!!!")
        return "dit-net_dim{}_depth{}_heads{}_dim-head{}_dropout{}_ff_mult{}".format(
            cfg["dim"],
            cfg["depth"],
            cfg["heads"],
            cfg["dim_head"],
            cfg["dropout"],
            cfg["ff_mult"],
        )
        
if __name__ == "__main__":
    dim = 768
    depth = 18
    heads = 12
    dim_head = 64
    dropout = 0.1
    ff_mult = 2
    mel_dim = 80
    long_skip_connection = False
    checkpoint_activations = False  # recompute activations and save memory for extra compute
    unconditional = False
    text_embed_prop = {
        "use_text_embed_rep": True,
        "text_dim": 512,
        "conv_layers": 4,
        "text_num_embeds": 256,
        "tokenizer": "phoneme",  # 'byte' #phoneme
        "tokenizer_path": "/home/dsi/moradim/SpeechRepainting/phoneme_to_number.json"  # none #  /home/dsi/moradim/SpeechRepainting/phoneme_to_number.json
    }    
    model = DiT(
        dim=dim,
        depth=depth,
        heads=heads,
        dim_head=dim_head,
        dropout=dropout,
        ff_mult=ff_mult,
        mel_dim=mel_dim,
        long_skip_connection=long_skip_connection,
        checkpoint_activations=checkpoint_activations,
        text_embed_prop=text_embed_prop,
        unconditional=unconditional,
    )
    print(model)
    print_modle_size(model, "dit")
    # The number of parameters of dit is: 157.915472M


