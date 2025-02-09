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
from dataloaders.stft import TacotronSTFT
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
)
from models.utils import print_modle_size, WeightedSum, get_tokenizer, list_str_to_tensor, list_str_to_idx

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
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: float["b n d"]):  # noqa: F722
        x = self.conv_pos_embed(x) + x
        return x



class StyleSpeechWrapper(nn.Module):
    def __init__(self, , tts_kw):
        super().__init__()
        _stft = TacotronSTFT(
                tts_kw.filter_length,
                tts_kw.hop_length,
                tts_kw.win_length,
                tts_kw.n_mel_channels,
                tts_kw.sampling_rate,
                tts_kw.mel_fmin,
                tts_kw.mel_fmax)
        style_speech_ch_path = tts_kw["style_speech_ch_path"]
        style_speech_config_path = tts_kw["style_speech_config_path"]
        self.style_speech = style_speech
        # preprocess audio and text
        ref_mel = preprocess_audio(args.ref_audio, _stft).transpose(0,1).unsqueeze(0)
        src = preprocess_english(args.text, args.lexicon_path).unsqueeze(0)
        src_len = torch.from_numpy(np.array([src.shape[1]])).to(device=device)
        
        save_path = args.save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        # Extract style vector
        style_vector = model.get_style_vector(ref_mel)

        # Forward
        masked_frame_number = ref_mel.shape[-2]
        mel_output = model.inference(style_vector, src, src_len, masked_frame_number=None)[0]
        
        mel_ref_ = ref_mel.cpu().squeeze().transpose(0, 1).detach()
        mel_ = mel_output.cpu().squeeze().transpose(0, 1).detach()

    def forward(self, phoneme_seq_ints, masked_audio_time):
        return self.style_speech(x, t)

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
        use_tts=False,
    ):
        super().__init__()
        self.unconditional = unconditional
        self.text_embed_prop = text_embed_prop
        self.use_tts = use_tts
        self.use_text_embed_rep = self.text_embed_prop["use_text_embed_rep"]
        if self.unconditional is False:
            if self.use_text_embed_rep:
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

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout) for _ in range(depth)]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNormZero_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

        self.checkpoint_activations = checkpoint_activations
        if use_tts:
            self.style_speech_tts
            
        

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
        if time.ndim == 0:
            time = time.repeat(batch)
        
        t = self.time_embed(time)
        # t: conditioning time, c: context (text + masked cond audio), x: noised input audio
        if self.unconditional is False:
            if self.use_text_embed_rep:
                if self.vocab_char_map is not None:
                    input_text = list_str_to_idx(input_text, self.vocab_char_map).to(device)
                else:
                    input_text = list_str_to_tensor(input_text).to(device)
                text_embed = self.text_embed(input_text, seq_len, drop_text=drop_text)
                x = self.input_embed(x, masked_melspec, text_embed)
            else:
                x = self.input_embed(x, masked_melspec)
        else:
            x = self.input_embed(x)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

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


