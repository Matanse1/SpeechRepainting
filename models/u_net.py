from __future__ import annotations
import numpy as np
import torch
from einops import rearrange
import math
from . import utils
import torch.nn as nn
from models.utils import print_modle_size, WeightedSum, get_tokenizer, list_str_to_tensor, list_str_to_idx
from transformers import AutoModel, WavLMModel
from models.modules import TextEmbedding, ConvPositionEmbedding

class BaseModule(torch.nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    @property
    def nparams(self):
        """
        Returns number of trainable parameters of the module.
        """
        num_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                num_params += np.prod(param.detach().cpu().numpy().shape)
        return num_params


    def relocate_input(self, x: list):
        """
        Relocates provided tensors to the same device set for the module.
        """
        device = next(self.parameters()).device
        for i in range(len(x)):
            if isinstance(x[i], torch.Tensor) and x[i].device != device:
                x[i] = x[i].to(device)
        return x


class Mish(BaseModule):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class Upsample(BaseModule):
    def __init__(self, dim):
        super(Upsample, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Downsample(BaseModule):
    def __init__(self, dim):
        super(Downsample, self).__init__()
        self.conv = torch.nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Rezero(BaseModule):
    def __init__(self, fn):
        super(Rezero, self).__init__()
        self.fn = fn
        self.g = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


class Block(BaseModule):
    def __init__(self, dim, dim_out, groups=8):
        super(Block, self).__init__()
        self.block = torch.nn.Sequential(torch.nn.Conv2d(dim, dim_out, 3, 
                                         padding=1), torch.nn.GroupNorm(
                                         groups, dim_out), Mish())

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


class ResnetBlock(BaseModule):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super(ResnetBlock, self).__init__()
        self.mlp = torch.nn.Sequential(Mish(), torch.nn.Linear(time_emb_dim, 
                                                               dim_out))

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        if dim != dim_out:
            self.res_conv = torch.nn.Conv2d(dim, dim_out, 1)
        else:
            self.res_conv = torch.nn.Identity()

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class LinearAttention(BaseModule):
    def __init__(self, dim, heads=4, dim_head=32):
        super(LinearAttention, self).__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = torch.nn.Conv2d(hidden_dim, dim, 1)            

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', 
                            heads = self.heads, qkv=3)            
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', 
                        heads=self.heads, h=h, w=w)
        return self.to_out(out)


class Residual(BaseModule):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        output = self.fn(x, *args, **kwargs) + x
        return output


class SinusoidalPosEmb(BaseModule):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: float["b n d"], cond: float["b n d"], text_embed: float["b n d"]):  # noqa: F722
        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x
    
    
@utils.register_model(name='unet')
class Unet(BaseModule):
    def __init__(self, dim, dim_mults=(1, 2, 4), groups=8,
                 n_spks=None, spk_emb_dim=64, n_feats=80, pe_scale=1000, unconditional=False, wavlm=False, text_embed_prop=None):
        super(Unet, self).__init__()
        self.dim = dim
        self.dim_mults = dim_mults
        self.groups = groups
        self.n_spks = n_spks if not isinstance(n_spks, type(None)) else 1
        self.spk_emb_dim = spk_emb_dim
        self.pe_scale = pe_scale
        self.unconditional = unconditional
        self.wavlm_prop = wavlm
        self.text_embed_prop = text_embed_prop
        self.use_text_embed_rep = self.text_embed_prop["use_text_embed_rep"]
        self.cat_cond = (not unconditional) and (not self.use_text_embed_rep)
        
        if self.use_text_embed_rep:
            vocab_char_map, vocab_size = get_tokenizer(tokenizer_path=self.text_embed_prop["tokenizer_path"], tokenizer=self.text_embed_prop["tokenizer"])
            self.vocab_char_map = vocab_char_map
            self.text_embed = TextEmbedding(vocab_size, self.text_embed_prop["text_dim"], conv_layers=self.text_embed_prop["conv_layers"])
            self.input_embed = InputEmbedding(self.text_embed_prop["mel_dim"], self.text_embed_prop["text_dim"], self.text_embed_prop["mel_dim"])
        if n_spks > 1:
            self.spk_mlp = torch.nn.Sequential(torch.nn.Linear(spk_emb_dim, spk_emb_dim * 4), Mish(),
                                               torch.nn.Linear(spk_emb_dim * 4, n_feats))
        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim, dim * 4), Mish(),
                                       torch.nn.Linear(dim * 4, dim))



        #### For WavLM representation model representation
        self.use_wavlm_rep = self.wavlm_prop["use_wavlm_rep"] 
        version_model = self.wavlm_prop["version_model"]
        
        if not self.unconditional:

            if self.use_wavlm_rep:
                print("Loading WavLM model")
                wavlm_model = AutoModel.from_pretrained(version_model)
                wavlm_model.eval()
                print("WavLM model loaded")
                print("Freeze WavLM model's parameters")
                for param in wavlm_model.parameters():
                    param.requires_grad = False
                print_modle_size(wavlm_model, version_model)

                self.use_wavlm_rep = self.wavlm_prop["use_wavlm_rep"] # representation
                self.use_weighted_sum_wavlm = self.wavlm_prop["use_weighted_sum_wavlm"]
                self.use_all_hidden_states = self.wavlm_prop["use_all_hidden_states"]
                version_model = self.wavlm_prop["version_model"]
                self.concat_mel_with_wavlm = self.wavlm_prop["concat_mel_with_wavlm"]
                self.reduce_channels = self.wavlm_prop["reduce_channels"]
                self.two_branch = self.wavlm_prop["two_branch"]
                self.wavlm_model = wavlm_model
                    
                if version_model == "microsoft/wavlm-large":
                    num_hs = 25 #number of hidden states
                    rep_dim_wavlm = 1024
                elif version_model == "microsoft/wavlm-base-plus":
                    num_hs = 13
                    rep_dim_wavlm = 768
                if self.use_weighted_sum_wavlm:
                    if self.wavlm_prop["specific_indices"]["use_indices_hidden_states"]:
                        self.indices_hidden_states = self.wavlm_prop["specific_indices"]["indices_hidden_states"]
                        num_hs = len(self.indices_hidden_states)
                    self.weighted_sum = WeightedSum(num_hs) # takes tuples and weights and returns the weighted sum of the tuples with learnable weights
                self.upsample_conv1d_wavlm = torch.nn.ConvTranspose1d(rep_dim_wavlm, n_feats, 4, 2, 1)
                torch.nn.init.kaiming_normal_(self.upsample_conv1d_wavlm.weight)
                    
        
        if self.cat_cond:
            dims = [2, *map(lambda m: dim * m, dim_mults)]
        # dims = [2 + (1 if n_spks > 1 else 0), *map(lambda m: dim * m, dim_mults)]
        else:
            dims = [1, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(torch.nn.ModuleList([
                       ResnetBlock(dim_in, dim_out, time_emb_dim=dim),
                       ResnetBlock(dim_out, dim_out, time_emb_dim=dim),
                       Residual(Rezero(LinearAttention(dim_out))),
                       Downsample(dim_out) if not is_last else torch.nn.Identity()]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(torch.nn.ModuleList([
                     ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim),
                     ResnetBlock(dim_in, dim_in, time_emb_dim=dim),
                     Residual(Rezero(LinearAttention(dim_in))),
                     Upsample(dim_in)]))
        self.final_block = Block(dim, dim)
        self.final_conv = torch.nn.Conv2d(dim, 1, 1)

    def forward(self, input_data, cond=None, mask_padding_time=None, mask_padding_frames=None, text=None, input_text=None, drop_text=None, spk=None):
        x, t = input_data
        device = x.device
        B, F, T = x.shape
        t = t.squeeze(1)
        if not self.unconditional:
            masked_melspec, masked_audio_time = cond
        mask = mask_padding_frames
        if not isinstance(spk, type(None)):
            s = self.spk_mlp(spk)
        
        t = self.time_pos_emb(t, scale=self.pe_scale)
        t = self.mlp(t)
        if self.use_text_embed_rep:
            if self.vocab_char_map is not None:
                input_text = list_str_to_idx(input_text, self.vocab_char_map).to(device)
            else:
                input_text = list_str_to_tensor(input_text).to(device)
            seq_len = T
            text_embed = self.text_embed(input_text, seq_len, drop_text=drop_text)
            x = x.transpose(-1, -2)
            masked_melspec = masked_melspec.transpose(-1, -2)
            x = self.input_embed(x, masked_melspec, text_embed)
            x = x.transpose(-1, -2)
            masked_melspec = masked_melspec.transpose(-1, -2)

        if self.cat_cond:
            if self.use_wavlm_rep:
                with torch.no_grad():
                    masked_audio_time_cond_wavlm = self.wavlm_model(masked_audio_time, output_hidden_states=self.use_all_hidden_states, attention_mask=mask_padding_time) #representation of the masked audio using wavlm
                    if self.use_weighted_sum_wavlm:
                        hidden_states = masked_audio_time_cond_wavlm.hidden_states
                        if self.wavlm_prop["specific_indices"]["use_indices_hidden_states"]:
                            hidden_states = [hidden_states[i] for i in self.indices_hidden_states]
                        wavlm_output = self.weighted_sum(hidden_states) #[B, T/2, F]
                    else:
                        wavlm_output = masked_audio_time_cond_wavlm.last_hidden_state
                wavlm_output = wavlm_output.transpose(-1, -2) #[B, F, T/2]
                if wavlm_output.shape[-1] % 2 != 0:
                    wavlm_output = torch.nn.functional.pad(wavlm_output, (0, 1), mode='replicate')
                wavlm_output = self.upsample_conv1d_wavlm(wavlm_output) #[B, F, T]
                x = torch.stack([x, wavlm_output], 1)
            else:
                x = torch.stack([x, masked_melspec], 1)
        else:
            x = x.unsqueeze(1)

        mask = mask.unsqueeze(1)

        hiddens = []
        masks = [mask]
        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)
            x = resnet2(x, mask_down, t)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]
        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)

        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            x = attn(x)
            x = upsample(x * mask_up)

        x = self.final_block(x, mask)
        output = self.final_conv(x * mask)

        return (output * mask).squeeze(1)
    
    @classmethod
    def name(cls, cfg):
        print("!!!The model is Unet!!!")
        return "unet_dim{}_dim_mults{}".format(
            cfg["dim"],
            "_".join(map(str, cfg["dim_mults"])),
        )