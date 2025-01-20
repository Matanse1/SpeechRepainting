import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import calc_diffusion_step_embedding, WeightedSum, match_and_concatenate, print_modle_size
from transformers import AutoModel, WavLMModel
from . import utils
def swish(x):
    return x * torch.sigmoid(x)


# dilated conv layer with kaiming_normal initialization
# from https://github.com/ksw0306/FloWaveNet/blob/master/modules.py
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        out = self.conv(x)
        return out


# conv1x1 layer with zero initialization
# from https://github.com/ksw0306/FloWaveNet/blob/master/modules.py but the scale parameter is removed
class ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ZeroConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        out = self.conv(x)
        return out


# every residual block
# contains one noncausal dilated conv
class Residual_block(nn.Module):
    def __init__(
            self, res_channels, skip_channels, dilation=1,
            diffusion_step_embed_dim_out=512,
            unconditional=True,
            mel_upsample=[16,16],
            cond_feat_size=640,
            representation_models={},
            number=0,
            **kwargs
        ):
        super(Residual_block, self).__init__()
        self.res_channels = res_channels

        # the layer-specific fc for diffusion step embedding
        self.fc_t = nn.Linear(diffusion_step_embed_dim_out, self.res_channels)

        # dilated conv layer
        self.dilated_conv_layer = Conv(self.res_channels, 2 * self.res_channels, kernel_size=3, dilation=dilation)

        self.unconditional = unconditional
        if not self.unconditional:
            
            self.wavlm_prop = kwargs.get("wavlm") #properties of wavlm
            self.use_wavlm_rep = self.wavlm_prop["use_wavlm_rep"] # representation
            self.use_weighted_sum_wavlm = self.wavlm_prop["use_weighted_sum_wavlm"]
            self.use_all_hidden_states = self.wavlm_prop["use_all_hidden_states"]
            version_model = self.wavlm_prop["version_model"]
            self.concat_mel_with_wavlm = self.wavlm_prop["concat_mel_with_wavlm"]
            self.reduce_channels = self.wavlm_prop["reduce_channels"]
            self.two_branch = self.wavlm_prop["two_branch"]
            if self.use_wavlm_rep:
                self.wavlm_model = representation_models["wavlm"]
                
                if version_model == "microsoft/wavlm-large":
                    num_hs = 25 #number of hidden states
                    rep_dim_wavlm = 1024
                elif version_model == "microsoft/wavlm-base-plus":
                    num_hs = 13
                    rep_dim_wavlm = 768
                if self.use_weighted_sum_wavlm:
                    if self.wavlm_prop["specific_indices"]["use_indices_hidden_states"]:
                        self.indices_hidden_states = self.wavlm_prop["specific_indices"]["indices_hidden_states"][number]
                        num_hs = len(self.indices_hidden_states)
                    self.weighted_sum = WeightedSum(num_hs) # takes tuples and weights and returns the weighted sum of the tuples with learnable weights
            
            # add mel spectrogram upsampler and conditioner conv1x1 layer
            self.upsample_conv2d = torch.nn.ModuleList()
            for s in mel_upsample:
                conv_trans2d = torch.nn.ConvTranspose2d(1, 1, (3, 2 * s), padding=(1, s // 2), stride=(1, s))
                conv_trans2d = torch.nn.utils.weight_norm(conv_trans2d)
                torch.nn.init.kaiming_normal_(conv_trans2d.weight)
                self.upsample_conv2d.append(conv_trans2d)
            if self.use_wavlm_rep:
                if self.two_branch:
                    self.mel_branch = nn.Sequential(Conv(cond_feat_size, cond_feat_size, kernel_size=3), nn.PReLU())
                    self.wavlm_branch = nn.Sequential(Conv(rep_dim_wavlm, cond_feat_size, kernel_size=3), nn.PReLU())
                    self.fusion_branch = nn.Sequential(Conv(2 * cond_feat_size, cond_feat_size, kernel_size=3), nn.PReLU())
                input_channel_concat = cond_feat_size + rep_dim_wavlm
                if self.reduce_channels:
                    self.mel_conv_reduce = Conv(cond_feat_size + rep_dim_wavlm, cond_feat_size, kernel_size=3)
                    input_channel_concat = cond_feat_size
                if self.concat_mel_with_wavlm:
                    self.mel_conv = Conv(input_channel_concat, 2 * self.res_channels, kernel_size=3)  # i chose the kernel in order to catch two consecutive frames of wavlm rep and melspec (the wamlm is half the size of the melspec)
                elif self.two_branch:
                    self.mel_conv = Conv(cond_feat_size, 2 * self.res_channels, kernel_size=3)
                else:
                    self.mel_conv = Conv(rep_dim_wavlm, 2 * self.res_channels, kernel_size=3)
            else:
                self.mel_conv = Conv(cond_feat_size, 2 * self.res_channels, kernel_size=1)  # 80 is mel bands

        # residual conv1x1 layer, connect to next residual layer
        self.res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.res_conv = nn.utils.weight_norm(self.res_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)

        # skip conv1x1 layer, add to all skip outputs through skip connections
        self.skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        self.skip_conv = nn.utils.weight_norm(self.skip_conv)
        nn.init.kaiming_normal_(self.skip_conv.weight)
        

    def forward(self, input_data, mel_spec=None, mask_padding=None):
        x, diffusion_step_embed = input_data
        h = x
        B, C, L = x.shape
        assert C == self.res_channels

        # add in diffusion step embedding
        part_t = self.fc_t(diffusion_step_embed)
        part_t = part_t.view([B, self.res_channels, 1])
        h = h + part_t
        # h += part_t

        # dilated conv layer
        h = self.dilated_conv_layer(h)

        # add mel spectrogram as (local) conditioner
        if mel_spec is not None:
            masked_melspec, masked_audio_time = mel_spec
            assert not self.unconditional
            
            # Upsample spectrogram to size of audio
            # mel_spec = torch.unsqueeze(mel_spec, dim=1)
            # mel_spec = F.leaky_relu(self.upsample_conv2d[0](mel_spec), 0.4)
            # mel_spec = F.leaky_relu(self.upsample_conv2d[1](mel_spec), 0.4)
            # mel_spec = torch.squeeze(mel_spec, dim=1)

            # assert(mel_spec.size(2) >= L)
            # if mel_spec.size(2) > L:
            #     mel_spec = mel_spec[:, :, :L]
            if self.use_wavlm_rep:
                with torch.no_grad():
                    masked_audio_time_cond_wavlm = self.wavlm_model(masked_audio_time, output_hidden_states=self.use_all_hidden_states, attention_mask=mask_padding) #representation of the masked audio using wavlm
                if self.use_weighted_sum_wavlm:
                    hidden_states = masked_audio_time_cond_wavlm.hidden_states
                    if self.wavlm_prop["specific_indices"]["use_indices_hidden_states"]:
                        hidden_states = [hidden_states[i] for i in self.indices_hidden_states]
                    wavlm_output = self.weighted_sum(hidden_states) #[B, T/2, F]
                else:
                    wavlm_output = masked_audio_time_cond_wavlm.last_hidden_state
                wavlm_output = wavlm_output.transpose(-1, -2) #[B, F, T/2]
                cond = match_and_concatenate(wavlm_output, masked_melspec, concat_mel_with_rep=self.concat_mel_with_wavlm)
                if self.two_branch:
                    mel_branch = self.mel_branch(masked_melspec)
                    wavlm_branch = self.wavlm_branch(cond)
                    cond = self.fusion_branch(torch.cat([mel_branch, wavlm_branch], dim=1)) #concatenate the two branches over the feature/freq dimension
                elif self.reduce_channels:
                    cond = self.mel_conv_reduce(cond)
            else:
                cond = masked_melspec
            
            mel_spec = self.mel_conv(cond)
            h = h + mel_spec

        # gated-tanh nonlinearity
        out = torch.tanh(h[:,:self.res_channels,:]) * torch.sigmoid(h[:,self.res_channels:,:])

        # residual and skip outputs
        res = self.res_conv(out)
        assert x.shape == res.shape
        skip = self.skip_conv(out)

        return (x + res) * math.sqrt(0.5), skip  # normalize for training stability


class Residual_group(nn.Module):
    def __init__(self, res_channels, skip_channels, num_res_layers=30, dilation_cycle=10,
                 diffusion_step_embed_dim_in=128,
                 diffusion_step_embed_dim_mid=512,
                 diffusion_step_embed_dim_out=512,
                 unconditional=False,
                 mel_upsample=[16,16],
                 cond_feat_size=640,
                 representation_models={},
                 **kwargs
                 ):
        super(Residual_group, self).__init__()
        self.num_res_layers = num_res_layers
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

        # the shared two fc layers for diffusion step embedding
        self.fc_t1 = nn.Linear(diffusion_step_embed_dim_in, diffusion_step_embed_dim_mid)
        self.fc_t2 = nn.Linear(diffusion_step_embed_dim_mid, diffusion_step_embed_dim_out)

        # stack all residual blocks with dilations 1, 2, ... , 512, ... , 1, 2, ..., 512
        self.residual_blocks = nn.ModuleList()
        for n in range(self.num_res_layers):
            self.residual_blocks.append(Residual_block(res_channels, skip_channels,
                                                       dilation=2 ** (n % dilation_cycle),
                                                       diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                                       unconditional=unconditional,
                                                       mel_upsample=mel_upsample,
                                                       cond_feat_size=cond_feat_size,
                                                       representation_models=representation_models,
                                                       number=n,
                                                       **kwargs))

    def forward(self, input_data, mel_spec=None, mask_padding=None):
        x, diffusion_steps = input_data

        # embed diffusion step t
        diffusion_step_embed = calc_diffusion_step_embedding(diffusion_steps, self.diffusion_step_embed_dim_in)
        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))

        # pass all residual layers
        h = x
        skip = 0
        for n in range(self.num_res_layers):
            h, skip_n = self.residual_blocks[n]((h, diffusion_step_embed), mel_spec=mel_spec, mask_padding=mask_padding)  # use the output from last residual layer
            skip = skip + skip_n  # accumulate all skip outputs
            # skip += skip_n  # accumulate all skip outputs

        return skip * math.sqrt(1.0 / self.num_res_layers)  # normalize for training stability

@utils.register_model(name='wavenet')
class WaveNet(nn.Module):
    def __init__(self, cond_feat_size, in_channels=1, res_channels=256, skip_channels=128, out_channels=1,
                 num_res_layers=30, dilation_cycle=10,
                 diffusion_step_embed_dim_in=128,
                 diffusion_step_embed_dim_mid=512,
                 diffusion_step_embed_dim_out=512,
                 unconditional=False,
                 mel_upsample=[16,16],
                 **kwargs):
        super().__init__()
        representation_models = {}
        #### For WavLM representation model
        self.text_embed_prop = kwargs.get("text_embed_prop")
        self.wavlm_prop = kwargs.get("wavlm") #properties of wavlm
        self.use_wavlm_rep = self.wavlm_prop["use_wavlm_rep"] # representation
        self.use_weighted_sum_wavlm = self.wavlm_prop["use_weighted_sum_wavlm"]
        self.use_all_hidden_states = self.wavlm_prop["use_all_hidden_states"]
        version_model = self.wavlm_prop["version_model"]
        if self.use_wavlm_rep:
            print("Loading WavLM model")
            wavlm_model = AutoModel.from_pretrained(version_model)
            wavlm_model.eval()
            print("WavLM model loaded")
            print("Freeze WavLM model's parameters")
            for param in wavlm_model.parameters():
                param.requires_grad = False
            print_modle_size(wavlm_model, version_model)
            representation_models["wavlm"] = wavlm_model

        self.res_channels = res_channels
        self.skip_channels = skip_channels
        self.num_res_layers = num_res_layers
        self.unconditional = unconditional

        # initial conv1x1 with relu
        self.init_conv = nn.Sequential(Conv(in_channels, res_channels, kernel_size=1), nn.ReLU())

        # all residual layers
        self.residual_layer = Residual_group(res_channels=res_channels,
                                             skip_channels=skip_channels,
                                             num_res_layers=num_res_layers,
                                             dilation_cycle=dilation_cycle,
                                             diffusion_step_embed_dim_in=diffusion_step_embed_dim_in,
                                             diffusion_step_embed_dim_mid=diffusion_step_embed_dim_mid,
                                             diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                             mel_upsample=mel_upsample,
                                             unconditional=unconditional,
                                             cond_feat_size=cond_feat_size,
                                             representation_models=representation_models,
                                             **kwargs)

        # final conv1x1 -> relu -> zeroconv1x1
        self.final_conv = nn.Sequential(Conv(skip_channels, skip_channels, kernel_size=1),
                                        nn.ReLU(),
                                        ZeroConv1d(skip_channels, out_channels))

    def forward(self, input_data, cond=None, mask_padding_time=None, mask_padding_frames=None, drop_text=None, text=None):
        audio, diffusion_steps = input_data

        x = audio
        x = self.init_conv(x)
        x = self.residual_layer((x, diffusion_steps), mel_spec=cond, mask_padding=mask_padding_time)
        x = self.final_conv(x)

        return x

    def __repr__(self):
        return f"wavenet_h{self.res_channels}_d{self.num_res_layers}_{'uncond' if self.unconditional else 'cond'}"

    @classmethod
    def name(cls, cfg):
        print("!!!The model is Wavenet!!!")
        return "wnet_h{}_d{}".format(
            cfg["res_channels"],
            cfg["num_res_layers"],
        )
