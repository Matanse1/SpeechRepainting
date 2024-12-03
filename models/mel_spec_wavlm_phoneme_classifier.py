import torch
from transformers import AutoModel, WavLMConfig, WavLMModel, WavLMPreTrainedModel
from transformers.models.wavlm.modeling_wavlm import WavLMGumbelVectorQuantizer, WavLMPositionalConvEmbedding, WavLMFeatureProjection
import numpy as np
from torch.nn import functional as F
from torch import nn
import os
os.sys.path.append('/home/dsi/moradim/SpeechRepainting/')
from transformers.activations import ACT2FN
from dataloaders import dataloader, CollateFn
from typing import Optional, Tuple, Union
import math

_HIDDEN_STATES_START_POSITION = 2


class WavLMGroupNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else config.mel_bin_num
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            padding=config.conv_kernel[layer_id] // 2,
            bias=config.conv_bias,
        )
        self.activation = ACT2FN[config.feat_extract_activation]

        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states
    
class WavLMNoLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else config.mel_bin_num
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            padding=config.conv_kernel[layer_id] // 2,
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class WavLMLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else config.mel_bin_num
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            padding=config.conv_kernel[layer_id] // 2,
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)

        hidden_states = hidden_states.transpose(-2, -1)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(-2, -1)

        hidden_states = self.activation(hidden_states)
        return hidden_states
    
    
class Custom_WavLMFeatureEncoder(nn.Module):
    """Construct the features from raw audio waveform"""

    def __init__(self, config):
        super().__init__()

        if config.feat_extract_norm == "group":
            conv_layers = [WavLMGroupNormConvLayer(config, layer_id=0)] + [
                WavLMNoLayerNormConvLayer(config, layer_id=i + 1) for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            conv_layers = [WavLMLayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)]
        else:
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        self.conv_layers = nn.ModuleList(conv_layers)
        self.gradient_checkpointing = False
        self._requires_grad = True

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def forward(self, input_values):
        hidden_states = input_values #[:, None]

        # make sure hidden_states require grad for gradient_checkpointing
        if self._requires_grad and self.training and not hidden_states.requires_grad:
            hidden_states.requires_grad = True

        for conv_layer in self.conv_layers:
            if self._requires_grad and self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    conv_layer.__call__,
                    hidden_states,
                )
            else:
                hidden_states = conv_layer(hidden_states)

        return hidden_states
    
    
def size_model(model):
    module_parameters = list(filter(lambda p: p[1].requires_grad, model.named_parameters()))
    params = sum([np.prod(p.size()) for n, p in module_parameters])
    print("The number of parameters of the model is: {:.6f}M".format(params/1e6))

class WavlmMelSpecPhonemeClassifier(WavLMPreTrainedModel):
    def __init__(self, config):
        wavlm_config_path = config.wavlm_config_path
        wavlm_config = WavLMConfig.from_pretrained(wavlm_config_path)
        wavlm_config.update(config)
        super(WavlmMelSpecPhonemeClassifier, self).__init__(wavlm_config)
        self.config = wavlm_config
        # config.mask_time_prob = 0
        # config.mask_feature_prob  = 0
        # config.conv_kernel = [9, 3, 3, 3, 3, 1, 1]# [10, 3, 3, 3, 3, 2, 2] #[9, 3, 3, 3, 3, 1, 1]
        # config.conv_stride = [1, 1, 1, 1, 1, 1, 1] # [5, 2, 2, 2, 2, 2, 2] #[1, 1, 1, 1, 1, 1, 1]
        # config.conv_kernel = [10, 3, 3, 3, 3, 2, 2]
        # config.conv_stride = [5, 2, 2, 2, 2, 2, 2]

        print(wavlm_config)
        if config.pre_trained_model:
            self.wavlm = WavLMModel.from_pretrained(config.wavlm_name, config=wavlm_config, ignore_mismatched_sizes=True)
        else:
            self.wavlm = WavLMModel(config=wavlm_config)
        self.wavlm.feature_extractor = Custom_WavLMFeatureEncoder(wavlm_config)
        print("-----------------------------")
        print(self.wavlm.config)
        # self.wavlm = AutoModel.from_pretrained("microsoft/wavlm-base-plus")
        print(f"The size before freezing is:")
        size_model(self.wavlm)
        if config.freeze_encoder:
            self.freeze_encoder(option="encoder")
        elif config.freeze_feature_extractor:
            self.freeze_encoder(option="feature_projection")
        elif config.freeze_feature_projection:
            self.freeze_encoder(option="feature_projection")
        
        print(f"The size after freezing is: ")
        size_model(self.wavlm)
        self.masked_spec_embed = nn.Parameter(torch.Tensor(wavlm_config.mel_bin_num).uniform_())
        
        
        ## For classifier
        num_layers = wavlm_config.num_hidden_layers + 1  # transformer layers + input embeddings
        if wavlm_config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.classifier_phoneme = nn.Linear(wavlm_config.hidden_size, wavlm_config.vocab_size)
        # self.classifier_word = nn.Conv1d(wavlm_config.vocab_size, wavlm_config.vocab_size)
        self.vocab_size = wavlm_config.vocab_size
        if not config.pre_trained_model:
            self.init_weights()
        
    def _init_weights(self, module):
        """Initialize the weights"""
        # gumbel softmax requires special init
        if isinstance(module, WavLMGumbelVectorQuantizer):
            module.weight_proj.weight.data.normal_(mean=0.0, std=1)
            module.weight_proj.bias.data.zero_()
            nn.init.uniform_(module.codevectors)
        elif isinstance(module, WavLMPositionalConvEmbedding):
            nn.init.normal_(
                module.conv.weight,
                mean=0,
                std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
            )
            nn.init.constant_(module.conv.bias, 0)
        elif isinstance(module, WavLMFeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)

            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)
        
    def freeze_component(self, option):

        if option == "feature_projection":
            for name, param in self.wavlm.named_parameters():
                if 'feature_projection' in name:
                    param.requires_grad = False
        elif option == "feature_extractor":
                self.wavlm.feature_extractor._freeze_parameters()
        elif option == "encoder":
            for name, param in self.wavlm.named_parameters():
                if 'encoder' in name:  # Adjust the name based on your specific wavlm
                    param.requires_grad = False
        

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        masked_region: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.vocab_size - 1]`. If `config.vocab_size == 1` a regression loss is computed (Mean-Square loss), If
            `config.vocab_size > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states
        if attention_mask is None:
            attention_mask = torch.ones(input_values.shape[0], input_values.shape[-1] , device=input_values.device)
        if self.config.mask_regions:
            masked_region_condition = attention_mask * (1 - masked_region)
            masked_region_condition = masked_region_condition > 0 # make it boolean
            masked_region_condition = torch.tensor(masked_region_condition, device=input_values.device, dtype=torch.bool) #[B, T]
            input_values = input_values.transpose(2, 1) #[B, T, mel_bin_num]
            input_values[masked_region_condition] = self.masked_spec_embed.to(input_values.dtype)
            input_values = input_values.transpose(2, 1) #[B, mel_bin_num, T]
        
        outputs = self.wavlm(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0] #TODO to check

        phonemes_estimated = self.classifier_phoneme(hidden_states) #[B, T, vocab_size]
        # word_estimated = self.classifier_word(hidden_states) #[B, T, vocab_size]
        phonemes_estimated = phonemes_estimated.transpose(2,1) #[[B, vocab_size, T]
        return phonemes_estimated# word_estimated
    
    @classmethod
    def name(cls, cfg):
        string = ""
        for key, value in cfg.items():
            if key == "_name_" or key == "wavlm_config_path" or key == "wavlm_name":
                continue
            string += f"_{key}={value}"
        return string
        # return "vocab_size={}_mel_bin_num={}_freeze_feature_extractor={}_freeze_feature_projection={}_freeze_encoder={}".format(
        #     cfg["vocab_size"],
        #     cfg["mel_bin_num"],
        #     cfg["freeze_feature_extractor"],
        #     cfg["freeze_feature_projection"],
        #     cfg["freeze_encoder"],
        # )