import torch
from transformers import AutoModel, WavLMConfig, WavLMModel
import numpy as np
from torch.nn import functional as F
from torch import nn
import os
from transformers.activations import ACT2FN

class WavLMGroupNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
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
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
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
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
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
    
    
class WavLMFeatureEncoder(nn.Module):
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
        hidden_states = input_values[:, None]

        # make sure hidden_states require grad for gradient_checkpointing
        if self._requires_grad and self.training:
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

class WavlmMelSpecPhonemeClassifier(nn.Module):
    def __init__(self):
        super(WavlmMelSpecPhonemeClassifier, self).__init__()
        configuration = WavLMConfig("microsoft/wavlm-base-plus")
        configuration.mask_time_prob = 0
        configuration.mask_feature_prob  = 0
        configuration.conv_kernel = [9, 3, 3, 3, 3, 1, 1]# [10, 3, 3, 3, 3, 2, 2] #[9, 3, 3, 3, 3, 1, 1]
        configuration.conv_stride = [1, 1, 1, 1, 1, 1, 1] # [5, 2, 2, 2, 2, 2, 2] #[1, 1, 1, 1, 1, 1, 1]
        configuration.save_pretrained("/home/dsi/moradim/SpeechRepainting/wavlm_utils/wavlm_config/")
        # configuration.conv_kernel = [10, 3, 3, 3, 3, 2, 2]
        # configuration.conv_stride = [5, 2, 2, 2, 2, 2, 2]

        print(configuration)
        self.model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus", config=configuration, ignore_mismatched_sizes=True)
        self.model.feature_extractor = WavLMFeatureEncoder(configuration)
        print("-----------------------------")
        print(self.model.config)
        # self.model = AutoModel.from_pretrained("microsoft/wavlm-base-plus")
        size_model(self.model)
        # self.freeze_encoder(option="feature_extractor")
        # self.freeze_encoder(option="feature_projection")
        # size_model(self.model)
        
    def freeze_encoder(self, option):
        # Freeze all parameters of the encoder and feature extraction layers
        if option == "all":
            for name, param in self.model.named_parameters():
                if 'feature_projection' in name or 'feature_projection' in name:
                    param.requires_grad = False
        elif option == "feature_projection":
            for name, param in self.model.named_parameters():
                if 'feature_projection' in name:
                    param.requires_grad = False
        elif option == "feature_extractor":
            for name, param in self.model.named_parameters():
                if 'feature_extractor' in name:  # Adjust the name based on your specific model
                    param.requires_grad = False
        elif option == "encoder":
            for name, param in self.model.named_parameters():
                if 'encoder' in name:  # Adjust the name based on your specific model
                    param.requires_grad = False
        
    def forward(self, input_values):
        return self.model(**input_values).last_hidden_state
    
    
if __name__ == "__main__":
    # Load the model
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    model_name = "microsoft/wavlm-base-plus"
    # model = AutoModel.from_pretrained(model_name).cuda()
    my_model = WavlmMelSpecPhonemeClassifier().cuda()
    my_model.train
    learning_rate = 1e-4
    batch_size = 8
    optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate)
    input_length = 1701  # 849 Example length of the input (1 second of audio at 16kHz)
    for i in range(16000):
        optimizer.zero_grad()
        input_values = torch.randn(batch_size, input_length).cuda()
        mask = torch.cat([torch.ones(batch_size, input_length - 400), torch.zeros(batch_size, 400)],  dim=-1).cuda()
        output = my_model({"input_values":input_values, "attention_mask":mask})
        print(output.shape)
        loss = output.mean()
        loss.backward()
        optimizer.step()