import torch
from transformers import AutoModel, WavLMConfig, WavLMModel
import numpy as np
from torch.nn import functional as F
from torch import nn
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '7'
def size_model(model):
    module_parameters = list(filter(lambda p: p[1].requires_grad, model.named_parameters()))
    params = sum([np.prod(p.size()) for n, p in module_parameters])
    print("The number of parameters of the model is: {:.6f}M".format(params/1e6))

class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()
        configuration = WavLMConfig("microsoft/wavlm-base-plus")
        print(configuration)
        configuration.mask_time_prob = 0
        configuration.mask_feature_prob  = 0
        configuration.conv_kernel = [9, 3, 3, 3, 3, 1, 1]
        configuration.conv_stride = [1, 1, 1, 1, 1, 1, 1]
        print(configuration)
        # self.model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus", config=configuration, ignore_mismatched_sizes=True)
        self.model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        size_model(self.model)

        self.freeze_encoder(option="feature_extractor")
        self.freeze_encoder(option="feature_projection")
        size_model(self.model)
        
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
    # def forward(
    #     self,
    #     input_values: Optional[torch.Tensor],
    #     attention_mask: Optional[torch.Tensor] = None,
    #     mask_time_indices: Optional[torch.FloatTensor] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    # ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
    #     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    #     output_hidden_states = (
    #         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    #     )
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    #     extract_features = self.feature_extractor(input_values)
    #     extract_features = extract_features.transpose(1, 2)

    #     if attention_mask is not None:
    #         # compute reduced attention_mask corresponding to feature vectors
    #         attention_mask = self._get_feature_vector_attention_mask(
    #             extract_features.shape[1], attention_mask, add_adapter=False
    #         )

    #     hidden_states, extract_features = self.feature_projection(extract_features)
    #     hidden_states = self._mask_hidden_states(
    #         hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
    #     )

    #     encoder_outputs = self.encoder(
    #         hidden_states,
    #         attention_mask=attention_mask,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=return_dict,
    #     )

    #     hidden_states = encoder_outputs[0]

    #     if self.adapter is not None:
    #         hidden_states = self.adapter(hidden_states)

    #     if not return_dict:
    #         return (hidden_states, extract_features) + encoder_outputs[1:]

    #     return Wav2Vec2BaseModelOutput(
    #         last_hidden_state=hidden_states,
    #         extract_features=extract_features,
    #         hidden_states=encoder_outputs.hidden_states,
    #         attentions=encoder_outputs.attentions,
        # )

if __name__ == "__main__":
    # Load the model
    model_name = "microsoft/wavlm-base-plus"
    # model = AutoModel.from_pretrained(model_name).cuda()
    my_model = Mymodel().cuda()
    my_model.train
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate)
    input_length = 17*16000  # Example length of the input (1 second of audio at 16kHz)
    batch_size = 16
    for i in range(16000):
        # optimizer.zero_grad()
        input_values = torch.randn(batch_size, input_length).cuda()
        mask = torch.cat([torch.ones(batch_size, 16000 * 15), torch.zeros(batch_size, 16000 * 2)],  dim=-1).cuda()
        output = my_model({"input_values":input_values, "attention_mask":mask})
        print(f" for i = {i} --> {output.shape}")
        loss = output.mean()
        # loss.backward()
        # optimizer.step()
    # # Create a dummy audio input (for the sake of this example, using random input data)
    # # You should replace this with actual waveform data (e.g., WAV or feature extraction)
    # sample_rate = 16000  # WAVL model typically uses 16kHz sample rate
    # input_length = 17*16000  # Example length of the input (1 second of audio at 16kHz)
    # input_values = torch.randn(1, input_length).cuda()  # Shape: (batch_size, sequence_length)



    # # Function to monitor memory usage
    # def print_memory_usage(stage=""):
    #     allocated_memory = torch.cuda.memory_allocated() / 1024**2  # In MB
    #     cached_memory = torch.cuda.memory_reserved() / 1024**2  # In MB
    #     print(f"{stage}: Allocated Memory: {allocated_memory:.2f} MB | Cached Memory: {cached_memory:.2f} MB")

    # # Print memory usage before the forward pass
    # print_memory_usage("Before Forward Pass")

    # # Forward pass
    # model.eval()  # Set model to evaluation mode (no dropout, etc.)
    # with torch.no_grad():  # We don't need gradients for the forward pass
    #     for i in range(16000):
    #         reverb_audio1_pad = F.pad(input_values, ((i, i)), "constant", 0)
    #         reverb_audio1_out = model(reverb_audio1_pad).last_hidden_state
    #         print(f" for i = {i} --> {reverb_audio1_out.shape}")
    #     # outputs = model(input_values)

    # Print memory usage after the forward pass
    # print_memory_usage("After Forward Pass")

    # # Now, let's compute gradients (simulating training) with a backward pass

    # # Create a dummy target for computing the loss (e.g., for classification tasks, etc.)
    # # For simplicity, we're using a dummy target. Replace this with your actual task.
    # dummy_labels = torch.randint(0, 2, (1, 1)).cuda()  # Binary classification dummy target

    # # Simulate a simple loss function, e.g., CrossEntropyLoss
    # criterion = torch.nn.CrossEntropyLoss()

    # # Compute loss
    # logits = outputs.last_hidden_state.mean(dim=1)  # Example: pooling over sequence
    # loss = criterion(logits, dummy_labels.squeeze())

    # # Print memory usage before the backward pass
    # print_memory_usage("Before Backward Pass")

    # # Backward pass
    # loss.backward()

    # # Print memory usage after the backward pass
    # print_memory_usage("After Backward Pass")

    # # Optional: To free memory (if needed, usually helpful when doing multiple runs)
    # torch.cuda.empty_cache()
