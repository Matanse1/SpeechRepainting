import numpy as np
import torch
import json
import torch.nn as nn

def load_json( json_fp ):
    with open( json_fp, 'r' ) as f:
        json_content = json.load(f)
    return json_content


def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    """
    Embed a diffusion step $t$ into a higher dimensional space
    E.g. the embedding vector in the 128-dimensional space is
    [sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)), cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]

    Parameters:
    diffusion_steps (torch.long tensor, shape=(batchsize, 1)):
                                diffusion steps for batch data
    diffusion_step_embed_dim_in (int, default=128):
                                dimensionality of the embedding space for discrete diffusion steps

    Returns:
    the embedding vectors (torch.tensor, shape=(batchsize, diffusion_step_embed_dim_in)):
    """

    assert diffusion_step_embed_dim_in % 2 == 0

    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)
    _embed = torch.exp(torch.arange(half_dim) * -_embed).cuda()
    _embed = diffusion_steps * _embed
    diffusion_step_embed = torch.cat((torch.sin(_embed),
                                      torch.cos(_embed)), 1)

    return diffusion_step_embed


class WeightedSum(nn.Module):
    def __init__(self, num_tensors):
        super(WeightedSum, self).__init__()
        # Initialize the learnable weights
        self.weights = nn.Parameter(torch.ones(num_tensors))

    def forward(self, tensor_tuple):
        # Ensure the input is a tuple of tensors
        if not isinstance(tensor_tuple, tuple):
            raise ValueError("Input must be a tuple of tensors")
        
        # Stack tensors to shape [num_tensors, B, T, F]
        stacked_tensors = torch.stack(tensor_tuple)
        
        # Reshape weights to be broadcastable: [num_tensors, 1, 1, 1]
        weights = self.weights.view(-1, 1, 1, 1)
        
        # Apply weights and sum along the first dimension
        weighted_sum = torch.sum(weights * stacked_tensors, dim=0)
        
        return weighted_sum
