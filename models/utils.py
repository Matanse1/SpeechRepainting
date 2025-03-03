from __future__ import annotations
import numpy as np
import torch
import json
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from StyleSpeech.models.StyleSpeech import StyleSpeech
from StyleSpeech.text import text_to_sequence
from StyleSpeech import utils 
_MODELS = {}


def get_phones_dict(file_path):
    phoneme_dict_p2d = {}
    phoneme_dict_d2p = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split()
            phoneme_dict_p2d[key] = int(value) #phone to digit
            phoneme_dict_d2p[int(value)] = key #digit to phone
    return phoneme_dict_p2d, phoneme_dict_d2p

def register_model(cls=None, *, name=None):
  """A decorator for registering model classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _MODELS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _MODELS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_model(name):
  return _MODELS[name]


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
        self.weights = nn.Parameter(torch.ones(num_tensors), requires_grad=True)

    def forward(self, tensor_tuple):
        # Ensure the input is a tuple of tensors
        if not isinstance(tensor_tuple, tuple) and not isinstance(tensor_tuple, list):
            raise ValueError("Input must be a tuple or array of tensors")
        
        # Stack tensors to shape [num_tensors, B, T, F]
        stacked_tensors = torch.stack(tensor_tuple)
        
        # Reshape weights to be broadcastable: [num_tensors, 1, 1, 1]
        weights = self.weights.view(-1, 1, 1, 1)
        
        # Apply weights and sum along the first dimension
        weighted_sum = torch.sum(weights * stacked_tensors, dim=0)
        
        return weighted_sum

def match_and_concatenate(H, Y, concat_mel_with_rep=True):
    """
    Duplicate the pre-trained representation H to match the time dimension of Y and concatenate them.
    
    Parameters:
    H (torch.Tensor): Pre-trained representation of shape (B, T/2, F_pretrained).
    Y (torch.Tensor): STFT representation of shape (B, T, F_stft).
    concat_mel_with_rep (bool): If True, concatenate the mel spectrogram with the repeated pre-trained representation.
    
    Returns:
    torch.Tensor: Concatenated representation of shape (B, T, F_stft + F_pretrained).
    """
    B, F_pretrained, T_half = H.size()
    B, F_stft, T = Y.size()
    
    # Duplicate each element of H to match the time dimension of Y
    H_expanded = H.repeat_interleave(2, dim=-1)[:, :, :T]
    if H_expanded.shape[-1 < Y.shape[-1]]:
        H_expanded = torch.nn.functional.pad(H_expanded, (0, Y.shape[-1] - H_expanded.shape[-1]), mode='replicate')
    Y = Y[:, :, :H_expanded.shape[-1]]
    # Concatenate along the feature dimension
    if concat_mel_with_rep:
        concatenated_representation = torch.cat((Y, H_expanded), dim=-2)
    else:
        concatenated_representation = H_expanded
    
    return concatenated_representation

def print_modle_size(model, model_name):
    params = sum([np.prod(p.size()) for n, p in model.named_parameters()])
    print("The number of parameters of {} is: {:.6f}M".format(model_name, params/1e6))
    
  # Get tokenizer


def get_tokenizer(tokenizer_path: str = None, tokenizer: str = "byte"):
    """
    tokenizer   - "pinyin" do g2p for only chinese characters, need .txt vocab_file
                - "char" for char-wise tokenizer, need .txt vocab_file
                - "byte" for utf-8 tokenizer
                - "custom" if you're directly passing in a path to the vocab.txt you want to use
    vocab_size  - if use "pinyin", all available pinyin types, common alphabets (also those with accent) and symbols
                - if use "char", derived from unfiltered character & symbol counts of custom dataset
                - if use "byte", set to 256 (unicode byte range)
    """
    if tokenizer in ["pinyin", "char"]:
        tokenizer_path = os.path.join(files("f5_tts").joinpath("../../data"), f"{dataset_name}_{tokenizer}/vocab.txt")
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)
        assert vocab_char_map[" "] == 0, "make sure space is of idx 0 in vocab.txt, cuz 0 is used for unknown char"

    elif tokenizer == "byte":
        vocab_char_map = None
        vocab_size = 256

    elif tokenizer == "phoneme":
        assert tokenizer_path is not None, "please provide path to vocab.json"
       # Load the phoneme-to-number dictionary from the JSON file
        with open(tokenizer_path, 'r') as f:
            phoneme_to_number_loaded = json.load(f)
        vocab_char_map = phoneme_to_number_loaded
        vocab_size = len(phoneme_to_number_loaded)
        
    elif tokenizer == "custom":
        with open(dataset_name, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)

    return vocab_char_map, vocab_size

# simple utf-8 tokenizer, since paper went character based
def list_str_to_tensor(text: list[str] | list[list[str]], padding_value=-1) -> int["b nt"]:  # noqa: F722
    list_tensors = [torch.tensor([*bytes(t[0], "UTF-8")]) for t in text]  # ByT5 style
    text = pad_sequence(list_tensors, padding_value=padding_value, batch_first=True)
    return text

# char tokenizer, based on custom dataset's extracted .txt file (for phoneme)
def list_str_to_idx(
    text: list[str] | list[list[str]],
    vocab_char_map: dict[str, int],  # {char: idx}
    padding_value=-1,
) -> int["b nt"]:  # noqa: F722
    list_idx_tensors = [torch.tensor([vocab_char_map.get(c, 0) for c in t]) for t in text] 
    text = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text

def list_str_to_idx_tts(
    text: list[str] | list[list[str]],
    padding_value=0,
) -> int["b nt"]:  # noqa: F722
    list_idx_tensors = [torch.tensor(text_to_sequence(("{" + " ".join(t) + "}").replace("space", "sp"), [])) for t in text] # the space of stylespeech (tts) is "sp", not "space"
    src_length = torch.tensor([t.shape[-1] for t in list_idx_tensors])
    text = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text, src_length
  
def get_StyleSpeech(config_path, checkpoint_path):
    with open(config_path) as f:
        data = f.read()
    json_config = json.loads(data)
    config = utils.AttrDict(json_config)
    model = StyleSpeech(config)
    state_dict = torch.load(checkpoint_path)['model']
    keys_to_remove = ['encoder.position_enc', 'variance_adaptor.length_regulator.position_enc', 'decoder.position_enc'] # in my case the audio can be much longer than 1000 frames, so i dont wwant each batch the position encoding to be created again
    for key in keys_to_remove:
        if key in state_dict:
            del state_dict[key]
    model.load_state_dict(state_dict, strict=False)
    # model.load_state_dict(torch.load(checkpoint_path)['model'], strict=False)
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {num_params}')
    return model, config
  