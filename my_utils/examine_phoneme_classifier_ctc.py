import sys
sys.path.append('/home/dsi/moradim/SpeechRepainting')
from ASR.nnet import CTCLoss
from pathlib import Path
import pickle
from dataloaders.stft import normalise_mel
from ASR import nnet
import torch
import json
from utils import get_diffusion_hyperparams

class DotDict(dict):
    """A dictionary that supports dot notation."""
    def __getattr__(self, key):
        return self[key] if key in self else None

    def __setattr__(self, key, value):
        self[key] = value

device = "cuda"
print("Loading the phoneme sequence and mel spectrogram")
basde_dir = '/dsi/gannot-lab/gannot-lab1/datasets/Librispeech_mfa'
mode = 'Test' # Test, Train
phonemes_dir = 'phoneme_seq2'
melspec_dir = 'mel_filter_length=640_hop_length=160'
file_name = '61/61-70968-0002' # 16-122827-0000, 61/61-70968-0002
input_text_path = Path(basde_dir) / phonemes_dir/ mode / f"{file_name}.phonemes"
melspec_path = Path(basde_dir) / melspec_dir/ mode / f"{file_name}.npz"

with open(input_text_path, 'rb') as file:
    input_text = pickle.load(file)  # Load the phoneme sequence from the file
    if input_text[-1] == 'space':
        input_text = input_text[:-1]
print("The Ground Truth Phoneme Sequence is: ", input_text)

melspec = torch.load(melspec_path)
mel = normalise_mel(melspec)
mel = mel.unsqueeze(0).to(device)


# Define the diffusion configuration
diffusion_cfg = DotDict({
    "name": "linear",
    "cosine": DotDict({"T": 400, "s": 0.008}),
    "linear": DotDict({"T": 400, "beta_0": 0.0001, "beta_T": 0.02, "beta": None}),
})

diffusion_hyperparams = get_diffusion_hyperparams(diffusion_cfg, fast=False)
Alpha_bar = diffusion_hyperparams['Alpha_bar'].to(device)

B = 1
t = 100
diffusion_steps = (t * torch.ones((mel.shape[0], 1))).cuda()
z = torch.normal(0, 1, size=mel.shape).cuda()
mel = torch.sqrt(Alpha_bar[diffusion_steps.int()]) * mel + torch.sqrt(1 - Alpha_bar[diffusion_steps.int()]) * z  # compute x_t from q(x_t|x_0)



#tokenizer
tokenizer_path = '/home/dsi/moradim/SpeechRepainting/phoneme_to_number.json'
with open(tokenizer_path, 'r') as f:
    phoneme_to_number_loaded = json.load(f)
    for key in phoneme_to_number_loaded.keys():
        phoneme_to_number_loaded[key] = phoneme_to_number_loaded[key] + 1 #blank is zero so we need to add one
        
num_to_phoneme = {v: k for k, v in phoneme_to_number_loaded.items()}
num_to_phoneme[0] = 'blank'
vocab_char_map = phoneme_to_number_loaded
vocab_size = len(phoneme_to_number_loaded)

interctc_blocks = []
loss_weights = None
att_type = "patch"
# Architecture



custom_tokenizer = True
tokenizer_path = None
decoder = nnet.CTCGreedySearchDecoder(tokenizer_path=tokenizer_path, custom_tokenizer=custom_tokenizer, num_to_phoneme=num_to_phoneme)


print("Loading the model")
asr_guidance_net = nnet.AudioEfficientConformerInterCTC(vocab_size=vocab_size, att_type=att_type, interctc_blocks=interctc_blocks, strides_subsampling=2)
# checkpoint_ao = '/dsi/gannot-lab/gannot-lab1/users/mordehay/phoneme_guidance_EffConfCTC/checkpoints_epoch_4_step_4127.ckpt'
checkpoint_ao = '/dsi/gannot-lab/gannot-lab1/users/mordehay/phoneme_guidance_EffConfCTC/checkpoints_epoch_9_step_9285.ckpt'
# checkpoint_ao = "/dsi/gannot-lab/gannot-lab1/users/mordehay/phoneme_guidance_EffConfCTC_with-space/checkpoints_epoch_3_step_4127.ckpt"
asr_guidance_net.compile(losses=CTCLoss(zero_infinity=True, assert_shorter=False), loss_weights=None, decoders=decoder)
asr_guidance_net = asr_guidance_net.cuda()
asr_guidance_net.load(checkpoint_ao)
asr_guidance_net.eval()

print("Model Loaded")
target_token = torch.tensor([vocab_char_map[phoneme] for phoneme in input_text]).unsqueeze(0).cuda()
target_len = torch.tensor([len(input_text)]).unsqueeze(0).cuda()
target = (target_token, target_len)
inputs = [mel, torch.tensor(mel.shape[-1]).unsqueeze(0).cuda()]
batch_losses, batch_metrics, batch_truths, batch_preds = asr_guidance_net.forward_model(inputs, diffusion_steps.view(B,1, 1), target, compute_metrics=True, verbose=0)
print("The Loss is: ", batch_losses["loss"])
logits, lengths = asr_guidance_net(inputs, diffusion_steps.view(B,1, 1))["outputs"]
outputs = logits, lengths
est_phonemes = decoder(outputs, from_logits=True)
print("The Estimated Phoneme Sequence is: ", est_phonemes)
    