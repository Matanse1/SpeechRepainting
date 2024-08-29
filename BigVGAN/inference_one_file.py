# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import numpy as np
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import MAX_WAV_VALUE
from bigvgan import BigVGAN as Generator

h = None
device = None
torch.backends.cudnn.benchmark = False
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print(f"Loading '{filepath}'")
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + "*")
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ""
    return sorted(cp_list)[-1]


def inference(a, h):
    generator = Generator(h, use_cuda_kernel=a.use_cuda_kernel).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g["generator"])


    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        # Load the mel spectrogram in .npy format
        x = torch.load(a.input_mel_path)
        x = torch.FloatTensor(x).to(device)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        y_g_hat = generator(x)

        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype("int16")

        output_file = os.path.join(
            os.path.splitext(a.input_mel_path)[0] + "_generated_bigvgan.wav"
        )
        write(output_file, h.sampling_rate, audio)
        print(output_file)


def main():
    print("Initializing Inference Process..")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_mel_path", default="/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/LibSp_wavlm-base-plus-rep_w_masked_pix=0.8_two_branch=True/wnet_h512_d12_T400_betaT0.02/as-train-gap_asr_guidance_9cp/w1=2_w2=1.5_asr_start=270/sample_2/generated_spec.npz")
    parser.add_argument("--output_dir", default="/dsi/gannot-lab1/users/mordehay/bigvgan/generated_file_from_mel_one-file")
    parser.add_argument("--checkpoint_file", required=False, default='/dsi/gannot-lab1/users/mordehay/bigvgan/g_00050000')
    parser.add_argument("--use_cuda_kernel", action="store_true", default=False)

    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], "config.json")
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"device: {device}")
    inference(a, h)


if __name__ == "__main__":
    main()
