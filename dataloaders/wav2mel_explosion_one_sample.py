
# This is script for creating the mel spectrograms from the audio files for the sppech with explosion dataset

import os
import torch
import torch.utils.data
from scipy.io.wavfile import read
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
# We're using the audio processing from TacoTron2 to make sure it matches
from stft import TacotronSTFT, normalise_mel
import pickle
import sys
import numpy as np 
sys.path.append("/home/dsi/moradim/SpeechRepainting/")
from utils import calc_diffusion_hyperparams


def get_all_filenames(data_path):
    """
    Load all .wav files in data_path
    """
    files = glob(os.path.join(data_path, '*.pkl'), recursive=True)
    return files

def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = read(full_path)
    return torch.from_numpy(data).float(), sampling_rate


class STFT():
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, filter_length, hop_length, win_length, sampling_rate, mel_fmin, mel_fmax):
        
        self.stft = TacotronSTFT(filter_length=filter_length,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 sampling_rate=sampling_rate,
                                 mel_fmin=mel_fmin, mel_fmax=mel_fmax)

    def get_mel(self, audio):
        audio = audio.unsqueeze(0)
        audio = torch.autograd.Variable(audio, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio)
        melspec = torch.squeeze(melspec, 0)
        return melspec

# ===================================================================
# Takes directory of clean audio and makes directory of spectrograms

@hydra.main(version_base=None, config_path="../configs/", config_name="config")
def main(cfg):
    add_noise = True
    diffusion_steps = 0
    dataset_type = cfg.dataset["dataset_type"]
    stft = STFT(**cfg.audio)
    wav_path = "/home/dsi/moradim/SpeechRepainting/temp_dir/example_5/speech.wav"
    rate, audio = read(wav_path)
    save_dir = Path("/home/dsi/moradim/SpeechRepainting/temp_dir/one_sample_mel_spectrum")
    save_dir.mkdir(parents=True, exist_ok=True)
    audio = torch.from_numpy(audio).float()
            # audio = audio / 1.1 / audio.abs().max()     # normalise max amplitude to be ~0.9
    melspectrogram = stft.get_mel(audio)

    melspectrogram = normalise_mel(melspectrogram)
    if add_noise:
        diffusion_hyperparams = calc_diffusion_hyperparams(**cfg.diffusion, fast=False)
        _dh = diffusion_hyperparams
        T, Alpha_bar = _dh["T"], _dh["Alpha_bar"].cpu()
        diffusion_steps = 50 #torch.randint(T, size=(1,))
        print(f"diffusion_steps: {diffusion_steps}")
        z = torch.normal(0, 1, size=melspectrogram.shape)
        melspectrogram = torch.sqrt(Alpha_bar[diffusion_steps]) * melspectrogram + torch.sqrt(1-Alpha_bar[diffusion_steps]) * z

    plt.imshow(melspectrogram.numpy(), cmap='jet', origin='lower')
    plt.axis('off')
    # plt.colorbar()
    # Save the image
    stem = Path(wav_path).stem
    image_path = save_dir / Path(f"{stem}_add_noise={add_noise}_step={diffusion_steps}.png")
    plt.savefig(str(image_path))
    plt.close()



if __name__ == "__main__":
    main()
