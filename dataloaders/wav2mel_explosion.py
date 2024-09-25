
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
# We're using the audio processing from TacoTron2 to make sure it matches
from stft import TacotronSTFT, normalise_mel
import pickle

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
    dataset_type = cfg.dataset["dataset_type"]
    stft = STFT(**cfg.audio)
    mode = 'Test'
    filepaths = get_all_filenames(Path(cfg.dataset[dataset_type]["base_data_dir"], mode, "audio")) #audio_final
    num_of_files =  len(filepaths)
    #filepaths = sorted(filepaths)
    
    max_val = {'speech_melspec': 0, 'mix_melspec': 0, 'masked_speech': 0}
    min_val = {'speech_melspec': 0, 'mix_melspec': 0, 'masked_speech': 0}
    
    mel_dir  = Path(cfg.dataset[dataset_type]["base_data_dir"], mode, "mel")
    mel_dir.mkdir(parents=True, exist_ok=True)
    
    # for filepath in tqdm(filepaths):
    for i in tqdm(range(num_of_files)):
        dict2save = {}
        mel_image_dir  = Path(cfg.dataset[dataset_type]["base_data_dir"], mode, "mel_image", f"example_{i}")
        mel_image_dir.mkdir(parents=True, exist_ok=True)
        filepath = Path(cfg.dataset[dataset_type]["base_data_dir"], mode, "audio", f"example_{i}.pkl") # audio_final
        with open(filepath, 'rb') as f:
            mix, _, masked_norm_speech, _, norm_speech = pickle.load(f)
# speech_melspec mix_melspec mix_time masked_speech
        run_dict = {'speech_melspec': norm_speech, 'mix_melspec': mix, 'masked_speech': masked_norm_speech}
        for key, value in run_dict.items():
            mel_image_dir.mkdir(parents=True, exist_ok=True)
            audio = torch.from_numpy(value).float()
            # audio = audio / 1.1 / audio.abs().max()     # normalise max amplitude to be ~0.9
            try:
                melspectrogram = stft.get_mel(audio)
            except:
                print(f"Error in {filepath}_{key}")
            if melspectrogram.max() > max_val[key]:
                max_val[key] = melspectrogram.max()
            if melspectrogram.min() < min_val[key]:
                min_val[key] = melspectrogram.min()

            dict2save[key] = melspectrogram
            melspectrogram = normalise_mel(melspectrogram)

            # Convert the spectrogram to an image
            plt.imshow(melspectrogram.numpy(), cmap='jet', origin='lower')
            plt.axis('off')
            plt.colorbar()
            # Save the image
            stem = Path(filepath).stem
            image_path = mel_image_dir / Path(f"{key}.png")
            plt.savefig(str(image_path))
            plt.close()
            
        mel_filepath = mel_dir / Path(f"{stem}.npz")      
        torch.save(dict2save, mel_filepath)
        
    print(f"max_val={max_val},\n min_val={min_val}")


if __name__ == "__main__":
    main()
