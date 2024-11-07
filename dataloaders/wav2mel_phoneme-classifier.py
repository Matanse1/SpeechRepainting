
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


def get_all_filenames(data_path):
    """
    Load all .wav files in data_path
    """
    files = glob(os.path.join(data_path, '*.wav'), recursive=True)
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

@hydra.main(version_base=None, config_path="../configs/", config_name="phoneme_classifier_config")
def main(cfg):
    dataset_type = cfg.dataset["dataset_type"]
    filter_length = cfg.audio["filter_length"]
    hop_length = cfg.audio["hop_length"]
    stft = STFT(**cfg.audio)
    mode = 'Test'
    wavfile_path_root = Path(cfg.dataset[dataset_type]["base_data_dir"], "data", mode) #audio_final


    
    mel_dir  = Path(cfg.dataset[dataset_type]["base_data_dir"],  f"mel_filter_length={filter_length}_hop_length={hop_length}", mode)
    mel_image_dir  = Path(cfg.dataset[dataset_type]["base_data_dir"], f"mel_image_filter_length={filter_length}_hop_length={hop_length}", mode)
    mel_dir.mkdir(parents=True, exist_ok=True)
    mel_image_dir.mkdir(parents=True, exist_ok=True)
    
    max_val = 0
    min_val = 0
    
    wav_files = glob(f"{wavfile_path_root}/**/*.wav", recursive=True)

    for wav_file in tqdm(wav_files):
        # try:
        audio, sr = load_wav_to_torch(wav_file)
        audio = 0.9 * audio / audio.abs().max()     # normalise max amplitude to be ~0.9, since the audio is not normalised yet
        melspectrogram = stft.get_mel(audio)
        if melspectrogram.max() > max_val:
            max_val = melspectrogram.max()
        if melspectrogram.min() < min_val:
            min_val = melspectrogram.min()
            
        # Convert the spectrogram to an image
        plt.imshow(melspectrogram.numpy(), cmap='jet', origin='lower')
        plt.axis('off')
        plt.colorbar()
        # Save the image
        ((Path(mel_image_dir) / (Path(wav_file).relative_to(wavfile_path_root))).parent).mkdir(parents=True, exist_ok=True)
        image_path = Path(mel_image_dir) / (Path(wav_file).relative_to(wavfile_path_root)).with_suffix('.png')
        plt.savefig(str(image_path))
        plt.close() 
        ((Path(mel_dir) / (Path(wav_file).relative_to(wavfile_path_root))).parent).mkdir(parents=True, exist_ok=True)
        mel_filepath = Path(mel_dir) / (Path(wav_file).relative_to(wavfile_path_root)).with_suffix('.npz')  
        torch.save(melspectrogram, mel_filepath)
        # except Exception as e:
        #     print(f"Error processing {wav_file}: {e}")
        #     continue
    
        
    print(f"max_val={max_val},\n min_val={min_val}")


if __name__ == "__main__":
    main()
