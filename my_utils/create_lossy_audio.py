import torch
import torchaudio
import numpy as np
import random
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt
from torchaudio.transforms import Resample
import sys
sys.path.append("/home/dsi/moradim/SpeechRepainting")
from dataloaders.wav2mel import STFT, load_wav_to_torch
from collections import Counter


def save_mel_as_image(mel, output_path):
    plt.figure(figsize=(10, 4))
    plt.imshow(mel.cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(output_path, format="png")
    plt.close()

def create_lossy_audio_and_save(
    input_path, output_dir, sample_rate=16000, lossy_rates=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], device="cpu"
):
    # Create output directories
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    lossy_dir = output_dir / "lossy_audio"
    original_dir = output_dir / "original_audio"
    mel_dir = output_dir / "mel_specs"
    
    lossy_dir.mkdir(parents=True, exist_ok=True)
    original_dir.mkdir(parents=True, exist_ok=True)
    mel_dir.mkdir(parents=True, exist_ok=True)
    
    # Load audio file
    audio, sr = torchaudio.load(input_path) #load_wav_to_torch(input_path)
    if sr != sample_rate:
        resampler = Resample(orig_freq=sr, new_freq=sample_rate)
        audio = resampler(audio)
        sr = sample_rate

    clean_waveform = audio.to(device)
    len_samples = clean_waveform.shape[1]
    lossy_waveform = torch.clone(clean_waveform)
    
    # Generate gap mask
    len_frames = len_samples // 160  # 160 samples = 1 frame = 10ms @ 16kHz
    current_lossy_rate = 0.6 #random.choice(lossy_rates)
    frame_mask = np.random.choice([0, 1], size=(len_frames,), p=[current_lossy_rate, 1-current_lossy_rate])
    sample_mask = torch.tensor(np.repeat(frame_mask, 160), dtype=torch.float32).to(device)
    # sample_mask = torch.tensor(frame_mask, dtype=torch.float32).to(device)    
    lossy_waveform *= sample_mask.unsqueeze(0)
    
    sampling_rate = 16000
    filter_length = 640
    hop_length = 160
    win_length = 640
    mel_fmin = 20.0
    mel_fmax = 8000.0
    
    fraction_counter = 0.8
    sample_mask = np.pad(sample_mask, (int(filter_length / 2), int(filter_length / 2)), constant_values=1)
    num_frames = (len(sample_mask) - filter_length) // hop_length + 1
    frames_masking = np.ones(num_frames)
    for i in range(num_frames):
        sunsamples = sample_mask[i * hop_length: i * hop_length + filter_length]
        counter = Counter(sunsamples)
        most_common = counter.most_common(2)[0][0]
        if counter[0] >= int(filter_length * fraction_counter):
            frames_masking[i] = 0
    # Normalizations
    clean_waveform = 0.9 * clean_waveform / clean_waveform.abs().max()
    lossy_waveform = 0.9 * lossy_waveform / lossy_waveform.abs().max()
    
    # Compute spectrograms
    audio_cfg =   {
        "sampling_rate": sample_rate,
        "filter_length": 640,
        "hop_length": 160,
        "win_length": 640,
        "mel_fmin": 20.0,
        "mel_fmax": 8000.0,
    }
    

    

    # Normalizations

    stft = STFT(**audio_cfg)
    S_clean = stft.get_mel(clean_waveform[0])
    S_lossy = stft.get_mel(lossy_waveform[0])
    
    S_lossy_freq = S_clean.clone()
    S_lossy_freq_min = S_lossy_freq.min()
    for i, frame in enumerate(frames_masking):
        if frame == 0:
            S_lossy_freq[:, i] = S_lossy_freq_min
    # S_lossy_freq = S_lossy_freq * torch.from_numpy(frames_masking).unsqueeze(0) + (S_lossy_freq - 10)* (1 - torch.from_numpy(frames_masking).unsqueeze(0))
    # Save WAV files using soundfile
    original_path = original_dir / f"{Path(input_path).stem}_original.wav"
    lossy_path = lossy_dir / f"{Path(input_path).stem}_lossy.wav"
    sf.write(original_path.as_posix(), clean_waveform[0].cpu().numpy().squeeze(), sample_rate)
    sf.write(lossy_path.as_posix(), lossy_waveform[0].cpu().numpy().squeeze(), sample_rate)
    
    # Save mel spectrograms as images
    clean_mel_image_path = mel_dir / f"{Path(input_path).stem}_original_mel.png"
    lossy_mel_image_path = mel_dir / f"{Path(input_path).stem}_lossy_mel.png"
    lossy_mel_freq_image_path = mel_dir / f"{Path(input_path).stem}_lossy_mel_freq.png"
    save_mel_as_image(S_clean, clean_mel_image_path)
    save_mel_as_image(S_lossy, lossy_mel_image_path)
    save_mel_as_image(S_lossy_freq, lossy_mel_freq_image_path)
    
    print(f"Saved: {original_path}, {lossy_path}, and mel spectrogram images.")

# Example usage

input_wav_path = "/dsi/gannot-lab1/datasets/lossy_audio/clean/sample_0.wav"
output_directory = "/home/dsi/moradim/SpeechRepainting/lossy_dir"
create_lossy_audio_and_save(input_wav_path, output_directory)
