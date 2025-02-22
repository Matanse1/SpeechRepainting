import matplotlib.pyplot as plt
import sys
sys.path.append('/home/dsi/moradim/SpeechRepainting/')
import numpy as np
from StyleSpeech.audio.stft import TacotronSTFT
import soundfile as sf
import matplotlib.image
import torch

path2audio = '/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/Unet_Anechoic_LibSp_wavlm-conditional_w-masked-pix=0.8/unet_dim64_dim_mults1_2_4_T400_betaT0.02/repeat_all_freq-length=100_skip=150_cp=1120000_mel_text=True_phoneme-without-space/w1=-1_w2=0.5_asr_start=270_mask=True/sample_69/masked_audio_time.wav'
audio, sr = sf.read(path2audio)
plt.figure(figsize=(8, 4))
plt.plot(audio)
# Remove axes and labels
plt.axis('off')  # Hide axes
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Adjust margins to remove extra space

# Save the plot with transparent background
save_path = "/home/dsi/moradim/SpeechRepainting/files4plots/signal_plot_sample=69.png"
plt.savefig(save_path, transparent=True, bbox_inches='tight', pad_inches=0)
# plt.savefig("/home/dsi/moradim/SpeechRepainting/files4plots/signal_plot_sample=69.png", transparent=True)

stft_kw = {
        'filter_length': 1024,
        'hop_length': 256,
        'win_length': 1024,
        'n_mel_channels': 80,
        'sampling_rate': 16000,
        'mel_fmin': 0.0,
        'mel_fmax': 8000.0
    }
stft = TacotronSTFT(
                stft_kw['filter_length'],
                stft_kw['hop_length'],
                stft_kw['win_length'],
                stft_kw['n_mel_channels'],
                stft_kw['sampling_rate'],
                stft_kw['mel_fmin'],
                stft_kw['mel_fmax'])
audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
mel, _ = stft.mel_spectrogram(audio_tensor)
melspec = mel[0].squeeze().cpu().numpy()
save_path_mel = save_path.replace('signal_plot', 'mel_plot')
matplotlib.image.imsave(save_path_mel, melspec[::-1])