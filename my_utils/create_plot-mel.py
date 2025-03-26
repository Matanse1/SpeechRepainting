import sys
sys.path.append('/home/dsi/moradim/SpeechRepainting/')
import torch
import os
import matplotlib.image
from dataloaders.stft import TacotronSTFT
import soundfile as sf


filter_length = 640
hop_length = 160
win_length = 640
n_mel_channels = 80
sampling_rate = 16000
mel_fmin = 20
mel_fmax = 8000
stft = TacotronSTFT(
            filter_length,
            hop_length,
            win_length,
            n_mel_channels,
            sampling_rate,
            mel_fmin,
            mel_fmax)

save_path = '/home/dsi/moradim/SpeechRepainting/files4plots/plot_masked_and_inpainted'
path2inpainted = '/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/DiT_Anechoic_LibSp_conditional-masked-melspec_w-masked-pix=1/dit-net_dim768_depth18_heads12_dim-head64_dropout0.1_ff_mult2_T400_betaT0.02/repeat_all_freq-length=100_skip=150_cp=112000_mel_text=True_phoneme-without-space/w1=1_w2=1.2_asr_start=320_mask=True/sample_244/generated_audio_hifi_gan.wav' 
path2masked = '/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/DiT_Anechoic_LibSp_conditional-masked-melspec_w-masked-pix=1/dit-net_dim768_depth18_heads12_dim-head64_dropout0.1_ff_mult2_T400_betaT0.02/repeat_all_freq-length=100_skip=150_cp=112000_mel_text=True_phoneme-without-space/w1=1_w2=1.2_asr_start=320_mask=True/sample_244/masked_audio_time.wav'
inpainted_audio, sr = sf.read(path2inpainted)
masked_audio, sr = sf.read(path2masked)
inpainted_audio = torch.tensor(inpainted_audio, dtype=torch.float32).unsqueeze(0)
masked_audio = torch.tensor(masked_audio, dtype=torch.float32).unsqueeze(0)
inpainted_mel = stft.mel_spectrogram(inpainted_audio)
masked_mel = stft.mel_spectrogram(masked_audio)


inpainted_mel = inpainted_mel[0].squeeze().cpu().numpy()
masked_mel = masked_mel[0].squeeze().cpu().numpy()
matplotlib.image.imsave(os.path.join(save_path, 'inpainted.png'), inpainted_mel[::-1])
matplotlib.image.imsave(os.path.join(save_path, 'masked_mel.png'), masked_mel[::-1])