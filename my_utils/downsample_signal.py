    
import sys
sys.path.append('/home/dsi/moradim/SpeechRepainting/')
from dataloaders.wav2mel import STFT, load_wav_to_torch
import torchaudio
import soundfile as sf

path_audio = "/dsi/gannot-lab/gannot-lab1/users/mordehay/speech_repainting/exp/LibSp_wavlm-base-plus-rep_w_masked_pix=0.8_two_branch=True_specific_hidden_states_randn-filled/wnet_h512_d12_T400_betaT0.02/lossy_audio/original_sample_0.wav"
audio, sr = load_wav_to_torch(path_audio)
# Define resampling parameters
target_sr = 16000
if sr != target_sr:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
    audio = resampler(audio)
audio = audio / 1.1 / audio.abs().max()
audio = audio.cpu().numpy()
sf.write('/dsi/gannot-lab/gannot-lab1/users/mordehay/speech_repainting/exp/LibSp_wavlm-base-plus-rep_w_masked_pix=0.8_two_branch=True_specific_hidden_states_randn-filled/wnet_h512_d12_T400_betaT0.02/lossy_audio/resample_original_sample_0.wav', audio, 16000)



