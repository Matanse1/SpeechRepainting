device = 'cuda'

import os
os.sys.path.append("/home/dsi/moradim/SpeechRepainting/BigVGAN")
import torch
import bigvgan
import librosa
from meldataset import get_mel_spectrogram

# instantiate the model. You can optionally set use_cuda_kernel=True for faster inference.
model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_24khz_100band_256x', use_cuda_kernel=False)

# remove weight norm in the model and set to eval mode
model.remove_weight_norm()
model = model.eval().to(device)

# load wav file and compute mel spectrogram
wav_path = '/dsi/gannot-lab/gannot-lab1/datasets/reverb_data/Train/audio_final/example_129797.wav'
wav, sr = librosa.load(wav_path, sr=model.h.sampling_rate, mono=True) # wav is np.ndarray with shape [T_time] and values in [-1, 1]
wav = torch.FloatTensor(wav).unsqueeze(0) # wav is FloatTensor with shape [B(1), T_time]
# wav = torch.nn.functional.pad(wav, (0, 65536*2 - wav.size(1))) 
wav = wav[..., :80000]
# compute mel spectrogram from the ground truth audio
mel = get_mel_spectrogram(wav, model.h).to(device) # mel is FloatTensor with shape [B(1), C_mel, T_frame]

# generate waveform from mel
with torch.inference_mode():
    wav_gen = model(mel) # wav_gen is FloatTensor with shape [B(1), 1, T_time] and values in [-1, 1]
wav_gen_float = wav_gen.squeeze(0).cpu() # wav_gen is FloatTensor with shape [1, T_time]
# save the generated audio
import scipy.io.wavfile as wavfile
print(wav_gen_float.shape)
output_path = '/home/dsi/moradim/SpeechRepainting/my_utils/audio.wav'
wavfile.write(output_path, model.h.sampling_rate, wav_gen_float[0].numpy())
# you can convert the generated waveform to 16 bit linear PCM
wav_gen_int16 = (wav_gen_float * 32767.0).numpy().astype('int16') # wav_gen is now np.ndarray with shape [1, T_time] and int16 dtype