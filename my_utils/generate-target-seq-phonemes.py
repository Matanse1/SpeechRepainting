# This code combines StyleSpeech and Glow-tts in order to get get the target phoneme sequence.


import torch
import numpy as np
import os
import sys
import re
import soundfile as sf
import json
from models.utils import list_str_to_idx_tts, get_StyleSpeech
from StyleSpeech.audio.stft import TacotronSTFT

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    style_speech_ch_path = "/dsi/gannot-lab1/users/mordehay/my_StyleSpeech/with-space_masked-mel4style-vec/ckpt/checkpoint_204500.pth.tar"
    style_speech_config_path =  "/dsi/gannot-lab1/users/mordehay/my_StyleSpeech/with-space_masked-mel4style-vec/config.json"

    style_speech_model, config = get_StyleSpeech(style_speech_config_path, style_speech_ch_path)
    stft = TacotronSTFT(
        config.filter_length,
        config.hop_length,
        config.win_length,
        config.n_mel_channels,
        config.sampling_rate,
        config.mel_fmin,
        config.mel_fmax,
        log="log")
    
    masked_audio_path = "/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/DiT_Anechoic_LibSp_conditional-masked-melspec_w-masked-pix=1/dit-net_dim768_depth18_heads12_dim-head64_dropout0.1_ff_mult2_T400_betaT0.02/repeat_all_freq-length=100_skip=150_cp=112000_mel_text=True_phoneme-without-space/w1=1_w2=1.5_asr_start=320_mask=True/sample_69/masked_audio_time.wav"
    masked_audio_time, sr = sf.read(masked_audio_path)
    phonemes_path = "/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/DiT_Anechoic_LibSp_conditional-masked-melspec_w-masked-pix=1/dit-net_dim768_depth18_heads12_dim-head64_dropout0.1_ff_mult2_T400_betaT0.02/repeat_all_freq-length=100_skip=150_cp=112000_mel_text=True_phoneme-without-space/w1=1_w2=1.5_asr_start=320_mask=True/sample_69/asr_text.txt"
    with open(phonemes_path, 'r') as file:
        content = file.read()

    match = re.search(r'asr_condition\s*:.*(?:\n.*){2}', content)
    if match:
        phonemes = match.group(0).splitlines()[2].strip()
        print("Extracted line:", phonemes)
    else:
        print("Pattern not found.")
    phonemes = phonemes.split()
    input_text, phoneme_length = list_str_to_idx_tts(phonemes) # already padded with zeros (zero = '_')
    input_text = input_text.to(device)
    phoneme_length = phoneme_length.to(device)
    ref_masked_mel, _ = stft.mel_spectrogram(masked_audio_time) #[B, F, T1]
    with torch.no_grad():
        # Extract style vector
        style_vector = style_speech_model.get_style_vector(ref_masked_mel.transpose(1, 2), mel_len=ref_masked_mel.shape[0]) # the input ref_masked_mel is [B, T, F], style_vector shape is [B, D=128]
        log_duration_prediction = style_speech_model.get_duration(style_vector, input_text, phoneme_length, masked_frame_number=None)
    
    
