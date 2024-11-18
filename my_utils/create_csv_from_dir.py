import os
import csv
from glob import glob
from tqdm import tqdm
from pathlib import Path
import librosa

"""This file generate the csv for the phoneme classifier model from the folders of
/dsi/gannot-lab1/datasets/Librispeech_mfa/phoneme-frames_filter_length=640_hop_length=160
/dsi/gannot-lab1/datasets/Librispeech_mfa/mel_filter_length=640_hop_length=160 
containing the phone for each frame and the mel spectrum respectively.
"""
def output_duration_sec(path):
    #print(path)
    try:
        wav_data, freq_sampling = librosa.load(path, sr=None)
        duration_audio = len(wav_data) / freq_sampling
    except:
        print("An exception occurre")
        duration_audio = -5
    
    return duration_audio

root_dir = "/dsi/gannot-lab1/datasets/Librispeech_mfa"
mode = "Train"
mel_dir = f'/dsi/gannot-lab1/datasets/Librispeech_mfa/mel_filter_length=640_hop_length=160/{mode}'
phoneme_dir = f'/dsi/gannot-lab1/datasets/Librispeech_mfa/phoneme-frames_filter_length=640_hop_length=160/{mode}'
wav_dir = f'/dsi/gannot-lab1/datasets/Librispeech_mfa/data/{mode}'
output_csv = f'/dsi/gannot-lab1/datasets/Librispeech_mfa/{mode}.csv'

mel_files = glob(f"{mel_dir}/**/*.npz", recursive=True)
max_duration = 17
min_duration = 2
max_duration_found = 0
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file, delimiter='|')
    writer.writerow(['mel_spectrum_path', 'phoneme_frame_path', 'wav_path'])
    short_count = 0
    long_count = 0
    for mel_file in tqdm(mel_files):
        phoneme_file = Path(phoneme_dir) / (Path(mel_file).relative_to(mel_dir)).with_suffix('.npy')
        wav_file = Path(wav_dir) / (Path(mel_file).relative_to(mel_dir)).with_suffix('.wav')
        duration_audio = output_duration_sec(wav_file)
        if duration_audio < min_duration:
            print(f"duration of {wav_file} is {duration_audio}")
            short_count += 1
            continue
        if duration_audio > max_duration:
            print(f"duration of {wav_file} is {duration_audio}")
            long_count += 1
            continue
        if duration_audio > max_duration_found:
            max_duration_found = duration_audio
            print(f"max_duration_found: {max_duration_found}")
        rel_phoneme_file = Path(phoneme_file).relative_to(root_dir)
        rel_mel_file = Path(mel_file).relative_to(root_dir)
        rel_wav_file = Path(wav_file).relative_to(root_dir)
        writer.writerow([rel_mel_file, rel_phoneme_file, rel_wav_file])

print(f"short_count: {short_count}, long_count: {long_count}")
print(f"max_duration_found: {max_duration_found}")