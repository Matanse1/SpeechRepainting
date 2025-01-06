import os
import csv
from glob import glob
from tqdm import tqdm
from pathlib import Path
import librosa
import textgrid
import numpy as np
from collections import Counter
import soundfile as sf


"""This file generate the csv for the phoneme classifier model from the folders of
/dsi/gannot-lab1/datasets/Librispeech_mfa/phoneme-frames_filter_length=640_hop_length=160
/dsi/gannot-lab1/datasets/Librispeech_mfa/mel_filter_length=640_hop_length=160 
containing the phone for each frame and the mel spectrum respectively.
"""

# Load the phoneme dictionary
def get_phones_dict(file_path):
    phoneme_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split()
            phoneme_dict[key] = int(value)
    return phoneme_dict

def samples2frames(samples, phoneme_dict_sample, phoneme_dict, filter_length, hop_length):
    """
    Convert samples to frames using frame_length and frame_shift
    """
    samples = np.pad(samples, (int(filter_length / 2), int(filter_length / 2)), constant_values=1)
    num_frames = (len(samples) - filter_length) // hop_length + 1
    frames = np.ones(num_frames)
    phoneme_sequence_list = []
    phoneme_sequence_list_with_silence = []
    phoneme_duration_list = []
    phoneme_duration_list_with_silence = []
    phoneme_int_list = []
    phoneme_int_list_with_silence = []
    
    for i in range(num_frames):
        sunsamples = samples[i * hop_length: i * hop_length + filter_length]
        counter = Counter(sunsamples)
        most_common = counter.most_common(1)[0] # number of most common element and its count
        phoneme_duration = most_common[1]
        frames[i] = most_common[0]
        
    counter_all = Counter(frames)
    for j in range(len(phoneme_dict_sample)):
        phoneme_duration = counter_all.get(j, None)
        if phoneme_duration is None:
            continue
        phoneme_detected = phoneme_dict_sample[j]
        
        if phoneme_detected == '':
            phoneme_sequence_list_with_silence.append("sil")
            phoneme_duration_list_with_silence.append(phoneme_duration)
            phoneme_int_list_with_silence.append(1)
            continue
        phoneme_int = phoneme_dict[phoneme_detected]
        ## with silence
        phoneme_sequence_list_with_silence.append(phoneme_detected)
        phoneme_duration_list_with_silence.append(phoneme_duration)
        phoneme_int_list_with_silence.append(phoneme_int)
        
        ## without silence
        phoneme_sequence_list.append(phoneme_detected)
        phoneme_duration_list.append(phoneme_duration)
        phoneme_int_list.append(phoneme_int)

        
        
    return phoneme_sequence_list, phoneme_sequence_list_with_silence, phoneme_duration_list, phoneme_duration_list_with_silence, phoneme_int_list, phoneme_int_list_with_silence

def textgrid_phoneme_to_numpy(file_path, num_samples, sampling_rate=16000):
    grid = textgrid.TextGrid.fromFile(file_path)
    audio_array = np.ones(num_samples) # one indicates silence accoriding to the phoneme_dict
    phones_tier = grid[1] # The second tier is the phoneme tier
    phoneme_dict_sample = {}
    for i, interval in enumerate(phones_tier):
        
        phoneme = interval.mark
        start_sample = int(interval.minTime * sampling_rate)
        end_sample = int(interval.maxTime * sampling_rate)

        phoneme_dict_sample[i] = phoneme
        audio_array[start_sample:end_sample] = i

    
    return audio_array, phoneme_dict_sample

def output_duration_sec(path):
    #print(path)
    try:
        wav_data, freq_sampling = librosa.load(path, sr=None)
        num_samples = len(wav_data)
        duration_audio = num_samples / freq_sampling
    except:
        print("An exception occurre")
        duration_audio = -5
        num_samples = 0
    
    return duration_audio, num_samples


if __name__ == '__main__':
    phoneme_dict_path = "/home/dsi/moradim/Documents/MFA/models/inspect/english_us_arpa_acoustic/phones.txt"
    phoneme_dict = get_phones_dict(phoneme_dict_path)
    
    root_dir = "/dsi/gannot-lab1/datasets/Librispeech_mfa"
    mode = "Test"
    mel_dir = f'/dsi/gannot-lab1/datasets/Librispeech_mfa/mel_filter_length=640_hop_length=160/{mode}'
    phoneme_dir = f'/dsi/gannot-lab1/datasets/Librispeech_mfa/phoneme-frames_filter_length=640_hop_length=160/{mode}'
    wav_dir = f'/dsi/gannot-lab1/datasets/Librispeech_mfa/data/{mode}'
    textgrid_dir = f'/dsi/gannot-lab1/datasets/Librispeech_mfa/mfa_text-grid/{mode}'
    output_csv = f'/dsi/gannot-lab1/datasets/Librispeech_mfa/{mode}_new.csv'

    mel_files = glob(f"{mel_dir}/**/*.npz", recursive=True)
    max_duration = 17
    min_duration = 2
    max_duration_found = 0
    
    hop_length = 160
    filter_length = 640
    
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter='|')
        writer.writerow(['mel_spectrum_path', 'phoneme_frame_path', 'wav_path',\
                         'phoneme_sequence_list', 'phoneme_duration_list', 'phoneme_int_list',\
                             'phoneme_sequence_list_with_silence', 'phoneme_duration_list_with_silence', 'phoneme_int_list_with_silence'])
        short_count = 0
        long_count = 0
        for num, mel_file in enumerate(tqdm(mel_files)):
            phoneme_file = Path(phoneme_dir) / (Path(mel_file).relative_to(mel_dir)).with_suffix('.npy')
            wav_file = Path(wav_dir) / (Path(mel_file).relative_to(mel_dir)).with_suffix('.wav')
            with sf.SoundFile(wav_file) as f:
                num_samples = len(f)
            textgrid_file = Path(textgrid_dir) / (Path(mel_file).relative_to(mel_dir)).with_suffix('.TextGrid')
            #get phoneme sequence and phoneme duration
            
            
            duration_audio, num_samples = output_duration_sec(wav_file)
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
                
            try:
                audio_array, phoneme_dict_sample = textgrid_phoneme_to_numpy(textgrid_file, num_samples)
                phoneme_sequence_list, phoneme_sequence_list_with_silence, phoneme_duration_list, phoneme_duration_list_with_silence,\
                    phoneme_int_list, phoneme_int_list_with_silence = samples2frames(audio_array, phoneme_dict_sample, phoneme_dict, filter_length, hop_length)
            except:
                print(f"Error in {textgrid_file}")
                continue
            rel_phoneme_file = Path(phoneme_file).relative_to(root_dir)
            rel_mel_file = Path(mel_file).relative_to(root_dir)
            rel_wav_file = Path(wav_file).relative_to(root_dir)
            writer.writerow([rel_mel_file, rel_phoneme_file, rel_wav_file, phoneme_sequence_list, phoneme_duration_list, phoneme_int_list,\
                             phoneme_sequence_list_with_silence, phoneme_duration_list_with_silence, phoneme_int_list_with_silence])
            # if num > 10:
            #     break
    print(f"short_count: {short_count}, long_count: {long_count}")
    print(f"max_duration_found: {max_duration_found}")