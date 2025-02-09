from pathlib import Path
import pickle
from glob import glob
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import math

mode = 'Test'
remove_space = True
min_num = math.inf
ratio_list = []
base_data_dir = f'/dsi/gannot-lab1/datasets/Librispeech_mfa/'
for file in tqdm(glob(str(Path(base_data_dir) / f'mel_filter_length=640_hop_length=160/{mode}/**/*.npz'), recursive=True)):
    melspec = torch.load(file)
    T = melspec.shape[-1]
    if T < 200: # Skip short mel spectrograms < 2s
        continue
    input_text_path = Path(file).with_suffix('.phonemes')
    input_text_path = Path(base_data_dir) / 'phoneme_seq2' / (Path(input_text_path).relative_to(Path(base_data_dir) / 'mel_filter_length=640_hop_length=160'))
    with open(input_text_path, 'rb') as file:
        input_text = pickle.load(file)  # Load the phoneme sequence from the file
        if input_text[-1] == 'space':
            input_text = input_text[:-1]
        if 'spn' in input_text:
            input_text = list(filter(lambda x: x != 'spn', input_text))
        if remove_space:
            input_text = [x for x in input_text if x != 'space']
            
    text_len = len(input_text)
    if text_len == 0:
        print(f"Empty phoneme sequence: {input_text_path}")
        continue
    # print(f"The length of the phoneme sequence is {text_len}")
    # print(f"The length of the mel spectrogram is {T}")
    # print(f"Ratio of mel spectrogram length to phoneme sequence length: {T/text_len:.2f}")
    ratio = T/text_len
    ratio_list.append(ratio)
    if ratio < min_num:
        min_num = ratio
        print(f"New min ratio: {min_num:.2f}")
       
plt.hist(ratio_list, bins=50, edgecolor='black')
plt.xlabel('Ratio of mel spectrogram length to phoneme sequence length')
plt.ylabel('Frequency')
plt.title(f'Histogram of Ratio (Min Ratio: {min_num:.2f})')
plt.savefig('/home/dsi/moradim/SpeechRepainting/ratio_histogram_without-space.png')