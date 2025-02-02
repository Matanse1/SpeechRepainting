from pathlib import Path
import pickle
from glob import glob
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import math
input_text_path = '/dsi/gannot-lab1/datasets/Librispeech_mfa/phoneme_seq2/Test/61/61-70968-0000.phonemes'
with open(input_text_path, 'rb') as file:
    input_text = pickle.load(file)  # Load the phoneme sequence from the file
    if input_text[-1] == 'space':
        input_text = input_text[:-1]
    if 'spn' in input_text:
        input_text = list(filter(lambda x: x != 'spn', input_text))
        
print(input_text)
print(f"The length of the phoneme sequence is {len(input_text)}")
