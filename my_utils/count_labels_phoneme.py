
""" 
This script counts the occurrences of each label in the phoneme-frames .npy files for weighting the cross entropy loss.
"""
import numpy as np
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from glob import glob
import sys
from torch.nn import functional as F
import torch
sys.path.append("/home/dsi/moradim/SpeechRepainting")
from models.utils import get_phones_dict

EPS = 1e-06

phoneme_dict_path = "/home/dsi/moradim/Documents/MFA/models/inspect/english_us_arpa_acoustic/phones.txt"
_, phoneme_dict_d2p = get_phones_dict(phoneme_dict_path)

# Directory containing .npy files
mode = 'Train'
npy_dir = Path(f"/dsi/gannot-lab/gannot-lab1/datasets/Librispeech_mfa/phoneme-frames_filter_length=640_hop_length=160/{mode.capitalize()}")

# Initialize a Counter to accumulate label counts
total_counter = Counter()

# Iterate through all .npy files
for npy_file in tqdm(glob(f"{npy_dir}/*/*.npy", recursive=True)):
    # Load the numpy array
    labels = np.load(npy_file)
    
    # Count occurrences of each label in the current file
    label_counts = Counter(labels)
    
    # Update the total counter
    total_counter.update(label_counts)

# Display the aggregated counts
for label, count in total_counter.items():
    print(f"Label {label}: {count}")
    
# Save the aggregated counts to a text file
output_file = Path(f"/home/dsi/moradim/SpeechRepainting/my_utils/label_counts_{mode.lower()}.txt")
total_count = sum(total_counter.values())
phoneme_dict_d2p_to_pop = phoneme_dict_d2p.copy()
for key, value in total_counter.items():
    phoneme_dict_d2p_to_pop.pop(key, None)
print(f"Total count: {phoneme_dict_d2p_to_pop.items()}")
# Save the counter as a list for cross entropy loss
label_counts = np.zeros(len(phoneme_dict_d2p))
for num, value in total_counter.items():
    label_counts[int(num)] = 1 / (value / total_count)
label_counts_divide_sum = label_counts / label_counts.sum()
print(f"Label counts sum: {label_counts_divide_sum}")

# print(f"Label counts softmax: {F.softmax(torch.tensor(label_counts))}")
# Save the list to a .npy file
np.save(output_file.with_suffix(".npy"), label_counts_divide_sum)
with output_file.open("w") as f:
    for num, phoneme in phoneme_dict_d2p.items():
        count = total_counter.get(num, 0)
        normalized_count = count / total_count
        f.write(f"Label {num} ({phoneme}): {count} (Normalized: {normalized_count:.10f})\n")
