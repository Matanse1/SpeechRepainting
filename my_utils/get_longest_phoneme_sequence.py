import csv
from tqdm import tqdm

"""This file contains a function that reads a csv file containing phoneme sequences and returns the longest phoneme sequence and its duration.
"""
def get_longest_phoneme_sequence(csv_file_path):
    longest_sequence_without_sil = []
    duration_without_sil = 0
    longest_sequence_with_sil = []
    duration_with_sil = 0

    with open(csv_file_path, mode='r') as file:
        reader = csv.DictReader(file, delimiter='|')
        for row in tqdm(reader):
            phoneme_list = eval(row['phoneme_int_list'])
            if len(phoneme_list) > len(longest_sequence_without_sil):
                longest_sequence_without_sil = phoneme_list
                duration_without_sil = len(phoneme_list) 
            phoneme_list = eval(row['phoneme_int_list_with_silence'])
            if len(phoneme_list) > len(longest_sequence_with_sil):
                longest_sequence_with_sil = phoneme_list
                duration_with_sil = len(phoneme_list) 

    return duration_without_sil, longest_sequence_without_sil, duration_with_sil, longest_sequence_with_sil

if __name__ == "__main__":
    csv_file_path = '/dsi/gannot-lab1/datasets/Librispeech_mfa/Train_new.csv'
    duration_without_sil, longest_sequence_without_sil, duration_with_sil, longest_sequence_with_sil = get_longest_phoneme_sequence(csv_file_path)
    print(f"Duration without silence: {duration_without_sil}")
    print(f"Longest Phoneme Sequence without silence: {longest_sequence_without_sil}")
    print(f"Duration with silence: {duration_with_sil}")
    print(f"Longest Phoneme Sequence with silence: {longest_sequence_with_sil}")
