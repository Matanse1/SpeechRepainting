
import ast
import pandas as pd
import commons
import numpy as np

def create_attention_matrix(sequence):
    num_frames = sum(sequence)
    sequence = np.array(sequence)
    num_rows = len(sequence)
    
    # Calculate start and end indices
    starts = np.cumsum(np.insert(sequence[:-1], 0, 0))  # Start indices
    ends = np.cumsum(sequence)                         # End indices
    
    # Generate the attention matrix
    attention_matrix = np.zeros((num_rows, num_frames), dtype=int)
    
    # Create a range for all columns
    columns = np.arange(num_frames)
    
    # Use broadcasting to create the matrix
    mask = (columns >= starts[:, None]) & (columns < ends[:, None])
    attention_matrix[mask] = 1
    
    return attention_matrix

split = 'Test'
save_dir = f'/dsi/gannot-lab1/datasets/Librispeech_mfa/attention_matrices_filter_length=640_hop_length=160/{split}'
csv_path = f"/dsi/gannot-lab1/datasets/Librispeech_mfa/{split}_new.csv"
csv_df = pd.read_csv(csv_path, delimiter="|")

for index in range(len(csv_df)):
    phoneme_duration_list_without_silence = ast.literal_eval(csv_df.loc[index, "phoneme_duration_list"])
    ##with silence
    phoneme_with_silence_list = ast.literal_eval(csv_df.loc[index, "phoneme_int_list_with_silence"])
    durations_with_silence = ast.literal_eval(csv_df.loc[index, "phoneme_duration_list_with_silence"])
    # phoneme_int_with_silence_list = ast.literal_eval(csv_df.loc[index, "durations_without_silence"])
    interspersed_phoneme_duration = commons.get_interspersed_phoneme_sequence(phoneme_with_silence_list, durations_with_silence, phoneme_duration_list_without_silence) # this list contain the duration of each phoneme interspersed with the silence token such that the duration of the silence token is also included
    interspersed_phoneme_duration = [1, 3, 10]
    num_frames = 20
    # num_frames = sum(interspersed_phoneme_duration)
    # num_rows = len(interspersed_phoneme_duration)
    # attn_matrix = np.zeros((num_rows, num_frames))
    attention_matrix = create_attention_matrix(interspersed_phoneme_duration, num_frames)
    print(attention_matrix)
    break
    