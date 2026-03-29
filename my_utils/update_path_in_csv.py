import pandas as pd
import os 

# Path to your CSV file
mode = 'Test_complete'
csv_file_path = f'/dsi/gannot-lab/gannot-lab1/datasets/reverb_data/{mode}/room_parameters.csv'  # Replace with your actual CSV file path

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file_path, delimiter='|')
# old_path = f'/dsi/gannot-lab/gannot-lab1/datasets/reverb_data/audio_final2/'
# new_path = f'/dsi/gannot-lab/gannot-lab1/datasets/reverb_data/{mode}/audio_final/'


def extract_transcript(audio_file_path):
    # Extract the base filename (without extension) to find the .trans.txt file
    base_filename = os.path.splitext(os.path.basename(audio_file_path))[0]
    # Extract the directory from the audio file path
    audio_dir = os.path.dirname(audio_file_path)
    # Construct the path to the corresponding .trans.txt file
    base_filename_without_last = '-'.join(base_filename.split('-')[:-1])
    trans_file_path = os.path.join(audio_dir, f"{base_filename_without_last}.trans.txt")
    
    # Read the transcript file and extract the relevant text
    if os.path.exists(trans_file_path):
        with open(trans_file_path, 'r') as file:
            for line in file:
                # Check if the line contains the transcript for the specific audio file
                if line.startswith(base_filename):
                    return line.split(' ', 1)[1].strip()  # Return text after the base filename
    return "Transcript not found"



# Function to replace the old path with the new path
def update_file_path(path):
    if pd.notna(path):
        return path.replace(old_path, new_path)
    return path
# Function to update the path in the 'wav_File' column
# def update_wav_file_path(row):
#     if pd.notna(row['wav_File']):
#         # Replace 'audio_final2' with 'audio_final'
#         return row['wav_File'].replace('audio_final2', 'audio_final')
#     return row['wav_File']

# Apply the function to update the 'wav_File' column
# df['wav_File'] = df.apply(update_wav_file_path, axis=1)
# df['wav_File'] = df['wav_File'].apply(update_file_path)

# Apply the function to extract the transcript for each row
df['transcript'] = df['original_librispeech_file'].apply(extract_transcript)

# Save the updated DataFrame back to a CSV file
updated_csv_file_path = f'/dsi/gannot-lab/gannot-lab1/datasets/reverb_data/{mode}/room_parameters_with_trans.csv'  # Replace with your desired output file path
df.to_csv(updated_csv_file_path, index=False, sep='|')

print(f"Updated CSV file saved to {updated_csv_file_path}")
