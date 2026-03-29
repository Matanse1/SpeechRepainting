import os
import random

def split_train_val_disjoint(train_dir, train_file, val_file, prefix, val_fraction=0.2, include_metadata=False):
    # List all WAV files in the Train directory
    wav_files = [
        os.path.splitext(file)[0]  # Remove the .wav extension
        for file in os.listdir(train_dir)
        if file.endswith(".wav")
    ]

    # Shuffle the list for random selection
    random.shuffle(wav_files)

    # Calculate the number of files for validation
    val_size = int(len(wav_files) * val_fraction)

    # Split the files into validation and training sets
    val_files = wav_files[:val_size]
    train_files = wav_files[val_size:]

    # Write validation files to the val_file
    with open(val_file, "w", encoding="utf-8") as f:
        for wav_file in val_files:
            file_path = f"{prefix}/{wav_file}"
            if include_metadata:
                f.write(f"{file_path}|metadata_placeholder\n")
            else:
                f.write(f"{file_path}\n")

    # Write the remaining files to the train_file
    with open(train_file, "w", encoding="utf-8") as f:
        for wav_file in train_files:
            file_path = f"{prefix}/{wav_file}"
            if include_metadata:
                f.write(f"{file_path}|metadata_placeholder\n")
            else:
                f.write(f"{file_path}\n")

def create_unseen_validation_file(test_dir, unseen_val_file, prefix, include_metadata=False):
    # Get a list of all WAV files in the Test directory
    wav_files = [
        os.path.splitext(file)[0]  # Remove the .wav extension
        for file in os.listdir(test_dir)
        if file.endswith(".wav")
    ]

    # Write the list to the unseen validation file
    with open(unseen_val_file, "w", encoding="utf-8") as f:
        for wav_file in wav_files:
            file_path = f"{prefix}/{wav_file}"
            if include_metadata:
                f.write(f"{file_path}|metadata_placeholder\n")
            else:
                f.write(f"{file_path}\n")

# Example usage
train_dir = "/dsi/gannot-lab/gannot-lab1/datasets/reverb_data/Train/audio_final"
test_dir = "/dsi/gannot-lab/gannot-lab1/datasets/reverb_data/Test/audio_final"
prefix = "Train/audio_final"
prefix_unseen = "Test/audio_final"
# Paths to the output files
train_file = "input_training_file.txt"
val_file = "input_validation_file.txt"
unseen_val_file = "list_input_unseen_validation_file.txt"

# Split the Train directory into disjoint 80% training and 20% validation sets
split_train_val_disjoint(train_dir, train_file, val_file, prefix, val_fraction=0.2, include_metadata=True)

# Create the unseen validation file from the Test directory with the same prefix
create_unseen_validation_file(test_dir, unseen_val_file, prefix_unseen, include_metadata=True)
