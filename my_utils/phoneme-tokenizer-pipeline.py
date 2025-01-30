


    
import sentencepiece as spm
from pathlib import Path
import glob
import os
import pickle

def read_phoneme_files(file_paths):
    """
    Reads phoneme files, with each file containing sequences of phonemes (one sentence per line).
    Ensures each phoneme (like 'IY0') is treated as an atomic unit.
    """
    sequences = []
    for file_path in file_paths:
        with open(file_path, 'rb') as f:
                data = pickle.load(f)  # Load pickle file
                if data[-1] == 'space':
                    data = data[:-1]
                s = " ".join(data).replace("space", "_")
                s = '_' + s + '_'
                sequences.extend([s])
    return sequences


def write_training_file(sequences, output_path):
    """
    Writes phoneme sequences to a file for SentencePiece training.
    Each line in the file represents one sequence.
    """
    with open(output_path, 'w') as f:
        for sequence in sequences:
            f.write(sequence + '\n')


def train_sentencepiece_model(training_file, model_prefix, vocab_size=46):
    """
    Trains a SentencePiece tokenizer that preserves phonemes as single tokens.
    """
    spm.SentencePieceTrainer.train(
        input=training_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='unigram',  # Better for phoneme-level tokenization
        character_coverage=1.0,
        input_sentence_size=1000000,  # Allow large input for diverse phoneme data
        normalization_rule_name='identity',  # Avoid normalization that could split phonemes
        split_by_whitespace=True
    )


# Paths and setup
phoneme_files_dir = "/dsi/gannot-lab1/datasets/Librispeech_mfa/phoneme_seq/Test"  # Directory containing phoneme text files
training_file = "/home/dsi/moradim/SpeechRepainting/my_utils/phoneme_sequences.txt"       # File to hold all phoneme sentences
model_prefix = "/home/dsi/moradim/SpeechRepainting/my_utils/phoneme_tokenizer"            # Prefix for the trained tokenizer

# Step 1: Gather phoneme sequences
file_paths = list(glob.glob(os.path.join(phoneme_files_dir, "**/*.phonemes"), recursive=True))  # All text files in the directory
phoneme_sequences = read_phoneme_files(file_paths)

# Step 2: Write the sequences into a training file
write_training_file(phoneme_sequences, training_file)

# Step 3: Train the tokenizer
train_sentencepiece_model(training_file, model_prefix)

print(f"Tokenizer model and vocab saved with prefix '{model_prefix}'")

