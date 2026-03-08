import pickle

def read_phoneme_sequence(phoneme_file):
    with open(phoneme_file, 'rb') as file:
        phoneme_sequence = pickle.load(file)  # Load the phoneme sequence from the file
    
    # Now you have the phonemes list directly
    return phoneme_sequence

# Example usage
phoneme_file = '/home/dsi/moradim/SpeechRepainting/Test_2delete/7176/7176-92135-0001.phonemes'
phonemes = read_phoneme_sequence(phoneme_file)

# Print the phonemes list
print(phonemes)
