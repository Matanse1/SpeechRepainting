# from pathlib import Path
# import pickle
# from tqdm import tqdm

# def extract_phonemes(textgrid_file):
#     with open(textgrid_file, 'r') as file:
#         lines = file.readlines()

#     phonemes = []
#     word_phonemes = []

#     # Parse phonemes from the "phones" tier
#     phone_mode = False
#     for line in lines:
#         line = line.strip()
#         if 'item [2]' in line:  # Start of phones tier
#             phone_mode = True
#         elif phone_mode and line.startswith('text ='):
#             phoneme = line.split('=')[1].strip().strip('"')
#             if phoneme == "":  # Marks the end of a word
#                 if word_phonemes:
#                     phonemes.extend(word_phonemes)  # Add the list of phonemes for this word
#                     phonemes.append("space")  # Add "space" to separate words
#                     word_phonemes = []
#             else:
#                 word_phonemes.append(phoneme)

#     # Add the last word (if any)
#     if word_phonemes:
#         phonemes.extend(word_phonemes)
#         phonemes.append("space")

#     return phonemes


# def process_nested_directory(input_root, output_root):
#     input_root = Path(input_root)
#     output_root = Path(output_root)

#     for textgrid_file in tqdm(input_root.glob("**/*.TextGrid"), desc="Processing TextGrid files"):  # Recursively find all TextGrid files
#         relative_path = textgrid_file.relative_to(input_root)  # Preserve folder structure
#         output_file = output_root / relative_path.with_suffix(".phonemes")

#         # Create parent directories for the output file
#         output_file.parent.mkdir(parents=True, exist_ok=True)

#         # Extract phoneme sequence and save
#         phoneme_sequence = extract_phonemes(textgrid_file)

#         # Save the phoneme list using pickle
#         with open(output_file, 'wb') as file:
#             pickle.dump(phoneme_sequence, file)
# # Replace these with your actual input and output root directories
# mode = 'Train'
# input_directory = f"/dsi/gannot-lab1/datasets/Librispeech_mfa/mfa_text-grid/{mode}"
# output_directory = f"/dsi/gannot-lab1/datasets/Librispeech_mfa/phoneme_seq/{mode}"

# process_nested_directory(input_directory, output_directory)


from pathlib import Path
import pickle
from tqdm import tqdm

def extract_phonemes(textgrid_file):
    with open(textgrid_file, 'r') as file:
        lines = file.readlines()

    phonemes = []
    word_boundaries = []
    extracting_words = False
    extracting_phones = False
    current_word_end = None
    
    # First pass: Extract word boundaries
    for line in lines:
        line = line.strip()
        if 'item [1]' in line:  # Start of words tier
            extracting_words = True
        elif 'item [2]' in line:  # Start of phones tier
            extracting_words = False
            extracting_phones = True
        
        if extracting_words and line.startswith('xmin ='):
            current_word_start = float(line.split('=')[1].strip())
        elif extracting_words and line.startswith('xmax ='):
            current_word_end = float(line.split('=')[1].strip())
        elif extracting_words and line.startswith('text ='):
            word_text = line.split('=')[1].strip().strip('"')
            if word_text:  # If the word is not empty, store the boundary
                word_boundaries.append(current_word_end)
    

        if extracting_phones and line.startswith('xmin ='):
            current_phone_start = float(line.split('=')[1].strip())
        elif extracting_phones and line.startswith('xmax ='):
            last_phone_end = float(line.split('=')[1].strip())
        elif extracting_phones and line.startswith('text ='):
            phoneme = line.split('=')[1].strip().strip('"')
            if phoneme:
                phonemes.append(phoneme)
                if last_phone_end in word_boundaries:
                    phonemes.append("space")
                
    
    return phonemes

def process_nested_directory(input_root, output_root):
    input_root = Path(input_root)
    output_root = Path(output_root)

    for textgrid_file in tqdm(input_root.glob("**/*.TextGrid"), desc="Processing TextGrid files"):  # Recursively find all TextGrid files
        relative_path = textgrid_file.relative_to(input_root)  # Preserve folder structure
        output_file = output_root / relative_path.with_suffix(".phonemes")

        # Create parent directories for the output file
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Extract phoneme sequence and save
        phoneme_sequence = extract_phonemes(textgrid_file)

        # Save the phoneme list using pickle
        with open(output_file, 'wb') as file:
            pickle.dump(phoneme_sequence, file)

# Replace these with your actual input and output root directories
mode = 'Train'
input_directory = f"/dsi/gannot-lab1/datasets/Librispeech_mfa/mfa_text-grid/{mode}"
output_directory = f"/dsi/gannot-lab1/datasets/Librispeech_mfa/phoneme_seq2/{mode}"

process_nested_directory(input_directory, output_directory)