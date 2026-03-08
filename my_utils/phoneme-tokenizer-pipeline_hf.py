
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

dataset_file = "/home/dsi/moradim/SpeechRepainting/my_utils/phoneme_sequences.txt"
# Step 4: Train a Tokenizer
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

# Configure Pre-tokenizer to split on spaces
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Define Trainer
trainer = trainers.BpeTrainer(
    vocab_size=1000,  # Adjust as needed
    min_frequency=2,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

# Train the Tokenizer
tokenizer.train(files=[dataset_file], trainer=trainer)

# Step 5: Save the Tokenizer
tokenizer.save("phoneme_tokenizer.json")
print("Tokenizer trained and saved as phoneme_tokenizer.json")

# # Step 6: Test the Tokenizer
# test_sentence = "I love machine learning"
# test_phonemes = text_to_phonemes(test_sentence)

# encoded = tokenizer.encode(test_phonemes)
# print("Test Sentence:", test_sentence)
# print("Phoneme Representation:", test_phonemes)
# print("Tokenized Output:", encoded.tokens)