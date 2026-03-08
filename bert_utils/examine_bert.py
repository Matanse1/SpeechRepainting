from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
text = "Replace me by any text you'd like and if you wish do it again and again."
encoded_input = tokenizer(text, return_tensors='pt')
print(encoded_input)
text = "Hi i am feeling good, thank you"
encoded_input = tokenizer(text, return_tensors='pt')
print(encoded_input)
output = model(**encoded_input)
print(output.last_hidden_state[:, 0].shape) #This is the CLS token embedding


from transformers import BertTokenizer, BertModel
import torch

# Initialize the BERT tokenizer and model from Hugging Face
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
method_list = [method for method in dir(tokenizer) if method.startswith('__') is False]
print("The methods of the tokenizer are:")
print(method_list)
model = BertModel.from_pretrained('bert-base-uncased')

# Example list of texts
texts = [
    "Hugging Face is creating a tool that democratizes AI.",
    "The quick brown fox jumps over the lazy dog."
]

# Tokenize the texts
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Ensure the model is in evaluation mode (important for layers like dropout)
model.eval()

# Using torch.no_grad to avoid computing gradients during inference
with torch.no_grad():
    # Feed the tokenized inputs into the BERT model
    outputs = model(**inputs)

    # Extract the embeddings (last hidden states)
    embeddings = outputs.last_hidden_state

    # The CLS token embedding is always the first token (index 0) in each sequence
    cls_embeddings = embeddings[:, 0, :]

# Print the CLS token embeddings
print("CLS Token Embeddings:", cls_embeddings)
print("CLS Token Embeddings Shape:", cls_embeddings.shape)
