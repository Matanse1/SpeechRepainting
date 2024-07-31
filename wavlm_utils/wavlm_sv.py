from transformers import AutoProcessor, WavLMModel
import torch
from datasets import load_dataset
import soundfile as sf
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from datasets import load_dataset
import torch

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-sv')
model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-sv')
module_parameters = list(filter(lambda p: p[1].requires_grad, model.named_parameters()))
params = sum([np.prod(p.size()) for n, p in module_parameters])
print(f"The number of parameters of the model is: {params}")
spk1_ut1_path = "/dsi/gannot-lab1/datasets/LibriSpeech/LibriSpeech/Train/480/123176/480-123176-0033.wav"
spk1_ut2_path = "/dsi/gannot-lab1/datasets/LibriSpeech/LibriSpeech/Train/480/123176/480-123176-0012.wav"
spk2_ut1_path = "/dsi/gannot-lab1/datasets/LibriSpeech/LibriSpeech/Train/5190/87791/5190-87791-0040.wav"
spk1_ut1, _ = sf.read(spk1_ut1_path) 
spk1_ut1_dict = {"input_values": torch.from_numpy(spk1_ut1.astype(np.float32)).unsqueeze(0), "output_hidden_states":True}
spk1_ut2, _ = sf.read(spk1_ut2_path)
spk1_ut2_dict = {"input_values": torch.from_numpy(spk1_ut2.astype(np.float32)).unsqueeze(0), "output_hidden_states":True}
spk2_ut1, _ = sf.read(spk2_ut1_path)
spk2_ut1_dict = {"input_values": torch.from_numpy(spk2_ut1.astype(np.float32)).unsqueeze(0), "output_hidden_states":True}

# audio files are decoded on the fly
with torch.no_grad():
    embeddings_spk1_ut1 = model(**spk1_ut1_dict).embeddings
    embeddings_spk2_ut1 = model(**spk2_ut1_dict).embeddings
    embeddings_spk1_ut1 = torch.nn.functional.normalize(embeddings_spk1_ut1, dim=-1).cpu()
    embeddings_spk2_ut1 = torch.nn.functional.normalize(embeddings_spk2_ut1, dim=-1).cpu()


# the resulting embeddings can be used for cosine similarity-based retrieval
cosine_sim = torch.nn.CosineSimilarity(dim=-1)
similarity = cosine_sim(embeddings_spk1_ut1[0], embeddings_spk2_ut1[0])
print(similarity)