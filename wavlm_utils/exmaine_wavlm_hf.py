from transformers import AutoProcessor, WavLMModel
import torch
from datasets import load_dataset
import soundfile as sf
import numpy as np

# dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation", trust_remote_code=True)
# dataset = dataset.sort("id")
# sampling_rate = dataset.features["audio"].sampling_rate

processor = AutoProcessor.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")
model = WavLMModel.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")
module_parameters = list(filter(lambda p: p[1].requires_grad, model.named_parameters()))
params = sum([np.prod(p.size()) for n, p in module_parameters])
print(f"The number of parameters of the model is: {params:.6f}M")

# model.freeze_feature_encoder()
# module_parameters = list(filter(lambda p: p[1].requires_grad, model.named_parameters()))
# params = sum([np.prod(p.size()) for n, p in module_parameters])
# print(f"The number of parameters of the model is: {params:.6f}M")

spk1_ut1_path = "/dsi/gannot-lab1/datasets/LibriSpeech/LibriSpeech/Train/480/123176/480-123176-0033.wav"
spk1_ut2_path = "/dsi/gannot-lab1/datasets/LibriSpeech/LibriSpeech/Train/480/123176/480-123176-0012.wav"
spk2_ut1_path = "/dsi/gannot-lab1/datasets/LibriSpeech/LibriSpeech/Train/5190/87791/5190-87791-0040.wav"
spk1_ut1, _ = sf.read(spk1_ut1_path) 
spk1_ut1 = torch.from_numpy(spk1_ut1.astype(np.float32))
spk1_ut1_dict = {"input_values": spk1_ut1.unsqueeze(0), "output_hidden_states":True}
spk1_ut2, _ = sf.read(spk1_ut2_path)
spk1_ut2 = torch.from_numpy(spk1_ut2.astype(np.float32))
spk1_ut2_dict = {"input_values": spk1_ut2.unsqueeze(0), "output_hidden_states":True}
spk2_ut1, _ = sf.read(spk2_ut1_path)
spk2_ut1 = torch.from_numpy(spk2_ut1.astype(np.float32))
spk2_ut1_dict = {"input_values": spk2_ut1.unsqueeze(0), "output_hidden_states":True}

batch = torch.stack((spk1_ut1, spk1_ut1), dim=0)
batch = batch[:, :16000*2]
# audio file is decoded on the fly
# inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
with torch.no_grad():
    batch_output = model(batch) #[B, T, fixed_dim]
    outputs11 = model(**spk1_ut1_dict).last_hidden_state
    outputs12 = model(**spk1_ut2_dict).last_hidden_state
    outputs21 = model(**spk2_ut1_dict).last_hidden_state
# outputs1 = outputs1[:, :outputs2.shape[1]]
# outputs2 = outputs2[:, :outputs1.shape[1]]
outputs11 = torch.mean(outputs11, dim=1)
outputs12 = torch.mean(outputs12, dim=1)
outputs21 = torch.mean(outputs21, dim=1)
similarity_same = torch.nn.functional.cosine_similarity(outputs11, outputs12)
similarity_diff1 = torch.nn.functional.cosine_similarity(outputs12, outputs21)
similarity_diff2 = torch.nn.functional.cosine_similarity(outputs11, outputs21)
print(f"For the same speakers the similarity is: {similarity_same}")
print(f"For different speakers the similarity is: {similarity_diff1}")
print(f"For different speakers the similarity is: {similarity_diff2}")
# last_hidden_states = outputs.last_hidden_state
# list(last_hidden_states.shape)