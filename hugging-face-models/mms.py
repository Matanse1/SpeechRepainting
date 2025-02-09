import torch
from transformers import VitsTokenizer, VitsModel, set_seed
import soundfile as sf
import time
save_dir_hf_models = '/dsi/gannot-lab1/users/mordehay/hf_models'
tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng", cache_dir=save_dir_hf_models)
model = VitsModel.from_pretrained("facebook/mms-tts-eng", cache_dir=save_dir_hf_models)

inputs = tokenizer(text="IN NOVELS THE HERO HAS OFTEN PUSHED HIS MEALS AWAY UNTASTED BUT NO STAGE HERO WOULD DO ANYTHING SO UNNATURAL AS THIS", return_tensors="pt")

set_seed(555)  # make deterministic
start = time.time()
with torch.no_grad():
   outputs = model(**inputs)
print(f"Time taken: {time.time()-start}")
speech = outputs.waveform[0]
sample_rate = model.config.sampling_rate 
print(f" sample_rate: {sample_rate}")
sf.write('output_speech_synthesis_mms.wav', speech, sample_rate)