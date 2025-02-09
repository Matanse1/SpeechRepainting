from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from transformers import SpeechT5HifiGan
import torch
import soundfile as sf
import time
from tqdm import tqdm
save_dir_hf_models = '/dsi/gannot-lab1/users/mordehay/hf_models'
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv', cache_dir=save_dir_hf_models)
model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv', cache_dir=save_dir_hf_models)

# audio files are decoded on the fly
audio_path = "/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/DiT_Anechoic_LibSp_conditional-masked-melspec_w-masked-pix=1/dit-net_dim768_depth18_heads12_dim-head64_dropout0.1_ff_mult2_T400_betaT0.02/repeat_all_freq-length=100_skip=150_cp=76000_mel_text=True_withoutLM_phoneme-with-space/w1=0.5_w2=0.8_asr_start=320_mask=True/sample_9/gt_audio_hifi_gan.wav"
audio, sample_rate = sf.read(audio_path)
# inputs = feature_extractor(audio, padding=True, return_tensors="pt")
# embeddings = model(**inputs).embeddings
# embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
embeddings = torch.load("/home/dsi/moradim/SpeechRepainting/hugging-face-models/xvector_7305.pt")
embeddings = embeddings.unsqueeze(0)


processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts", cache_dir=save_dir_hf_models)
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts", cache_dir=save_dir_hf_models)
inputs = processor(text="BUT SUPPOSE YOU SAID I'M FOND OF WRITING MY PEOPLE ALWAYS SAY MY LETTERS HOME ARE GOOD ENOUGH FOR PUNCH", return_tensors="pt")
inputs["input_ids"] = inputs["input_ids"].repeat(32, 1)
inputs["attention_mask"] = inputs["attention_mask"].repeat(32, 1)
# processor(text=["BUT SUPPOSE YOU SAID I'M FOND OF WRITING MY PEOPLE ALWAYS SAY MY LETTERS HOME ARE GOOD ENOUGH FOR PUNCH", "hi, how are you"], return_tensors="pt")
speaker_embeddings = torch.tensor(embeddings).repeat(32, 1)
# speaker_embeddings = torch.randn(1, 512)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan", cache_dir=save_dir_hf_models)
start = time.time()
for i in tqdm(range(10)):
    spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)
print(f"Time taken: {time.time()-start}")
with torch.no_grad():
    speech = vocoder(spectrogram)
print(f"Time taken: {time.time()-start}")
sf.write('output_speech_synthesis_f5.wav', speech[0].numpy(), sample_rate)
# start = time.time()
# speech = model.generate_speech(inputs["input_ids"], vocoder=vocoder, speaker_embeddings=speaker_embeddings)
# print(f"Time taken: {time.time()-start}")
# sf.write('output_speech_synthesis_f5.wav', speech.numpy(), sample_rate)

