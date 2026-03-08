from transformers import AutoProcessor, BarkModel
import soundfile as sf
import time
save_dir_hf_models = '/dsi/gannot-lab1/users/mordehay/hf_models'
processor = AutoProcessor.from_pretrained("suno/bark-small", cache_dir=save_dir_hf_models)
model = BarkModel.from_pretrained("suno/bark-small", cache_dir=save_dir_hf_models)

voice_preset = "v2/en_speaker_6"

inputs = processor("IN NOVELS THE HERO HAS OFTEN PUSHED HIS MEALS AWAY UNTASTED BUT NO STAGE HERO WOULD DO ANYTHING SO UNNATURAL AS THIS", voice_preset=voice_preset)

start = time.time()
audio_array = model.generate(**inputs)
print(f"Time taken: {time.time()-start}")
speech = audio_array.cpu().numpy().squeeze()
sample_rate = model.generation_config.sample_rate
print(f" sample_rate: {sample_rate}")
sf.write('output_speech_synthesis_bark.wav', speech, sample_rate)