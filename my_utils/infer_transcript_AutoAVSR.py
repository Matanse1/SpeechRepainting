# this is the main script for inference of the AutoAVSR model in this github repository:
# [ https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages?tab=readme-ov-file]


import os
os.sys.path.append("/home/dsi/moradim/SpeechRepainting")
from mouthroi_processing.pipelines.pipeline import InferencePipeline
from scipy.io.wavfile import write, read
import tempfile
import numpy as np

conf_path = "/home/dsi/moradim/SpeechRepainting/mouthroi_processing/configs/LRS3_A_WER1.0.ini"
pipeline_asr = InferencePipeline(conf_path, device='cuda')
audio_path = "/dsi/gannot-lab1/users/mordehay/speech_repainting/asr_eval_recordings/sample_0/time_masking_audio.wav"
rate, audio = read(audio_path)
# Create a temporary file
sample_rate = 16000
with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
    # Save the masked audio array as a WAV file in the temporary file
    write(temp_wav.name, sample_rate, audio.astype(np.float32))

    # Send the temporary WAV file to the pipeline
    transcript_from_condition = pipeline_asr(temp_wav.name)
    text = transcript_from_condition

print(f"The transcript of the file is: {transcript_from_condition}")
transcript_from_condition = pipeline_asr(audio_path)
print(f"The transcript of the file is: {transcript_from_condition}")