import scipy.io.wavfile as wav
# Specify the path to the WAV file
wav_file = '/dsi/gannot-lab/gannot-lab1/datasets/reverb_data/Train/audio_final/example_129797.wav'

# Load the WAV file
sample_rate, data = wav.read(wav_file)

# Print the size of the WAV file
print(f"Sample rate: {sample_rate}")
print(f"Data size: {data.shape}")