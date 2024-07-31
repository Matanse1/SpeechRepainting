import pyroomacoustics as pra
import numpy as np
import csv
import os
import random
import yaml
from scipy.io.wavfile import read, write
from scipy.signal import fftconvolve
import glob
from tqdm import tqdm
from types import SimpleNamespace
from dotmap import DotMap
from pathlib import Path

np.random.seed(0)
# Load parameters from the YAML file
with open('/home/dsi/moradim/SpeechRepainting/create_data/parameters.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

cfg = DotMap(cfg)


# Function to generate a random room and parameters
def generate_random_room():
    while True:
        # Random room dimensions
        room_dim = np.random.uniform([cfg.min_w, cfg.min_l, cfg.min_h], [cfg.max_w, cfg.max_l, cfg.max_h])

        # Random reverberation time (RT60)
        rt60 = np.random.uniform(cfg.rt60_min, cfg.rt60_max)
        
        # Calculate absorption coefficient from RT60 using Sabine's formula
        absorption_coefficient, max_order = pra.inverse_sabine(rt60, room_dim)

        # Random speaker position
        speaker_position = np.random.uniform(cfg.min_wall_distance, room_dim - cfg.min_wall_distance, 3)

        # Random microphone position ensuring the minimum distance to the speaker
        while True:
            mic_position = np.random.uniform(cfg.min_wall_distance, room_dim - cfg.min_wall_distance, 3)
            if np.linalg.norm(mic_position - speaker_position) >= cfg.min_mic_speaker_distance and np.linalg.norm(mic_position - speaker_position) <= cfg.max_mic_speaker_distance:
                break
        
        return room_dim, rt60, absorption_coefficient, max_order, speaker_position, mic_position

Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
librispeech_wav_paths = glob.glob(os.path.join(cfg.librispeech_path, cfg.mode, "**/**/*.wav"), recursive=True)
print(f"Found {len(librispeech_wav_paths)} LibriSpeech audio files.")
# Generate examples and save to files

# Write the parameters to a CSV file 
csv_filename = os.path.join(cfg.save_dir, "room_parameters.csv")
params = []
num_save_iter = 10000
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file, delimiter='|')
    writer.writerow(["example", "amplitude_value", "spk2mic_distance", "room_dimensions", "rt60", "absorption_coefficient", "speaker_position", "mic_position", "wav_File", "original_librispeech_file"])
for i in tqdm(range(cfg.num_examples)):
    while True:
        try:
            room_dim, rt60, absorption_coefficient, max_order, speaker_position, mic_position = generate_random_room()
            # Create the room
            room = pra.ShoeBox(room_dim, fs=cfg.sample_rate, absorption=absorption_coefficient, max_order=max_order)

            # Add a source (speaker)
            room.add_source(speaker_position)

            # Add a microphone
            mic_array = np.array(mic_position).reshape(3, 1)
            room.add_microphone_array(pra.MicrophoneArray(mic_array, room.fs))

            # Simulate
            room.compute_rir()
            break
        except Exception as e:
            print(f"Error occurred: {e}. Retrying...")
            print(f"Room dimensions: {room_dim}, RT60: {rt60}, Absorption coefficient: {absorption_coefficient}, Speaker position: {speaker_position}, Mic position: {mic_position}, Max order: {max_order}")
            
    rir = room.rir[0][0]  # Get the room impulse response

    # Load a random LibriSpeech audio signal
    while True:
        librispeech_wav_path = np.random.choice(librispeech_wav_paths)
        fs, audio_signal = read(librispeech_wav_path)
        if len(audio_signal) / fs >= cfg.audio_len:
            # Truncate  the audio signal to the desired length
            if len(audio_signal) / fs > cfg.audio_len:
                if cfg.model == 'Test':
                    start_idx = 0
                else:
                    start_idx = np.random.randint(0, len(audio_signal) - int(cfg.audio_len * fs))
                end_idx = start_idx + int(cfg.audio_len * fs)
                audio_signal = audio_signal[start_idx:end_idx]
                break
    
    amplitude_value = np.random.uniform(0.6, 1.0)
    audio_signal = amplitude_value * 0.9 * audio_signal.astype(np.float32) / np.max(np.abs(audio_signal))  # Normalize to [-1, 1]
    if fs != cfg.sample_rate:
        raise ValueError(f"Sample rate of {librispeech_wav_path} does not match the room sample rate.")

    # Convolve the audio signal with the RIR
    convolved_signal = fftconvolve(audio_signal, rir)[:len(audio_signal)]  # Truncate to the original signal length
    convolved_signal = pra.highpass(convolved_signal, cfg.sample_rate, fc=cfg.highpass_cutoff)
    convolved_signal = 0.9 * convolved_signal.astype(np.float32) / np.max(np.abs(convolved_signal))
    # Save the convolved signal as a WAV file
    wav_filename = os.path.join(cfg.save_dir, f"example_{i}.wav")
    write(wav_filename, cfg.sample_rate, convolved_signal.astype(np.float32))  # Save as float32
    spk2mic_distance = np.round(np.linalg.norm(speaker_position - mic_position), 2)
    # Save the parameters
    param = [i, amplitude_value, spk2mic_distance, np.round(room_dim, 2).tolist(), np.round(rt60, 2), np.round(absorption_coefficient, 2), np.round(speaker_position, 2).tolist(), np.round(mic_position, 2).tolist(), wav_filename, librispeech_wav_path]
    params.append(param)
    if i % num_save_iter == 0 or i == cfg.num_examples - 1:
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file, delimiter='|')
            for param in params:
                writer.writerow(param)
        params = []
print("Simulation complete. WAV files and parameters saved.")
