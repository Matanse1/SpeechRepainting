# this file mix speech and explosion sounds, just for testing purposes. the explosion and speech files are randomly picked from the dataset.

import pandas as pd
import numpy as np
import os
import random
import scipy.io.wavfile as wavfile
import glob
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import csv
from pathlib import Path

np.random.seed(42)
random.seed(42)

def numpy_to_audiosegment(audio: np.ndarray, sample_rate: int = 16000) -> AudioSegment:
    """
    Convert a NumPy array to an AudioSegment object.
    
    :param audio: NumPy array containing the audio data.
    :param sample_rate: Sample rate of the audio (samples per second).
    :return: AudioSegment object.
    """
    audio = (audio * 32767).astype(np.int16)  # Convert to 16-bit PCM format
    return AudioSegment(audio.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1)


def remove_leading_silence(audio: AudioSegment, silence_thresh: int = -50, min_silence_len: int = 10) -> AudioSegment:
    """
    Remove leading silence from an audio file.

    :param audio: AudioSegment object containing the audio data.
    :param silence_thresh: Silence threshold in dBFS. Audio quieter than this will be considered silence.
    :param min_silence_len: Minimum length of silence (in ms) to detect.
    :return: AudioSegment with leading silence removed.
    """
    rate1 = 16000
    # Convert NumPy array to AudioSegment
    audio = numpy_to_audiosegment(audio)
    # Detect the non-silent chunks of the audio
    nonsilent_ranges_start = detect_nonsilent(audio, silence_thresh=-20, min_silence_len=50) #for start
    nonsilent_ranges_end = detect_nonsilent(audio, silence_thresh=-40, min_silence_len=50) #for end
    
    if nonsilent_ranges_start:
                # Get the first non-silent range
        start_trim = random.sample(nonsilent_ranges_start, 1)[0][0]
        for start, end in nonsilent_ranges_end:
            if start <= start_trim and end >= start_trim:
                end_trim = end
        
        start_trim = min(0, start_trim - 50)
        # Trim the leading silence
        trimmed_audio = audio[start_trim:]
    else:
        # If no non-silent part is found, return the original audio
        trimmed_audio = audio
    explosion_length = min(int(1.5 * 1000), len(trimmed_audio), end_trim - start_trim)
    trimmed_audio = trimmed_audio[:explosion_length]
    # Convert the AudioSegment to a NumPy array
    audio_np = np.array(trimmed_audio.get_array_of_samples())
    return audio_np, explosion_length
    

def apply_delay(audio, delay):
    """Apply delay to the audio signal by introducing zeros at the beginning."""
    if delay > 0:
        return np.pad(audio, (delay, 0), mode='constant')
    return audio


def add_delay(speech, explosion, total_interval=2, interval_num=1):


    # Calculate the range for random delay
    min_delay = int(0.2 * len(speech)) / total_interval + (interval_num - 1) / total_interval * len(speech)
    max_delay = int(0.7 * len(speech)) / total_interval + (interval_num - 1) / total_interval * len(speech)

    # Generate a random delay within the specified range
    delay = random.randint(min_delay, max_delay)

    # Apply the random delay to the explosion audio
    explosion_delayed = apply_delay(explosion, delay)
    
    return explosion_delayed, delay


def adjust_snr(speech, explosion, snr):
    # snr = 0
    speech_std = np.std(speech)
    explosion_std = np.std(explosion)
    explosion_gain = np.sqrt(10 ** (-snr / 10) * np.power(speech_std, 2) / np.power(explosion_std, 2))
    explosion = explosion_gain * explosion
    return explosion


def pad_with_zeros(audio, target_length):
    """Pad the audio array with zeros to match the target length."""
    padding_length = target_length - len(audio)
    if padding_length > 0:
        audio = np.pad(audio, (0, padding_length), 'constant')
    return audio

def truncate_audio(audio, target_length):
    """Truncate the audio array to the target length."""
    if len(audio) > target_length:
        audio = audio[:target_length]
    return audio

def mix_sppech_and_explosion(df_explosion, df_speech, num_explosions=2, silence_thresh=-20):

    delays = []
    explosions_length = []
    # Load the speech 
    speech_path = get_speech(df_speech)
    rate2, speech = wavfile.read(speech_path)
    speech = speech / max(abs(speech))
    explosions = np.zeros_like(speech)
    for n_e in range(1, num_explosions+1):
        explosion_path = get_explosion(df_explosion, wav_dir)
        rate1, explosion = wavfile.read(explosion_path)
        explosion = explosion / max(abs(explosion))
        explosion, explosion_length = remove_leading_silence(explosion, silence_thresh)
        
        explosions_length.append(explosion_length)
        # explosion = truncate_audio(explosion, desired_explosion_length) 
        snr = random.randint(-5, 0)
        explosion = adjust_snr(speech, explosion, snr)
        explosion, delay = add_delay(speech, explosion, total_interval=num_explosions, interval_num=n_e)
        delays.append(delay)
        
        # speech *= 0.2
        if len(speech) > len(explosion):
            explosion = pad_with_zeros(explosion, len(speech))
        else:
            explosion = truncate_audio(explosion, len(speech))
        # Make sure the sample rates match
        if rate1 != rate2:
            raise ValueError("Sample rates of the input WAV files do not match.")

        explosions +=  explosion
        
    mix = explosions + speech
    mix = mix / max(abs(mix))
    masked_mix = mix.copy()
    norm_speech = speech.copy() / max(abs(mix))
    masked_norm_speech = speech.copy() / max(abs(mix))
    for delay, explosion_length in zip(delays, explosions_length):
        masked_mix[delay: delay + explosion_length] = 0
        masked_norm_speech[delay: delay + explosion_length] = 0
        
    return mix, masked_mix, norm_speech, masked_norm_speech, delays, explosions_length
    
    
def get_explosion(df_explosions, wav_dir):
    # Randomly select a row
    random_row = df_explosions.sample(n=1, random_state=42).iloc[0]

    # Get the corresponding filename (without extension) from the 'fname' column
    file_name = f"{random_row['fname']}.wav"
    
    # Construct the full path to the WAV file
    wav_path = os.path.join(wav_dir, file_name)

    # # Check if the WAV file exists
    # if os.path.exists(wav_path):
    #     # Read the WAV file
    #     sample_rate, audio_data = wavfile.read(wav_path)
    #     print(f"Loaded WAV file: {wav_path}")
    #     print(f"Sample Rate: {sample_rate}, Audio Data Shape: {audio_data.shape}")
    # else:
    #     print(f"WAV file not found: {wav_path}")
    print("picked explosion file: ", wav_path)
    # return '/dsi/gannot-lab1/datasets/FSD50K/FSD50K.dev_audio_16k/184418.wav'
    return wav_path


def get_speech(df_speech):
    # wav_files = [
    #     file  # Remove the .wav extension
    #     for file in os.listdir(speech_dir)
    #     if file.endswith(".wav")
    # ]
    wav_path = df_speech.sample(n=1, random_state=42).iloc[0]["wav_File"]

    # if os.path.exists(wav_path):
    #     # Read the WAV file
    #     sample_rate, audio_data = wavfile.read(wav_path)
    #     print(f"Loaded WAV file: {wav_path}")
    #     print(f"Sample Rate: {sample_rate}, Audio Data Shape: {audio_data.shape}")
    # else:
    #     print(f"WAV file not found: {wav_path}")
    print("picked speech file: ", wav_path)
    # return "/dsi/gannot-lab1/datasets/reverb_data/Test_complete/audio_final/example_1736.wav"
    return wav_path
    
    
if __name__ == '__main__':
    output_dir = "/home/dsi/moradim/SpeechRepainting/temp_dir/" #'/dsi/gannot-lab1/datasets/Speech_with_Explosions'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    num_examples = 10
    rate = 16000
    num_explosions = 2
    silence_thresh = -20 #20*np.log10(0.1)
    save_interval = 5
    wav_dir = '/dsi/gannot-lab1/datasets/FSD50K/FSD50K.dev_audio_16k'
    csv_explosion = '/dsi/gannot-lab1/datasets/FSD50K/FSD50K.ground_truth/dev_explosion_labels.csv'  # Replace with your actual CSV file path
    csv_speech = '/dsi/gannot-lab1/datasets/reverb_data/Test_complete/room_parameters_with_trans.csv'
    df_explosion = pd.read_csv(csv_explosion)
    print(f"The number of the explosions in the dataset is: {len(df_explosion)}")
    df_speech = pd.read_csv(csv_speech, delimiter='|')
    print(f"The number of the speech files in the dataset is: {len(df_speech)}")

    titles = ["exmaple",  "delays", "explosions_length"]
    with open(os.path.join(output_dir, 'explosions.csv'), mode='w') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerow(titles)

    
    delays_list = []
    explosions_length_list = []
    examples = []
    for example in range(num_examples):
        mix, masked_mix, norm_speech, masked_norm_speech, delays, explosions_length = mix_sppech_and_explosion(df_explosion, df_speech, num_explosions, silence_thresh)
        delays_list.append(delays)
        explosions_length_list.append(explosions_length)
        examples.append(example)
        
        (Path(output_dir) / Path(f'example_{example}')).mkdir(parents=True, exist_ok=True)
        output_file =  Path(output_dir)/ Path(f'example_{example}/mix.wav')
        wavfile.write(output_file, rate, mix)
        output_file =  Path(output_dir)/ Path(f'example_{example}/masked_mix.wav')
        wavfile.write(output_file, rate, masked_mix)
        output_file =  Path(output_dir)/ Path(f'example_{example}/speech.wav')
        wavfile.write(output_file, rate, norm_speech)
        output_file =  Path(output_dir)/ Path(f'example_{example}/masked_speech.wav')
        wavfile.write(output_file, rate, masked_norm_speech)
        
        if example % save_interval ==0 or example == num_examples - 1:
            with open(os.path.join(output_dir, 'explosions.csv'), mode='a') as f:
                writer = csv.writer(f, delimiter='|')
                writer.writerows(zip(examples, delays_list, explosions_length_list))
            delays_list = []
            explosions_length_list = []
            examples = []






