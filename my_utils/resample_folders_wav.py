import os
import librosa
import soundfile as sf
from tqdm import tqdm

def resample_wav_files(src_dir, dst_dir, target_sr=16000):
    """
    Resample all WAV files in the source directory and save them to the destination directory.

    Parameters:
    src_dir (str): Path to the source directory containing WAV files.
    dst_dir (str): Path to the destination directory where resampled WAV files will be saved.
    target_sr (int): Target sampling rate (in Hz). Default is 16000 Hz.
    """
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    wav_files = []
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                wav_files.append(os.path.join(root, file))

    for src_path in tqdm(wav_files, desc="Processing WAV files"):
        file_name = os.path.basename(src_path)
        dst_path = os.path.join(dst_dir, file_name)
        
        # Load the audio file with librosa
        y, sr = librosa.load(src_path, sr=None)  # Load with original sampling rate
        
        # Resample audio to target sampling rate
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        
        # Save the resampled audio to the destination directory
        sf.write(dst_path, y_resampled, target_sr)

# Example usage
src_directory = '/dsi/gannot-lab1/datasets/FSD50K/FSD50K.eval_audio'
dst_directory = '/dsi/gannot-lab1/datasets/FSD50K/FSD50K.eval_audio_16k'
target_sr = 16000
resample_wav_files(src_directory, dst_directory, target_sr=target_sr)
