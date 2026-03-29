import pickle
import scipy.io.wavfile as wav
import os

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

# Example usage
for i in range(10):
    filename = f'example_{i}'
    pickle_file = f'/dsi/gannot-lab/gannot-lab1/datasets/speech_with_explosions/Test/audio_rn/{filename}.pkl'
    mix, masked_mix, masked_norm_speech, explosions, norm_speech = load_pickle(pickle_file)
    sample_rate = 16000
    os.makedirs(f'/home/dsi/moradim/sgmse/target_speech/{filename}', exist_ok=True) 
    # os.makedirs(f'/home/dsi/moradim/sgmse/speech_explosions/{filename}', exist_ok=True)
    # dict2save = {'mix': mix, 'masked_mix': masked_mix, 'masked_norm_speech': masked_norm_speech, 'explosions': explosions, 'norm_speech': norm_speech}
    dict2save = {'norm_speech': norm_speech}
    # dict2save = {'mix': mix}
    for key, value in dict2save.items(): 
        wav_file = f'/home/dsi/moradim/sgmse/target_speech/{filename}/{key}.wav'
        # wav_file = f'/home/dsi/moradim/sgmse/speech_explosions/{filename}/{key}.wav'
        wav.write(wav_file, sample_rate, value)
        print(f"Saved wav file: {wav_file}")

