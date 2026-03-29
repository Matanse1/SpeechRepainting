import numpy as np
import matplotlib.pyplot as plt
import os

base_dir = '/dsi/gannot-lab/gannot-lab1/datasets/libri_tts/LibriTTS/preprocessed_original/mel'
path2file = "libritts-mel-3551_7887_000049_000002.npy"

array_original = np.load(os.path.join(base_dir, path2file)).T

base_dir = '/dsi/gannot-lab/gannot-lab1/datasets/libri_tts/LibriTTS/preprocessed/mel'
path2file = "libritts-mel-3551_7887_000049_000002.npy"

array = np.load(os.path.join(base_dir, path2file)).T

base_dir = '/dsi/gannot-lab/gannot-lab1/datasets/libri_tts/LibriTTS/preprocessed_with-masked-mel/masked-mel'
path2file = "libritts-masked-mel-3551_7887_000049_000002.npy"

array = np.load(os.path.join(base_dir, path2file)).T

base_dir = '/dsi/gannot-lab/gannot-lab1/datasets/libri_tts/LibriTTS/preprocessed/mel'
path2file = "libritts-mel-3551_7887_000049_000002.npy"

array = np.load(os.path.join(base_dir, path2file)).T

fig = plt.figure(figsize=(30, 10))

plt.imshow(array, cmap='viridis', aspect='auto', origin='lower')
plt.colorbar()
plt.title('Attention Weights')
plt.xlabel('Key Positions')
plt.ylabel('Query Positions')
plt.savefig("mel_original.png")