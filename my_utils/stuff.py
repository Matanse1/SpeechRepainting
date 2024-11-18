import numpy as np


filter_length = 640
hop_length = 160
edge = int(0.5*16000/hop_length)
print(f"edge={edge}")
min_spacing = 30
max_block_size = 65
samples = np.random.randn(int(2*16000))
samples = np.pad(samples, (int(filter_length / 2), int(filter_length / 2)), constant_values=1)
num_frames = (len(samples) - filter_length) // hop_length + 1
print(f"num_frames={num_frames}")

num_blocks = (num_frames - 2 * edge) // (max_block_size + min_spacing)
print(f"num_blocks={num_blocks}")