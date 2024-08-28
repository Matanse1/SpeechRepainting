import torch
import matplotlib.pyplot as plt
import numpy as np


# Load the checkpoint
iter_num = 50000
checkpoint_path = f'/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/LibSp_wavlm-base-plus-rep_w_masked_pix=0.8_two_branch=True/wnet_h512_d12_T400_betaT0.02/checkpoint/{iter_num}.pkl'  # Replace with the actual path
checkpoint = torch.load(checkpoint_path)

# Initialize a list to store the weights from all blocks
weights = []

# Loop through each block and extract the weights
for i in range(12):
    weight_key = f'net_diffwave.residual_layer.residual_blocks.{i}.weighted_sum.weights'
    block_weights = checkpoint["model_state_dict"][weight_key].cpu().numpy()
    weights.append(block_weights)

# Convert the list to a NumPy array for easier manipulation
weights = np.array(weights)

# Plot the weights
num_blocks = 12
num_weights = 13

fig, ax = plt.subplots()

for i in range(num_weights):
    ax.plot(range(1, num_blocks + 1), weights[:, i], marker='o', label=f'Weight {i+1}')

ax.set_xlabel('Block Number')
ax.set_ylabel('Weights')
ax.set_title('Weights of weighted_sum vs. Blocks')
ax.legend(title='Weights', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f'/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/LibSp_wavlm-base-plus-rep_w_masked_pix=0.8_two_branch=True_last_hidden/wnet_h512_d12_T400_betaT0.02/weights_vs_blocks_{iter_num}.png')

plt.close()
plt.figure(figsize=(10, 8))
plt.imshow(weights.T, aspect='auto', cmap='viridis')
plt.colorbar(label='Weight Value')
plt.xticks(ticks=range(12), labels=[f'Block {i+1}' for i in range(12)], rotation=45)
plt.yticks(ticks=range(13), labels=[f'Weight {i+1}' for i in range(13)])
plt.xlabel('Block Number')
plt.ylabel('Weight Index')
plt.title('Heatmap of weighted_sum Weights across Blocks')
plt.savefig(f'/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/LibSp_wavlm-base-plus-rep_w_masked_pix=0.8_two_branch=True_last_hidden/wnet_h512_d12_T400_betaT0.02/weights_vs_blocks_heatmap_{iter_num}.png')
