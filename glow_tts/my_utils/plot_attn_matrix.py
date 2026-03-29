import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import os

def plot_attention_matrix(npy_file_path, path2true_attn, save_path):
    # Load the numpy file
    attn_matrix = np.load(npy_file_path)
    true_attn_matrix = np.load(path2true_attn)
    # Plot the attention matrix
    fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.imshow(attn_matrix, aspect='auto', origin='lower',
                    interpolation='none')
    im = ax.imshow(true_attn_matrix, aspect='auto', origin='lower',
                    interpolation='none')
    fig.colorbar(im, ax=ax)
    plt.title('Attention Matrix')
    plt.xlabel('Source Sequence')
    plt.ylabel('Target Sequence')
    plt.savefig(save_path)
    plt.close()



def plot_attention_matrix2(npy_file_path, path2true_attn, path2est_attn, save_path):
    # Load the numpy files
    attn_matrix = np.load(npy_file_path)  # Binary matrix (0s and 1s)
    true_attn_matrix = np.load(path2true_attn)  # True attention matrix
    est_attn_matrix = np.load(path2est_attn)  # Est attention matrix

    # Create a masked version of the binary matrix (mask zeros)
    true_attn_matrix = np.where(true_attn_matrix == 0, np.nan, true_attn_matrix)
    est_attn_matrix = np.where(est_attn_matrix == 0, np.nan, est_attn_matrix)

    # Custom colormap for the binary attention matrix (1: red)
    binary_cmap_true_attn = ListedColormap(['red'])
    binary_cmap_est_attn = ListedColormap(['blue'])

    # Plot both matrices
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the true attention matrix
    im = ax.imshow(attn_matrix, aspect='auto', origin='lower', interpolation='none',
                   cmap='viridis')
    
    # Overlay the masked binary attention matrix (only ones are shown)
    ax.imshow(true_attn_matrix, aspect='auto', origin='lower', interpolation='none',
              cmap=binary_cmap_true_attn, vmin=0, vmax=1, alpha=0.7)
    
    ax.imshow(est_attn_matrix, aspect='auto', origin='lower', interpolation='none',
              cmap=binary_cmap_est_attn, vmin=0, vmax=1, alpha=0.7)
    # Create custom legend
    legend_elements = [Line2D([0], [0], color='red', lw=4, label='True Attention'),
                       Line2D([0], [0], color='blue', lw=4, label='Estimated Attention')]
    ax.legend(handles=legend_elements, loc='upper right')
    # Add a colorbar for the true attention matrix
    fig.colorbar(im, ax=ax)
    
    # Titles and labels
    plt.title('Overlayed Attention Matrices (Ones Only)')
    plt.xlabel('Source Sequence')
    plt.ylabel('Target Sequence')

    # Save and show the figure
    plt.savefig(save_path)
# Example usage

path2model = '/dsi/gannot-lab/gannot-lab1/users/mordehay/glow_tts_alignment/mel-spec-as-input_without-silenece-token_with-blank-token_true_duration_mean-only_true-attn_ce_weight=0p8_c-non-simple-head_npz=2_warmup_and_constant/alignment_results/G_155'
sample = 0
path2true_attn = os.path.join(path2model, f"sample_{sample}", 'true_attn.npy')
path2logp = os.path.join(path2model, f"sample_{sample}", 'logp.npy')
path2est_attn = os.path.join(path2model, f"sample_{sample}", 'est_attn.npy')
save_path = os.path.join(path2model, f"sample_{sample}", 'logp_plot.png')

plot_attention_matrix2(path2logp, path2true_attn, path2est_attn, save_path)