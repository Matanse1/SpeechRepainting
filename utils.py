import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from collections import Counter
from models import model_identifier
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
import math




def pad_last_dim(tensor, pad_size, pad_value=0):
    # Create the padding tuple dynamically based on the number of dimensions
    pad = [0, pad_size]  # Only pad the last dimension
    pad = tuple(pad) + (0,) * (2 * (tensor.dim() - 1)) 
    
    # Apply padding
    return F.pad(tensor, pad, value=pad_value)

def fix_len_compatibility(length, num_downsamplings_in_unet=2):
    while True:
        if length % (2**num_downsamplings_in_unet) == 0:
            return length
        length += 1

def flatten(v):
    """
    Flatten a list of lists/tuples
    """

    return [x for y in v for x in y]


def rescale(x):
    """
    Rescale a tensor to 0-1
    """

    return (x - x.min()) / (x.max() - x.min())


def find_max_epoch(path):
    """
    Find maximum epoch/iteration in path, formatted ${n_iter}.pkl
    E.g. 100000.pkl

    Parameters:
    path (str): checkpoint path

    Returns:
    maximum iteration, -1 if there is no (valid) checkpoint
    """

    files = os.listdir(path)
    epoch = -1
    for f in files:
        if len(f) <= 4:
            continue
        if f[-4:]  == '.pkl':
            try:
                epoch = max(epoch, int(f[:-4]))
            except:
                continue
    return epoch

def smooth_ckpt(path, min_ckpt, max_ckpt, alpha=None):
    print(f"finding checkpoints in ({min_ckpt}, {max_ckpt}] in {path}")
    files = os.listdir(path)
    ckpts = []
    for f in files:
        if len(f) <= 4:
            continue
        if f[-4:]  == '.pkl':
            print(f)
            try:
                it = int(f[:-4])
                if min_ckpt < it and it <= max_ckpt:
                    ckpts.append(it)
            except:
                continue
    ckpts = sorted(ckpts)
    print("found ckpts", ckpts)
    state_dict = None
    for n, it in enumerate(ckpts):
        model_path = os.path.join(path, '{}.pkl'.format(it))
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            # net.load_state_dict(checkpoint['model_state_dict'])
            state_dict = smooth_dict(state_dict, checkpoint['model_state_dict'], n, alpha=alpha)
            print('Successfully loaded model at iteration {}'.format(it))
        except:
            raise Exception(f'No valid model found at iteration {it}, path {model_path}')
    return state_dict


def print_size(net, verbose=False):
    """
    Print the number of parameters of a network
    """

    if net is not None and isinstance(net, torch.nn.Module):
        # module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        module_parameters = list(filter(lambda p: p[1].requires_grad, net.named_parameters()))

        if verbose:
            for n, p in module_parameters:
                print(n, p.numel())

        params = sum([np.prod(p.size()) for n, p in module_parameters])
        print("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6), flush=True)



def local_directory(name, model_cfg, diffusion_cfg, save_dir, output_directory):

    # generate experiment (local) path
    model_name = model_identifier(model_cfg)
    if not isinstance(diffusion_cfg, DictConfig):
        diffusion_name = diffusion_cfg
    else:
        if diffusion_cfg.name == 'linear':
            diffusion_name = f"_T{diffusion_cfg.linear['T']}_betaT{diffusion_cfg.linear['beta_T']}"
        elif diffusion_cfg.name == 'cosine':
            diffusion_name = f"_T{diffusion_cfg.cosine['T']}_s{diffusion_cfg.cosine['s']}"
    local_path = model_name + diffusion_name


    # Get shared output_directory ready
    if save_dir is None:
        save_dir = os.getcwd()
    if not (name is None or name == ""):
        output_directory = os.path.join(save_dir, 'exp', name, local_path, output_directory)
    else:
        output_directory = os.path.join(save_dir, 'exp', local_path, output_directory)
    if not os.path.isdir(output_directory):
        print(f"Creating output directory {output_directory}")
        os.makedirs(output_directory, exist_ok=True)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)
    return local_path, output_directory


# Utilities for diffusion models

def diffwave_fast_inference_schedule(T, beta_0, beta_T, beta=None):
    Beta = torch.tensor(beta)
    inference_timesteps = []

    # training schedules
    tBeta = torch.linspace(beta_0, beta_T, T)   # training values for Beta
    tAlpha = 1 - tBeta
    tAlpha_bar = torch.cumprod(tAlpha, dim=0)
    
    # inference schedules
    Alpha = 1 - Beta
    Alpha_bar = torch.cumprod(Alpha, dim=0)

    for s in range(len(beta)):
      for t in range(len(tBeta) - 1):
        if tAlpha_bar[t+1] <= Alpha_bar[s] <= tAlpha_bar[t]:
          twiddle = (tAlpha_bar[t]**0.5 - Alpha_bar[s]**0.5) / (tAlpha_bar[t]**0.5 - tAlpha_bar[t+1]**0.5)
          inference_timesteps.append(t + twiddle)
          break

    Beta_tilde = Beta + 0
    for t in range(1, len(Beta)):
        Beta_tilde[t] *= (1-Alpha_bar[t-1]) / (1-Alpha_bar[t])

    Sigma = torch.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t
    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"], _dh["inference_timesteps"] = T, Beta.cuda(), Alpha.cuda(), Alpha_bar.cuda(), Sigma, inference_timesteps
    return _dh


def calc_diffusion_hyperparams_linear(T, beta_0, beta_T, beta=None, fast=False):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value,
                                where any beta_t in the middle is linearly interpolated

    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
        These cpu tensors are changed to cuda tensors on each individual gpu
    """

    if fast and beta is not None:
        Beta = torch.tensor(beta)
        T = len(beta)
    else:
        Beta = torch.linspace(beta_0, beta_T, T)
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t-1]  # \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
        Beta_tilde[t] *= (1-Alpha_bar[t-1]) / (1-Alpha_bar[t])  # \tilde{\beta}_t = \beta_t * (1-\bar{\alpha}_{t-1}) / (1-\bar{\alpha}_t)
    Sigma = torch.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t

    _dh = {"name": "linear"}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = T, Beta.cuda(), Alpha.cuda(), Alpha_bar.cuda(), Sigma
    return _dh


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas)

def calc_diffusion_hyperparams_cosine(T, s=0.008):
    """
    Compute diffusion process hyperparameters using a cosine-based beta schedule.

    Parameters:
    T (int): Number of diffusion steps
    s (float): Small offset to prevent beta values from being too small at the start

    Returns:
    dict: Diffusion hyperparameters including T, Beta, Alpha, Alpha_bar, and Sigma
    """

    Beta =  betas_for_alpha_bar(T, lambda t: (math.cos((t + s) / (s + 1) * math.pi / 2) ** 2) / (math.cos((0 + s) / (s + 1) * math.pi / 2) ** 2))
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t-1]  # \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
        Beta_tilde[t] *= (1-Alpha_bar[t-1]) / (1-Alpha_bar[t])  # \tilde{\beta}_t = \beta_t * (1-\bar{\alpha}_{t-1}) / (1-\bar{\alpha}_t)
    Sigma = torch.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t

    _dh = {"name": "cosine"}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = T, Beta.cuda(), Alpha.cuda(), Alpha_bar.cuda(), Sigma
    
    return _dh


def get_diffusion_hyperparams(diffusion_cfg, fast=False):
    if diffusion_cfg.name == 'linear':
        dh = calc_diffusion_hyperparams_linear(**diffusion_cfg.linear, fast=fast)
    elif diffusion_cfg.name == 'cosine':
        dh = calc_diffusion_hyperparams_cosine(**diffusion_cfg.cosine)
    return dh


def find_linear_t_given_cosine_t():
    dh_linear = calc_diffusion_hyperparams_linear(T=400, beta_0=0.0001, beta_T=0.02)
    Alpha_bar_linear = dh_linear["Alpha_bar"]
    dh_cosine = calc_diffusion_hyperparams_cosine(T=400, s=0.008)
    Alpha_bar_cosine = dh_cosine["Alpha_bar"]
    def linear_t_given_cosine_t(t_prime):
        alpha_bar_t = Alpha_bar_cosine[t_prime]
        # Calculate the absolute difference between alpha_bar_t_prime and alpha_bar_t
        diff = np.abs(Alpha_bar_linear - alpha_bar_t)
        # Find the timestep in the linear schedule that minimizes the difference
        corresponding_linear_t = np.argmin(diff)
        return corresponding_linear_t
    return linear_t_given_cosine_t

linear_t_given_cosine_t = find_linear_t_given_cosine_t()

def plot_melspec(melspec):
    fig = plt.figure()
    plt.imshow(melspec[::-1])
    return fig

def preprocess_text(txt):
    txt = txt.replace("{LG}", "")  # remove laughter
    txt = txt.replace("{NS}", "")  # remove noise
    txt = txt.replace("\n", "")
    txt = txt.replace("  ", " ")
    txt = txt.lower().strip()
    return txt


def samples2frames(samples, filter_length, hop_length):
    """
    Convert samples to frames using frame_length and frame_shift
    """
    samples = np.pad(samples, (int(filter_length / 2), int(filter_length / 2)), constant_values=1)
    num_frames = (len(samples) - filter_length) // hop_length + 1
    frames = np.ones(num_frames)
    for i in range(num_frames):
        sunsamples = samples[i * hop_length: i * hop_length + filter_length]
        counter = Counter(sunsamples)
        most_common = counter.most_common(2)[0][0]
        if counter[0] > int(filter_length * 0.5):
            frames[i] = 0


        # most_common = counter.most_common(2)[0][0]
        # frames[i] = most_common
        # if 0 in sunsamples:
        #     frames[i] =  0 
    return frames

def insert_values(melspec, frame_mask, mean=0.0, std=1.0, num='randn'):
    """
    Insert random values into a mel-spectrogram at indices where the frame mask is `1`.

    Args:
        melspec (np.ndarray): Input mel-spectrogram (2D array).
        frame_mask (np.ndarray): Frame-based mask (1D array of 0s and 1s).
        mean (float): Mean of the normal distribution.
        std (float): Standard deviation of the normal distribution.

    Returns:
        np.ndarray: Modified mel-spectrogram with random values inserted.
    """
    # Ensure frame mask is the same length as the number of frames in the mel-spectrogram
    assert melspec.shape[1] == len(frame_mask), "Frame mask length must match the mel-spectrogram's frame count."
    
    # Create a copy of the mel-spectrogram to modify
    modified_melspec = melspec.copy()
    
    # Identify indices where frame mask is 0
    indices = np.where(frame_mask == 0)[0]
    
    # Replace values in the identified frames with random values
    if num == 'randn':
        for idx in indices:
            modified_melspec[:, idx] = np.random.normal(loc=mean, scale=std, size=melspec.shape[0])
    elif num == 'zeros':
        modified_melspec = modified_melspec * frame_mask
    
    return modified_melspec

def find_zero_regions(signal, min_length=10, max_length=1.5*16000):
    """
    Generate a mask indicating locations in the signal where 
    the audio is zero for at least `min_length` consecutive samples.

    Args:
        signal (np.ndarray): Input audio signal (1D array).
        min_length (int): Minimum number of consecutive zero samples.

    Returns:
        np.ndarray: Boolean mask of the same length as the input signal.
                    True for samples part of zero regions >= min_length.
    """
    # Convert signal to binary mask: True for zeros
    zero_mask = (signal == 0)
    
    # Find transitions in the zero mask
    zero_diff = np.diff(zero_mask.astype(int))
    starts = np.where(zero_diff == 1)[0] + 1  # Start of zero regions
    ends = np.where(zero_diff == -1)[0] + 1   # End of zero regions
    
    # Handle edge cases where signal starts or ends with zeros
    if zero_mask[0]:
        starts = np.insert(starts, 0, 0)
    if zero_mask[-1]:
        ends = np.append(ends, len(signal))
        
    durations = ends - starts
    count = np.sum(durations > min_length)
    print(f"Found {count} regions with length > {min_length}")
    # Initialize the output mask
    output_mask = np.ones_like(signal, dtype=bool)
    
    # Mark regions with at least `min_length` consecutive zeros
    for start, end in zip(starts, ends):
        if (end - start >= min_length) and (end - start <= max_length):
            output_mask[start:end] = 0
    
    return output_mask

def plot_masked_melspec_with_activity(masked_melspec, frame_mask, output_directory, sample_idx):
    """
    Plot masked mel-spectrogram and frame mask activity side by side.

    Args:
        masked_melspec (np.ndarray): The masked mel-spectrogram (2D array).
        frame_mask (np.ndarray): Frame mask activity (1D array of 0s and 1s).
        output_directory (str): Directory to save the plot.
        sample_idx (int): Sample index for naming the output file.
    """
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Create a subplot for the mel-spectrogram and activity plot
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [4, 1]})

    # Plot the masked mel-spectrogram
    ax[0].imshow(masked_melspec[::-1], aspect='auto', cmap='viridis', origin='lower')
    ax[0].set_title("Masked Mel-Spectrogram with Noise")
    ax[0].set_ylabel("Frequency Bins")
    ax[0].set_xlabel("Time Frames")
    
    # Plot the frame mask activity
    ax[1].plot(frame_mask, drawstyle='steps-mid', color='red', label='Activity Mask')
    ax[1].set_title("Frame Mask Activity")
    ax[1].set_ylabel("Activity (0 or 1)")
    ax[1].set_xlabel("Frames")
    ax[1].set_ylim(-0.1, 1.1)
    ax[1].grid(True, linestyle='--', alpha=0.5)
    ax[1].legend()

    # Save the combined plot
    output_path = os.path.join(output_directory, f'sample_{sample_idx}', 'masked_melspectrogram_with_noise_and_activity.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def plot_signal_with_activity(signal, frame_mask, sr, output_directory, sample_idx, idx_bool=False):
    """
    Plot the time-domain signal and frame mask activity side by side.

    Args:
        signal (np.ndarray): The time-domain signal (1D array).
        frame_mask (np.ndarray): Frame mask activity (1D array of 0s and 1s).
        sr (int): Sampling rate of the signal (e.g., 16000 Hz).
        output_directory (str): Directory to save the plot.
        sample_idx (int): Sample index for naming the output file.
    """
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Time vector for the signal
    time_vector = np.arange(len(signal)) / sr

    # Create a subplot for the signal and activity plot
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [4, 1]})

    # Plot the signal in the time domain
    ax[0].plot(time_vector, signal, color='blue', label='Signal')
    ax[0].set_title("Time-Domain Signal")
    ax[0].set_ylabel("Amplitude")
    ax[0].set_xlabel("Time (seconds)")
    ax[0].grid(True, linestyle='--', alpha=0.5)
    ax[0].legend()

    # Plot the frame mask activity
    ax[1].plot(frame_mask, drawstyle='steps-mid', color='red', label='Activity Mask')
    ax[1].set_title("Frame Mask Activity")
    ax[1].set_ylabel("Activity (0 or 1)")
    ax[1].set_xlabel("Frames")
    ax[1].set_ylim(-0.1, 1.1)
    ax[1].grid(True, linestyle='--', alpha=0.5)
    ax[1].legend()

    # Save the combined plot
    if idx_bool:
        output_path = os.path.join(output_directory, f'sample_{sample_idx}', 'signal_with_activity.png')
    else:
        output_path = os.path.join(output_directory, sample_idx, 'signal_with_activity.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    
    
def save_melspectrogram_with_colorbar(melspectrogram, output_directory, sample_idx):
    """
    Save the melspectrogram image with a colorbar.

    Args:
        melspectrogram (np.ndarray or torch.Tensor): The mel spectrogram to save.
        output_directory (str): The directory where the image will be saved.
        sample_idx (int): The sample index for naming the file.
    """
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Convert melspectrogram to NumPy if it's a PyTorch tensor
    if not isinstance(melspectrogram, np.ndarray):
        melspectrogram = melspectrogram.numpy()

    # Create the figure
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(melspectrogram[::-1], aspect='auto', interpolation='nearest', cmap='viridis')
    fig.colorbar(cax, ax=ax, orientation='vertical', label='Magnitude')

    # Add titles and labels
    ax.set_title("Masked Mel Spectrogram")
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Mel Bands")

    # Save the image
    save_path = os.path.join(output_directory, f'sample_{sample_idx}', 'masked_melspectrogram_image.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    
    
def plot_masked_melspec_and_spec_with_activity(masked_melspec, frame_mask, melspectrogram, output_directory, sample_idx, idx_bool=True):
    """
    Plot masked mel-spectrogram, frame mask activity, and raw mel-spectrogram with a shared x-axis.
    """
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Convert melspectrogram to numpy if needed
    if not isinstance(melspectrogram, np.ndarray):
        melspectrogram = melspectrogram.numpy()

    # Create figure with a shared colorbar position
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[4, 1, 4], width_ratios=[20, 1])
    
    # Create three main subplot axes and two colorbar axes
    ax0 = fig.add_subplot(gs[0, 0])  # Top spectrogram
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)  # Middle activity plot
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)  # Bottom spectrogram
    cax0 = fig.add_subplot(gs[0, 1])  # Colorbar for top
    cax1 = fig.add_subplot(gs[2, 1])  # Colorbar for bottom
    
    # Plot the masked mel-spectrogram
    im0 = ax0.imshow(masked_melspec, aspect='auto', cmap='viridis', origin='lower')
    ax0.set_title("Masked Mel-Spectrogram with Noise")
    ax0.set_ylabel("Frequency Bins")
    fig.colorbar(im0, cax=cax0)

    # Plot the frame mask activity
    ax1.step(np.arange(len(frame_mask)), frame_mask, color='red', label='Activity Mask', where='post')
    ax1.set_title("Frame Mask Activity")
    ax1.set_ylabel("Activity (0 or 1)")
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    # Plot the raw mel-spectrogram
    im2 = ax2.imshow(melspectrogram, aspect='auto', cmap='viridis', origin='lower')
    ax2.set_title("Raw Mel-Spectrogram")
    ax2.set_ylabel("Frequency Bins")
    ax2.set_xlabel("Time Frames")
    fig.colorbar(im2, cax=cax1)

    # Set consistent x-axis limits
    n_frames = masked_melspec.shape[1]
    ax0.set_xlim(0, n_frames-1)
    
    # Hide x-label and ticks for top plots
    ax0.xaxis.set_visible(False)
    ax1.xaxis.set_visible(False)

    # Save the combined plot
    if idx_bool:
        output_path = os.path.join(output_directory, f'sample_{sample_idx}', 'masked_melspectrogram_with_activity_and_raw_shared_xaxis.png')
    else:
        output_path = os.path.join(output_directory, sample_idx, 'masked_melspectrogram_with_activity_and_raw_shared_xaxis.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    
    
# Method for creating masks for zero regions in audio signals
def mask_time_all_frequencies_mask(mel_spec, audio, x_ratio, y_ratio, hop_length, noise_type='zeros'):
    """
    Creates a mask for a fraction of X_ratio of total time steps and skips Y_ratio of total time steps for both 
    mel spectrogram and audio (time domain) signals.
    """
    # Mel spectrogram masking
    length = mel_spec.shape[1]
    if x_ratio > 1 or y_ratio > 1:
        x = x_ratio
        y = y_ratio
    else:
        x = int(x_ratio * length)
        y = int(y_ratio * length)
    offset_frames = 1 * 16000 / hop_length #number of frames for one sec
    mel_mask = torch.ones_like(mel_spec)
    for i in range(0, length, x + y):
        if i + x + y + offset_frames > length:
            break
        mel_mask[:, i + y:i + x + y] = 0  # Mask X samples

    # Audio (time domain) masking
    audio_length = len(audio)
    audio_mask = torch.ones_like(audio)
    offset_samples = 1 * 16000
    for i in range(0, audio_length, hop_length * (x + y)):
        if i + hop_length * (x + y) + offset_samples > audio_length:
            break
        audio_mask[i + hop_length * y:i + hop_length * (x + y)] = 0  # Mask X samples

    # Apply noise (zeros for audio, optional for mel spectrogram)
    if noise_type == 'randn':
        mel_noise = torch.randn_like(mel_spec)
    else:
        mel_noise = noise_type * torch.ones_like(mel_spec)

    audio_noise = torch.zeros_like(audio)

    masked_mel_spec = mel_spec * mel_mask + mel_noise * (1 - mel_mask)
    masked_audio = audio * audio_mask + audio_noise * (1 - audio_mask)

    return masked_mel_spec, masked_audio, mel_mask, audio_mask


def mask_time_specific_frequencies_mask(mel_spec, audio, x_ratio, y_ratio, hop_length, freq_range, noise_type='zeros'):
    """
    Creates a mask for a fraction of X_ratio of total time steps and skips Y_ratio of total time steps for specific frequencies 
    and applies the same masking to both mel spectrogram and audio (time domain) signals.
    """
    # Mel spectrogram masking
    length = mel_spec.shape[1]
    x = int(x_ratio * length)
    y = int(y_ratio * length)

    mel_mask = torch.ones_like(mel_spec)
    for i in range(0, length, x + y):
        for start_freq, end_freq in freq_range:
            mel_mask[start_freq:end_freq, i:i + x] = 0  # Mask specific frequencies

    # Audio (time domain) masking
    audio_length = len(audio)
    audio_mask = torch.ones_like(audio)
    for i in range(0, audio_length, hop_length * (x + y)):
        audio_mask[i + hop_length * y:i + hop_length * (x + y)] = 0  # Mask X samples

    # Apply noise (zeros for audio, optional for mel spectrogram)
    if noise_type == 'randn':
        mel_noise = torch.randn_like(mel_spec)
    else:
        mel_noise = noise_type * torch.ones_like(mel_spec)
    audio_noise = torch.zeros_like(audio)

    masked_mel_spec = mel_spec * mel_mask + mel_noise * (1 - mel_mask)
    masked_audio = audio * audio_mask + audio_noise * (1 - audio_mask)

    return masked_mel_spec, masked_audio, mel_mask, audio_mask


def mask_specific_frequencies_all_time_mask(mel_spec, audio, freq_range, noise_type='zeros'):
    """
    Creates a mask for specific frequency ranges for all time steps in both mel spectrogram and audio (time domain) signals.
    """
    mel_mask = torch.ones_like(mel_spec)
    for start_freq, end_freq in freq_range:
        mel_mask[start_freq:end_freq, :] = 0  # Mask the frequency bands for all time steps
    
    audio_mask = torch.ones_like(audio)
    audio_mask[:] = 0  # Mask all of the audio (adjust as needed for specific logic)

    # Apply noise (zeros for audio, optional for mel spectrogram)
    if noise_type == 'randn':
        mel_noise = torch.randn_like(mel_spec)
    else:
        mel_noise = noise_type * torch.ones_like(mel_spec)
    audio_noise = torch.zeros_like(audio)

    masked_mel_spec = mel_spec * mel_mask + mel_noise * (1 - mel_mask)
    masked_audio = audio * audio_mask + audio_noise * (1 - audio_mask)

    return masked_mel_spec, masked_audio, mel_mask, audio_mask


def mask_combined_mask(mel_spec, audio, x_ratio, y_ratio, hop_length, freq_range, noise_type='zeros'):
    """
    Creates a combined mask for specific frequencies and periodic time masking for both mel spectrogram and audio (time domain).
    """
    masked_mel_spec_time, masked_audio_time, mel_mask_time, audio_mask_time = mask_time_all_frequencies_mask(
        mel_spec, audio, x_ratio, y_ratio, hop_length, noise_type)
    
    masked_mel_spec_freq, masked_audio_freq, mel_mask_freq, audio_mask_freq = mask_specific_frequencies_all_time_mask(
        mel_spec, audio, freq_range, noise_type)

    combined_mel_mask = mel_mask_time * mel_mask_freq
    combined_audio_mask = audio_mask_time * audio_mask_freq

    # Apply combined masking
    if noise_type == 'randn':
        mel_noise = torch.randn_like(mel_spec)
    else:
        mel_noise = noise_type * torch.ones_like(mel_spec)
    masked_mel_spec_combined = masked_mel_spec_time * combined_mel_mask + mel_noise * (1 - combined_mel_mask)
    masked_audio_combined = masked_audio_time * combined_audio_mask + (torch.zeros_like(audio)) * (1 - combined_audio_mask)

    return masked_mel_spec_combined, masked_audio_combined, combined_mel_mask, combined_audio_mask


    
def mask_with_shape_mask(mel_spec, number):
    """
    Creates a binary mask for the mel spectrogram based on a drawn number using matplotlib.
    The number will be flipped over the Y-axis.
    """
    freq_dim, time_dim = mel_spec.shape
    
    # Dynamically set the font size based on the spectrogram dimensions
    font_size = max(freq_dim, time_dim) * 5  # Adjust the scaling factor for larger numbers
    
    # Create a matplotlib figure
    fig, ax = plt.subplots(figsize=(time_dim / 10, freq_dim / 10))  # Scale the figure size
    ax.text(0.5, 0.5, str(number), color="black", fontsize=font_size, ha="center", va="center", alpha=1.0)
    
    # Set axis properties
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")  # Remove axes
    
    # Save the figure to a buffer and load it as an image
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)  # Close the figure to free memory
    
    # Convert the image to grayscale
    img = Image.fromarray(img).convert("L")
    
    # Flip the image horizontally (over the Y-axis)
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Resize the image to the spectrogram's shape
    img = img.resize((time_dim, freq_dim), Image.Resampling.LANCZOS)
    shape_mask = (np.array(img) < 128).astype(np.float32)  # Binary mask: 0 for mask, 1 otherwise
    mask = torch.tensor(shape_mask)
    mask = mask.flip((0, 1))
    mask = 1 - mask
    # Add random noise to the masked regions
    noise = torch.randn_like(mel_spec)
    return mel_spec * mask + noise * (1 - mask), mask