# this file is an adapated version https://github.com/albertfgu/diffwave-sashimi, licensed
# under https://github.com/albertfgu/diffwave-sashimi/blob/master/LICENSE

##############################################################################################################################
########### Check in this file the on_noisy_masked_melspec if it is good and the same meaning  as other files!!! ##################
##############################################################################################################################
# this is the full test without asr: mel condition, free-classifier and vocoder(mel2audio)
import json
import os
import subprocess
import time
import warnings
warnings.filterwarnings("ignore")
from itertools import product
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
# from functools import partial
# import multiprocessing as mp
import re
import soundfile as sf
import matplotlib.image
import numpy as np
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from models.model_builder import ModelBuilder
from models.audiovisual_model import AudioVisualModel
# import ASR.asr_models as asr_models
from dataloaders.dataset_lipvoicer import get_dataset
from dataloaders.stft import denormalise_mel, normalise_mel
from hifi_gan.generator import Generator as Vocoder
from BigVGAN.bigvgan import BigVGAN as Generator
from BigVGAN.inference_e2e import load_checkpoint as load_checkpoint_vgan
from dataloaders.wav2mel import STFT, load_wav_to_torch
from hifi_gan import utils as vocoder_utils
from hifi_gan.env import AttrDict

from utils import find_max_epoch, print_size, get_diffusion_hyperparams, linear_t_given_cosine_t, local_directory, preprocess_text
import csv
from mouthroi_processing.pipelines.pipeline import InferencePipeline
from scipy.io.wavfile import write
import tempfile
import ASR.asr_models as asr_models
from utils import find_zero_regions, samples2frames, insert_values, plot_masked_melspec_with_activity, plot_signal_with_activity, \
    save_melspectrogram_with_colorbar, plot_masked_melspec_and_spec_with_activity
import torchaudio


def get_g2p_pipeline(g2p_model, with_space=False):
    p2n = '/home/dsi/moradim/SpeechRepainting/phoneme_to_number.json'
    with open(p2n, 'r') as f:
        valid_chars = json.load(f)
        valid_chars = list(valid_chars.keys())
    def g2p(text):
        phonemes = g2p_model(text)
        processed_list = []
        for item in phonemes:
            if item == ' ':
                if with_space:
                    processed_list.append('space')
                # else: don't append, effectively removing it
            elif item in valid_chars:
                processed_list.append(item)
            # else: don't append, removing invalid char.
        return processed_list
    return g2p
    

def sampling(net, diffusion_hyperparams,
            w_mel_cond, condition=None, 
            asr_guidance_net=None,
            w_asr=None,
            asr_start=None,
            guidance_text=None,
            tokenizer=None,
            decoder=None,
            without_condtion=False,
            mask=None,
            on_noisy_masked_melspec=False,
            mask_frames=None,
            masked_audio_time_mask=None,
            text=None, 
            input_text=None,
            phoneme4guidance=None,
            type_input_guidance='text',
            skip_step=1,
            output_directory=None,
            ):
    """
    Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

    Parameters:
    net (torch network):            the model
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors

    Returns:
    the generated melspec(s) in torch.tensor, shape=size
    """

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    preds_ao = 'None'

    # torch.set_anomaly_enabled(True)
    # tokenize text
    if asr_guidance_net is not None:
        if type_input_guidance == 'text':
            tokens = torch.LongTensor(tokenizer.encode(guidance_text))
            tokens = tokens.unsqueeze(0).cuda()
        elif type_input_guidance == 'phoneme':
            tokens = torch.tensor([tokenizer[t] for t in phoneme4guidance[0]]).unsqueeze(0).cuda()
        elif type_input_guidance == 'frame_level_phoneme':
            pass

    masked_melspec, _ = condition
    matplotlib.image.imsave(os.path.join(output_directory, f'masked_melspec.png'), masked_melspec[0].cpu().numpy()[::-1])
    # This is Algorithm 1 in the paper of classifier-free
    B, C, L = masked_melspec.shape  # B is batchsize, C=80, L is number of melspec frames
    mask = mask.cuda()  
    generator = torch.Generator().manual_seed(42)
    x = torch.normal(0, 1, size=masked_melspec.shape, generator=generator).cuda()
    repeat_asr = 1
    asr_finish = -1
    add_noise = True
    same_scale_noise = True
    tho = 1/3
    with torch.no_grad():
        for t in tqdm(range(T - 1, -1, -skip_step)):
            if t < skip_step:  # Ensure the last step is exactly zero
                t = 0
            if _dh["name"] == "cosine":
                t_linear_asr = linear_t_given_cosine_t(t)
            else:
                t_linear_asr = t
                
            diffusion_steps_asr_linear = (t_linear_asr * torch.ones((x.shape[0], 1))).cuda()  # use the corresponding reverse step
            diffusion_steps = (t * torch.ones((x.shape[0], 1))).cuda()  # use the corresponding reverse step
            # if asr_guidance_net is not None and (t <= asr_start and t > asr_finish):
                # repeat_asr = 1
            # for _ in range(repeat_asr):
            if on_noisy_masked_melspec is not None and same_scale_noise:
                z = torch.normal(0, 1, size=masked_melspec.shape, generator=generator).cuda()
                # noisy_masked_melspec = torch.sqrt(Alpha_bar[diffusion_steps.int()]) * masked_melspec + (torch.sqrt(1-Alpha_bar[diffusion_steps.int()]) - sub) * z
                if t < (T - 1):
                    noisy_masked_melspec = torch.sqrt(Alpha_bar[t + 1]) * masked_melspec + torch.sqrt(1-Alpha_bar[t + 1]) * z
                    noisy_masked_melspec = (noisy_masked_melspec - (1-Alpha[t + 1])/torch.sqrt(1-Alpha_bar[t + 1]) * z) / torch.sqrt(Alpha[t + 1]) 
                    z2 = torch.normal(0, 1, size=masked_melspec.shape, generator=generator).cuda()
                    noisy_masked_melspec = noisy_masked_melspec + tho * Sigma[t + 1] * z2
                    # noisy_masked_melspec = torch.sqrt(Alpha_bar[diffusion_steps.int()]) * masked_melspec + (torch.sqrt(1-Alpha_bar[diffusion_steps.int()]) - sub) * z
                    x = noisy_masked_melspec * mask + x * (1 - mask)
            else:
                z = torch.normal(0, 1, size=masked_melspec.shape, generator=generator).cuda()
                noisy_masked_melspec = torch.sqrt(Alpha_bar[t]) * masked_melspec + torch.sqrt(1-Alpha_bar[t]) * z
                x = noisy_masked_melspec * mask + x * (1 - mask)
            if without_condtion:
                epsilon_theta_mel = net(x, condition, diffusion_steps, cond_drop_prob=1, mask_padding_frames=mask_frames, text=text, input_text=input_text,  mask_padding_time=masked_audio_time_mask)
                if net.g_model_cfg.predict_type =='speech':
                    epsilon_theta_mel = (x - torch.sqrt(Alpha_bar[diffusion_steps.int()]) * epsilon_theta_mel) / torch.sqrt(1-Alpha_bar[diffusion_steps.int()])
            else:
                epsilon_theta_cond = net(x, condition, diffusion_steps, cond_drop_prob=0, mask_padding_frames=mask_frames, text=text, input_text=input_text,  mask_padding_time=masked_audio_time_mask)   # predict \epsilon according to \epsilon_\theta
                if net.g_model_cfg.predict_type =='speech':
                    epsilon_theta_cond = (x - torch.sqrt(Alpha_bar[diffusion_steps.int()]) * epsilon_theta_cond) / torch.sqrt(1-Alpha_bar[diffusion_steps.int()])
                epsilon_theta_uncond = net(x, condition, diffusion_steps, cond_drop_prob=1, mask_padding_frames=mask_frames, text=text, input_text=input_text,  mask_padding_time=masked_audio_time_mask)
                if net.g_model_cfg.predict_type =='speech':
                    epsilon_theta_uncond = (x - torch.sqrt(Alpha_bar[diffusion_steps.int()]) * epsilon_theta_uncond) / torch.sqrt(1-Alpha_bar[diffusion_steps.int()])
                epsilon_theta_mel = (1+w_mel_cond) * epsilon_theta_cond - w_mel_cond * epsilon_theta_uncond
            
            if (asr_guidance_net is not None) and (t <= asr_start and t > asr_finish):
                for r in range(repeat_asr):
                    if r > 1:
                        if on_noisy_masked_melspec is not None:
                            z = torch.normal(0, 1, size=masked_melspec.shape, generator=generator).cuda()
                            noisy_masked_melspec = torch.sqrt(Alpha_bar[diffusion_steps_asr_linear.int()]) * masked_melspec + torch.sqrt(1-Alpha_bar[diffusion_steps_asr_linear.int()]) * z
                            x = noisy_masked_melspec * mask + x * (1 - mask)
                    with torch.enable_grad():
                        # torch.set_anomaly_enabled(True)
                        if type_input_guidance == 'text' or type_input_guidance == 'phoneme':
                            length_input = torch.tensor([x.shape[2]]).cuda()
                            inputs = x.detach().requires_grad_(True), length_input
                            targets = tokens, torch.tensor([tokens.shape[1]]).cuda()
                            asr_guidance_net.device = torch.device("cuda")
                            batch_losses = asr_guidance_net.forward_model(inputs, diffusion_steps_asr_linear, targets, compute_metrics=False, verbose=0)[0] #batch_losses, batch_metrics, batch_truths, batch_preds
                            asr_grad = torch.autograd.grad(batch_losses["loss"], inputs[0])[0]
                        elif type_input_guidance == 'frame_level_phoneme':
                            inputs = x.detach().requires_grad_(True), length_input
                            outputs_ao = asr_guidance_net(inputs, diffusion_steps_asr_linear)["outputs"]
                            asr_grad = torch.autograd.grad(preds_ao, inputs[0])[0]
                            
                        if torch.sum(asr_grad) == 0 and type_input_guidance == 'phoneme':
                            print("Zero grad")
                            grad_normaliser = 0
                        else:
                            grad_normaliser = torch.norm(epsilon_theta_mel / torch.sqrt(1 - Alpha_bar[t])) / torch.norm(asr_grad)
                        # if torch.isnan(asr_grad).any():
                        #     print("NAN in asr_grad")
                        asr_guidance_net.device = torch.device("cpu")
                    num_tokens = tokens.shape[1]
                    epsilon_theta = epsilon_theta_mel + torch.sqrt(1 - Alpha_bar[t]) * w_asr * grad_normaliser * asr_grad #the asr_grad include the minus
                    # print(f"grad_normaliser: {grad_normaliser} \t w_asr: {w_asr} \t grad: {torch.norm(asr_grad)} \t 1-alpha_bar: {torch.sqrt(1 - Alpha_bar[t])}")
                    x = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])  # update x_{t-1} to \mu_\theta(x_t)
                    if t > 0 and add_noise:
                        x = x + tho * Sigma[t] * torch.normal(0, 1, size=x.shape, generator=generator).cuda()  # add the variance term to x_{t-1}
            else:
                x = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * epsilon_theta_mel) / torch.sqrt(Alpha[t])  # update x_{t-1} to \mu_\theta(x_t)
                if t > 0 and add_noise:
                    x = x + tho * Sigma[t] * torch.normal(0, 1, size=x.shape, generator=generator).cuda()  # add the variance term to x_{t-1}
            # x = x.clip(-1, 1.5)
            if t % 50 == 0 and output_directory is not None:
                matplotlib.image.imsave(os.path.join(output_directory, f'generated_melspec_{t}.png'), x[0].cpu().numpy()[::-1])
            if t % 10 == 0:
                if (asr_guidance_net is not None) and (t <= asr_start) and (type_input_guidance != 'frame_level_phoneme'):
                    inputs = x, length_input
                    outputs_ao = asr_guidance_net(inputs, diffusion_steps_asr_linear)["outputs"]
                    preds_ao = decoder(outputs_ao)[0]
                    print(preds_ao)
    if on_noisy_masked_melspec is not None:
        x = masked_melspec * mask + x * (1 - mask)
    if mask_frames is not None:
        x = x[..., :int(torch.sum(mask_frames, dim=-1).item())]
    return x, preds_ao



@torch.no_grad()
def generate(
        rank,
        diffusion_cfg,
        model_cfg,
        audio_cfg,
        g_model_cfg,
        ckpt_path,
        w_mel_cond=0,
        w_asr=1.1,
        asr_start=250,
        save_dir=None,
        n_samples_test = 20,
        inference_mel_only_name_dir='generated_mels',
        without_condtion=False,
        config_filename_asr_cond=None,
        apply_asr_guidance=False,
        type_input_guidance = 'text',
        lipread_text_dir=None,
        on_noisy_masked_melspec=False,
        mask_info=None,
        mel_text=None,
        with_space=False,
        skip_step=1,
        **kwargs
    ):

    if rank is not None:
        print(f"rank {rank} {torch.cuda.device_count()} GPUs")
        torch.cuda.set_device(rank % torch.cuda.device_count())

    # map diffusion hyperparameters to gpu
    # diffusion_hyperparams  = calc_diffusion_hyperparams(**diffusion_cfg, fast=True)  # dictionary of all diffusion hyperparameters
    diffusion_hyperparams = get_diffusion_hyperparams(diffusion_cfg, fast=True)
    # predefine MelGen model
    builder = ModelBuilder()
    net_diffwave = builder.build_model(model_cfg)
    net = AudioVisualModel(g_model_cfg, net_diffwave).cuda()
    print_size(net)
    net.eval()

    # load checkpoint
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model_weights = checkpoint['model_state_dict']
        model_weights = {k: v for k, v in model_weights.items() if 'wavlm_model' not in k}
        missing_keys , _ = net.load_state_dict(model_weights, strict=False)
        filtered_missing_keys = [key for key in missing_keys if 'wavlm_model' not in key]
        if not filtered_missing_keys:
            print('All keys loaded successfully')
            print('Successfully loaded MelGen checkpoint')
        else:
            raise Exception(f'The following keys were not loaded: {filtered_missing_keys}')
    except Exception as e:
        print(e)
        raise Exception('No valid model found')
        

    if save_dir is None:
        save_dir = os.getcwd()
    output_directory = os.path.join(save_dir, inference_mel_only_name_dir)
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        print("saving to output directory", output_directory)

    # print('Loading ASR, tokenizer and decoder')
    ds_name = 'LRS3' # 'LRS2'
    if apply_asr_guidance:
        if type_input_guidance == "text":
                print(f'Apply {type_input_guidance} guidance')
        elif type_input_guidance == "phoneme":
            from g2p_en import G2p
            g2p_model = G2p()
            g2p = get_g2p_pipeline(g2p_model, with_space=with_space)
            print(f'Apply {type_input_guidance} guidance with space={with_space}')    
        asr_guidance_net, tokenizer, decoder = asr_models.get_models(ds_name, type_input_guidance=type_input_guidance, with_space=with_space)
        print('ASR, tokenizer and decoder loaded')
        
    else:
        asr_guidance_net, tokenizer, decoder, text, g2p = None, None, None, None, None
    
    vocoders = {}
    # HiFi-GAN
    print('Load HiFi-GAN')
    config_file = 'hifi_gan/config.json'
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    vocoder = Vocoder(h).cuda()
    checkpoint_file = '/dsi/gannot-lab1/users/mordehay/hifi_gan/g_02400000'
    state_dict_g = vocoder_utils.load_checkpoint(checkpoint_file, 'cuda')
    vocoder.load_state_dict(state_dict_g['generator'])
    vocoder.eval()
    vocoder.remove_weight_norm()
    print('Finish Loading HiFi-GAN')
    vocoders['hifi_gan'] = vocoder
    # BigVGAN
    checkpoint_file = '/dsi/gannot-lab1/users/mordehay/bigvgan/g_00550000'
    config_file = '/dsi/gannot-lab1/users/mordehay/bigvgan/config.json'
    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    device_bigvgan = torch.device("cuda")
    generator = Generator(h, use_cuda_kernel=False).to(device_bigvgan)
    state_dict_g = load_checkpoint_vgan(checkpoint_file, device_bigvgan)
    generator.load_state_dict(state_dict_g["generator"])
    generator.eval()
    generator.remove_weight_norm()
    vocoders['bigvgan'] = generator
    






    stft = STFT(**audio_cfg)

    # ASR based on audio-only model, this is used for getting transcription for guidance, so the input is the masked audio in time domain
    pipeline_asr = InferencePipeline(config_filename_asr_cond, device='cuda')
    main_folder_path_audio = "/dsi/gannot-lab1/users/mordehay/exmaples_my_dataset/data/100_frames"
    # main_folder_path_audio = "/dsi/gannot-lab1/users/mordehay/google_baseline_exmaples/data"
    pathes_folder = [os.path.join(main_folder_path_audio, folder) for folder in os.listdir(main_folder_path_audio) if os.path.isdir(os.path.join(main_folder_path_audio, folder))]
    n_samples = len(pathes_folder)
    #my_data:
    full_audio = False # TODO afjust the code for False
    known_regions_gap = True
    known_text = True
    my_dataset = True
    num_sil_sec = 0.8
    #google_data:
    # full_audio = True # TODO afjust the code for False
    # known_regions_gap = True
    # known_text = True
    # my_dataset = False
    # num_sil_sec = 0.8
    dataset_indices = torch.arange(n_samples)
    w_asr_list = w_asr
    asr_start_list = asr_start
    w_mel_cond_list = w_mel_cond if OmegaConf.is_list(w_mel_cond) else [w_mel_cond]
    for w_asr, asr_start, w_mel_cond in product(w_asr_list, asr_start_list, w_mel_cond_list):
        guidance_dir_name = f'w1={w_mel_cond}'
        guidance_dir_name += f'_w2={w_asr}_asr_start={asr_start}'
        guidance_dir_name += f'_mask={on_noisy_masked_melspec}'
        _output_directory = os.path.join(output_directory, guidance_dir_name)
        os.makedirs(_output_directory, exist_ok=True)
        print("saving to output directory", _output_directory)
        for i, path_folder in enumerate(pathes_folder):
            # if (os.path.exists(os.path.join(_output_directory, f'sample_{indx_data}', 'generated_audio_hifi_gan.wav'))) and (os.path.exists(os.path.join(_output_directory, f'sample_{indx_data}', 'generated_audio_bigvgan.wav'))):
            #     print(f"{os.path.join(_output_directory, f'sample_{indx_data}')} already exists")
            #     progress.update(1)  # Manually updating tqdm
            #     i += 1 
            #     continue
            name_folder = os.path.basename(path_folder)
            os.makedirs(os.path.join(_output_directory, name_folder), exist_ok=True)
            if my_dataset:
                path_audio =  os.path.join(path_folder, 'masked_audio_time.wav') #os.path.join(path_folder, 'gt_audio_hifi_gan.wav')
                masked_audio_time, sample_rate = load_wav_to_torch(path_audio)
                path_audio =  os.path.join(path_folder, 'gt_audio_hifi_gan.wav') #os.path.join(path_folder, 'gt_audio_hifi_gan.wav')
                audio, sr = load_wav_to_torch(path_audio)
                masked_audio_time = masked_audio_time[:audio.shape[0]]
                masked_audio_time = masked_audio_time / torch.abs(masked_audio_time).max()
                audio = audio[:masked_audio_time.shape[0]]
                path_text = os.path.join(path_folder, 'asr_text.txt')
                with open(path_text, 'r') as f:
                    lines = f.readlines()
                    text_guidance = lines[0].strip()
                    text_guidance = re.sub(r'^asr_generated_signal:\s+', '', text_guidance)
                
                start_time, duration_time = 1.5, 1
            else:
                path_audio = os.path.join(path_folder, 'audio.wav')
                audio, sr = load_wav_to_torch(path_audio)
                path_text = os.path.join(path_folder, 'text.txt')
                with open(path_text, 'r') as f:
                    lines = f.readlines()
                    text_guidance = lines[0].strip()
                    start_time, duration_time = lines[1].strip().split(',')
                    start_time, duration_time = float(start_time), float(duration_time)
            target_sr = 16000
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                audio = resampler(audio)
                sr = target_sr  # Update the sample rate
            audio = audio / torch.abs(audio).max()
            if model_cfg._name_ == 'dit':
                type_mask = 'zeros'
            elif model_cfg._name_ == 'unet':
                type_mask = 'randn'
            
            audio = torch.tensor(audio, dtype=torch.float32)
            melspectrogram = stft.get_mel(audio)
            melspectrogram = normalise_mel(melspectrogram)
            if full_audio:
                if known_regions_gap:
                    mask_time = torch.ones_like(audio)
                    start_frame = int(start_time * target_sr)
                    end_frame = int(start_frame + duration_time * target_sr)
                    mask_time[start_frame:end_frame] = 0
                    masked_audio_time = audio.clone()
                    masked_audio_time = masked_audio_time * mask_time
                    frame_mask = samples2frames(mask_time, audio_cfg.filter_length, audio_cfg.hop_length)
                    masked_melspec = insert_values(melspectrogram.numpy(), frame_mask, num=type_mask)
                    mask_time_int = mask_time.numpy().astype(int)
            else:
                min_length = num_sil_sec * sample_rate
                mask_time = find_zero_regions(masked_audio_time.numpy(), min_length=min_length) #this return a mask indicatiing where the signal should be inpainted(mask=0)
                mask_time_int = mask_time.astype(int)
                masked_audio_time = masked_audio_time * mask_time
                frame_mask = samples2frames(mask_time_int, audio_cfg.filter_length, audio_cfg.hop_length)
                masked_melspec = insert_values(melspectrogram.numpy(), frame_mask, num=type_mask)

            
                
            
            # Define resampling parameters

            # save_melspectrogram_with_colorbar(melspectrogram.numpy(), _output_directory, i)
            
            for vocoder_name, vocoder in vocoders.items():
                audio = vocoder(melspectrogram.unsqueeze(0).cuda())
                audio = audio.squeeze()
                audio = audio / 1.1 / audio.abs().max()
                audio = audio.cpu().numpy()
                sf.write(os.path.join(_output_directory, name_folder, f'vocoder_on_melspectrogram_{vocoder_name}.wav'), audio, 16000)
                
            


            frame_mask = torch.tensor(torch.from_numpy(frame_mask), dtype=torch.float32)
            plot_masked_melspec_and_spec_with_activity(masked_melspec, frame_mask, melspectrogram.numpy(), _output_directory, name_folder, idx_bool=False)
            # plot_masked_melspec_with_activity(masked_melspec, frame_mask.numpy(), _output_directory, i)
            plot_signal_with_activity(masked_audio_time.numpy(), mask_time_int, target_sr, _output_directory, name_folder, idx_bool=False)
            masked_melspec = torch.from_numpy(masked_melspec)
            masked_melspec = torch.tensor(masked_melspec, dtype=torch.float32)
            masked_melspec = masked_melspec.unsqueeze(0).cuda()
            masked_audio_time = masked_audio_time.unsqueeze(0).cuda()
            masked_cond = [masked_melspec, masked_audio_time]
            masked_audio_time4text = masked_audio_time.squeeze().cpu()
            text = 'None'
            phoneme4guidance = 'None'

            
            if apply_asr_guidance:
                if known_text:
                    text = text_guidance
                # Create a temporary file
                else:
                    sample_rate = 16000  # Example value
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
                        # Save the masked audio array as a WAV file in the temporary file
                        write(temp_wav.name, sample_rate, masked_audio_time4text.numpy().astype(np.float32)) # TODO maybe we need to do something more clever here
                        # Send the temporary WAV file to the pipeline
                        transcript_from_condition = pipeline_asr(temp_wav.name)
                        text = transcript_from_condition
                        print(f"The transcript is: {text}")
                        text = preprocess_text(text)
                        print(f"The normalized transcript is: {text}")
            
            if type_input_guidance == "phoneme":
                phoneme4guidance = g2p(text)
                print(f"phoneme4guidance: {phoneme4guidance}")
                phoneme4guidance = [phoneme4guidance]
                
        


            masked_audio_time4saveing = masked_audio_time.squeeze().cpu().numpy()
            sf.write(os.path.join(_output_directory, name_folder, 'masked_audio_time.wav'), masked_audio_time4saveing, 16000)



            
            # inference
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            melspec, preds_ao = sampling(net, 
                                diffusion_hyperparams,
                                w_mel_cond,
                                condition=masked_cond,
                                asr_guidance_net=asr_guidance_net,
                                w_asr=w_asr,
                                asr_start=asr_start,
                                guidance_text=text,
                                tokenizer=tokenizer,
                                decoder=decoder,
                                without_condtion=without_condtion,
                                mask=frame_mask,
                                on_noisy_masked_melspec=on_noisy_masked_melspec,
                                mask_frames=None, #that for batching in training phase
                                masked_audio_time_mask=None,
                                text=text, 
                                input_text=None, #in current model there is no input text to the model but for the guidance
                                phoneme4guidance=phoneme4guidance,
                                type_input_guidance=type_input_guidance,
                                skip_step=skip_step,
                                output_directory=os.path.join(_output_directory, name_folder)
                                )
            
            
            melspec = denormalise_mel(melspec)
            end.record()
            torch.cuda.synchronize()
            print('generated sample_{} in {} seconds'.format(i, int(start.elapsed_time(end)/1000)))

            # save text
            text_filename = os.path.join(_output_directory, name_folder, 'asr_text.txt')
            with open(text_filename, 'w') as f:
                if type_input_guidance == 'text':
                    f.write("asr_condition       :  " +text+"\n")
                    f.write("asr_generated_signal:  " + preds_ao)
                elif type_input_guidance == 'phoneme':
                    f.write("asr_generated_signal:  " + text + "\n")
                    f.write("asr_condition       :  " +" ".join(phoneme4guidance[0])+"\n")
                    f.write("asr_generated_signal:  " + " ".join(preds_ao))
            
            # generate audio from masked melspec
            masked_melspec = denormalise_mel(masked_melspec)
            for vocoder_name, vocoder in vocoders.items():
                masked_audio = vocoder(masked_melspec.cuda())
                masked_audio = masked_audio.squeeze()
                masked_audio = masked_audio / 1.1 / masked_audio.abs().max()
                masked_audio = masked_audio.cpu().numpy()
                sf.write(os.path.join(_output_directory, name_folder, f'spec_masking_audio_{vocoder_name}.wav'), masked_audio, 16000)

            # generate audio from generated melspec
            for vocoder_name, vocoder in vocoders.items():
                audio = vocoder(melspec)
                audio = audio.squeeze()
                audio = audio / 1.1 / audio.abs().max()
                audio = audio.cpu().numpy()
                sf.write(os.path.join(_output_directory, name_folder, f'generated_audio_{vocoder_name}.wav'), audio, 16000)
                

            

            

            # save as file
            melspec = melspec.squeeze(0).cpu()
            torch.save(melspec, os.path.join(_output_directory, name_folder, 'generated_spec.npz'))
            
            # save as image
            melspec = melspec.numpy()
            masked_melspec = masked_melspec.squeeze(0).cpu().numpy()

            matplotlib.image.imsave(os.path.join(_output_directory, name_folder, 'generated_spec_image.png'), melspec[::-1])
            matplotlib.image.imsave(os.path.join(_output_directory, name_folder, 'masked_spec_image.png'), masked_melspec[::-1])
    
        
    return

# /home/dsi/moradim/SpeechRepainting/configs/4testing/config_dit_without-space-phoneme.yaml
@hydra.main(version_base=None, config_path="configs/4testing/", config_name="config_dit_without-space-phoneme")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    generate(0,
        g_model_cfg=cfg.g_model,
        diffusion_cfg=cfg.diffusion,
        model_cfg=cfg[cfg.melgen],
        audio_cfg=cfg.audio,
        **cfg.generate,
    )



if __name__ == "__main__":
    main()
