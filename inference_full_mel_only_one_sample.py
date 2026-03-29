# this file is an adapated version https://github.com/albertfgu/diffwave-sashimi, licensed
# under https://github.com/albertfgu/diffwave-sashimi/blob/master/LICENSE

# this is the full test without asr: mel condition, free-classifier and vocoder(mel2audio)
import json
import os
import subprocess
import time
import warnings
warnings.filterwarnings("ignore")

# from functools import partial
# import multiprocessing as mp

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

from utils import find_max_epoch, print_size, get_diffusion_hyperparams, local_directory, preprocess_text
import csv
from mouthroi_processing.pipelines.pipeline import InferencePipeline
from scipy.io.wavfile import write
import tempfile
import ASR.asr_models as asr_models
from utils import zero_regions_mask, samples2frames, insert_random_values, plot_masked_melspec_with_activity, plot_signal_with_activity, \
    save_melspectrogram_with_colorbar, plot_masked_melspec_and_spec_with_activity
import torchaudio




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
            output_directory=None
            ):
    """
    Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

    Parameters: 
    net (torch network):            the model
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by get_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors

    Returns:
    the generated melspec(s) in torch.tensor, shape=size
    """

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T

    preds_ao = "None"
    
    # tokenize text
    if asr_guidance_net is not None:
        text_tokens = torch.LongTensor(tokenizer.encode(guidance_text))
        text_tokens = text_tokens.unsqueeze(0).cuda()

    masked_melspec, _ = condition
    # This is Algorithm 1 in the paper of classifier-free
    B, C, L = masked_melspec.shape  # B is batchsize, C=80, L is number of melspec frames
    mask = mask.cuda()  
    
    x = torch.normal(0, 1, size=masked_melspec.shape).cuda()
    with torch.no_grad():
        for t in tqdm(range(T-1, -1, -1)):
            diffusion_steps = (t * torch.ones((x.shape[0], 1))).cuda()  # use the corresponding reverse step
            
            if on_noisy_masked_melspec:
                z = torch.normal(0, 1, size=masked_melspec.shape).cuda()
                noisy_masked_melspec = torch.sqrt(Alpha_bar[diffusion_steps.int()]) * masked_melspec + torch.sqrt(1-Alpha_bar[diffusion_steps.int()]) * z
                x = noisy_masked_melspec * mask + x * (1 - mask)
            
            if without_condtion:
                epsilon_theta = net(x, condition, diffusion_steps, cond_drop_prob=1)
            else:
                epsilon_theta = net(x, condition, diffusion_steps, cond_drop_prob=0)   # predict \epsilon according to \epsilon_\theta
                epsilon_theta_uncond = net(x, condition, diffusion_steps, cond_drop_prob=1)
                epsilon_theta = (1+w_mel_cond) * epsilon_theta - w_mel_cond * epsilon_theta_uncond
            
            if asr_guidance_net is not None and t <= asr_start:
                with torch.enable_grad():
                    length_input = torch.tensor([x.shape[2]]).cuda()
                    inputs = x.detach().requires_grad_(True), length_input
                    targets = text_tokens, torch.tensor([text_tokens.shape[1]]).cuda()
                    asr_guidance_net.device = torch.device("cuda")
                    batch_losses = asr_guidance_net.forward_model(inputs, diffusion_steps, targets, compute_metrics=True, verbose=0)[0]
                    asr_grad = torch.autograd.grad(batch_losses["loss"], inputs[0])[0]
                    asr_guidance_net.device = torch.device("cpu")
                grad_normaliser = torch.norm(epsilon_theta / torch.sqrt(1 - Alpha_bar[t])) / torch.norm(asr_grad)
                epsilon_theta = epsilon_theta + torch.sqrt(1 - Alpha_bar[t]) * w_asr * grad_normaliser * asr_grad #the asr_grad include the minus

            x = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])  # update x_{t-1} to \mu_\theta(x_t)
            if t > 0:
                x = x + Sigma[t] * torch.normal(0, 1, size=x.shape).cuda()  # add the variance term to x_{t-1}
            # x = x.clip(-1, 1.5)
            if t % 50 == 0 and output_directory is not None:
                matplotlib.image.imsave(os.path.join(output_directory, f'generated_melspec_{t}.png'), x[0].cpu().numpy()[::-1])
            if t % 10 == 0:
                if asr_guidance_net is not None and t <= asr_start:
                    inputs = x, length_input
                    outputs_ao = asr_guidance_net(inputs, diffusion_steps)["outputs"]
                    preds_ao = decoder(outputs_ao)[0]
                    print(preds_ao)

    x = masked_melspec * mask + x * (1 - mask)
    return x, preds_ao


@torch.no_grad()
def generate(
        rank,
        diffusion_cfg,
        model_cfg,
        audio_cfg,
        dataset_cfg,
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
        lipread_text_dir=None,
        on_noisy_masked_melspec=False,
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
    net_diffwave = builder.build_diffwave_model(model_cfg)
    net = AudioVisualModel(net_diffwave).cuda()
    print_size(net)
    net.eval()

    # load checkpoint
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
        print('Successfully loaded MelGen checkpoint')
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
        asr_guidance_net, tokenizer, decoder = asr_models.get_models(ds_name)
        print('ASR, tokenizer and decoder loaded')
    else:
        asr_guidance_net, tokenizer, decoder, text = None, None, None, None
    
    vocoders = {}
    # HiFi-GAN
    print('Load HiFi-GAN')
    config_file = 'hifi_gan/config.json'
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    vocoder = Vocoder(h).cuda()
    checkpoint_file = '/dsi/gannot-lab/gannot-lab1/users/mordehay/hifi_gan/g_02400000'
    state_dict_g = vocoder_utils.load_checkpoint(checkpoint_file, 'cuda')
    vocoder.load_state_dict(state_dict_g['generator'])
    vocoder.eval()
    vocoder.remove_weight_norm()
    print('Finish Loading HiFi-GAN')
    vocoders['hifi_gan'] = vocoder
    # BigVGAN
    checkpoint_file = '/dsi/gannot-lab/gannot-lab1/users/mordehay/bigvgan/g_00550000'
    config_file = '/dsi/gannot-lab/gannot-lab1/users/mordehay/bigvgan/config.json'
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
    
    dataset_type = dataset_cfg['dataset_type']
    dataset = get_dataset(dataset_cfg, split='test', return_mask_properties=True)
    criterion = nn.L1Loss(reduction='none')
    guidance_dir_name = f'w1={w_mel_cond}'
    guidance_dir_name += f'_w2={w_asr}_asr_start={asr_start}'
    guidance_dir_name += f'_mask={on_noisy_masked_melspec}_zero-duration=0p5'
    _output_directory = os.path.join(output_directory, guidance_dir_name)
    os.makedirs(_output_directory, exist_ok=True)
    print("saving to output directory", _output_directory)



    stft = STFT(**audio_cfg)

    # ASR based on audio-only model, this is used for getting transcription for guidance, so the input is the masked audio in time domain
    pipeline_asr = InferencePipeline(config_filename_asr_cond, device='cuda')
    path_audio = ["/dsi/gannot-lab/gannot-lab1/datasets/lossy_audio/lossy/sample_1.wav",
                  "/dsi/gannot-lab/gannot-lab1/datasets/lossy_audio/lossy/sample_1.wav"]
    n_samples = len(path_audio)
    dataset_indices = torch.arange(n_samples)
    for i in dataset_indices:
        if i==0:
            continue
        os.makedirs(os.path.join(_output_directory, f'sample_{i}'), exist_ok=True)
        audio, sr = load_wav_to_torch(path_audio[i])
        # Define resampling parameters
        target_sr = 16000
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            audio = resampler(audio)
            sr = target_sr  # Update the sample rate
        masked_audio_time = audio / torch.abs(audio).max()
        masked_audio_time = torch.tensor(masked_audio_time, dtype=torch.float32)
        melspectrogram = stft.get_mel(masked_audio_time)
        # save_melspectrogram_with_colorbar(melspectrogram.numpy(), _output_directory, i)
        
        for vocoder_name, vocoder in vocoders.items():
            audio = vocoder(melspectrogram.unsqueeze(0).cuda())
            audio = audio.squeeze()
            audio = audio / 1.1 / audio.abs().max()
            audio = audio.cpu().numpy()
            sf.write(os.path.join(_output_directory, f'sample_{i}', f'vocoder_on_melspectrogram_{vocoder_name}.wav'), audio, 16000)
            
        melspectrogram = normalise_mel(melspectrogram)
        
        mask_time = zero_regions_mask(masked_audio_time.numpy(), min_length=300)
        mask_time_int = mask_time.astype(int)
        frame_mask = samples2frames(mask_time_int, audio_cfg.filter_length, audio_cfg.hop_length)
        
        masked_melspec = insert_random_values(melspectrogram.numpy(), frame_mask)
        frame_mask = torch.tensor(torch.from_numpy(frame_mask), dtype=torch.float32)
        plot_masked_melspec_and_spec_with_activity(masked_melspec, frame_mask, melspectrogram.numpy(), _output_directory, i)
        # plot_masked_melspec_with_activity(masked_melspec, frame_mask.numpy(), _output_directory, i)
        plot_signal_with_activity(masked_audio_time.numpy(), mask_time_int, target_sr, _output_directory, i)
        masked_melspec = torch.from_numpy(masked_melspec)
        masked_melspec = torch.tensor(masked_melspec, dtype=torch.float32)
        masked_melspec = masked_melspec.unsqueeze(0).cuda()
        masked_audio_time = masked_audio_time.unsqueeze(0).cuda()
        masked_cond = [masked_melspec, masked_audio_time]
        masked_audio_time4text = masked_audio_time.squeeze().cpu()
        text = 'None'
        if apply_asr_guidance:
            # Create a temporary file
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
    


        masked_audio_time4saveing = masked_audio_time.squeeze().cpu().numpy()
        sf.write(os.path.join(_output_directory, f'sample_{i}', 'masked_audio_time.wav'), masked_audio_time4saveing, 16000)



        
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
                        output_directory=os.path.join(_output_directory, f'sample_{i}')
                        )
        melspec = denormalise_mel(melspec)
        end.record()
        torch.cuda.synchronize()
        print('generated sample_{} in {} seconds'.format(i, int(start.elapsed_time(end)/1000)))

        # save text
        text_filename = os.path.join(_output_directory, f'sample_{i}', 'asr_text.txt')
        with open(text_filename, 'w') as f:
            f.write("asr_condition       :  " +text+"\n")
            f.write("asr_generated_signal:  " + preds_ao)
        
        # generate audio from masked melspec
        masked_melspec = denormalise_mel(masked_melspec)
        for vocoder_name, vocoder in vocoders.items():
            masked_audio = vocoder(masked_melspec.cuda())
            masked_audio = masked_audio.squeeze()
            masked_audio = masked_audio / 1.1 / masked_audio.abs().max()
            masked_audio = masked_audio.cpu().numpy()
            sf.write(os.path.join(_output_directory, f'sample_{i}', f'spec_masking_audio_{vocoder_name}.wav'), masked_audio, 16000)

        # generate audio from generated melspec
        for vocoder_name, vocoder in vocoders.items():
            audio = vocoder(melspec)
            audio = audio.squeeze()
            audio = audio / 1.1 / audio.abs().max()
            audio = audio.cpu().numpy()
            sf.write(os.path.join(_output_directory, f'sample_{i}', f'generated_audio_{vocoder_name}.wav'), audio, 16000)
            

        

        

        # save as file
        melspec = melspec.squeeze(0).cpu()
        torch.save(melspec, os.path.join(_output_directory, f'sample_{i}', 'generated_spec.npz'))
        
        # save as image
        melspec = melspec.numpy()
        masked_melspec = masked_melspec.squeeze(0).cpu().numpy()

        matplotlib.image.imsave(os.path.join(_output_directory, f'sample_{i}', 'generated_spec_image.png'), melspec[::-1])
        matplotlib.image.imsave(os.path.join(_output_directory, f'sample_{i}', 'masked_spec_image.png'), masked_melspec[::-1])
    
        
    return

#  config_plc
# config
@hydra.main(version_base=None, config_path="configs/", config_name="config_plc")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    generate(0,
        diffusion_cfg=cfg.diffusion,
        model_cfg=cfg.melgen,
        audio_cfg=cfg.audio,
        dataset_cfg=cfg.dataset,
        **cfg.generate,
    )



if __name__ == "__main__":
    main()
