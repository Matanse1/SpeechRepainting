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
from dataloaders.stft import denormalise_mel
from hifi_gan.generator import Generator as Vocoder
from BigVGAN.bigvgan import BigVGAN as Generator
from BigVGAN.inference_e2e import load_checkpoint as load_checkpoint_vgan
from hifi_gan import utils as vocoder_utils
from hifi_gan.env import AttrDict
from utils import find_max_epoch, print_size, calc_diffusion_hyperparams, local_directory, preprocess_text, fix_len_compatibility, pad_last_dim
import csv
from mouthroi_processing.pipelines.pipeline import InferencePipeline
from scipy.io.wavfile import write
import tempfile
# from my_utils.compute_metrics import Metrics
import ASR.asr_models as asr_models
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from utils import mask_time_all_frequencies_mask, mask_time_specific_frequencies_mask, mask_specific_frequencies_all_time_mask, mask_combined_mask, mask_with_shape_mask
print("finished imports")


def training_loss(net, loss_fn, melspec, masked_melspec, mask, diffusion_hyperparams, w_masked_pix=0.8, mask_frames=None):
    """
    Compute the training loss of epsilon and epsilon_theta

    Parameters:
    net (torch network):            the wavenet model
    loss_fn (torch loss function):  the loss function, default is nn.MSELoss()
    X (torch.tensor):               training data, shape=(batchsize, 1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors

    Returns:
    training loss
    """
    # Predict melspectrogram from visual features using diffusion model
    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]
    # This is Algorithm 1 in the paper of classifier-free
    B, C, L = melspec.shape  # B is batchsize, C=80, L is number of melspec frames
    diffusion_steps = torch.randint(T, size=(B,1,1)).cuda()  # randomly sample diffusion steps from 1~T
    z = torch.normal(0, 1, size=melspec.shape).cuda()
    transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * melspec + torch.sqrt(1-Alpha_bar[diffusion_steps]) * z  # training from Denoising Diffusion Probabilistic Models paper compute x_t from q(x_t|x_0)
    cond_drop_prob = 0.2
    epsilon_theta = net(transformed_X, masked_melspec, diffusion_steps.view(B,1), cond_drop_prob, mask_padding_frames=mask_frames)
    loss = loss_fn(epsilon_theta, z) #[B, F, T]
    mean_loss = round(torch.mean(loss).item(), 2)
    unmaksed_loss = torch.sum(mask * loss) / torch.sum(mask)
    masked_loss = torch.sum((1-mask) * loss) / torch.sum(1-mask)
    weighted_loss = (1 - w_masked_pix) * unmaksed_loss + w_masked_pix * masked_loss
    est_X = (transformed_X - torch.sqrt(1-Alpha_bar[diffusion_steps]) * epsilon_theta) / torch.sqrt(Alpha_bar[diffusion_steps])
    return weighted_loss, est_X, transformed_X, diffusion_steps, mean_loss



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
            on_masked_melspec=False,
            mask_frames=None
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

    
    # tokenize text
    if asr_guidance_net is not None:
        text_tokens = torch.LongTensor(tokenizer.encode(guidance_text))
        text_tokens = text_tokens.unsqueeze(0).cuda()

    masked_melspec, _ = condition
    # This is Algorithm 1 in the paper of classifier-free
    B, C, L = masked_melspec.shape  # B is batchsize, C=80, L is number of melspec frames
    if on_masked_melspec is not None:
        mask = mask.cuda()  
    
    x = torch.normal(0, 1, size=masked_melspec.shape).cuda()
    repeat_asr = 1
    asr_finish = -1
    with torch.no_grad():
        for t in tqdm(range(T-1, -1, -1)):
            diffusion_steps = (t * torch.ones((x.shape[0], 1))).cuda()  # use the corresponding reverse step
            # if asr_guidance_net is not None and (t <= asr_start and t > asr_finish):
                # repeat_asr = 1
            # for _ in range(repeat_asr):
            if on_masked_melspec is not None:
                z = torch.normal(0, 1, size=masked_melspec.shape).cuda()
                noisy_masked_melspec = torch.sqrt(Alpha_bar[diffusion_steps.int()]) * masked_melspec + torch.sqrt(1-Alpha_bar[diffusion_steps.int()]) * z
                x = noisy_masked_melspec * mask + x * (1 - mask)
            
            if without_condtion:
                epsilon_theta_mel = net(x, condition, diffusion_steps, cond_drop_prob=1, mask_padding_frames=mask_frames)
            else:
                epsilon_theta_cond = net(x, condition, diffusion_steps, cond_drop_prob=0, mask_padding_frames=mask_frames)   # predict \epsilon according to \epsilon_\theta
                epsilon_theta_uncond = net(x, condition, diffusion_steps, cond_drop_prob=1, mask_padding_frames=mask_frames)
                epsilon_theta_mel = (1+w_mel_cond) * epsilon_theta_cond - w_mel_cond * epsilon_theta_uncond
            
            if asr_guidance_net is not None and (t <= asr_start and t > asr_finish):
                for r in range(repeat_asr):
                    if r > 1:
                        if on_masked_melspec is not None:
                            z = torch.normal(0, 1, size=masked_melspec.shape).cuda()
                            noisy_masked_melspec = torch.sqrt(Alpha_bar[diffusion_steps.int()]) * masked_melspec + torch.sqrt(1-Alpha_bar[diffusion_steps.int()]) * z
                            x = noisy_masked_melspec * mask + x * (1 - mask)
                    with torch.enable_grad():
                        length_input = torch.tensor([x.shape[2]]).cuda()
                        inputs = x.detach().requires_grad_(True), length_input
                        targets = text_tokens, torch.tensor([text_tokens.shape[1]]).cuda()
                        asr_guidance_net.device = torch.device("cuda")
                        batch_losses = asr_guidance_net.forward_model(inputs, diffusion_steps, targets, compute_metrics=True, verbose=0)[0]
                        asr_grad = torch.autograd.grad(batch_losses["loss"], inputs[0])[0]
                        asr_guidance_net.device = torch.device("cpu")
                    grad_normaliser = torch.norm(epsilon_theta_mel / torch.sqrt(1 - Alpha_bar[t])) / torch.norm(asr_grad)
                    epsilon_theta = epsilon_theta_mel + torch.sqrt(1 - Alpha_bar[t]) * w_asr * grad_normaliser * asr_grad #the asr_grad include the minus
                    x = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])  # update x_{t-1} to \mu_\theta(x_t)
                    if t > 0:
                        x = x + Sigma[t] * torch.normal(0, 1, size=x.shape).cuda()  # add the variance term to x_{t-1}
            else:
                x = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * epsilon_theta_mel) / torch.sqrt(Alpha[t])  # update x_{t-1} to \mu_\theta(x_t)
                if t > 0:
                    x = x + Sigma[t] * torch.normal(0, 1, size=x.shape).cuda()  # add the variance term to x_{t-1}
            # x = x.clip(-1, 1.5)
            
            if t % 10 == 0:
                if asr_guidance_net is not None and t <= asr_start:
                    inputs = x, length_input
                    outputs_ao = asr_guidance_net(inputs, diffusion_steps)["outputs"]
                    preds_ao = decoder(outputs_ao)[0]
                    print(preds_ao)
    if on_masked_melspec is not None:
        x = masked_melspec * mask + x * (1 - mask)
    if mask_frames is not None:
        x = x[..., :int(torch.sum(mask_frames, dim=-1).item())]
    return x, preds_ao


@torch.no_grad()
def generate(
        rank,
        diffusion_cfg,
        model_cfg,
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
        on_masked_melspec=False,
        mask_info=None,
        mel_text=None,
        **kwargs
    ):

    if rank is not None:
        print(f"rank {rank} {torch.cuda.device_count()} GPUs")
        torch.cuda.set_device(rank % torch.cuda.device_count())

    # map diffusion hyperparameters to gpu
    diffusion_hyperparams  = calc_diffusion_hyperparams(**diffusion_cfg, fast=True)  # dictionary of all diffusion hyperparameters

    # predefine MelGen model
    builder = ModelBuilder()
    net_diffwave = builder.build_model(model_cfg)
    net = AudioVisualModel(net_diffwave).cuda()
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
    
    dataset_type = dataset_cfg['dataset_type']
    criterion = nn.L1Loss(reduction='none')
    w_asr_list = w_asr
    asr_start_list = asr_start
    for w_asr in w_asr_list:
        for asr_start in asr_start_list:
            dataset = get_dataset(dataset_cfg, split='test', return_mask_properties=True, return_true_text=True)
            guidance_dir_name = f'w1={w_mel_cond}'
            guidance_dir_name += f'_w2={w_asr}_asr_start={asr_start}' #_asr_finish=80'
            guidance_dir_name += f'_mask={on_masked_melspec}' #_repeat=5_same-theta_-mel'
            _output_directory = os.path.join(output_directory, guidance_dir_name)
            os.makedirs(_output_directory, exist_ok=True)
            print("saving to output directory", _output_directory)

            # Create a CSV file
            csv_file = open(os.path.join(_output_directory, 'samples_info.csv'), 'w', newline='')
            csv_writer = csv.writer(csv_file, delimiter='|')
            
            # compute_metrics = Metrics()

            # Write the header row
            if dataset_type == 'explosion_speech_inpainting':
                csv_writer.writerow(['Sample', 'start_explosions', 'explosions_length'])
            elif dataset_type == 'speech_inpainting' or dataset_type == 'speech_inpainting_anechoic':
                titles = ['Sample', 'block_size_list', 'num_blocks']#, 'plcmos_masked_init'] #+  \
                    # [met + '_' + voc for met in ['WER_init', 'plcmos_target_init', 'LSD_init', 'STOI_init', 'PESQ_init'] for voc in vocoders.keys()] + \
                    #     [met + '_' + voc for met in ['WER', 'plcmos_pred' 'LSD', 'STOI', 'PESQ'] for voc in vocoders.keys()]
                csv_writer.writerow(titles)
                # csv_writer.writerow(['Sample', 'block_size_list', 'num_blocks'])

            # ASR based on audio-only model, this is used for getting transcription for guidance, so the input is the masked audio in time domain
            pipeline_asr = InferencePipeline(config_filename_asr_cond, device='cuda')
            
            for i in tqdm(range(n_samples_test)):
                os.makedirs(os.path.join(_output_directory, f'sample_{i}'), exist_ok=True)
                
                if dataset_type == 'explosion_speech_inpainting':
                    speech_melspec, mix_melspec, mix_time, masked_speech, masked_speech_time, explosions_activity, start_explosions, explosions_length = dataset[i]
                    mask = 1 - explosions_activity # zero = explosion, one = no explosion
                    # for j in range(len(masked_cond)):
                    #     masked_cond[j] = masked_cond[j].unsqueeze(0).cuda()
                    # row_dict = {'Sample': i, 'start_explosions': start_explosions, 'explosions_length': explosions_length}
                    csv_writer.writerow([i, start_explosions, explosions_length]) # in samples
                    gt_melspec = speech_melspec.unsqueeze(0)
                    
                    masked_melspec, masked_audio_time = mix_melspec.unsqueeze(0).cuda(), mix_time.unsqueeze(0).cuda()
                    masked_audio_time4text = masked_speech_time
                    masked_cond = [masked_melspec, masked_audio_time]
                    
                
                elif dataset_type == 'speech_inpainting':
                    gt_melspec, *masked_cond, mask, block_size_list, num_blocks = dataset[i]
                    # row_dict = {'Sample': i, 'block_size_list': block_size_list, 'num_blocks': num_blocks}
                    masked_cond = [masked_cond[i].unsqueeze(0).cuda() for i in range(len(masked_cond))]
                    # for j in range(len(masked_cond)):
                    #     masked_cond[j] = masked_cond[j].unsqueeze(0).cuda()
                    csv_writer.writerow([i, block_size_list, num_blocks])
                    gt_melspec = gt_melspec.unsqueeze(0)
                    masked_melspec, masked_audio_time = masked_cond
                    masked_audio_time4text = masked_audio_time.squeeze().cpu()
                
                elif dataset_type == 'plc_task':
                    gt_melspec, masked_melspec, masked_audio_time, frame_mask, sample_mask = dataset[i]
                    mask = frame_mask
                    gt_melspec = gt_melspec.unsqueeze(0)
                    masked_cond = [masked_melspec.cuda(), masked_audio_time.cuda()]
                    masked_cond = [masked_cond[i].unsqueeze(0) for i in range(len(masked_cond))]
                    masked_melspec, masked_audio_time = masked_cond
                    masked_audio_time4text = masked_audio_time.squeeze().cpu()
                elif dataset_cfg.dataset_type == 'speech_inpainting_anechoic':
                    #melspec, masked_melspec, masked_audio_time, mask, block_size_list, num_blocks
                    gt_melspec, masked_melspec, masked_audio_time, mask, block_size_list, num_blocks, true_text = dataset[i]
                    csv_writer.writerow([i, block_size_list, num_blocks])
                    gt_melspec = gt_melspec.unsqueeze(0)
                    mask = mask.unsqueeze(0).cuda()
                    masked_cond = [masked_melspec.cuda(), masked_audio_time.cuda()]
                    masked_cond = [masked_cond[i].unsqueeze(0) for i in range(len(masked_cond))]
                    masked_melspec, masked_audio_time = masked_cond
                    masked_audio_time4text = masked_audio_time.squeeze().cpu()
                
                        #For Unet we need to fix the length of the input to be divided with 4 (2**2)
                if model_cfg._name_ == 'unet':
                    freq_siganl, time_signal = masked_cond
                    desired_num_frames = fix_len_compatibility(gt_melspec.shape[-1])
                    gt_melspec = pad_last_dim(gt_melspec, desired_num_frames - gt_melspec.shape[-1])
                    time_signal = pad_last_dim(time_signal, (desired_num_frames - freq_siganl.shape[-1]) * dataset_cfg[dataset_type]["audio_stft_hop"])
                    freq_siganl = pad_last_dim(freq_siganl, desired_num_frames - freq_siganl.shape[-1]).cuda()
                    mask_frames = torch.zeros((list(mask.shape[:-1]) + [desired_num_frames]))
                    mask_frames[..., :mask.shape[-1]] = 1
                    mask_frames = mask_frames.cuda()
                    mask = pad_last_dim(mask, desired_num_frames - mask.shape[-1], pad_value=1)
                    masked_cond = [freq_siganl, time_signal]
                else:
                    mask_frames = None
                
                if mask_info['mask_type'] == 'repeat_all_freq':
                    masked_melspec, mask = mask_time_all_frequencies_mask(gt_melspec[0], mask_info['repeat_all_freq']['length'], mask_info['repeat_all_freq']['skip'])
                    masked_melspec = masked_melspec.unsqueeze(0).cuda()
                    mask = mask.unsqueeze(0).cuda()
                    masked_cond = [masked_melspec, masked_audio_time]
                elif mask_info['mask_type'] == 'repeat_specific_freq':
                    masked_melspec, mask = mask_time_specific_frequencies_mask(gt_melspec[0], mask_info['repeat_specific_freq']['length'], mask_info['repeat_specific_freq']['skip'], mask_info['repeat_specific_freq']['freq'])
                    masked_melspec = gt_melspec[0] * mask
                    masked_melspec = masked_melspec.unsqueeze(0).cuda()
                    mask = mask.unsqueeze(0).cuda()
                    masked_cond = [masked_melspec, masked_audio_time]
                elif mask_info['mask_type'] == 'by_number':
                    masked_melspec, mask = mask_with_shape_mask(gt_melspec[0], mask_info['by_number']['number'])
                    masked_melspec = masked_melspec.unsqueeze(0).cuda()
                    mask = mask.unsqueeze(0).cuda()
                    masked_cond = [masked_melspec, masked_audio_time]
                elif mask_info['mask_type'] == 'all_time_specific_freq':
                    masked_melspec, mask = mask_specific_frequencies_all_time_mask(gt_melspec[0], mask_info['all_time_specific_freq']['freq'])
                    masked_melspec = gt_melspec[0] * mask
                    masked_melspec = masked_melspec.unsqueeze(0).cuda()
                    mask = mask.unsqueeze(0).cuda()
                    
                text = 'None'
                if apply_asr_guidance:
                    if mel_text:
                        text = true_text
                        print(f"The transcript is: {text}")
                        text = preprocess_text(text)
                        print(f"The normalized transcript is: {text}")
                    else:
                        # Create a temporary file
                        audio4text = masked_audio_time4text
                        sample_rate = 16000  # Example value
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
                            # Save the masked audio array as a WAV file in the temporary file
                            write(temp_wav.name, sample_rate, audio4text.numpy().astype(np.float32)) # TODO maybe we need to do something more clever here
                            # Send the temporary WAV file to the pipeline
                            transcript_from_condition = pipeline_asr(temp_wav.name)
                            text = transcript_from_condition
                            print(f"The transcript is: {text}")
                            text = preprocess_text(text)
                            print(f"The normalized transcript is: {text}")
            
                
                if dataset_type == 'explosion_speech_inpainting':
                    ## save the masked audio in time domain
                    masked_audio_time4saveing = masked_speech_time.squeeze().cpu().numpy()
                    sf.write(os.path.join(_output_directory, f'sample_{i}', 'speech_time_masking_audio.wav'), masked_audio_time4saveing, 16000)
                    ## save the masked audio in time domain
                    masked_audio_time4saveing = masked_audio_time.squeeze().cpu().numpy()
                    sf.write(os.path.join(_output_directory, f'sample_{i}', 'mix_explsions.wav'), masked_audio_time4saveing, 16000)
                elif dataset_type == 'speech_inpainting' or dataset_type == 'plc_task' or dataset_type == 'speech_inpainting_anechoic':
                    ## save the masked audio in time domain
                    masked_audio_time4saveing = masked_audio_time.squeeze().cpu().numpy()
                    sf.write(os.path.join(_output_directory, f'sample_{i}', 'masked_audio_time.wav'), masked_audio_time4saveing, 16000)
                ## get the the clean version of the noisy melspec and the noisy melspec
                weighted_loss, est_X, transformed_X, diffusion_steps, mean_loss = training_loss(net, criterion, gt_melspec.cuda(), masked_cond,  mask.cuda(), diffusion_hyperparams, w_masked_pix=0.8, mask_frames=mask_frames)
                # save the est audio
                est_X = denormalise_mel(est_X)
                est_audio = vocoder(est_X)
                est_audio = est_audio.squeeze()
                est_audio = est_audio / 1.1 / est_audio.abs().max()
                est_audio = est_audio.cpu().numpy()
                sf.write(os.path.join(_output_directory, f'sample_{i}', f'est_audio_after_clean_loss={mean_loss}.wav'), est_audio, 16000)
                
                est_X = est_X.squeeze(0).cpu().numpy()
                matplotlib.image.imsave(os.path.join(_output_directory, f'sample_{i}', 'est_melspec_after_clean_image.png'), est_X[::-1])
                
                #save the noisy audio
                transformed_X = denormalise_mel(transformed_X)
                transformed_X_audio = vocoder(transformed_X)
                transformed_X_audio = transformed_X_audio.squeeze()
                transformed_X_audio = transformed_X_audio / 1.1 / transformed_X_audio.abs().max()
                transformed_X_audio = transformed_X_audio.cpu().numpy()
                sf.write(os.path.join(_output_directory, f'sample_{i}', f'noisy_audio={diffusion_steps.item()}.wav'), transformed_X_audio, 16000)
                
                transformed_X = transformed_X.squeeze(0).cpu().numpy()
                matplotlib.image.imsave(os.path.join(_output_directory, f'sample_{i}', 'noisy_melspec_image.png'), transformed_X[::-1])
                
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
                                mask=mask,
                                on_masked_melspec=on_masked_melspec,
                                mask_frames=mask_frames
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
                
                # plcmos_masked_init = compute_metrics.compute_plcmos(masked_audio_time.squeeze().cpu().numpy())
                # row_dict.update({'plcmos_masked_init': plcmos_masked_init})
                
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

                
            # generate audio from gt melspec
                gt_melspec = denormalise_mel(gt_melspec)
                for vocoder_name, vocoder in vocoders.items():
                    gt_audio = vocoder(gt_melspec.cuda())
                    gt_audio = gt_audio.squeeze()
                    gt_audio = gt_audio / 1.1 / gt_audio.abs().max()
                    gt_audio = gt_audio.cpu().numpy()
                    sf.write(os.path.join(_output_directory, f'sample_{i}', f'gt_audio_{vocoder_name}.wav'), gt_audio, 16000)

                # save as file
                melspec = melspec.squeeze(0).cpu()
                torch.save(melspec, os.path.join(_output_directory, f'sample_{i}', 'generated_spec.npz'))
                
                mask_cpu = mask.squeeze(0).cpu()
                torch.save(mask_cpu, os.path.join(_output_directory, f'sample_{i}', 'mask.npz'))
                # save as image
                melspec = melspec.numpy()
                masked_melspec = masked_melspec.squeeze(0).cpu().numpy()
                gt_melspec = gt_melspec.squeeze(0).numpy()
                matplotlib.image.imsave(os.path.join(_output_directory, f'sample_{i}', 'generated_spec_image.png'), melspec[::-1])
                matplotlib.image.imsave(os.path.join(_output_directory, f'sample_{i}', 'gt_spec_image.png'), gt_melspec[::-1])
                matplotlib.image.imsave(os.path.join(_output_directory, f'sample_{i}', 'masked_spec_image.png'), masked_melspec[::-1])
                
                
                
            
            # Close the CSV file
            csv_file.close()
            
    return


@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    generate(0,
        diffusion_cfg=cfg.diffusion,
        model_cfg=cfg[cfg.melgen],
        dataset_cfg=cfg.dataset,
        **cfg.generate,
    )



if __name__ == "__main__":
    main()
