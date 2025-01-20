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
from hifi_gan.env import AttrDict

from utils import find_max_epoch, print_size, calc_diffusion_hyperparams, local_directory, preprocess_text
from models.utils import get_phones_dict
from models.mel_spec_wavlm_phoneme_classifier import WavlmMelSpecPhonemeClassifier
import csv
from scipy.io.wavfile import write
import tempfile
from losses.losses import get_loss_func

from dataloaders import dataloader, CollateFn



def calc_phoneme(net, loss_fn, masked_melspec, mask, mask_mask, phoneme_target, 
                  phoneme_target_mask, w_masked_pix=0.7, masked_audio_time_mask=None):
    """
    Compute the training loss of epsilon and epsilon_theta

    Parameters:
    net (torch network):            the wavenet model
    loss_fn (torch loss function):  the loss function, default is nn.MSELoss()
    X (torch.tensor):               training data, shape=(batchsize, 1, length of audio)

    Returns:
    training loss
    """

    phonemes_estimated = net(**{"input_values":masked_melspec, "attention_mask":mask_mask, "masked_region":mask}) 
    # print(phoneme_estimated)
    loss = loss_fn(phonemes_estimated, phoneme_target) #[B, T]
    loss = loss * mask_mask
    unmaksed_loss = torch.sum(mask * loss) / torch.sum(mask * mask_mask)
    masked_loss = torch.sum((1-mask) * loss) / torch.sum((1-mask) * mask_mask)
    weighted_loss = (1 - w_masked_pix) * unmaksed_loss + w_masked_pix * masked_loss
    return weighted_loss, phonemes_estimated




@torch.no_grad()
def inference(
        rank,
        model_cfg,
        dataset_cfg,
        ckpt_path,
        cfg_loss,
        # w_mel_cond=0,
        # w_asr=1.1,
        # asr_start=250,
        save_dir=None,
        n_samples_test = 20,
        inference_phoneme_only_name_dir='inferenced_phonemes',
        on_masked_melspec=False,
        **kwargs
    ):

    if rank is not None:
        print(f"rank {rank} {torch.cuda.device_count()} GPUs")
        torch.cuda.set_device(rank % torch.cuda.device_count())

    phoneme_dict_path = "/home/dsi/moradim/Documents/MFA/models/inspect/english_us_arpa_acoustic/phones.txt"
    _, phoneme_dict_d2p = get_phones_dict(phoneme_dict_path)


    net = WavlmMelSpecPhonemeClassifier(model_cfg).cuda()
    print_size(net)
    net.eval()

    # load checkpoint
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
        print('Successfully loaded wavlm phoneme classifier checkpoint')
    except Exception as e:
        print(e)
        raise Exception('No valid model found')
        

    if save_dir is None:
        save_dir = os.getcwd()
    output_directory = os.path.join(save_dir, inference_phoneme_only_name_dir)
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        print("saving to output directory", output_directory)


    
    dataset_type = dataset_cfg['dataset_type']
        # load training data
    collate_fn = CollateFn(inputs_params=[{"axis": 2, "end_number": 'min', 'max_length':1701},
                                          {"axis": 4, "end_number": 1, 'max_length':1701}],
                           targets_params=[{"axis": 0, "end_number": 1, 'max_length':1701}])
    # (phoneme_target, melspec, masked_melspec, masked_audio_time, mask)
    trainloader_test = dataloader(dataset_cfg, batch_size=1, num_gpus=1, collate_fn=collate_fn, split='Test')
    dataset = get_dataset(dataset_cfg, split='test', return_mask_properties=True)
    criterion = get_loss_func(cfg_loss).cuda()
    # guidance_dir_name = f'w1={w_mel_cond}'
    # guidance_dir_name += f'_w2={w_asr}_asr_start={asr_start}'

    _output_directory = output_directory
    os.makedirs(_output_directory, exist_ok=True)
    print("saving to output directory", _output_directory)

    # Create a CSV file
    csv_file = open(os.path.join(_output_directory, 'results.csv'), 'w', newline='')
    csv_writer = csv.writer(csv_file, delimiter='|')

    # Write the header row
    if dataset_type == 'explosion_speech_inpainting':
        csv_writer.writerow(['Sample', 'start_explosions', 'explosions_length'])
    elif dataset_type == 'speech_inpainting':
        csv_writer.writerow(['Sample', 'block_size_list', 'num_blocks'])
    elif dataset_type == 'speech_inpainting_phoneme_classifier':
        csv_writer.writerow(['Sample', 'block_size_list', 'num_blocks'])

    with torch.no_grad():
        for i in tqdm(range(n_samples_test)):
            os.makedirs(os.path.join(_output_directory, f'sample_{i}'), exist_ok=True)
            
            if dataset_type == 'explosion_speech_inpainting':
                speech_melspec, mix_melspec, mix_time, masked_speech, masked_speech_time, explosions_activity, start_explosions, explosions_length = dataset[i]
                mask = 1 - explosions_activity # zero = explosion, one = no explosion
                # for j in range(len(masked_cond)):
                #     masked_cond[j] = masked_cond[j].unsqueeze(0).cuda()
                csv_writer.writerow([i, start_explosions, explosions_length]) # in samples
                gt_melspec = speech_melspec.unsqueeze(0)
                
                masked_melspec, masked_audio_time = mix_melspec.unsqueeze(0).cuda(), mix_time.unsqueeze(0).cuda()
                masked_audio_time4text = masked_speech_time
                masked_cond = [masked_melspec, masked_audio_time]
                
            
            elif dataset_type == 'speech_inpainting':
                gt_melspec, *masked_cond, mask, block_size_list, num_blocks = dataset[i]
                masked_cond = [masked_cond[i].unsqueeze(0).cuda() for i in range(len(masked_cond))]
                # for j in range(len(masked_cond)):
                #     masked_cond[j] = masked_cond[j].unsqueeze(0).cuda()
                csv_writer.writerow([i, block_size_list, num_blocks])
                gt_melspec = gt_melspec.unsqueeze(0)
                masked_melspec, masked_audio_time = masked_cond
                masked_audio_time4text = masked_audio_time.squeeze().cpu()
                
            elif dataset_type == 'speech_inpainting_phoneme_classifier':
                dataloader_bool = False
                if dataloader_bool:
                    #TODO unfinished
                    data = trainloader_test[i]
                    phoneme_target, phoneme_target_mask = data["targets"]
                    phoneme_target, phoneme_target_mask = phoneme_target.cuda(), phoneme_target_mask.cuda()
                    # melspec, masked_melspec, masked_audio_time, mask
                    inputs, inputs_masks = data["inputs"]
                    gt_melspec, masked_melspec, masked_audio_time, mask = inputs[0].cuda(), inputs[1].cuda(), inputs[2].cuda(), inputs[3].cuda()
                    melspec_mask, masked_melspec_mask, masked_audio_time_mask, mask_mask = inputs_masks[0].cuda(), inputs_masks[1].cuda(), inputs_masks[2].cuda(), inputs_masks[3].cuda()
                    masked_cond = [masked_melspec, masked_audio_time]
                else:
                    phoneme_target, gt_melspec, masked_melspec, masked_audio_time, mask, block_size_list, num_blocks = dataset[i]
                    phoneme_target, gt_melspec, mask, masked_melspec = phoneme_target.unsqueeze(0).cuda(), gt_melspec.unsqueeze(0).cuda(), mask.unsqueeze(0).cuda(), masked_melspec.unsqueeze(0).cuda()
                    mask_mask = torch.ones_like(mask)
                    melspec_mask, masked_melspec_mask, masked_audio_time_mask, phoneme_target_mask = None, None, None, None
                    csv_writer.writerow([i, block_size_list, num_blocks])
                    if not model_cfg.mask_regions:
                        masked_melspec = gt_melspec

        
            
            if dataset_type == 'explosion_speech_inpainting':
                ## save the masked audio in time domain
                masked_audio_time4saveing = masked_speech_time.squeeze().cpu().numpy()
                sf.write(os.path.join(_output_directory, f'sample_{i}', 'speech_time_masking_audio.wav'), masked_audio_time4saveing, 16000)
                ## save the masked audio in time domain
                masked_audio_time4saveing = masked_audio_time.squeeze().cpu().numpy()
                sf.write(os.path.join(_output_directory, f'sample_{i}', 'mix_explsions.wav'), masked_audio_time4saveing, 16000)
            elif dataset_type == 'speech_inpainting':
                ## save the masked audio in time domain
                masked_audio_time4saveing = masked_audio_time.squeeze().cpu().numpy()
                sf.write(os.path.join(_output_directory, f'sample_{i}', 'masked_audio_time.wav'), masked_audio_time4saveing, 16000)
            elif dataset_type == 'speech_inpainting_phoneme_classifier':
                pass

                    
            weighted_loss, phoneme_estimated = calc_phoneme(net, criterion, masked_melspec, mask, mask_mask, phoneme_target, 
                  phoneme_target_mask, w_masked_pix=0.5, masked_audio_time_mask=masked_audio_time_mask)
            phoneme_estimated_prob = torch.nn.functional.softmax(phoneme_estimated, dim=-2)
            est_phoneme_digits = torch.argmax(phoneme_estimated_prob, dim=-2)
            est_phoneme_string = [phoneme_dict_d2p[est_phoneme_digit] for est_phoneme_digit in est_phoneme_digits.tolist()[0]]
            # est_phoneme_string = ' '.join(est_phoneme_string)
            true_phoneme_string = [phoneme_dict_d2p[true_phoneme_digit] for true_phoneme_digit in phoneme_target.tolist()[0]]
            # print(f"Ths loss is: {weighted_loss} /n true_phoneme_string: {true_phoneme_string}, /n est_phoneme_string: {est_phoneme_string}")
            
                    # save as image
            masked_melspec = masked_melspec.squeeze(0).cpu().numpy()
            gt_melspec = gt_melspec.squeeze(0).cpu().numpy()
            matplotlib.image.imsave(os.path.join(_output_directory, f'sample_{i}', 'gt_spec_image.png'), gt_melspec[::-1])
            matplotlib.image.imsave(os.path.join(_output_directory, f'sample_{i}', 'masked_spec_image.png'), masked_melspec[::-1])

            
            # save phoneme estimation
            phoneme_filename = os.path.join(_output_directory, f'sample_{i}', 'phoneme_gt_and_estimated.html')
            def colorize_html(char, color):
                return f'<span style="color:red">{char}</span>' if color == 0 else char

            # Create HTML table rows with aligned elements
            rows = []
            color_list = mask.tolist()[0]
            hop_length = 160
            sampling_rate = 16000
            for i, (true_char, est_char, color) in enumerate(zip(true_phoneme_string, est_phoneme_string, color_list)):
                true_colored = colorize_html(true_char, color)
                est_colored = colorize_html(est_char, color)
                rows.append(f"<tr><td>{true_colored}</td><td>{est_colored}</td><td>{i * hop_length / sampling_rate}</td></tr>")

            # Write the aligned and colorized lists to an HTML file
            with open(phoneme_filename, 'w') as f:
                f.write("<html><body>\n")
                f.write("<table border='1' style='border-collapse: collapse; text-align: center;'>\n")
                f.write("<tr><th>true_phoneme_string</th><th>est_phoneme_string</th><th>time[s]</th></tr>\n")
                f.write("\n".join(rows))
                f.write("</table>\n")
                f.write("</body></html>\n")
            # RED = "\033[91m"
            # RESET = "\033[0m"

            # # Function to color elements based on the binary list
            # def colorize_list(string_list, color_list):
            #     return [
            #         f"{RED}{char}{RESET}" if color == 0 else char
            #         for char, color in zip(string_list, color_list)
            #     ]

            # # Apply colorization
            # colored_true = colorize_list(true_phoneme_string, mask.tolist()[0])
            # colored_est = colorize_list(est_phoneme_string, mask.tolist()[0])
            
            
            # with open(text_filename, 'w') as f:
            #     f.write(f"{'true_phoneme_string:':25} " + str(true_phoneme_string)  + "\n")
            #     f.write(f"{'est_phoneme_string:':25} " + str(est_phoneme_string))
        
    # Close the CSV file
    csv_file.close()
        
    return



@hydra.main(version_base=None, config_path="configs/", config_name="wavlm_phoneme_classifier_config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    inference(0,
        model_cfg=cfg.wavlm_phoneme_classifier_masked_speech,
        dataset_cfg=cfg.dataset,
        cfg_loss=cfg.cfg_loss,
        **cfg.inference,
        
    )



if __name__ == "__main__":
    main()
