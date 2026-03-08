# this file is an adapated version https://github.com/albertfgu/diffwave-sashimi, licensed
# under https://github.com/albertfgu/diffwave-sashimi/blob/master/LICENSE

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #'1,2,4,5,6,7'
import time
import warnings
warnings.filterwarnings("ignore")
from functools import partial
import multiprocessing as mp
# import torch.multiprocessing as mp
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import hydra
# import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from dataloaders import dataloader, CollateFn
from utils import find_max_epoch, print_size, get_diffusion_hyperparams, local_directory, plot_melspec, fix_len_compatibility

from distributed_util import init_distributed, apply_gradient_allreduce, reduce_tensor
from inference_melgen import generate

from models.model_builder import ModelBuilder
from models.audiovisual_model import AudioVisualModel
from models import WaveNet, Unet, DiT

def distributed_train(rank, num_gpus, group_name, cfg):

    # Distributed running initialization
    dist_cfg = cfg.pop("distributed")
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_cfg)

    train(
        rank=rank, num_gpus=num_gpus,
        diffusion_cfg=cfg.diffusion,
        model_cfg=cfg[cfg.melgen],
        g_model_cfg=cfg.g_model,
        dataset_cfg=cfg.dataset,
        generate_cfg=cfg.generate,
        **cfg.train,
        cfg=cfg
    )

def train(
    rank, num_gpus, save_dir,
    diffusion_cfg, model_cfg, g_model_cfg, dataset_cfg, generate_cfg, # dist_cfg, wandb_cfg, # train_cfg,
    ckpt_iter, n_iters, iters_per_ckpt, iters_per_logging,
    learning_rate, batch_size_per_gpu, w_masked_pix, on_noisy_masked_melspec,
    name=None, cfg=None
):
    
    """
    Parameters:
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded;
                                    automitically selects the maximum iteration if 'max' is selected
    n_iters (int):                  number of iterations to train, default is 1M
    iters_per_ckpt (int):           number of iterations to save checkpoint,
                                    default is 10k, for models with residual_channel=64 this number can be larger
    iters_per_logging (int):        number of iterations to save training log and compute validation loss, default is 100
    learning_rate (float):          learning rate
    batch_size_per_gpu (int):       batchsize per gpu, default is 2 so total batchsize is 16 with 8 gpus
    name (str):                     prefix in front of experiment name
    """
    

    dataset_type = dataset_cfg.dataset_type
    ignore_keys = ['wavlm_model', 'style_speech_model']
    local_path, checkpoint_directory = local_directory(name, model_cfg, diffusion_cfg, save_dir, 'checkpoint')
    
    if rank == 0:
        if not (name is None or name == ""):
            path_log = os.path.join(save_dir, 'exp', name, local_path, "logs")
            path_config = os.path.join(save_dir, 'exp', name, local_path, "config")
            Path(path_config).mkdir(parents=True, exist_ok=True)
        else:
            path_log = os.path.join(save_dir, 'exp', local_path, "logs")
            path_config = os.path.join(save_dir, 'exp', local_path, "config")
            Path(path_config).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=path_log)
    
    if rank == 0:
        config_path = os.path.join(path_config, 'config.yaml')
        with open(config_path, 'w') as f:
            OmegaConf.save(cfg, f)
        print('Configuration saved')
    

    # map diffusion hyperparameters to gpu
    # diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_cfg, fast=False)  # dictionary of all diffusion hyperparameters
    diffusion_hyperparams = get_diffusion_hyperparams(diffusion_cfg)
    # load training data
        # load training data
    max_num_frame = 1701 #989 # 1701
    time_samples = 16000 * 17#251200 # 16000 * 17
    if model_cfg._name_ == 'unet':
        new_max_num_frame = fix_len_compatibility(max_num_frame)
        time_samples = time_samples + (new_max_num_frame - max_num_frame) * dataset_cfg[dataset_type]["audio_stft_hop"]
        max_num_frame = new_max_num_frame
    if dataset_cfg.dataset_type == 'explosion_speech_inpainting':
        pass
    elif dataset_cfg.dataset_type == 'speech_inpainting':
        collate_fn = None 
    elif dataset_cfg.dataset_type == 'plc_task':
        #melspec, masked_melspec, masked_audio_time, frame_mask, sample_mask
        collate_fn = CollateFn(inputs_params=[{"axis": 0, "end_number": 0, 'max_length':max_num_frame}, {"axis": 1, "end_number": 0, 'max_length':max_num_frame},
                                            {"axis": 2, "end_number": 0, 'max_length':time_samples},
                                            {"axis": 3, "end_number": 1, 'max_length':max_num_frame},
                                            {"axis": 4, "end_number": 1, 'max_length':max_num_frame}],
                            targets_params=[])
    elif dataset_cfg.dataset_type == 'speech_inpainting_anechoic':
        inputs_params = [{"axis": 0, "end_number": 0, 'max_length':max_num_frame}, {"axis": 1, "end_number": 0, 'max_length':max_num_frame},
                                            {"axis": 3, "end_number": 1, 'max_length':max_num_frame},
                                            {"axis": 2, "end_number": 0, 'max_length':time_samples}, {"axis": 4, "text":True}]
        if model_cfg.text_embed_prop.use_text_embed_rep or model_cfg.tts_kw.use_tts:
            inputs_params.append({"axis": 5, "text":True})
            #melspec, masked_melspec, masked_audio_time, mask, text, input_text
        collate_fn = CollateFn(inputs_params=inputs_params,
                            targets_params=[])
    else:
        collate_fn = None
    # (phoneme_target, melspec, masked_melspec, masked_audio_time, mask)
    trainloader = dataloader(dataset_cfg, batch_size=batch_size_per_gpu, num_gpus=num_gpus, collate_fn=collate_fn, split='Train', return_true_text=True)
    trainloader_test = dataloader(dataset_cfg, batch_size=batch_size_per_gpu, num_gpus=num_gpus, collate_fn=collate_fn, split='Test', return_true_text=True)
    print('Data loaded')

    # predefine model
    builder = ModelBuilder()
    #net_lipreading = builder.build_lipreadingnet()
    #net_facial = builder.build_facial(fc_out=128, with_fc=True)
    net_diffwave = builder.build_model(model_cfg)
    #net = AudioVisualModel((net_lipreading, net_facial, net_diffwave)).cuda()
    net = AudioVisualModel(g_model_cfg, net_diffwave).cuda()
    # net = torch.compile(net)
    print_size(net, verbose=False)

    criterion = nn.L1Loss(reduction='none')

    # apply gradient all reduce
    if num_gpus > 1:
        net = apply_gradient_allreduce(net)

    # define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # load checkpoint
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(checkpoint_directory)
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(checkpoint_directory, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')

            # feed model dict and optimizer state
            model_weights = checkpoint['model_state_dict']
            model_weights = {k: v for k, v in model_weights.items() if k not in ignore_keys}
            missing_keys , _ = net.load_state_dict(model_weights, strict=False)
            filtered_missing_keys = [key for key in missing_keys if key not in ignore_keys]
            if not filtered_missing_keys:
                print('All keys loaded successfully')
            else:
                raise Exception(f'The following keys were not loaded: {filtered_missing_keys}')
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # HACK to reset learning rate
                optimizer.param_groups[0]['lr'] = learning_rate

            print('Successfully loaded model at iteration {}'.format(ckpt_iter))
        except:
            print(f"Model checkpoint found at iteration {ckpt_iter}, but was not successfully loaded - training from scratch.")
            ckpt_iter = -1
    else:
        print('No valid checkpoint model found - training from scratch.')
        ckpt_iter = -1

    # training

    n_iter = ckpt_iter + 1
    while n_iter < n_iters + 1:
        epoch_loss = 0.
        for data in tqdm(trainloader, desc=f'Epoch {n_iter // len(trainloader)}') if rank==0 else trainloader:
        # for data in tqdm(trainloader, desc=f'Epoch {n_iter // len(trainloader)}'):
            text = None
            input_text = None
            if dataset_type == 'explosion_speech_inpainting':
                speech_melspec, mix_melspec, mix_time, masked_speech, masked_speech_time, explosions_activity, start_explosions, explosions_length = data
                mask = 1 - explosions_activity # zero = explosion, one = no explosion
                mask = mask.cuda()
                melspec = speech_melspec.cuda()
                mix_melspec, mix_time = mix_melspec.cuda(), mix_time.cuda()
                masked_cond = [mix_melspec, mix_time]
            elif dataset_type == 'speech_inpainting':
                melspec, *masked_cond, mask = data
                masked_cond = [masked_cond[i].cuda() for i in range(len(masked_cond))]
                melspec, mask = melspec.cuda(), mask.cuda()
                mask_mask = torch.ones_like(mask).cuda()
                masked_audio_time_mask = None
                # for i in range(len(masked_cond)):
                #     masked_cond[i] = masked_cond[i].cuda()
            elif dataset_type == 'speech_inpainting_anechoic':
                data_list, mask_list = data["inputs"]
                if model_cfg.text_embed_prop.use_text_embed_rep or model_cfg.tts_kw.use_tts:
                    input_text = data_list[5]
                text = data_list[4]
                melspec, masked_melspec, mask, masked_audio_time = data_list[0].cuda(), data_list[1].cuda(), data_list[2].cuda(), data_list[3].cuda()
                melspec_mask, masked_melspec_mask, mask_mask, masked_audio_time_mask = mask_list[0].cuda(), mask_list[1].cuda(), mask_list[2].cuda(), mask_list[3].cuda()
                masked_cond = [masked_melspec, masked_audio_time]
            elif dataset_type == 'plc_task':
                data_list, mask_list = data["inputs"]
                melspec, masked_melspec, masked_audio_time, frame_mask, sample_mask = \
                        data_list[0].cuda(), data_list[1].cuda(), data_list[2].cuda(), data_list[3].cuda(), data_list[4].cuda()
                melspec_mask, masked_melspec_mask, masked_audio_time_mask, frame_mask_mask, sample_mask_mask = \
                        mask_list[0].cuda(), mask_list[1].cuda(), mask_list[2].cuda(), mask_list[3].cuda(), mask_list[4].cuda()
                masked_cond = [masked_melspec, masked_audio_time]
                mask = frame_mask
                mask_mask = frame_mask_mask
            # back-propagation
            # print("The max of melspec is: ", torch.max(melspec))
            # print("The min of melspec is: ", torch.min(melspec))
            optimizer.zero_grad()
            loss = training_loss(net, criterion, melspec, masked_cond, mask, mask_mask, diffusion_hyperparams, text, input_text,
                  masked_audio_time_mask, on_noisy_masked_melspec, w_masked_pix)
            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.item()
            # print("doing the backward nor")
            loss.backward()
            optimizer.step()

            epoch_loss += reduced_loss

            # output to log
            if n_iter % iters_per_logging == 0 and rank == 0:
                # save training loss to tensorboard
                print("iteration: {} \tloss: {}".format(n_iter, reduced_loss))

            # save checkpoint
            if n_iter % iters_per_ckpt == 0 and rank == 0:
                checkpoint_name = '{}.pkl'.format(n_iter)
                model_weights = net.state_dict()
                model_weights = {k: v for k, v in model_weights.items() if k not in ignore_keys}
                torch.save({'model_state_dict': model_weights,
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(checkpoint_directory, checkpoint_name))
                print('model at iteration %s is saved' % n_iter)

                # Generate samples
                generate_cfg["ckpt_iter"] = n_iter
                samples = generate(
                    rank, # n_iter,
                    diffusion_cfg, model_cfg, g_model_cfg, dataset_cfg,
                    name=name,
                    save_dir=save_dir,
                    ckpt_iter="max",
                    n_samples=generate_cfg.n_samples,
                    w_mel_cond=generate_cfg.w_mel_cond,
                    on_noisy_masked_melspec=generate_cfg.on_noisy_masked_melspec
                )
                
                # send images to log
                for i, (mel, mel_gt, masked_cond) in enumerate(zip(*samples)):
                    writer.add_figure(f'spec/{i+1}_gen', plot_melspec(mel[0].cpu().numpy()), n_iter)
                    writer.add_figure(f'spec/{i+1}_gt', plot_melspec(mel_gt[0].cpu().numpy()), n_iter)
                    writer.add_figure(f'spec/{i+1}_masked_melspec', plot_melspec(masked_cond[0][0].cpu().numpy()), n_iter) #this is the masked mel spectrogram
                    writer.add_audio(f'audio/{i+1}_masked_audio_time', masked_cond[1].cpu().numpy(), n_iter, sample_rate=16000) # this is the masked audio in time domain

            n_iter += 1
        if rank == 0:
            epoch_loss /= len(trainloader)
            writer.add_scalar('train_loss', epoch_loss, n_iter)
            
        ###################################### TEST ######################################
        epoch_loss = 0.
        net.eval()
        with torch.no_grad():
            for data in tqdm(trainloader_test, desc=f'Test Epoch {n_iter // len(trainloader_test)}') if rank==0 else trainloader_test:
            # for data in tqdm(trainloader, desc=f'Epoch {n_iter // len(trainloader)}'):
                text = None
                input_text = None
                if dataset_type == 'explosion_speech_inpainting':
                    speech_melspec, mix_melspec, mix_time, masked_speech, masked_speech_time, explosions_activity, start_explosions, explosions_length = data
                    mask = 1 - explosions_activity # zero = explosion, one = no explosion
                    mask = mask.cuda()
                    melspec = speech_melspec.cuda()
                    mix_melspec, mix_time = mix_melspec.cuda(), mix_time.cuda()
                    masked_cond = [mix_melspec, mix_time]
                elif dataset_type == 'speech_inpainting':
                    melspec, *masked_cond, mask = data
                    masked_cond = [masked_cond[i].cuda() for i in range(len(masked_cond))]
                    melspec, mask = melspec.cuda(), mask.cuda()
                    # for i in range(len(masked_cond)):
                    #     masked_cond[i] = masked_cond[i].cuda()
                elif dataset_type == 'plc_task':
                    data_list, mask_list = data["inputs"]
                    melspec, masked_melspec, masked_audio_time, frame_mask, sample_mask = \
                        data_list[0].cuda(), data_list[1].cuda(), data_list[2].cuda(), data_list[3].cuda(), data_list[4].cuda()
                    melspec_mask, masked_melspec_mask, masked_audio_time_mask, frame_mask_mask, sample_mask_mask = \
                        mask_list[0].cuda(), mask_list[1].cuda(), mask_list[2].cuda(), mask_list[3].cuda(), mask_list[4].cuda()
                    masked_cond = [masked_melspec, masked_audio_time]
                    mask = frame_mask
                    mask_mask = frame_mask_mask
                elif dataset_type == 'speech_inpainting_anechoic':
                    data_list, mask_list = data["inputs"]
                    if model_cfg.text_embed_prop.use_text_embed_rep or model_cfg.tts_kw.use_tts:
                        input_text = data_list[5]
                    text = data_list[4]
                    melspec, masked_melspec, mask, masked_audio_time = data_list[0].cuda(), data_list[1].cuda(), data_list[2].cuda(), data_list[3].cuda()
                    melspec_mask, masked_melspec_mask, mask_mask, masked_audio_time_mask = mask_list[0].cuda(), mask_list[1].cuda(), mask_list[2].cuda(), mask_list[3].cuda()
                    masked_cond = [masked_melspec, masked_audio_time]
                    
                loss = test_loss(net, criterion, melspec, masked_cond, mask, mask_mask, diffusion_hyperparams, text, input_text,
                  masked_audio_time_mask, on_noisy_masked_melspec, w_masked_pix)
                if num_gpus > 1:
                    reduced_loss = reduce_tensor(loss.data, num_gpus).item()
                else:
                    reduced_loss = loss.item()

                epoch_loss += reduced_loss

            if rank == 0:
                epoch_loss /= len(trainloader_test)
                print("Test loss: {}".format(epoch_loss))
                writer.add_scalar('test_loss', epoch_loss, n_iter)

    # Close logger
    if rank == 0:
        writer.close()

def training_loss(net, loss_fn, melspec, masked_cond, mask, mask_mask, diffusion_hyperparams, text, input_text,
                  masked_audio_time_mask, on_noisy_masked_melspec, w_masked_pix=0.7):
    """
    Compute the training loss of epsilon and epsilon_theta

    Parameters:
    net (torch network):            the wavenet model
    loss_fn (torch loss function):  the loss function, default is nn.MSELoss()
    X (torch.tensor):               training data, shape=(batchsize, 1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by get_diffusion_hyperparams
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
    if on_noisy_masked_melspec:
        transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * melspec + torch.sqrt(1-Alpha_bar[diffusion_steps]) * z
        transformed_X = melspec * mask + transformed_X * (1-mask)
    else:
        transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * melspec + torch.sqrt(1-Alpha_bar[diffusion_steps]) * z  # training from Denoising Diffusion Probabilistic Models paper compute x_t from q(x_t|x_0)
    cond_drop_prob = 0.2 # 0.2
    epsilon_theta = net(transformed_X, masked_cond, diffusion_steps.view(B,1), cond_drop_prob, text=text, input_text=input_text,  mask_padding_time=masked_audio_time_mask, mask_padding_frames=mask_mask)
    if net.g_model_cfg.predict_type =='speech':
        epsilon_theta = (transformed_X - torch.sqrt(Alpha_bar[diffusion_steps]) * epsilon_theta) / torch.sqrt(1-Alpha_bar[diffusion_steps])
    loss = loss_fn(epsilon_theta, z) #[B, F, T]    
    loss = loss * mask_mask
    unmaksed_loss = torch.sum(mask * loss) / (torch.sum(mask * mask_mask) * loss.shape[1])
    masked_loss = torch.sum((1-mask) * loss) / (torch.sum((1-mask) * mask_mask) * loss.shape[1])
    weighted_loss = (1 - w_masked_pix) * unmaksed_loss + w_masked_pix * masked_loss
    
    
    return weighted_loss

def test_loss(net, loss_fn, melspec, masked_cond, mask, mask_mask, diffusion_hyperparams, text, input_text,
                  masked_audio_time_mask, on_noisy_masked_melspec, w_masked_pix=0.7):
    return training_loss(net, loss_fn, melspec, masked_cond, mask, mask_mask, diffusion_hyperparams, text, input_text,
                  masked_audio_time_mask, on_noisy_masked_melspec, w_masked_pix)

#small_my-tts-dit_with-space_without-sma_tts-output=mel
# small_my-tts-dit_with-space_without-sma_tts-output=phoneme
# small_my-tts-dit_with-space_without-sma_tts-output=phoneme_with_energy_pitch
# config_dit_without-space-phoneme_on-masked-mel
# small_my-tts-dit_with-space_without-sma_tts-output=phoneme_with_energy_pitch_nnter_attention
@hydra.main(version_base=None, config_path="configs/4exp/", config_name="config_dit_without-space-phoneme_on-masked-mel")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    if not os.path.isdir("exp/"):
        os.makedirs("exp/")
        os.chmod("exp/", 0o775)

    num_gpus = torch.cuda.device_count()
    print(f'there are {num_gpus} gpus')
    train_fn = partial(
        distributed_train,
        num_gpus=num_gpus,
        group_name=time.strftime("%Y%m%d-%H%M%S"),
        cfg=cfg,
    )

    if num_gpus <= 1:
        train_fn(0)
    else:
        mp.set_start_method("spawn", force=True)
        # mp.set_start_method("fork", force=True)
        processes = []
        for i in range(num_gpus):
            p = mp.Process(target=train_fn, args=(i,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

if __name__ == "__main__":
    main()
