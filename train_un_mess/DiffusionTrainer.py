import os
import sys
main_dir_path = os.path.abspath("/home/dsi/kenanal/SpeechRepainting")

if main_dir_path not in sys.path:
    sys.path.append(main_dir_path)

from SDE import VPSDE, VESDE
from dataloaders import CollateFn, dataloader
from utils import find_max_epoch, print_size, get_diffusion_hyperparams, local_directory, plot_melspec, fix_len_compatibility
import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf
from models.model_builder import ModelBuilder
from models.audiovisual_model import AudioVisualModel
import torch.nn as nn
from distributed_util import init_distributed, apply_gradient_allreduce, reduce_tensor
import torch
from tqdm import tqdm
from inference_melgen_continuous import generate


class DiffusionTrainer:
    """ The trainer class for the continuous SDE path, which will handle the training loop, including logging, checkpointing, and sample generation.
      It will use the train_class to calculate the training loss and test loss. """
    def __init__(self, rank, name, model_cfg, diffusion_cfg, save_dir, cfg, dataset_cfg, g_model_cfg, num_gpus, batch_size_per_gpu
                 , learning_rate, ckpt_iter, on_noisy_masked_melspec, w_masked_pix, iters_per_logging, iters_per_ckpt,
                   generate_cfg = None, n_iters = None):
        self._rank = rank
        self._name = name
        self._model_cfg = model_cfg # config for the dit model
        self._diffusion_cfg = diffusion_cfg
        self._save_dir = save_dir
        self._cfg = cfg
        self._dataset_cfg = dataset_cfg
        self._dataset_type = dataset_cfg.dataset_type
        self._g_model_cfg = g_model_cfg
        self._num_gpus = num_gpus
        self._batch_size_per_gpu = batch_size_per_gpu
        self._learning_rate = learning_rate
        self._ckpt_iter = ckpt_iter
        self._on_noisy_masked_melspec = on_noisy_masked_melspec
        self._w_masked_pix = w_masked_pix
        self._iters_per_logging = iters_per_logging
        self._iters_per_ckpt = iters_per_ckpt
        self._generate_cfg = generate_cfg
        self._n_iters = n_iters
        self._writer = None
        self._checkpoint_directory = None
        self._net = None
        self._optimizer = None
        self._criterion = None
        self._model_weights = None

    # PART 1: ENVIRONMENT & RESOURCE SETUP

    def setup_experiment_resources(self):
        """ create directiories for logs config and checkpoints, and save the config file.
            create a SummaryWriter for Tensorboard logging (only on rank 0)
            args:
                none
            updates:
                writer: the SummaryWriter object for Tensorboard logging (only on rank 0)
                checkpoint_directory: the directory where checkpoints will be saved
                """
        
        local_path, checkpoint_directory = local_directory(self._name, self._model_cfg, self._diffusion_cfg, self._save_dir, 'checkpoint')
        self._checkpoint_directory = checkpoint_directory
        
        if self._rank == 0:

            if not (self._name is None or self._name == ""):
                path_log = os.path.join(self._save_dir, 'exp', self._name, local_path, "logs")
                path_config = os.path.join(self._save_dir, 'exp', self._name, local_path, "config")
                Path(path_config).mkdir(parents=True, exist_ok=True)
            else:
                path_log = os.path.join(self._save_dir, 'exp', local_path, "logs")
                path_config = os.path.join(self._save_dir, 'exp', local_path, "config")
                Path(path_config).mkdir(parents=True, exist_ok=True)

            self._writer = SummaryWriter(log_dir=path_log)

            config_path = os.path.join(path_config, 'config.yaml')
            with open(config_path, 'w') as f:
                OmegaConf.save(self._cfg, f)
            print('Configuration saved')



    def get_sde(self):
        """ initialize the SDE object based on the diffusion hyperparameters
            args:
                none (reads from self._diffusion_cfg and self._diffusion_hyperparams)
            returns:
                sde: the initialized SDE object (VPSDE or VESDE)
        """
        # map diffusion hyperparameters to gpu
        # diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_cfg, fast=False)  # dictionary of all diffusion hyperparameters
        diffusion_hyperparams = get_diffusion_hyperparams(self._diffusion_cfg)
        
        # SDE init - reads directly from what get_diffusion_hyperparams already parsed
        if diffusion_hyperparams["name"] == "VPSDE":
            return VPSDE(diffusion_hyperparams["beta_min"],
                        diffusion_hyperparams["beta_max"],
                        diffusion_hyperparams["N"])
        elif diffusion_hyperparams["name"] == "VESDE":
            return VESDE(diffusion_hyperparams["sigma_min"],
                        diffusion_hyperparams["sigma_max"],
                        diffusion_hyperparams["N"])
        else:
            raise ValueError(f"Unsupported diffusion/SDE name: {diffusion_hyperparams['name']}")  # DDPM is commented out for now since we are focusing on the continuous SDE path, but can be added back in if we want to do ablation comparing the two paths

 
    def get_dataloaders(self,):
        """ create dataloader for training and testing
            args:
                none
            returns:
                trainloader: the dataloader for training
                testloader: the dataloader for testing
                note: the dataloader should return a tuple of (phoneme_target, melspec, masked_melspec, masked_audio_time, mask) when iterated through
        """

            
        max_num_frame = 1701 #989 # 1701
        time_samples = 16000 * 17 #251200 # 16000 * 17
        if self._model_cfg._name_ == 'unet':
            new_max_num_frame = fix_len_compatibility(max_num_frame)
            time_samples = time_samples + (new_max_num_frame - max_num_frame) * self._dataset_cfg[self._dataset_type]["audio_stft_hop"]
            max_num_frame = new_max_num_frame


        if self._dataset_type == 'speech_inpainting_anechoic':
            inputs_params = [{"axis": 0, "end_number": 0, 'max_length':max_num_frame}, {"axis": 1, "end_number": 0, 'max_length':max_num_frame},
                                                {"axis": 3, "end_number": 1, 'max_length':max_num_frame},
                                                {"axis": 2, "end_number": 0, 'max_length':time_samples}, {"axis": 4, "text":True}]
            if self._model_cfg.text_embed_prop.use_text_embed_rep or self._model_cfg.tts_kw.use_tts:
                inputs_params.append({"axis": 5, "text":True})
                #melspec, masked_melspec, masked_audio_time, mask, text, input_text
            collate_fn = CollateFn(inputs_params=inputs_params,
                                targets_params=[])
        else:
            collate_fn = None
        # (phoneme_target, melspec, masked_melspec, masked_audio_time, mask)
        trainloader = dataloader(self._dataset_cfg, batch_size=self._batch_size_per_gpu, num_gpus=self._num_gpus, collate_fn=collate_fn, split='Train', return_true_text=True)
        trainloader_test = dataloader(self._dataset_cfg, batch_size=self._batch_size_per_gpu, num_gpus=self._num_gpus, collate_fn=collate_fn, split='Test', return_true_text=True)
        print('Data loaded')
        return trainloader, trainloader_test

    def set_net_and_optimizer(self):
        """ create the model - net and optimizer
            args:
                none
            updates:
                net: the model
                optimizer: the optimizer
        """
        
        # predefine model
        builder = ModelBuilder()
        #net_lipreading = builder.build_lipreadingnet()
        #net_facial = builder.build_facial(fc_out=128, with_fc=True)
        net_diffwave = builder.build_model(self._model_cfg) # create the dit model according to the config file
        #net = AudioVisualModel((net_lipreading, net_facial, net_diffwave)).cuda()
        self._net = AudioVisualModel(self._g_model_cfg, net_diffwave).cuda() # get the net of the dit model, i think?
        # net = torch.compile(net)
        print_size(self._net, verbose=False) # sanity check for model size.

        self._criterion = nn.L1Loss(reduction='none')

        # apply gradient all reduce
        if self._num_gpus > 1:
            self._net = apply_gradient_allreduce(self._net)

        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=self._learning_rate)


       
    def load_checkpoint(self):
        """ load checkpoint if exists, and feed the model weights and optimizer state dict
            args:
                none (reads from self._ckpt_iter and self._checkpoint_directory)
            updates:
                model_weights: the loaded model weights from checkpoint, which will be fed into the model
                optimizer state dict: the loaded optimizer state dict from checkpoint, which will be fed into the optimizer
                ckpt_iter: the iteration number of the loaded checkpoint, which will be used to determine the starting iteration for training
        """

        ignore_keys = ['wavlm_model', 'style_speech_model']
        # load checkpoint
        if self._ckpt_iter == 'max':
            self._ckpt_iter = find_max_epoch(self._checkpoint_directory)
        if self._ckpt_iter >= 0:
            try:
                # load checkpoint file
                model_path = os.path.join(self._checkpoint_directory, '{}.pkl'.format(self._ckpt_iter))
                checkpoint = torch.load(model_path, map_location='cpu')

                # feed model dict and optimizer state
                self._model_weights = checkpoint['model_state_dict']
                self._model_weights = {k: v for k, v in self._model_weights.items() if k not in ignore_keys}
                missing_keys , _ = self._net.load_state_dict(self._model_weights, strict=False)
                filtered_missing_keys = [key for key in missing_keys if key not in ignore_keys]
                if not filtered_missing_keys:
                    print('All keys loaded successfully')
                else:
                    raise Exception(f'The following keys were not loaded: {filtered_missing_keys}')
                if 'optimizer_state_dict' in checkpoint:
                    self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    # HACK to reset learning rate
                    self._optimizer.param_groups[0]['lr'] = self._learning_rate

                print('Successfully loaded model at iteration {}'.format(self._ckpt_iter))
            except:
                print(f"Model checkpoint found at iteration {self._ckpt_iter}, but was not successfully loaded - training from scratch.")
                self._ckpt_iter = -1
        else:
            print('No valid checkpoint model found - training from scratch.')
            self._ckpt_iter = -1


    # PART 2: TRAINING LOOP FUNCTIONS
    

    def get_ckpt_iter(self):
        """ get the current checkpoint iteration number, which will be used to determine the starting iteration for training
            args:
                none (reads from self._ckpt_iter)
            returns:
                ckpt_iter: the current checkpoint iteration number, which will be used to determine the starting iteration for training
        """
        return self._ckpt_iter
    
    def parse_batch(self, data):
        """ parse the batch data from dataloader, and move the data to gpu
            args:
                data: the batch data from dataloader, which is a tuple of (phoneme_target, melspec, masked_melspec, masked_audio_time, mask)
            returns:
                parsed_data: the parsed data after moving to gpu, which is a tuple of (melspec, masked_cond, mask, mask_mask, masked_audio_time_mask)
        """
        text = None
        input_text = None
        if self._dataset_type == 'speech_inpainting_anechoic':
                data_list, mask_list = data["inputs"]
                if self._model_cfg.text_embed_prop.use_text_embed_rep or self._model_cfg.tts_kw.use_tts:
                    input_text = data_list[5]
                text = data_list[4]
                melspec, masked_melspec, mask, masked_audio_time = data_list[0].cuda(), data_list[1].cuda(), data_list[2].cuda(), data_list[3].cuda()
                melspec_mask, masked_melspec_mask, mask_mask, masked_audio_time_mask = mask_list[0].cuda(), mask_list[1].cuda(), mask_list[2].cuda(), mask_list[3].cuda()
                masked_cond = [masked_melspec, masked_audio_time]

        return dict(melspec_key=melspec, masked_cond_key=masked_cond, mask_key=mask, mask_mask_key=mask_mask, masked_audio_time_mask_key=masked_audio_time_mask
                    , text_key=text, input_text_key=input_text, melspec_mask_key=melspec_mask, masked_melspec_mask_key=masked_melspec_mask)

    def total_gpu_loss(self, loss):
        """ calculate the total loss across all gpus for distributed training
            args:
                loss: the loss value calculated on the current gpu, which is a scalar tensor
            returns:
                reduced_loss: the total loss value averaged across all gpus, which is a scalar
        """
        if self._num_gpus > 1:
            reduced_loss = reduce_tensor(loss.data, self._num_gpus).item()
        else:
            reduced_loss = loss.item()
        return reduced_loss


    def log_to_tensorboard(self, reduced_loss, n_iter):
        """ dosent actually log to tensorboard."""
        if n_iter % self._iters_per_logging == 0 and self._rank == 0:
                # save training loss to tensorboard
                print("iteration: {} \tloss: {}".format(n_iter, reduced_loss))




    def save_checkpoint(self, n_iter):
        """ save checkpoint of the model and optimizer state dict at regular intervals
            args:
                n_iter: the current iteration number, which is used to determine when to save checkpoint and to name the checkpoint file
            updates:                 
            checkpoint file: a file saved in the checkpoint directory, which contains the model state dict and optimizer
        """

        ignore_keys = ['wavlm_model', 'style_speech_model']
         # save checkpoint
        if n_iter % self._iters_per_ckpt == 0 and self._rank == 0:
            checkpoint_name = '{}.pkl'.format(n_iter)
            model_weights = self._net.state_dict()
            model_weights = {k: v for k, v in model_weights.items() if k not in ignore_keys}
            torch.save({'model_state_dict': model_weights,
                        'optimizer_state_dict': self._optimizer.state_dict()},
                        os.path.join(self._checkpoint_directory, checkpoint_name))
            print('model at iteration %s is saved' % n_iter)


    def generate_and_log_samples(self, n_iter):
        """ generate samples with the current model at regular intervals
            args:
                none (reads from self._generate_cfg and self._net)
            updates:
                generated samples: the generated samples will be saved in the specified directory, and can be visualized in Tensorboard
        """
        
        if n_iter % self._iters_per_ckpt == 0 and self._rank == 0:
            # Generate samples
            samples = generate(
                self._rank, # n_iter,
                self._diffusion_cfg, self._model_cfg, self._g_model_cfg, self._dataset_cfg,
                name=self._name,
                save_dir=self._save_dir,
                ckpt_iter=n_iter, # Use current iteration instead of "max"?
                n_samples=self._generate_cfg.n_samples,
                w_mel_cond=self._generate_cfg.w_mel_cond,
                on_noisy_masked_melspec=self._generate_cfg.on_noisy_masked_melspec
            )
            
            # send images to log
            for i, (mel, mel_gt, masked_cond) in enumerate(zip(*samples)):
                self._writer.add_figure(f'spec/{i+1}_gen', plot_melspec(mel[0].cpu().numpy()), n_iter)
                self._writer.add_figure(f'spec/{i+1}_gt', plot_melspec(mel_gt[0].cpu().numpy()), n_iter)
                self._writer.add_figure(f'spec/{i+1}_masked_melspec', plot_melspec(masked_cond[0][0].cpu().numpy()), n_iter) #this is the masked mel spectrogram
                self._writer.add_audio(f'audio/{i+1}_masked_audio_time', masked_cond[1].cpu().numpy(), n_iter, sample_rate=16000) # this is the masked audio in time domain

    
    def log_loss_to_tensorboard(self, trainloader, epoch_loss, n_iter, is_test=False):
        """ log the training loss to tensorboard
            args:
                reduced_loss: the loss value calculated on the current gpu, which is a scalar tensor
                epoch_loss: the average loss value for the current epoch, which is a scalar
            updates:
                writer: the SummaryWriter object for Tensorboard logging (only on rank 0) will log the reduced_loss and epoch_loss to Tensorboard
        """

        if self._rank == 0:
            epoch_loss /= len(trainloader)
            if is_test:
                print("Test loss: {}".format(epoch_loss))
                self._writer.add_scalar('test_loss', epoch_loss, n_iter)
            else:
                print("Train loss: {}".format(epoch_loss))
                self._writer.add_scalar('train_loss', epoch_loss, n_iter)
            

    
    def train(self, trainer, trainloader, n_iter):
        # 6. main training loop
        epoch_loss = 0
        self._net.train()
        for data in tqdm(trainloader, desc=f'Epoch {n_iter // len(trainloader)}') if self._rank==0 else trainloader:
        # for data in tqdm(trainloader, desc=f'Epoch {n_iter // len(trainloader)}'):
            text = None
            input_text = None
            # 1. parse_batch(self, data): parse the data
            parsed_data = self.parse_batch(data)
            text = parsed_data['text_key']
            input_text = parsed_data['input_text_key']
                

            # 2. somthing with optimizer.zero_grad()
            self._optimizer.zero_grad()

            # 3. calculate training loss with training_loss function.
            loss = trainer.training_loss(self._net, text, input_text, self._on_noisy_masked_melspec, parsed_data) # calculate the training loss.

            # 4. calculate total loss across gpus for logging
            reduced_loss = self.total_gpu_loss(loss)


            # 5. do backward and optimizer step
            loss.backward()
            self._optimizer.step()

            # accumulate epoch loss for logging
            epoch_loss += reduced_loss

            # 6. output to log
            self.log_to_tensorboard(reduced_loss, n_iter)

            # 7. save checkpoint, generate samples and log to tensorboard at regular intervals
            self.save_checkpoint(n_iter)
            self.generate_and_log_samples(n_iter)
            n_iter += 1
        # 8. log epoch loss to tensorboard
        self.log_loss_to_tensorboard(trainloader, epoch_loss, n_iter, is_test=False)
        return n_iter

    def test(self, trainloader_test, n_iter, trainer):
        epoch_loss = 0.
        self._net.eval()
        with torch.no_grad():
            for data in tqdm(trainloader_test, desc=f'Test Epoch {n_iter // len(trainloader_test)}') if self._rank==0 else trainloader_test:
            # for data in tqdm(trainloader, desc=f'Epoch {n_iter // len(trainloader)}'):
                text = None
                input_text = None
               
                # 1. parse_batch(self, data): parse the data
                parsed_test_data = self.parse_batch(data)
                text = parsed_test_data['text_key']
                input_text = parsed_test_data['input_text_key']
                
                # 2. calculate test loss with test_loss function.   
                loss = trainer.test_loss(self._net, text, input_text, self._on_noisy_masked_melspec, parsed_test_data) # calculate the training loss.

                # 3. calculate total loss across gpus for logging
                reduced_loss = self.total_gpu_loss(loss)

                epoch_loss += reduced_loss


            # 4. log epoch loss to tensorboard
            self.log_loss_to_tensorboard(trainloader_test, epoch_loss, n_iter, is_test=True)
        
    