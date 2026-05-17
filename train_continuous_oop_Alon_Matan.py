from distributed_util import init_distributed, apply_gradient_allreduce, reduce_tensor
from train_un_mess.DiffusionTrainer import DiffusionTrainer
from train_un_mess.train_class import train_class
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import os
from functools import partial
import time
import multiprocessing as mp
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #'1,2,4,5,6,7'


""" The main training script for the continuous SDE path, which will be used to train the model on the continuous SDE path.
 It will be called by the command line with the appropriate config file.
   The config file will specify the hyperparameters for the training, including the diffusion hyperparameters, model hyperparameters, dataset hyperparameters, and training hyperparameters.
     The main function will initialize the distributed training if necessary, create an instance of the DiffusionTrainer class, and call the train and test methods of the DiffusionTrainer class in a loop until the specified number of iterations is reached.
     The DiffusionTrainer class will handle the training loop, including logging, checkpointing, and sample generation. The train_class will be used to calculate the training loss and test loss. """
def distributed_train(rank, num_gpus, group_name, cfg):

    # Distributed running initialization
    dist_cfg = cfg.pop("distributed")
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_cfg)

    diff_trainer = DiffusionTrainer(
        rank=rank, num_gpus=num_gpus,
        diffusion_cfg=cfg.diffusion,
        model_cfg=cfg[cfg.melgen],
        g_model_cfg=cfg.g_model,
        dataset_cfg=cfg.dataset,
        generate_cfg=cfg.generate,
        **cfg.train,
        cfg=cfg
    )

    

    # 1. create directiories for logs config and checkpoints, and save the config file. create a SummaryWriter for Tensorboard logging
    diff_trainer.setup_experiment_resources()

    # 2. initialize the SDE object based on the diffusion hyperparameters
    sde = diff_trainer.get_sde()
    
    # 3. create dataloader for training and testing
    trainloader, trainloader_test = diff_trainer.get_dataloaders()

    # 4. create the model - net and optimizer
    diff_trainer.set_net_and_optimizer()

    # 5. load checkpoint if exists, and feed the model weights and optimizer state dict
    diff_trainer.load_checkpoint()

    # 6. main training loop
    n_iter = diff_trainer.get_ckpt_iter() + 1
    n_iters = diff_trainer._n_iters
    while n_iter < n_iters + 1:
        trainer = train_class(sde, diff_trainer._criterion, diff_trainer._w_masked_pix) # create an instance of the training class, which will be used to calculate the training loss and test loss.
        ###################################### TRAIN ######################################
        n_iter = diff_trainer.train(trainer, trainloader, n_iter)
        ###################################### TEST ######################################
        diff_trainer.test(trainer, trainloader_test, n_iter)

    # Close logger
    if rank == 0:
        diff_trainer._writer.close()


@hydra.main(version_base=None, config_path="configs_Alon_Matan", config_name="config_dit_without-space-phoneme_on-masked-mel_for_inference")
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