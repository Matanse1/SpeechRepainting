# this file is an adapated version https://github.com/albertfgu/diffwave-sashimi, licensed
# under https://github.com/albertfgu/diffwave-sashimi/blob/master/LICENSE


import torch
from torch.utils.data.distributed import DistributedSampler
from .dataset_lipvoicer import get_dataset

def dataloader(dataset_cfg, batch_size, num_gpus):
    # train
    dataset = get_dataset(dataset_cfg, split='Old_train', return_mask_properties=False)
    # distributed sampler
    train_sampler = DistributedSampler(dataset) if num_gpus > 1 else None

    trainloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=False,
        drop_last=True,
        # collate_fn=custom_collate_fn
    )
    return trainloader

def custom_collate_fn(batch):
    melspecs, masked_conds, masks = zip(*batch)
    
    # Process masked_melspecs and masked_audio_times separately
    masked_melspecs = [cond[0] for cond in masked_conds]
    masked_audio_times = [cond[1] for cond in masked_conds]
    
    return torch.stack(list(melspecs)), [torch.stack(masked_melspecs), torch.stack(masked_audio_times)], torch.stack(list(masks))
