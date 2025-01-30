# this file is an adapated version https://github.com/albertfgu/diffwave-sashimi, licensed
# under https://github.com/albertfgu/diffwave-sashimi/blob/master/LICENSE


import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from .dataset_lipvoicer import get_dataset
import torch.nn.functional as F

def dataloader(dataset_cfg, batch_size, num_gpus, collate_fn=None, split='Train', return_true_text=False):
    # train
    dataset = get_dataset(dataset_cfg, split=split, return_mask_properties=False, return_true_text=return_true_text)
    # distributed sampler
    train_sampler = DistributedSampler(dataset) if num_gpus > 1 else None

    trainloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_fn
    )
    return trainloader

def custom_collate_fn(batch):
    melspecs, masked_conds, masks = zip(*batch)
    
    # Process masked_melspecs and masked_audio_times separately
    masked_melspecs = [cond[0] for cond in masked_conds]
    masked_audio_times = [cond[1] for cond in masked_conds]
    
    return torch.stack(list(melspecs)), [torch.stack(masked_melspecs), torch.stack(masked_audio_times)], torch.stack(list(masks))


class CollateFn(nn.Module):

    """ Collate samples to List / Dict

    Args:
        - inputs_params_: List / Dict of collate param for inputs
        - targets_params: List / Dict of collate param for targets

    Collate Params Dict:
        - axis: axis to select samples
        - padding: whether to pad samples
        - padding_value: padding token, default 0

    """

    def __init__(self, inputs_params=[{"axis": 0}], targets_params=[{"axis": 1}]):
        super(CollateFn, self).__init__()

        assert isinstance(inputs_params, dict) or isinstance(inputs_params, list) or isinstance(inputs_params, tuple)
        self.inputs_params = inputs_params
        assert isinstance(targets_params, dict) or isinstance(targets_params, list) or isinstance(targets_params, tuple)
        self.targets_params = targets_params

        # Default Params
        if  isinstance(inputs_params, dict):
            for params in self.inputs_params.values():
                if not "padding" in params:
                    params["padding"] = False
                if not "padding_value" in params:
                    params["padding_value"] = -1
                if not "start_token" in params:
                    params["start_token"] = None
                if not "end_number" in params:
                    params["end_number"] = None

            for params in self.targets_params.values():
                if not "padding" in params:
                    params["padding"] = False
                if not "padding_value" in params:
                    params["padding_value"] = 0
                if not "start_token" in params:
                    params["start_token"] = None
                if not "end_number" in params:
                    params["end_number"] = None

        else:
            for params in self.inputs_params:
                if not "padding" in params:
                    params["padding"] = False
                if not "padding_value" in params:
                    params["padding_value"] = -1
                if not "start_token" in params:
                    params["start_token"] = None
                if not "end_number" in params:
                    params["end_number"] = None

            for params in self.targets_params:
                if not "padding" in params:
                    params["padding"] = False
                if not "padding_value" in params:
                    params["padding_value"] = 0
                if not "start_token" in params:
                    params["start_token"] = None
                if not "end_number" in params:
                    params["end_number"] = None

    def forward(self, samples):
        #the samples are [(phoneme_target, melspec, masked_melspec, masked_audio_time, mask)1, (phoneme_target, melspec, masked_melspec, masked_audio_time, mask)2]
        return {"inputs": self.collate(samples, self.inputs_params), "targets": self.collate(samples, self.targets_params)}
    
    def collate(self, samples, collate_params):

        def pad_last_dim(tensor, pad_size, pad_value=0):
            # Create the padding tuple dynamically based on the number of dimensions
            pad = [0, pad_size]  # Only pad the last dimension
            pad = tuple(pad) + (0,) * (2 * (tensor.dim() - 1)) 
            
            # Apply padding
            return F.pad(tensor, pad, value=pad_value)
    
        def process_single_collate(collate, params):
            original_lengths = torch.tensor([item.shape[-1] for item in collate])
            max_length = params["max_length"] #max(original_lengths)
            # # Start Token
            # if params["start_token"]:
            #     collate = [torch.cat([params["start_token"] * item.new_ones(1), item]) for item in collate]
            #     original_lengths += 1

            # End Token
            # print(f"The params are: {params}")
            if params["end_number"] is not None:
                if params["end_number"] == 'min':
                    collate_new = [pad_last_dim(item, max_length - item.shape[-1], pad_value=item.min()) for item in collate]
                # elif params["end_number"]:
                else:
                    collate_new = [pad_last_dim(item, max_length - item.shape[-1], pad_value=params["end_number"]) for item in collate]
                mask = [pad_last_dim(torch.ones_like(item), max_length - item.shape[-1], pad_value=0) for item in collate] # this mask is used to mask the padding

            # # Padding
            # if params["padding"]:
            #     collate = torch.nn.utils.rnn.pad_sequence(collate, batch_first=True, padding_value=params["padding_value"])
            #     mask = create_mask(collate, original_lengths)
            # else:
            #     collate = torch.stack(collate, axis=0)
            #     mask = None

            return collate_new, mask

        # # Dict
        # if isinstance(collate_params, dict):
        #     collates = {}
        #     masks = {}
        #     for name, params in collate_params.items():
        #         collate = [sample[params["axis"]] for sample in samples]
        #         collate, mask = process_single_collate(collate, params)
        #         collates[name] = collate
        #         if mask is not None:
        #             masks[name] = mask

        # List
        
        collates = []
        masks = []
        for params in collate_params:
            collate = [sample[params["axis"]] for sample in samples]
            if "text" in params: #for text i dont want to pad, i want just list of str so the tokenizer can be applied on this
                collates.append(collate)
                masks.append(None)
            else:
                collate, mask = process_single_collate(collate, params)
                collates.append(torch.stack(collate, dim=0))
                # if mask is not None:
                masks.append(torch.stack(mask, dim=0))

        # Tuple
        # elif isinstance(collate_params, tuple):
        #     collates = []
        #     masks = []
        #     for params in collate_params:
        #         collate = [sample[params["axis"]] for sample in samples]
        #         collate, mask = process_single_collate(collate, params)
        #         collates.append(collate)
        #         if mask is not None:
        #             masks.append(mask)
        #     collates = tuple(collates)
        #     masks = tuple(masks) if masks else None

        collates = collates[0] if len(collates) == 1 else collates
        masks = masks[0] if len(masks) == 1 else masks if masks else None

        return collates, masks
