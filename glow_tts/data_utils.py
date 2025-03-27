import random
import numpy as np
import torch
import torch.utils.data
import json
import commons 
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence, cmudict
from text.symbols import symbols
import os
from utils import HParams
from pathlib import Path
import math
import pandas as pd
from scipy.io.wavfile import read
from torch import nn
import ast
from torch.nn import functional as F
from torch.utils.data import DataLoader

def set_seed(seed):
    # Set the seed for Python's built-in random module
    random.seed(seed)
    
    # Set the seed for NumPy's random module
    np.random.seed(seed)
    
    # Set the seed for PyTorch's random module
    torch.manual_seed(seed)
    
    # If you are using CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU.
    
    # Ensure reproducibility in some other operations in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def normalise_mel(melspec, min_val=math.log(1e-5)): 
    melspec = ((melspec - min_val) / (-min_val / 2)) - 1    #log(1e-5)~2 --> -1~1
    return melspec

class MyTextMelLoader(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, split, sampling_rate, min_block_size, max_block_size, min_spacing, csv_loc, add_blank,
                 audio_stft_hop, base_data_dir, num4empty_str, num_blocks, rand_num_blocks, return_mask_properties, return_target_time=False, return_full_phoneme_squence=False,  **kwargs):
        split = split.capitalize()
        self.base_data_dir = base_data_dir
        self.add_blank = add_blank
        self.return_target_time = return_target_time
        self.return_mask_properties = return_mask_properties
        self.num_blocks = num_blocks
        self.rand_num_blocks = rand_num_blocks
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size
        self.min_spacing = min_spacing
        self.return_full_phoneme_squence = return_full_phoneme_squence
        self.num4empty_str = num4empty_str
        try:
            float(num4empty_str)
            is_number = True  # Conversion succeeded, it's a number
        except ValueError:
            is_number = False  # Conversion failed, not a number
        if is_number:
            self.num4empty = float(num4empty_str)
        
        self.audio_stft_hop = audio_stft_hop
        set_seed(1234)
        self.sampling_rate = sampling_rate
        self.split = split
        self.csv_path = Path(csv_loc) / f'{split}_new.csv'
        self.csv_df = pd.read_csv(self.csv_path, delimiter="|")
    
    def __len__(self):
        #return 100000
        # return 100
        # if self.split=='Train':
        #     return 300
        return len(self.csv_df)
    
    def get_phonemes(self, index):
        phoneme_sequence_list_without_silence = ast.literal_eval(self.csv_df.loc[index, "phoneme_sequence_list"])
        phoneme_duration_list_without_silence = ast.literal_eval(self.csv_df.loc[index, "phoneme_duration_list"])
        phoneme_int_list_without_silence = ast.literal_eval(self.csv_df.loc[index, "phoneme_int_list"])
        
        ##with silence
        phoneme_with_silence_list = ast.literal_eval(self.csv_df.loc[index, "phoneme_int_list_with_silence"])
        durations_with_silence = ast.literal_eval(self.csv_df.loc[index, "phoneme_duration_list_with_silence"])
        # phoneme_int_with_silence_list = ast.literal_eval(self.csv_df.loc[index, "durations_without_silence"])
        if self.add_blank:
            interspersed_phoneme_int_list_without_silence = commons.intersperse(phoneme_int_list_without_silence, 1) # add a blank token, whose id number is 1 #TODO try 0 (pay attention the padding is also zero so maybe it's wrong to do that)
            interspersed_phoneme_duration = commons.get_interspersed_phoneme_sequence(phoneme_with_silence_list, durations_with_silence, phoneme_duration_list_without_silence) # this list contain the duration of each phoneme interspersed with the silence token such that the duration of the silence token is also included
            true_attention_matrix = commons.create_attention_matrix(interspersed_phoneme_duration)
            true_attention_matrix = torch.FloatTensor(true_attention_matrix)
            interspersed_phoneme_duration = torch.IntTensor(interspersed_phoneme_duration)
            interspersed_phoneme_int_list_without_silence = torch.IntTensor(interspersed_phoneme_int_list_without_silence)
            return interspersed_phoneme_duration, interspersed_phoneme_int_list_without_silence, true_attention_matrix
        
        true_attention_matrix = commons.create_attention_matrix(phoneme_duration_list_without_silence)
        true_attention_matrix = torch.FloatTensor(true_attention_matrix)
        phoneme_duration_list_without_silence = torch.IntTensor(phoneme_duration_list_without_silence)
        phoneme_int_list_without_silence = torch.IntTensor(phoneme_int_list_without_silence)
        return phoneme_duration_list_without_silence, phoneme_int_list_without_silence, true_attention_matrix
    
    def __getitem__(self, index):
        phoneme_duration_list, phoneme_int_list, true_attention_matrix = self.get_phonemes(index)

        melspec_path =  Path(self.base_data_dir) / self.csv_df.loc[index, "mel_spectrum_path"] #npz file
        melspec = torch.load(melspec_path, weights_only=False)
        melspec = normalise_mel(melspec)
        
        audio_path =  Path(self.base_data_dir) / self.csv_df.loc[index, "wav_path"] # wav file
        _, audio_time = read(audio_path)
        audio_time = torch.from_numpy(audio_time.astype(np.float32))
        audio_time =  0.9 * audio_time / audio_time.abs().max() #because we took the original librispeech dataset, the audio is not norlaized to [-1, 1]
        if self.return_full_phoneme_squence:
                full_phoneme_squence_path = Path(self.base_data_dir) / self.csv_df.loc[index, "phoneme_frame_path"] #npy file
                full_phoneme_squence = np.load(full_phoneme_squence_path)
                full_phoneme_squence = torch.from_numpy(full_phoneme_squence)
        if self.return_mask_properties:
            masked_melspec, mask, masked_audio_time, block_size_list, num_blocks = self.create_masked_melspec(melspec, audio_time)
            if self.return_target_time:
                return {"phonemes": [phoneme_duration_list, phoneme_int_list, true_attention_matrix], "mel_spectrum": [audio_time, melspec, masked_melspec, masked_audio_time, mask, block_size_list, num_blocks]}
            if self.return_full_phoneme_squence:
                return {"phonemes": [phoneme_duration_list, phoneme_int_list, full_phoneme_squence, true_attention_matrix], "mel_spectrum": [melspec, masked_melspec, masked_audio_time, mask, block_size_list, num_blocks]}
            return {"phonemes": [phoneme_duration_list, phoneme_int_list], "mel_spectrum": [melspec, masked_melspec, masked_audio_time, mask, block_size_list, num_blocks]}
        else:  
            masked_melspec, mask, masked_audio_time = self.create_masked_melspec(melspec, audio_time)
            if self.return_target_time:
                return {"phonemes": [phoneme_duration_list, phoneme_int_list, true_attention_matrix], "mel_spectrum": [audio_time, melspec, masked_melspec, masked_audio_time, mask]}      
            if self.return_full_phoneme_squence:
                return {"phonemes": [phoneme_duration_list, phoneme_int_list, full_phoneme_squence, true_attention_matrix], "mel_spectrum": [melspec, masked_melspec, masked_audio_time, mask]} 
            return {"phonemes": [phoneme_duration_list, phoneme_int_list, true_attention_matrix], "mel_spectrum": [melspec, masked_melspec, masked_audio_time, mask]}

    def create_masked_melspec(self, melspec, audio_time):
        
        melspec = melspec.clone()
        audio_time = audio_time.clone()
        # print(audio_time.shape)
        mask = torch.ones(melspec.shape[-1])
        hop_length = self.audio_stft_hop
        edge = int(0.5*self.sampling_rate/self.audio_stft_hop) # 0.5sec of edge
        if self.num4empty_str == 'min':
            min_melspec = torch.min(melspec)
            # min_melspec -= 1
            self.num4empty = min_melspec 
        shape = melspec.shape
        # Determine the number of blocks
        block_size_list = []
        if self.rand_num_blocks:
            # self.num_blocks = torch.randint(2, (shape[1] - self.min_block_size) // (self.max_block_size + self.min_spacing) + 1, (1,)).item()
            if (shape[-1] - 2 * edge) // (self.max_block_size + self.min_spacing) + 1 <= 1:
                self.num_blocks = 0
                if self.return_mask_properties:
                    return melspec, mask, audio_time, block_size_list, self.num_blocks
                else:
                    return melspec, mask, audio_time
                
            else:
                # self.num_blocks = torch.randint(max(((shape[1] - 2 * edge) // (self.max_block_size + self.min_spacing))//2, 1),
                #                             (shape[1] - 2 * edge) // (self.max_block_size + self.min_spacing) + 1, (1,)).item()
                self.num_blocks = torch.randint(max(((shape[-1] - 2 * edge) // (self.max_block_size + self.min_spacing))//2, 1),
                                            (shape[-1] - 2 * edge) // (self.max_block_size + self.min_spacing) + 1, (1,)).item()
                # print(f"num_blocks is {self.num_blocks}. /t from {max(((shape[1] - 2 * edge) // (self.max_block_size + self.min_spacing))//2, 1)} to {(shape[1] - 2 * edge) // (self.max_block_size + self.min_spacing) + 1}")
        # print(num_blocks)
        # Keep track of the end position of the last block to ensure spacing
        
        # edge = int(0.5*16000/hop_length) # 0.5sec of edge
        last_block_end = -self.min_spacing + edge
        for i in range(self.num_blocks):
            #print(i)
            # Random block size
            block_size = torch.randint(self.min_block_size, self.max_block_size + 1, (1,)).item()
            
            # Ensure valid start_pos to avoid overlapping
            if last_block_end + self.min_spacing + block_size >= shape[-1]-edge:
                # print("break")
                break
            if i == self.num_blocks-1:
                # start_pos = torch.randint(last_block_end + self.min_spacing, last_block_end + self.min_spacing + shape[1]//self.num_blocks - block_size - edge + 1, (1,)).item()
                start_pos = torch.randint(last_block_end + self.min_spacing, last_block_end + self.min_spacing + (shape[-1] - 2 * edge)//self.num_blocks - block_size + 1, (1,)).item()
            else:
                start_pos = torch.randint(last_block_end + self.min_spacing, last_block_end + self.min_spacing + (shape[-1] - 2 * edge)//self.num_blocks - block_size + 1, (1,)).item()
            # print(f"start is {start_pos}")
            # Set the mask to 0 for the current block
            # mask[:, start_pos:start_pos + block_size] = 0
            if start_pos + block_size > shape[-1]:
                block_size = shape[-1] - start_pos
            if self.num4empty_str == 'randn' and block_size > 0:
                self.num4empty = torch.randn((shape[0], block_size))
                # print("num4empty is randn")
            if block_size > 0:
                melspec[:, start_pos:start_pos + block_size] = self.num4empty
            mask[start_pos:start_pos + block_size] = 0
            audio_time[int(start_pos*hop_length):int((start_pos + block_size)*hop_length)] = 0
            # Update the end position of the last block
            block_size_list.append(block_size)
            last_block_end = start_pos + block_size
        
        if self.return_mask_properties:
            return melspec, mask, audio_time, block_size_list, self.num_blocks
        else:
            return melspec, mask, audio_time


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.add_noise = hparams.add_noise
        self.add_blank = getattr(hparams, "add_blank", False) # improved version
        if getattr(hparams, "cmudict_path", None) is not None:
          self.cmudict = cmudict.CMUDict(hparams.cmudict_path)
        self.stft = commons.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return (text, mel)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            if self.add_noise:
                audio = audio + torch.rand_like(audio)
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text):
        text_norm = text_to_sequence(text, self.text_cleaners, getattr(self, "cmudict", None))
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, len(symbols)) # add a blank token, whose id number is len(symbols)
        text_norm = torch.IntTensor(text_norm)
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


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

    def __init__(self, inputs_params=[{"type": "mel_spectrum", "axis": 0, "max_length": 1701}], targets_params=[{"type": "phonemes", "axis": 1, "max_length": 300}]):
        super(CollateFn, self).__init__()

        assert isinstance(inputs_params, dict) or isinstance(inputs_params, list) or isinstance(inputs_params, tuple)
        self.inputs_params = inputs_params
        assert isinstance(targets_params, dict) or isinstance(targets_params, list) or isinstance(targets_params, tuple)
        self.targets_params = targets_params

        # Default Params
        
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
        
        def pad_2last_dim(tensor, pad_size_m1, pad_size_m2, pad_value=0):
            # Create the padding tuple dynamically based on the number of dimensions
            pad_m1 = [0, pad_size_m1]
            pad_m2 = [0, pad_size_m2]
            pad = tuple(pad_m1) + tuple(pad_m2) + (0,) * (2 * (tensor.dim() - 2))
            return F.pad(tensor, pad, value=pad_value)
    
        def process_single_collate(collate, params):
            original_lengths = torch.tensor([item.shape[-1] for item in collate])
            
            # # Start Token
            # if params["start_token"]:
            #     collate = [torch.cat([params["start_token"] * item.new_ones(1), item]) for item in collate]
            #     original_lengths += 1

            # End Token
            # print(f"The params are: {params}")
            if params["end_number"] is not None:
                max_length = params["max_length"] #max(original_lengths)
                if params["end_number"] == 'min':
                    collate_new = [pad_last_dim(item, max_length - item.shape[-1], pad_value=item.min()) for item in collate]
                # elif params["end_number"]:
                else:
                    collate_new = [pad_last_dim(item, max_length - item.shape[-1], pad_value=params["end_number"]) for item in collate]
                mask = [pad_last_dim(torch.ones_like(item), max_length - item.shape[-1], pad_value=0) for item in collate] # this mask is used to mask the padding
                input_length_original = [torch.tensor(item.shape[-1]) for item in collate]
            elif (params["end_number_m1"] is not None) and (params["end_number_m2"] is not None):
                max_length_m1 = params["max_length_m1"] #max(original_lengths)
                max_length_m2 = params["max_length_m2"] #max(original_lengths)
                collate_new = [pad_2last_dim(item, max_length_m1 - item.shape[-1], max_length_m2 - item.shape[-2], pad_value=params["end_number_m1"]) for item in collate]
                mask = [pad_2last_dim(torch.ones_like(item), max_length_m1 - item.shape[-1], max_length_m2 - item.shape[-2], pad_value=0) for item in collate] # this mask is used to mask the padding
                input_length_original = [torch.tensor([item.shape[-2], item.shape[-1]]) for item in collate]
                
            return collate_new, mask, input_length_original


        # List
        if isinstance(collate_params, list):
            collates = []
            masks = []
            inputs_length_original = []
            for params in collate_params:
                collate = [sample[params["type"]][params["axis"]] for sample in samples]
                collate, mask, input_length_original = process_single_collate(collate, params)
                collates.append(torch.stack(collate, dim=0))
                # if mask is not None:
                masks.append(torch.stack(mask, dim=0))
                inputs_length_original.append(torch.stack(input_length_original, dim=0))



        # collates = collates[0] if len(collates) == 1 else collates
        # masks = masks[0] if len(masks) == 1 else masks if masks else None
        # inputs_length_original = inputs_length_original[0] if len(inputs_length_original) == 1 else inputs_length_original if inputs_length_original else None

        return collates, masks, inputs_length_original


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per step
    """
    def __init__(self, n_frames_per_step=1):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, output_lengths


"""Multi speaker version"""
class TextMelSpeakerLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_sid_text, hparams):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.add_noise = hparams.add_noise
        self.add_blank = getattr(hparams, "add_blank", False) # improved version
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)
        if getattr(hparams, "cmudict_path", None) is not None:
          self.cmudict = cmudict.CMUDict(hparams.cmudict_path)
        self.stft = commons.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)

        self._filter_text_len()
        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)

    def _filter_text_len(self):
      audiopaths_sid_text_new = []
      for audiopath, sid, text in self.audiopaths_sid_text:
        if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
          audiopaths_sid_text_new.append([audiopath, sid, text])
      self.audiopaths_sid_text = audiopaths_sid_text_new

    def get_mel_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath, sid, text = audiopath_sid_text[0], audiopath_sid_text[1], audiopath_sid_text[2]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        sid = self.get_sid(sid)
        return (text, mel, sid)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            if self.add_noise:
                audio = audio + torch.rand_like(audio)
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text):
        text_norm = text_to_sequence(text, self.text_cleaners, getattr(self, "cmudict", None))
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, len(symbols)) # add a blank token, whose id number is len(symbols)
        text_norm = torch.IntTensor(text_norm)
        return text_norm

    def get_sid(self, sid):
        sid = torch.IntTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        return self.get_mel_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)


class TextMelSpeakerCollate():
    """ Zero-pads model inputs and targets based on number of frames per step
    """
    def __init__(self, n_frames_per_step=1):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded & sid
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            output_lengths[i] = mel.size(1)
            sid[i] = batch[ids_sorted_decreasing[i]][2]

        return text_padded, input_lengths, mel_padded, output_lengths, sid


if __name__=='__main__':
    config_save_path = "/home/dsi/moradim/SpeechRepainting/glow-tts/configs/my_base_blank.json"
    with open(config_save_path, "r") as f:
      data = f.read()
    config = json.loads(data)
    hps = HParams(**config)
    train_dataset = MyTextMelLoader(**hps.data, split='Train',  return_mask_properties=False) 
    data = train_dataset[0]
    n_gpus = torch.cuda.device_count()
    train_sampler = torch.utils.data.distributed.DistributedSampler(
      train_dataset,
      num_replicas=n_gpus,
      rank=0,
      shuffle=True)
    collate_fn = CollateFn(inputs_params=[{"type": "mel_spectrum", "axis": 0, "max_length": 1701, "end_number": 0}, # correspond to melspec
                                    {"type": "mel_spectrum", "axis": 1, "max_length": 1701, "end_number": 0}, # correspond to masked_melspec
                                    {"type": "mel_spectrum", "axis": 3, "max_length": 1701, "end_number": 1}], # correspond to mask
                        targets_params=[{"type": "phonemes", "axis": 2, "max_length": 300, "end_number": 0}]) # correspond to phoneme_int_list
    dataloader = DataLoader(train_dataset, num_workers=8, shuffle=False,
      batch_size=hps.train.batch_size, pin_memory=True,
      drop_last=True, collate_fn=collate_fn)#, sampler=train_sampler)
    
    for i, data in enumerate(dataloader):
        print(i)
        print(data)
        break