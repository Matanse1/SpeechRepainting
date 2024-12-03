# some parts of this code were borrowed from https://github.com/facebookresearch/VisualVoice/tree/main
# under the licence https://github.com/facebookresearch/VisualVoice/blob/main/LICENSE


import os
import random
import torch
import torch.nn as nn
import torch.utils.data
from scipy.io.wavfile import read
from glob import glob
from pathlib import Path
import numpy as np
# from PIL import Image, ImageEnhance
# from .video_reader import VideoReader
# from .lipreading_utils import *
# import cv2
import torchaudio
# import torchvision.transforms as transforms
from .stft import normalise_mel
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
import pickle
import ast


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

def create_mask(shape, min_block_size=35, max_block_size=65, min_spacing=30):
        # Initialize the mask with ones
        mask = torch.ones((shape[0], shape[1]))
        
        # Determine the number of blocks
        num_blocks = torch.randint(2, (shape[1] - min_block_size) // (max_block_size + min_spacing) + 1, (1,)).item()
        # Keep track of the end position of the last block to ensure spacing
        edge = int(0.5*16000/160) #0.5sec of edge
        last_block_end = -min_spacing + edge
        
        for i in range(num_blocks):
            #print(i)
            # Random block size
            block_size = torch.randint(min_block_size, max_block_size + 1, (1,)).item()
            
            # Ensure valid start_pos to avoid overlapping
            if last_block_end + min_spacing + block_size >= shape[1]-edge:
                print("break")
                break
            if i == num_blocks-1:
                start_pos = torch.randint(last_block_end + min_spacing, last_block_end + min_spacing + shape[1]//num_blocks - block_size - edge + 1, (1,)).item()
            else:
                start_pos = torch.randint(last_block_end + min_spacing, last_block_end + min_spacing + shape[1]//num_blocks - block_size + 1, (1,)).item()
            print(f"start is {start_pos}")
            # Set the mask to 0 for the current block
            mask[:, start_pos:start_pos + block_size] = 0
            
            # Update the end position of the last block
            last_block_end = start_pos + block_size
        
        return mask

def files_to_list(data_path, suffix):
    """
    Load all .wav files in data_path
    """
    files = glob(os.path.join(data_path, f'**/*.{suffix}'), recursive=True)
    return files

def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = read(full_path)
    return torch.from_numpy(data).float(), sampling_rate


class LipVoicerDataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, split, videos_dir, mouthrois_dir, audios_dir, sampling_rate, min_block_size, max_block_size, 
                 min_spacing, audio_stft_hop):
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size
        self.min_spacing = min_spacing
        self.mouthrois_dir = mouthrois_dir
        if "LRS3" in videos_dir:
            self.ds_name = "LRS3"
            split_dir = ['pretrain','trainval'] if split in ['train', 'val'] else ['test']
            self.videos_dir = videos_dir
            self.audios_dir = audios_dir
            
            self.moutroi_files = []
            for s in split_dir:
                _mouthrois_dir = os.path.join(mouthrois_dir, s)
                self.moutroi_files += files_to_list(_mouthrois_dir, 'npz')

        elif "LRS2" in videos_dir:
            self.ds_name = "LRS2"
            split_dir = ['main']
            if split == 'train':
                split_dir.append('pretrain')
            
            self.moutroi_files = []
            for s in split_dir:
                if s == "main":
                    if split == "pretrain":
                        videos_list_file = os.path.join(videos_dir, "train.txt")
                    else:
                        videos_list_file = os.path.join(videos_dir, split+".txt")
                    with open(videos_list_file, "r") as f:
                        _video_ids = f.readlines()
                    for _vid in _video_ids:
                        mouthroi_file = os.path.join(mouthrois_dir, s, _vid.strip("\n")+".npz")
                        mouthroi_file = mouthroi_file.replace(" NF", "").replace(" MV", "")
                        if os.path.isfile(mouthroi_file):
                            self.moutroi_files.append(mouthroi_file)
                elif s == "pretrain":
                    self.moutroi_files += glob(os.path.join(mouthrois_dir, "pretrain", "**/*.npz"), recursive=True)
                    
            self.videos_dir = videos_dir
            self.audios_dir = audios_dir
            self.moutroi_files = sorted(self.moutroi_files)        
        
        self.test = True if split=='test' else False
        self.videos_window_size = videos_window_size
        self.audio_stft_hop = audio_stft_hop
        random.seed(1234)
        random.shuffle(self.moutroi_files)
        self.sampling_rate = sampling_rate

        self.mouthroi_transform = self.get_mouthroi_transform()[split]
        self.face_image_transform = self.get_face_image_transform()

    def __getitem__(self, index):
        
        while True:
            # Get paths
            mouthroi_filename = self.moutroi_files[index]
            pfilename = Path(mouthroi_filename)
            if self.ds_name in ["LRS3", "LRS2"]:
                video_id = '/'.join([pfilename.parts[-2], pfilename.stem])
                video_filename = mouthroi_filename.replace(self.mouthrois_dir, self.videos_dir).replace('.npz','.mp4')
                melspec_filename = mouthroi_filename.replace(self.mouthrois_dir, self.audios_dir).replace('.npz','.wav.spec')
            
            # Get mouthroi
            mouthroi = np.load(mouthroi_filename)['data']
            if mouthroi.shape[0] >= self.videos_window_size or self.test:
                break
            else:
                index = random.randrange(len(self.moutroi_files))
        melspec = torch.load(melspec_filename)
        face_image = self.load_frame(video_filename)
        
        video = cv2.VideoCapture(video_filename)
        info = {'audio_fps': self.sampling_rate, 'video_fps': video.get(cv2.CAP_PROP_FPS)}

        if self.test:
            audio, fs = torchaudio.load(melspec_filename.replace('.spec', ''))
            text_filename = video_filename.replace(".mp4", ".txt")
            text = self.preprocess_text(text_filename)

            # Normalisations & transforms
            audio = audio / 1.1 / audio.abs().max()
            face_image = self.face_image_transform(face_image)
            mouthroi = torch.FloatTensor(self.mouthroi_transform(mouthroi)).unsqueeze(0)
            melspec = normalise_mel(melspec)
            return (melspec, audio, mouthroi, face_image, text, video_id)
        else:

            # Get corresponding crops
            mouthroi, melspec = self.extract_window(mouthroi, melspec, info)
            if mouthroi.shape[0] < self.videos_window_size:
                return self.__getitem__(random.randrange(len(self)))
            
            # Augmentations
            face_image = self.augment_image(face_image)

            # Noramlisations & Transforms
            face_image = self.face_image_transform(face_image)
            mouthroi = torch.FloatTensor(self.mouthroi_transform(mouthroi)).unsqueeze(0)   # add channel dim
            melspec = normalise_mel(melspec)
            return (melspec, mouthroi, face_image)

    def __len__(self):
        return len(self.moutroi_files)

    
    

    def extract_window(self, mouthroi, mel, info):
        hop = self.audio_stft_hop

        # vid : T,C,H,W
        vid_2_aud = info['audio_fps'] / info['video_fps'] / hop

        st_fr = random.randint(0, mouthroi.shape[0] - self.videos_window_size)
        mouthroi = mouthroi[st_fr:st_fr + self.videos_window_size]

        st_mel_fr = int(st_fr * vid_2_aud)
        mel_window_size = int(self.videos_window_size * vid_2_aud)

        mel = mel[:, st_mel_fr:st_mel_fr + mel_window_size]

        return mouthroi, mel

    @staticmethod
    def load_frame(clip_path):
        video_reader = VideoReader(clip_path, 1)
        start_pts, time_base, total_num_frames = video_reader._compute_video_stats()
        end_frame_index = total_num_frames - 1
        if end_frame_index < 0:
            clip = video_reader.read_video_only(start_pts, 1)
        else:
            clip = video_reader.read_video_only(random.randint(0, end_frame_index) * time_base, 1)
        frame = Image.fromarray(np.uint8(clip[0].to_rgb().to_ndarray())).convert('RGB')
        return frame

    @staticmethod
    def augment_image(image):
        if(random.random() < 0.5):
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.random()*0.6 + 0.7)
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(random.random()*0.6 + 0.7)
        return image

    @staticmethod
    def get_mouthroi_transform():
        # -- preprocess for the video stream
        preprocessing = {}
        # -- LRW config
        crop_size = (88, 88)
        (mean, std) = (0.421, 0.165)
        preprocessing['train'] = Compose([
                                    Normalize( 0.0,255.0 ),
                                    RandomCrop(crop_size),
                                    HorizontalFlip(0.5),
                                    Normalize(mean, std) ])
        preprocessing['val'] = Compose([
                                    Normalize( 0.0,255.0 ),
                                    CenterCrop(crop_size),
                                    Normalize(mean, std) ])
        preprocessing['test'] = preprocessing['val']
        return preprocessing
    
    @staticmethod
    def get_face_image_transform():
        normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )
        vision_transform_list = [transforms.Resize(224), transforms.ToTensor(), normalize]
        vision_transform = transforms.Compose(vision_transform_list)
        return vision_transform
    
    @staticmethod
    def preprocess_text(txt_filename):
        with open(txt_filename, "r") as f:
            txt = f.readline()[7:]  # discard 'Text:  ' prefix
        txt = txt.replace("{LG}", "")  # remove laughter
        txt = txt.replace("{NS}", "")  # remove noise
        txt = txt.replace("\n", "")
        txt = txt.replace("  ", " ")
        txt = txt.lower().strip()
        return txt


class SpeechRepaingingDataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, split, sampling_rate, min_block_size, max_block_size, min_spacing,
                 audio_stft_hop, base_data_dir, num4empty_str, num_blocks, rand_num_blocks, return_mask_properties, return_target_time=False):
        split = split.capitalize()
        self.mel_dir = Path(base_data_dir, split, "mel")
        self.audio_dir = Path(base_data_dir, split, "audio_final")
        self.return_target_time = return_target_time
        self.return_mask_properties = return_mask_properties
        self.num_blocks = num_blocks
        self.rand_num_blocks = rand_num_blocks
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size
        self.min_spacing = min_spacing
        self.num4empty_str = num4empty_str
        try:
            float(num4empty_str)
            is_number = True  # Conversion succeeded, it's a number
        except ValueError:
            is_number = False  # Conversion failed, not a number
        if is_number:
            self.num4empty = float(num4empty_str)
        
        self.test = True if split=='Test' else False
        self.audio_stft_hop = audio_stft_hop
        set_seed(1234)
        self.sampling_rate = sampling_rate
        self.csv_path = Path(base_data_dir, split, "room_parameters.csv")
        self.audio_csv = pd.read_csv(self.csv_path)
    
    def __len__(self):
        return 100000
        #return len(self.audio_csv)
    
    def __getitem__(self, index):
        
        melspec_filename = Path(self.mel_dir, f"example_{index}.npz")
        melspec = torch.load(melspec_filename)
        melspec = normalise_mel(melspec)
        
        audio_dir_filename = Path(self.audio_dir, f"example_{index}.wav")
        _, audio_time = read(audio_dir_filename)
        audio_time = torch.from_numpy(audio_time.astype(np.float32))
        if self.return_mask_properties:
            masked_melspec, mask, masked_audio_time, block_size_list, num_blocks = self.create_masked_melspec(melspec, audio_time)
            if self.return_target_time:
                return (audio_time, melspec, masked_melspec, masked_audio_time, mask, block_size_list, num_blocks)
            return (melspec, masked_melspec, masked_audio_time, mask, block_size_list, num_blocks)
        else:  
            masked_melspec, mask, masked_audio_time = self.create_masked_melspec(melspec, audio_time)
            if self.return_target_time:
                return (audio_time, melspec, masked_melspec, masked_audio_time, mask)
            return (melspec, masked_melspec, masked_audio_time, mask)

    

    def create_mask(self, shape):
        # Initialize the mask with ones
        mask = torch.ones((shape[0], shape[1]))
        
        # Determine the number of blocks
        num_blocks = torch.randint(1, (shape[1] - self.min_block_size) // (self.max_block_size + self.min_spacing) + 1, (1,)).item()
        # Keep track of the end position of the last block to ensure spacing
        edge = int(0.5*16000/160) #0.5sec of edge
        last_block_end = -self.min_spacing + edge
        
        for i in range(num_blocks):
            #print(i)
            # Random block size
            block_size = torch.randint(self.min_block_size, self.max_block_size + 1, (1,)).item()
            
            # Ensure valid start_pos to avoid overlapping
            if last_block_end + self.min_spacing + block_size >= shape[1]-edge:
                #print("break")
                break
            if i == num_blocks-1:
                start_pos = torch.randint(last_block_end + self.min_spacing, last_block_end + self.min_spacing + shape[1]//num_blocks - block_size - edge + 1, (1,)).item()
            else:
                start_pos = torch.randint(last_block_end + self.min_spacing, last_block_end + self.min_spacing + shape[1]//num_blocks - block_size + 1, (1,)).item()
            #print(f"start is {start_pos}")
            # Set the mask to 0 for the current block
            mask[:, start_pos:start_pos + block_size] = 0
            
            # Update the end position of the last block
            last_block_end = start_pos + block_size
        
        return mask
    
    def create_masked_melspec(self, melspec, audio_time):
        
        melspec = melspec.clone()
        audio_time = audio_time.clone()
        mask = torch.ones(melspec.shape)
        if self.num4empty_str == 'min':
            min_melspec = torch.min(melspec)
            min_melspec -= 1
            self.num4empty = min_melspec 
        shape = melspec.shape
        # Determine the number of blocks
        if self.rand_num_blocks:
            self.num_blocks = torch.randint(2, (shape[1] - self.min_block_size) // (self.max_block_size + self.min_spacing) + 1, (1,)).item()
        # print(num_blocks)
        # Keep track of the end position of the last block to ensure spacing
        hop_length = 160
        edge = int(0.5*16000/hop_length) # 0.5sec of edge
        last_block_end = -self.min_spacing + edge
        block_size_list = []
        for i in range(self.num_blocks):
            #print(i)
            # Random block size
            block_size = torch.randint(self.min_block_size, self.max_block_size + 1, (1,)).item()
            block_size_list.append(block_size)
            # Ensure valid start_pos to avoid overlapping
            if last_block_end + self.min_spacing + block_size >= shape[1]-edge:
                # print("break")
                break
            if i == self.num_blocks-1:
                start_pos = torch.randint(last_block_end + self.min_spacing, last_block_end + self.min_spacing + shape[1]//self.num_blocks - block_size - edge + 1, (1,)).item()
            else:
                start_pos = torch.randint(last_block_end + self.min_spacing, last_block_end + self.min_spacing + shape[1]//self.num_blocks - block_size + 1, (1,)).item()
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
            mask[:, start_pos:start_pos + block_size] = 0
            audio_time[int(start_pos*hop_length):int((start_pos + block_size)*hop_length)] = 0
            # Update the end position of the last block
            last_block_end = start_pos + block_size
        
        if self.return_mask_properties:
            return melspec, mask, audio_time, block_size_list, self.num_blocks
        else:
            return melspec, mask, audio_time
        
        
class ExplosionSpeechRepaingingDataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, split, sampling_rate, audio_stft_hop, base_data_dir):
        split = split.capitalize()
        self.mel_dir = Path(base_data_dir, split, "mel")
        self.audio_dir = Path(base_data_dir, split, "audio")

        
        self.test = True if split=='Test' else False
        self.audio_stft_hop = audio_stft_hop
        set_seed(1234)
        self.sampling_rate = sampling_rate
        self.csv_info = pd.read_csv(Path(base_data_dir, split, "explosions.csv"), delimiter="|")
    
    def __len__(self):
        return 100000
        #return len(self.audio_csv)
    
    def __getitem__(self, index):
        explosions_length = self.csv_info.loc[index, "explosions_length"] # in samples
        start_explosions = self.csv_info.loc[index, "start_explosions_original"] # in samples
        filename = Path(self.mel_dir, f"example_{index}.npz")
        data = torch.load(filename)
        speech_melspec = data["speech_melspec"]
        mix_melspec = data["mix_melspec"] # mix of speech and explosions
        filepath = Path(self.audio_dir, f"example_{index}.pkl")
        with open(filepath, 'rb') as f:
            mix, _, masked_speech_time, _, _ = pickle.load(f)
        mix_time = torch.from_numpy(mix)
        masked_speech_time = torch.from_numpy(masked_speech_time)
        masked_speech = data["masked_speech"]
        speech_melspec = normalise_mel(speech_melspec)
        mix_melspec = normalise_mel(mix_melspec)
        masked_speech = normalise_mel(masked_speech)
        
        # For activity of the explosion
        start_explosions = ast.literal_eval(start_explosions)
        explosions_length = ast.literal_eval(explosions_length)
        
        explosions_activity = torch.zeros_like(speech_melspec)

        
        for start, length in zip(start_explosions, explosions_length):
            start_frame = self.time_to_frames(start, hop_length=self.audio_stft_hop)
            length_frame = self.time_to_frames(length, hop_length=self.audio_stft_hop)
            explosions_activity[..., start_frame: start_frame + length_frame] = 1 # assume the shape is [..., T]
        
        return (speech_melspec, mix_melspec, mix_time, masked_speech, masked_speech_time, explosions_activity, start_explosions, explosions_length)
        
        
    def time_to_frames(self, time_in_seconds, hop_length=160):
        return int(time_in_seconds / hop_length)



class SpeechRepaingingPhonemeClassifierDataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, split, sampling_rate, min_block_size, max_block_size, min_spacing, csv_loc, 
                 audio_stft_hop, base_data_dir, num4empty_str, num_blocks, rand_num_blocks, return_mask_properties, return_target_time=False):
        split = split.capitalize()
        self.base_data_dir = base_data_dir
        self.return_target_time = return_target_time
        self.return_mask_properties = return_mask_properties
        self.num_blocks = num_blocks
        self.rand_num_blocks = rand_num_blocks
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size
        self.min_spacing = min_spacing
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
        self.csv_path = Path(csv_loc) / f'{split}.csv'
        self.csv_df = pd.read_csv(self.csv_path, delimiter="|")
    
    def __len__(self):
        #return 100000
        # return 100
        # if self.split=='Train':
        #     return 300
        return len(self.csv_df)
    
    def __getitem__(self, index):
        phoneme_frame_path = Path(self.base_data_dir) / self.csv_df.loc[index, "phoneme_frame_path"] #npy file
        phoneme_target = np.load(phoneme_frame_path)
        phoneme_target = torch.from_numpy(phoneme_target).type(torch.LongTensor)
        melspec_path =  Path(self.base_data_dir) / self.csv_df.loc[index, "mel_spectrum_path"] #npz file
        melspec = torch.load(melspec_path)
        melspec = normalise_mel(melspec)
        
        audio_path =  Path(self.base_data_dir) / self.csv_df.loc[index, "wav_path"] # wav file
        _, audio_time = read(audio_path)
        audio_time = torch.from_numpy(audio_time.astype(np.float32))
        audio_time =  0.9 * audio_time / audio_time.abs().max() #because we took the original librispeech dataset, the audio is not norlaized to [-1, 1]
        if self.return_mask_properties:
            masked_melspec, mask, masked_audio_time, block_size_list, num_blocks = self.create_masked_melspec(melspec, audio_time)
            if self.return_target_time:
                return (phoneme_target, audio_time, melspec, masked_melspec, masked_audio_time, mask, block_size_list, num_blocks)
            return (phoneme_target, melspec, masked_melspec, masked_audio_time, mask, block_size_list, num_blocks)
        else:  
            masked_melspec, mask, masked_audio_time = self.create_masked_melspec(melspec, audio_time)
            if self.return_target_time:
                return (phoneme_target, audio_time, melspec, masked_melspec, masked_audio_time, mask)
            return (phoneme_target, melspec, masked_melspec, masked_audio_time, mask)

    

    # def create_mask(self, shape):
    #     # Initialize the mask with ones
    #     mask = torch.ones((shape[0], shape[1]))
        
    #     # Determine the number of blocks
    #     num_blocks = torch.randint(1, (shape[1] - self.min_block_size) // (self.max_block_size + self.min_spacing) + 1, (1,)).item()
    #     # Keep track of the end position of the last block to ensure spacing
    #     edge = int(0.5*16000/160) #0.5sec of edge
    #     last_block_end = -self.min_spacing + edge
        
    #     for i in range(num_blocks):
    #         #print(i)
    #         # Random block size
    #         block_size = torch.randint(self.min_block_size, self.max_block_size + 1, (1,)).item()
            
    #         # Ensure valid start_pos to avoid overlapping
    #         if last_block_end + self.min_spacing + block_size >= shape[1]-edge:
    #             #print("break")
    #             break
    #         if i == num_blocks-1:
    #             start_pos = torch.randint(last_block_end + self.min_spacing, last_block_end + self.min_spacing + shape[1]//num_blocks - block_size - edge + 1, (1,)).item()
    #         else:
    #             start_pos = torch.randint(last_block_end + self.min_spacing, last_block_end + self.min_spacing + shape[1]//num_blocks - block_size + 1, (1,)).item()
    #         #print(f"start is {start_pos}")
    #         # Set the mask to 0 for the current block
    #         mask[:, start_pos:start_pos + block_size] = 0
            
    #         # Update the end position of the last block
    #         last_block_end = start_pos + block_size
        
    #     return mask
    
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
        
        

    
def get_dataset(cfg, split='Train', return_mask_properties=False, return_target_time=False):
    if cfg['dataset_type'] == 'speech_inpainting':
        return SpeechRepaingingDataset(**cfg['speech_inpainting'], split=split, return_mask_properties=return_mask_properties, return_target_time=return_target_time)
    elif cfg['dataset_type'] == 'explosion_speech_inpainting':
        return ExplosionSpeechRepaingingDataset(**cfg['explosion_speech_inpainting'], split=split)
    elif cfg['dataset_type'] == 'speech_inpainting_phoneme_classifier':
        return SpeechRepaingingPhonemeClassifierDataset(**cfg['speech_inpainting_phoneme_classifier'], split=split, return_mask_properties=return_mask_properties, return_target_time=return_target_time)
    else:
        raise Exception('Invalid dataset type')
    