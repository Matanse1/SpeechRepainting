import os
os.sys.path.append("/home/dsi/moradim/SpeechRepainting")
from ASR import asr_models as asr_models
import torch
from dataloaders.dataset_lipvoicer import get_dataset
from omegaconf import DictConfig, OmegaConf
import hydra
from utils import calc_diffusion_hyperparams
import soundfile as sf
import numpy as np


@hydra.main(version_base=None, config_path="../configs/", config_name="asr_checking_config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys
    dataset_cfg = cfg.dataset
    diffusion_cfg = cfg.diffusion
    save_dir = cfg.save_dir
    ds_name = 'LRS3' # 'LRS2'
        
    print('Loading ASR, tokenizer and decoder')
    asr_guidance_net, tokenizer, decoder = asr_models.get_models(ds_name)
    guidance_text = " hi"
    text_tokens = torch.LongTensor(tokenizer.encode(guidance_text))
    text_tokens = text_tokens.unsqueeze(0).cuda()
    diffusion_hyperparams  = calc_diffusion_hyperparams(**diffusion_cfg, fast=True)  # dictionary of all diffusion hyperparameters
    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    t = list(range(T-1, -1, -1))[-1]
    print(f"t = {t}")
    diffusion_steps = (t * torch.ones((1, 1))).cuda()
    
    dataset = get_dataset(dataset_cfg, split='test', return_mask_properties=True, return_target_time=True)
    i = 0
    audio_time, gt_melspec, *masked_cond, mask, block_size_list, num_blocks = dataset[i]
    # masked_cond = [masked_cond[i].unsqueeze(0).cuda() for i in range(len(masked_cond))]
    masked_melspec = masked_cond[0]
    masked_audio_time = masked_cond[1]
    # save the masked audio in time domain
    os.makedirs(os.path.join(save_dir, f'sample_{i}'), exist_ok=True)
    masked_audio_time4saveing = masked_audio_time.squeeze().cpu().numpy()
    sf.write(os.path.join(save_dir, f'sample_{i}', 'time_masking_audio.wav'), masked_audio_time4saveing, 16000)
    # save the target audio in time domain
    audio_time_time4saveing = audio_time.squeeze().cpu().numpy()
    sf.write(os.path.join(save_dir, f'sample_{i}', 'audio_time.wav'), audio_time_time4saveing, 16000)
    
    # x = gt_melspec.unsqueeze(0).cuda()
    x = masked_melspec.unsqueeze(0).cuda()
    
    # spec_generated = torch.load("/dsi/gannot-lab/gannot-lab1/users/mordehay/speech_repainting/exp/LibSp_wavlm-base-plus-rep_w_masked_pix=0.8_two_branch=True/wnet_h512_d12_T400_betaT0.02/as-train-gap_asr-guidance/w1=2_w2=1.5_asr_start=270/sample_0/generated_spec.npz")
    spec_generated = torch.load("/dsi/gannot-lab/gannot-lab1/users/mordehay/speech_repainting/exp/LibSp_wavlm-base-plus-rep_w_masked_pix=0.8_two_branch=True/wnet_h512_d12_T400_betaT0.02/as-train-gap_2cp/w1=2_w2=0_asr_start=270/sample_1/generated_spec.npz")
    
    x = spec_generated.unsqueeze(0).cuda()
    inputs = x, torch.tensor([x.shape[-1]]).cuda() 
    outputs_ao = asr_guidance_net(inputs, diffusion_steps)["outputs"]
    preds_ao = decoder(outputs_ao)[0]
    print(preds_ao)
        
    print('ASR checking done')
if __name__ == "__main__":
    main()