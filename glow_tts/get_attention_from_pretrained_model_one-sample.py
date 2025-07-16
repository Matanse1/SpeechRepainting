import os
import json
import argparse
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
# from apex.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DistributedDataParallel as DDP
# from apex import amp
import sys
sys.path.append('/home/dsi/moradim/SpeechRepainting/')
from glow_tts.data_utils import TextMelLoader, TextMelCollate, CollateFn, MyTextMelLoader
from glow_tts import models
from glow_tts import commons
from glow_tts import utils
from glow_tts.text.symbols import symbols
from glow_tts.text import _id_to_symbol, _symbol_to_id
import os
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

global_step = 0
original_model = False
save_logp_attn = True

def main():
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."
  dict_phoneme_path = "/home/dsi/moradim/Documents/MFA/models/inspect/english_us_arpa_acoustic/phones.txt"
  phoneme_dict_p2d, phoneme_dict_d2p = utils.get_phones_dict(dict_phoneme_path)
  if original_model:
    phoneme_dict_d2p_or = phoneme_dict_d2p
    phoneme_dict_d2p = _id_to_symbol
    phoneme_dict_d2p[148] = 'sil'
  else:
    phoneme_dict_d2p_or = phoneme_dict_d2p
  model_dir = '/home/dsi/moradim/SpeechRepainting/glow_tts/masked-mel-spec-as-input_silent-and-space-between-words_true_duration_mean-only_true-attn_ce_weight=0p8_c-non-simple-head_npz=2_warmup_and_constant_without-weighted-loss' #'/dsi/gannot-lab1/users/mordehay/glow_tts_alignment/masked-mel-spec-as-input_without-silenece-token_with-blank-token_true_duration_mean-only_true-attn_ce_weight=0p8_c-non-simple-head_npz=2_warmup_and_constant_without-weighted-loss'
  pretrained_model_path = '/home/dsi/moradim/SpeechRepainting/glow_tts/masked-mel-spec-as-input_silent-and-space-between-words_true_duration_mean-only_true-attn_ce_weight=0p8_c-non-simple-head_npz=2_warmup_and_constant_without-weighted-loss/G_97.pth' #'/dsi/gannot-lab1/users/mordehay/glow_tts_alignment/masked-mel-spec-as-input_without-silenece-token_with-blank-token_true_duration_mean-only_true-attn_ce_weight=0p8_c-non-simple-head_npz=2_warmup_and_constant_without-weighted-loss/G_126.pth'
  cp_num = Path(pretrained_model_path).stem
  hps = utils.get_hparams_from_dir(model_dir)
  torch.manual_seed(hps.train.seed)



  
  # {"phonemes": [phoneme_sequence_list, phoneme_duration_list, phoneme_int_list, full_phoneme_squence], "mel_spectrum": [melspec, masked_melspec, masked_audio_time, mask]}

  
  collate_fn = CollateFn(inputs_params=[{"type": "mel_spectrum", "axis": 0, "max_length": 1701, "end_number": 0}, # correspond to melspec
                                      {"type": "mel_spectrum", "axis": 1, "max_length": 1701, "end_number": 0}, # correspond to masked_melspec
                                      {"type": "mel_spectrum", "axis": 3, "max_length": 1701, "end_number": 1}], # correspond to mask
                        targets_params=[{"type": "phonemes", "axis": 0, "max_length": 491, "end_number": 0},
                                        {"type": "phonemes", "axis": 1, "max_length": 491, "end_number": 0}, # correspond to phoneme_int_list, the max seuqene length is 245 for 'without sil token' and 250 for 'with sil token'. after the interweaving of the phoneme_int_list the lognest sequence is len(lst) * 2 + 1, i.e. 491 for 'without sil token' and 501 for 'with sil token'
                                        {"type": "phonemes", "axis": 3, "max_length_m1": 1701,  "max_length_m2": 491, "end_number_m1": 0, "end_number_m2": 0},
                                        {"type": "phonemes", "axis": 2, "max_length": 1701, "end_number": 0}])

  val_dataset = MyTextMelLoader(**hps.data, split='Test', return_mask_properties=False, return_full_phoneme_squence=True)
  val_loader = DataLoader(val_dataset, num_workers=8, shuffle=False,
      batch_size=1, pin_memory=True,
      drop_last=True, collate_fn=collate_fn)
  if original_model:
    num_symbol = 148
  else:
    num_symbol = 72
  generator = models.FlowGenerator(
      n_vocab=num_symbol + getattr(hps.data, "add_blank", False), 
      out_channels=hps.data.n_mel_channels, 
      **hps.model).cuda(0)
  utils.size_model(generator)
  optimizer_g = commons.Adam(generator.parameters(), scheduler=hps.train.scheduler, dim_model=hps.model.hidden_channels, warmup_steps=hps.train.warmup_steps, lr=hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
  # if hps.train.fp16_run:
  #   generator, optimizer_g._optim = amp.initialize(generator, optimizer_g._optim, opt_level="O1")
  # generator = DDP(generator)


  # checkpoint = torch.load(pretrained_model_path)
  # saved_param_groups = checkpoint['optimizer']['param_groups']
  # current_param_groups = optimizer_g.state_dict()['param_groups']

  # print("Saved parameter groups:")
  # print(saved_param_groups)
  # print("\nCurrent parameter groups:")
  # print(current_param_groups)
  # for idx, param in enumerate(checkpoint['model'].keys()):
  #   if idx == 520:
  #       print(f"Parameter 520: {param}")
  #   if idx == 521:
  #       print(f"Parameter 521: {param}")
  #       break
  generator, optimizer, learning_rate, epoch_str = utils.load_checkpoint(pretrained_model_path, generator, optimizer_g)

  generator.eval()
  num_samples = 20
  losses_tot = []
  with torch.no_grad():
    for batch_idx, data in tqdm(enumerate(val_loader)):
      inputs_collates, inputs_masks, inputs_length_original = data["inputs"]
      targets_collates, targets_masks, targets_length_original = data["targets"]
      
      melspec, masked_melspec, mask = inputs_collates[0], inputs_collates[1], inputs_collates[2]
      melspec_mask, masked_melspec_mask, mask_mask = inputs_masks[0], inputs_masks[1], inputs_masks[2]
      melspec_inputs_length_original, masked_melspec_inputs_length_original, mask_inputs_length_original = inputs_length_original[0], inputs_length_original[1], inputs_length_original[2]
      
      # phoneme_int_list, phoneme_duration_list, phoneme_sequence_list = targets_collates[0], targets_collates[1], targets_collates[2]
      phoneme_duration, phoneme_int, true_attention_matrix, full_phoneme_squence =\
        targets_collates[0], targets_collates[1], targets_collates[2], targets_collates[3]
        
      phoneme_duration_mask, phoneme_int_mask, true_attention_matrix_mask, full_phoneme_squence_mask \
        = targets_masks[0], targets_masks[1], targets_masks[2], targets_masks[3]
        
      phoneme_duration_length_original, phoneme_int_length_original, true_attention_matrix_length_original, full_phoneme_squence_length_original \
        = targets_length_original[0], targets_length_original[1], targets_length_original[2], targets_length_original[3]
        
            #saving results
      save_dir = Path(os.path.join(model_dir, "alignment_results", cp_num, f"sample_{batch_idx}")) 
      # save_dir = Path(os.path.join(model_dir, "alignment_results_updated-backward-and-forward", cp_num, f"sample_{batch_idx}"))
      save_dir.mkdir(parents=True, exist_ok=True)
      
      if save_logp_attn:
        np.save(os.path.join(save_dir, 'true_attn.npy'), true_attention_matrix[0].cpu().numpy())
        
      if hps.train.insert_masked_melspec_bool:
        y = masked_melspec
        y_lengths = masked_melspec_inputs_length_original
        attention_mask = mask_mask
        masked_region = mask
      else:
        y = melspec
        y_lengths = melspec_inputs_length_original
        attention_mask = None
        masked_region = None
      if  original_model:
        phoneme_int_np = np.zeros(phoneme_int.shape)
        for j in range(phoneme_int.shape[0]):
          for i in range(phoneme_int.shape[-1]):
            ph = phoneme_dict_d2p_or[phoneme_int[j, i].item()]
            if 'sil' == ph or ph == 'spn':
              phoneme_int_np[j, i] = num_symbol
            elif phoneme_int[j, i].item() == 0:
              phoneme_int_np[j, i] = 0
            else:
              phoneme_int_np[j, i] = _symbol_to_id['@' + ph]
        phoneme_int = torch.from_numpy(phoneme_int_np.astype(np.int32))
        x = phoneme_int
      else:
        x = phoneme_int
      x_lengths = phoneme_int_length_original
      x, x_lengths = x.cuda(0, non_blocking=True), x_lengths.cuda(0, non_blocking=True)
      y, y_lengths = y.cuda(0, non_blocking=True), y_lengths.cuda(0, non_blocking=True)
      phoneme_duration, phoneme_duration_mask = phoneme_duration.cuda(0, non_blocking=True), phoneme_duration_mask.cuda(0, non_blocking=True)
      # if hps.train.use_true_attn:
      #   true_attention_matrix = true_attention_matrix.cuda(0, non_blocking=True)
      # else:
      true_attention_matrix = None
      
      (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_), (vocab_classification, logp) = generator(x, x_lengths, y, y_lengths, gen=False, attention_mask=attention_mask, masked_region=masked_region, true_attention_matrix=true_attention_matrix)
      
      
      l_mle = commons.mle_loss(z, z_m, z_logs, logdet, z_mask)
      l_length = commons.duration_loss(logw, logw_, x_lengths)

      loss_gs = [l_mle, l_length]
      loss_g = sum(loss_gs)
      

      if save_logp_attn:
        np.save(os.path.join(save_dir, 'logp.npy'), logp[0].cpu().numpy())
        np.save(os.path.join(save_dir, 'est_attn.npy'), attn[0, 0].cpu().numpy())
        
      full_phoneme_squence = full_phoneme_squence[0].tolist()
      estimated_full_phoneme_squence = []
      num_occurance_each_phoneme = torch.sum(attn[0,0], dim=-1)
      for i in range(num_occurance_each_phoneme.shape[0]):
        estimated_full_phoneme_squence += [phoneme_int[0][i].item()] * int(num_occurance_each_phoneme[i].item())
      if len(estimated_full_phoneme_squence) < len(full_phoneme_squence):
        estimated_full_phoneme_squence += [phoneme_int[0][-1].item()] * (len(full_phoneme_squence) - len(estimated_full_phoneme_squence))
      

      for i in range(len(estimated_full_phoneme_squence)):
        estimated_full_phoneme_squence[i] = phoneme_dict_d2p[estimated_full_phoneme_squence[i]]
      for i in range(len(full_phoneme_squence)):
        full_phoneme_squence[i] = phoneme_dict_d2p_or[full_phoneme_squence[i]]
      
      # save phoneme estimation
      phoneme_filename = os.path.join(save_dir, f'phoneme_gt_and_estimated_{l_mle.item():.2f}.html')
      def colorize_html(char, color):
          return f'<span style="color:red">{char}</span>' if color == 0 else char

      # Create HTML table rows with aligned elements
      rows = []
      color_list = mask.tolist()[0]
      hop_length = 160
      sampling_rate = 16000
      for i, (true_char, est_char, color) in enumerate(zip(full_phoneme_squence, estimated_full_phoneme_squence, color_list)):
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
                
                
      alignment_image  = utils.plot_alignment_to_numpy(attn[0,0].data.cpu().numpy())
      # Convert the NumPy array to a PIL Image and save it
      image = Image.fromarray(alignment_image)
      
      image.save(os.path.join(save_dir, "alignment_image.png"))  # Save as PNG
      if batch_idx == num_samples:
        break


                           
if __name__ == "__main__":
  main()
