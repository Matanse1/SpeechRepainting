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
from data_utils import TextMelLoader, TextMelCollate, CollateFn, MyTextMelLoader
from glow_tts import models
from glow_tts import commons
from glow_tts import utils
from text.symbols import symbols
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

global_step = 0



def main():
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."

  n_gpus = 1 #torch.cuda.device_count()
  # os.environ['MASTER_ADDR'] = 'localhost'
  # os.environ['MASTER_PORT'] = '8004'

  hps = utils.get_hparams(config_path='/home/dsi/moradim/SpeechRepainting/glow_tts/configs/my_base_with-space.json', model_root_dir='/home/dsi/moradim/SpeechRepainting/glow_tts/')
  train_and_eval(0, n_gpus, hps)
  # mp.spawn(train_and_eval, nprocs=n_gpus, args=(n_gpus, hps,))


def train_and_eval(rank, n_gpus, hps):
  global global_step
  if rank == 0:
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    # utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

  # dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
  torch.manual_seed(hps.train.seed)
  torch.cuda.set_device(rank)

  train_dataset = MyTextMelLoader(**hps.data, split='Train', return_mask_properties=False, return_full_phoneme_squence=hps.model.classification_head)
  train_sampler = torch.utils.data.distributed.DistributedSampler(
      train_dataset,
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True)
  
  #  {"phonemes": [phoneme_duration_list_without_silence, phoneme_int_list_without_silence, true_attention_matrix], "mel_spectrum": [melspec, masked_melspec, masked_audio_time, mask]} 
  targets_params = [{"type": "phonemes", "axis": 0, "max_length": 491, "end_number": 0},
                                         {"type": "phonemes", "axis": 1, "max_length": 491, "end_number": 0}, # correspond to phoneme_int_list, the max seuqene length is 245 for 'without sil token' and 250 for 'with sil token'. after the interweaving of the phoneme_int_list the lognest sequence is len(lst) * 2 + 1, i.e. 491 for 'without sil token' and 501 for 'with sil token'
                                         {"type": "phonemes", "axis": 2, "max_length_m1": 1701,  "max_length_m2": 491, "end_number_m1": 0, "end_number_m2": 0}]
  inputs_params = [{"type": "mel_spectrum", "axis": 0, "max_length": 1701, "end_number": 0}, # correspond to melspec
                                        {"type": "mel_spectrum", "axis": 1, "max_length": 1701, "end_number": 0}, # correspond to masked_melspec
                                        {"type": "mel_spectrum", "axis": 3, "max_length": 1701, "end_number": 1}]
  if hps.model.classification_head:
    targets_params = [{"type": "phonemes", "axis": 0, "max_length": 491, "end_number": 0},
                                         {"type": "phonemes", "axis": 1, "max_length": 491, "end_number": 0}, # correspond to phoneme_int_list, the max seuqene length is 245 for 'without sil token' and 250 for 'with sil token'. after the interweaving of the phoneme_int_list the lognest sequence is len(lst) * 2 + 1, i.e. 491 for 'without sil token' and 501 for 'with sil token'
                                         {"type": "phonemes", "axis": 3, "max_length_m1": 1701,  "max_length_m2": 491, "end_number_m1": 0, "end_number_m2": 0},
                                         {"type": "phonemes", "axis": 2, "max_length": 1701, "end_number": 0}]
    
  collate_fn = CollateFn(inputs_params=inputs_params, # correspond to mask
                         targets_params=targets_params)
  # collate_fn = TextMelCollate(1)
  train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False,
      batch_size=hps.train.batch_size, pin_memory=True,
      drop_last=True, collate_fn=collate_fn, sampler=train_sampler)
  if rank == 0:
    val_dataset = MyTextMelLoader(**hps.data, split='Test', return_mask_properties=False, return_full_phoneme_squence=hps.model.classification_head)
    val_loader = DataLoader(val_dataset, num_workers=8, shuffle=False,
        batch_size=hps.train.batch_size, pin_memory=True,
        drop_last=True, collate_fn=collate_fn)
  num_symbol = 72
  generator = models.FlowGenerator(
      n_vocab=num_symbol + getattr(hps.data, "add_blank", False), 
      out_channels=hps.data.n_mel_channels,
      **hps.model).cuda(rank)
  utils.size_model(generator)
  optimizer_g = commons.Adam(generator.parameters(), scheduler=hps.train.scheduler, dim_model=hps.model.hidden_channels, warmup_steps=hps.train.warmup_steps, lr=hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
  if hps.train.fp16_run:
    generator, optimizer_g._optim = amp.initialize(generator, optimizer_g._optim, opt_level="O1")
  # generator = DDP(generator)
  epoch_str = 1
  global_step = 0
  try:
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), generator, optimizer_g)
    epoch_str += 1
    optimizer_g.step_num = (epoch_str - 1) * len(train_loader)
    optimizer_g._update_learning_rate()
    global_step = (epoch_str - 1) * len(train_loader)
  except:
    if hps.train.ddi and os.path.isfile(os.path.join(hps.model_dir, "ddi_G.pth")):
      _ = utils.load_checkpoint(os.path.join(hps.model_dir, "ddi_G.pth"), generator, optimizer_g)
  
  for epoch in range(epoch_str, hps.train.epochs + 1):
    if rank==0:
      train(rank, epoch, hps, generator, optimizer_g, train_loader, logger, writer)
      evaluate(rank, epoch, hps, generator, optimizer_g, val_loader, logger, writer_eval)
      utils.save_checkpoint(generator, optimizer_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(epoch)))
    else:
      train(rank, epoch, hps, generator, optimizer_g, train_loader, None, None)


def train(rank, epoch, hps, generator, optimizer_g, train_loader, logger, writer):
  train_loader.sampler.set_epoch(epoch)
  global global_step

  generator.train()
  for batch_idx, data in enumerate(train_loader):
    # (x, x_lengths, y, y_lengths) . x and y is the text and mel-spectrogram respectively
    inputs_collates, inputs_masks, inputs_length_original = data["inputs"]
    targets_collates, targets_masks, targets_length_original = data["targets"]
    
    melspec, masked_melspec, mask = inputs_collates[0], inputs_collates[1], inputs_collates[2]
    melspec_mask, masked_melspec_mask, mask_mask = inputs_masks[0], inputs_masks[1], inputs_masks[2]
    melspec_inputs_length_original, masked_melspec_inputs_length_original, mask_inputs_length_original = inputs_length_original[0], inputs_length_original[1], inputs_length_original[2]
    
    # phoneme_int_list, phoneme_duration_list, phoneme_sequence_list = targets_collates[0], targets_collates[1], targets_collates[2]
    phoneme_duration, phoneme_int, true_attention_matrix = targets_collates[0], targets_collates[1], targets_collates[2]
    phoneme_duration_mask, phoneme_int_mask, true_attention_matrix_mask = targets_masks[0], targets_masks[1], targets_masks[2]
    phoneme_duration_length_original, phoneme_int_length_original, true_attention_matrix_length_original = targets_length_original[0], targets_length_original[1], targets_length_original[2]
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
      
    x = phoneme_int
    x_lengths = phoneme_int_length_original
    
    x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
    y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
    phoneme_duration, phoneme_duration_mask = phoneme_duration.cuda(rank, non_blocking=True), phoneme_duration_mask.cuda(rank, non_blocking=True)
    if hps.train.use_true_attn:
      true_attention_matrix = true_attention_matrix.cuda(rank, non_blocking=True)
    else:
      true_attention_matrix = None

    if hps.model.classification_head:
      full_phoneme_squence = targets_collates[3]
      full_phoneme_squence_mask = targets_masks[3]
      full_phoneme_squence = full_phoneme_squence.cuda(rank, non_blocking=True)
      full_phoneme_squence_mask = full_phoneme_squence_mask.cuda(rank, non_blocking=True)
    # Train Generator
    optimizer_g.zero_grad()
    
    (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_), (vocab_classification, logp) = generator(x, x_lengths, y, y_lengths, gen=False, attention_mask=attention_mask, masked_region=masked_region, true_attention_matrix=true_attention_matrix)
    l_mle = commons.mle_loss(z, z_m, z_logs, logdet, z_mask)
    if hps.train.use_true_duration:
      true_duration = torch.log(1e-8 + phoneme_duration) * phoneme_duration_mask
      true_duration = true_duration.unsqueeze(1)
      l_length = commons.duration_loss(logw, true_duration, x_lengths)
    else:
      l_length = commons.duration_loss(logw, logw_, x_lengths)
    if hps.model.classification_head:
      cel = commons.ce_loss(vocab_classification, full_phoneme_squence, full_phoneme_squence_mask)
      loss_gs = [l_mle, l_length, cel]
      loss_g = hps.train.mle_weight * l_mle + hps.train.length_weight * l_length + hps.train.ce_weight * cel
    else:
      loss_gs = [l_mle, l_length]
      loss_g = hps.train.mle_weight * l_mle + hps.train.length_weight * l_length

    if hps.train.fp16_run:
      with amp.scale_loss(loss_g, optimizer_g._optim) as scaled_loss:
        scaled_loss.backward()
      grad_norm = commons.clip_grad_value_(amp.master_params(optimizer_g._optim), 5)
    else:
      loss_g.backward()
      grad_norm = commons.clip_grad_value_(generator.parameters(), 5)
    optimizer_g.step()
    
    if rank==0:
      if batch_idx % hps.train.log_interval == 0:
        # (y_gen, *_), *_ = generator(x[:1], x_lengths[:1], gen=True)
        logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(x), len(train_loader.dataset),
          100. * batch_idx / len(train_loader),
          loss_g.item()))
        logger.info([x.item() for x in loss_gs] + [global_step, optimizer_g.get_lr()])
        
        scalar_dict = {"loss/g/total": loss_g, "learning_rate": optimizer_g.get_lr(), "grad_norm": grad_norm}
        scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(loss_gs)})
        utils.summarize(
          writer=writer,
          global_step=global_step, 
          images={"y_org": utils.plot_spectrogram_to_numpy(y[0].data.cpu().numpy()), 
            # "y_gen": utils.plot_spectrogram_to_numpy(y_gen[0].data.cpu().numpy()), 
            "attn": utils.plot_alignment_to_numpy(attn[0,0].data.cpu().numpy()),
            },
          scalars=scalar_dict)
    global_step += 1
  
  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch))

 
def evaluate(rank, epoch, hps, generator, optimizer_g, val_loader, logger, writer_eval):
  if rank == 0:
    global global_step
    generator.eval()
    losses_tot = []
    with torch.no_grad():
      for batch_idx, data in enumerate(val_loader):
        # (x, x_lengths, y, y_lengths) . x and y is the text and mel-spectrogram respectively
        inputs_collates, inputs_masks, inputs_length_original = data["inputs"]
        targets_collates, targets_masks, targets_length_original = data["targets"]
        
        melspec, masked_melspec, mask = inputs_collates[0], inputs_collates[1], inputs_collates[2]
        melspec_mask, masked_melspec_mask, mask_mask = inputs_masks[0], inputs_masks[1], inputs_masks[2]
        melspec_inputs_length_original, masked_melspec_inputs_length_original, mask_inputs_length_original = inputs_length_original[0], inputs_length_original[1], inputs_length_original[2]
        
        # phoneme_int_list, phoneme_duration_list, phoneme_sequence_list = targets_collates[0], targets_collates[1], targets_collates[2]
        phoneme_duration, phoneme_int, true_attention_matrix = targets_collates[0], targets_collates[1], targets_collates[2]
        phoneme_duration_mask, phoneme_int_mask, true_attention_matrix_mask = targets_masks[0], targets_masks[1], targets_masks[2]
        phoneme_duration_length_original, phoneme_int_length_original, true_attention_matrix_length_original = targets_length_original[0], targets_length_original[1], targets_length_original[2]
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
        x = phoneme_int
        x_lengths = phoneme_int_length_original
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
        phoneme_duration, phoneme_duration_mask = phoneme_duration.cuda(rank, non_blocking=True), phoneme_duration_mask.cuda(rank, non_blocking=True)
        if hps.train.use_true_attn:
          true_attention_matrix = true_attention_matrix.cuda(rank, non_blocking=True)
        else:
          true_attention_matrix = None
        
        if hps.model.classification_head:
          full_phoneme_squence = targets_collates[3]
          full_phoneme_squence_mask = targets_masks[3]
          full_phoneme_squence = full_phoneme_squence.cuda(rank, non_blocking=True)
          full_phoneme_squence_mask = full_phoneme_squence_mask.cuda(rank, non_blocking=True)
        
        (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_), (vocab_classification, logp) = generator(x, x_lengths, y, y_lengths, gen=False, attention_mask=attention_mask, masked_region=masked_region, true_attention_matrix=true_attention_matrix)
        l_mle = commons.mle_loss(z, z_m, z_logs, logdet, z_mask)
        if hps.train.use_true_duration:
          true_duration = torch.log(1e-8 + phoneme_duration) * phoneme_duration_mask
          true_duration = true_duration.unsqueeze(1)
          l_length = commons.duration_loss(logw, true_duration, x_lengths)
        else:
          l_length = commons.duration_loss(logw, logw_, x_lengths)

        if hps.model.classification_head:
          cel = commons.ce_loss(vocab_classification, full_phoneme_squence, full_phoneme_squence_mask)
          loss_gs = [l_mle, l_length, cel]
          loss_g = hps.train.mle_weight * l_mle + hps.train.length_weight * l_length + hps.train.ce_weight * cel
        else:
          loss_gs = [l_mle, l_length]
          loss_g = hps.train.mle_weight * l_mle + hps.train.length_weight * l_length

        if batch_idx == 0:
          losses_tot = loss_gs
        else:
          losses_tot = [x + y for (x, y) in zip(losses_tot, loss_gs)]

        if batch_idx % hps.train.log_interval == 0:
          logger.info('Eval Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(x), len(val_loader.dataset),
            100. * batch_idx / len(val_loader),
            loss_g.item()))
          logger.info([x.item() for x in loss_gs])
           
    
    losses_tot = [x/len(val_loader) for x in losses_tot]
    loss_tot = sum(losses_tot)
    scalar_dict = {"loss/g/total": loss_tot}
    scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_tot)})
    utils.summarize(
      writer=writer_eval,
      global_step=global_step, 
      scalars=scalar_dict)
    logger.info('====> Epoch: {}'.format(epoch))

                           
if __name__ == "__main__":
  
  main()
