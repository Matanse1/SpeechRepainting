#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from random import random
import torch
import torch.nn as nn

class AudioVisualModel(torch.nn.Module):
    def name(self):
        return "AudioVisualModel"

    def __init__(self, g_model_cfg, nets):
        super(AudioVisualModel, self).__init__()

        # initialize model
        # self.net_lipreading, self.net_facial, self.net_diffwave = nets
        self.g_model_cfg = g_model_cfg
        self.net_diffwave = nets

        # classifier guidance null conditioners
        torch.manual_seed(0)        # so we have the same null tokens on all nodes

    def forward(self, melspec, masked_cond, diffusion_steps, cond_drop_prob, mask_padding_time=None, mask_padding_frames=None, text=None):
        # classifier guidance
        if self.net_diffwave.unconditional:
            cond = None
        else:
            masked_melspec, masked_audio_time = masked_cond
            batch, C, L = melspec.shape
            if self.g_model_cfg.null_type == 'zeros':
                masked_melspec_null = torch.zeros(1, C, L, device=melspec.device)
                masked_audio_time_null = torch.zeros(1, masked_audio_time.shape[-1], device=melspec.device) 
            elif self.g_model_cfg.null_type == 'randn':
                masked_melspec_null = torch.randn(1, C, L, device=melspec.device)
                masked_audio_time_null = torch.randn(1, masked_audio_time.shape[-1], device=melspec.device) 
            elif 'param':
                masked_melspec_null = nn.Parameter(torch.randn(1, C, L, device=melspec.device))
                masked_audio_time_null = nn.Parameter(torch.randn(1, masked_audio_time.shape[-1], device=melspec.device))
            if cond_drop_prob > 0:
                # for melspec cond
                prob_keep_mask = self.prob_mask_like((batch, 1, 1), 1.0 - cond_drop_prob, melspec.device)
                _masked_melspec = torch.where(prob_keep_mask, masked_melspec, masked_melspec_null)
                
                # for audio time cond
                prob_keep_mask = self.prob_mask_like((batch, 1), 1.0 - cond_drop_prob, melspec.device)
                _masked_audio_time = torch.where(prob_keep_mask, masked_audio_time, masked_audio_time_null)
                
            else:
                _masked_melspec = masked_melspec
                _masked_audio_time = masked_audio_time
            
        drop_text = random() < cond_drop_prob  # p_drop in voicebox paper
        cond = (_masked_melspec, _masked_audio_time)

        # pass through visual stream and extract lipreading features
        # lipreading_feature = self.net_lipreading(_mouthroi)
        
        # # pass through visual stream and extract identity features
        # identity_feature = self.net_facial(_face_image)

        # what type of visual feature to use
        # identity_feature = identity_feature.repeat(1, 1, 1, lipreading_feature.shape[-1])
        # visual_feature = torch.cat((identity_feature, lipreading_feature), dim=1)
        # visual_feature = visual_feature.squeeze(2)  # so dimensions are B, C, num_frames
        

        output = self.net_diffwave((melspec, diffusion_steps), cond=cond, text=text, mask_padding_time=mask_padding_time, mask_padding_frames=mask_padding_frames, drop_text=drop_text)
        return output

    @staticmethod
    def prob_mask_like(shape, prob, device):
        if prob == 1:
            return torch.ones(shape, device=device, dtype=torch.bool)
        elif prob == 0:
            return torch.zeros(shape, device=device, dtype=torch.bool)
        else:
            return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob
