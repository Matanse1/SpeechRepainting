#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchvision
from .face_model import Resnet18
from .lipreading_models.lipreading_model import Lipreading
import os
import sys
# sys.path.insert(0, '..')
from .utils import load_json
from .utils import get_model

class ModelBuilder():
    def build_model(self, model_cfg):
        # cond_feat_size = 640        # size of feature dimension for the conditioner
        name = model_cfg.pop("_name_")
        cls_model = get_model(name)
        # model = WaveNet(cond_feat_size, **model_cfg)
        model = cls_model(**model_cfg)
        model_cfg["_name_"] = name # restore
        return model
