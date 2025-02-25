from .base import SurrogateModelAttackBase, softmax_entropy
from utils.registry import ATTACK_REGISTRY, ATTACKING_OBJECTIVE

import torch
import numpy as np

from copy import deepcopy
from collections import OrderedDict
from functools import partial

from .attacking_objective import *
from autoattack import AutoAttack


@ATTACK_REGISTRY.register()
class SurrogateModelAutoAttack(SurrogateModelAttackBase):
    def __init__(self, cfg, surrogate_model):
        super().__init__(cfg, surrogate_model)

        self.eps = 0.3
        self.alpha = 0.01

    @torch.enable_grad()
    def forward(self, clean_data, label):
        self.surrogate_model.train()
        self.surrogate_model.zero_grad()

        self.adversary = AutoAttack(self.surrogate_model, norm='Linf', eps=self.eps, version='standard', verbose=False)

        image_final = self.adversary.run_standard_evaluation(clean_data, label, bs=clean_data.shape[0]).detach()
        
        self.surrogate_model.zero_grad()
        return image_final
