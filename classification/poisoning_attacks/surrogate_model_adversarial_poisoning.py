from .base import SurrogateModelAttackBase, softmax_entropy
from utils.registry import ATTACK_REGISTRY, ATTACKING_OBJECTIVE

import torch
import numpy as np

from copy import deepcopy
from collections import OrderedDict
from functools import partial

from .attacking_objective import *
from advertorch.attacks import LinfPGDAttack
from copy import deepcopy
import torch.nn.functional as F


@ATTACK_REGISTRY.register()
class SurrogateModelAdvPoisoningAttack(SurrogateModelAttackBase):
    def __init__(self, cfg, surrogate_model):
        super().__init__(cfg, surrogate_model)

        self.eps = 0.3
        self.alpha = 0.01

    @torch.enable_grad()
    def forward(self, clean_data, label):
        self.surrogate_model.train()
        self.surrogate_model.zero_grad()
        
        self.adversary = LinfPGDAttack(
            self.surrogate_model, loss_fn=self.loss_fn, eps=self.eps,
            nb_iter=40, eps_iter=self.alpha, rand_init=True, clip_min=0.0, clip_max=1.0,
            targeted=False)
        
        image_final = self.adversary.perturb(clean_data, label).detach()
        
        self.surrogate_model.zero_grad()
        return image_final

    def loss_fn(self, x: torch.Tensor, y) -> torch.Tensor:
        loss = -F.cross_entropy(x, (y + 1) % x.shape[-1])
        return loss
