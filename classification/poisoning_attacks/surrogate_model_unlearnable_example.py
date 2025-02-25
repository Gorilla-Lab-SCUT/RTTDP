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
class SurrogateModelUnlearnableAttack(SurrogateModelAttackBase):
    def __init__(self, cfg, surrogate_model):
        super().__init__(cfg, surrogate_model)

        self.eps = 0.3
        self.alpha = 0.01
        self.random_init_model = deepcopy(self.surrogate_model)
        self.random_init_model.__init__()
        self.random_init_model.cuda()


    @torch.enable_grad()
    def forward(self, clean_data, label):
        self.random_init_model.train()
        self.random_init_model.zero_grad()
        
        self.adversary = LinfPGDAttack(
            self.random_init_model, loss_fn=self.loss_fn, eps=self.eps,
            nb_iter=40, eps_iter=self.alpha, rand_init=True, clip_min=0.0, clip_max=1.0,
            targeted=False)
        
        image_final = self.adversary.perturb(clean_data, label).detach()
        
        self.random_init_model.zero_grad()
        return image_final

    def loss_fn(self, x: torch.Tensor, y) -> torch.Tensor:
        loss = -F.cross_entropy(x, y)
        return loss
