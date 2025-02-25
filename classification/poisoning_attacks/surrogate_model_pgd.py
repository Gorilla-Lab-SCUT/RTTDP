from .base import SurrogateModelAttackBase, softmax_entropy
from utils.registry import ATTACK_REGISTRY, ATTACKING_OBJECTIVE

import torch
import numpy as np
from advertorch.attacks import LinfPGDAttack

from copy import deepcopy
from collections import OrderedDict
from functools import partial

from .attacking_objective import *


@ATTACK_REGISTRY.register()
class SurrogateModelPGDAttack(SurrogateModelAttackBase):
    def __init__(self, cfg, surrogate_model):
        super().__init__(cfg, surrogate_model)

        self.eps = 0.3
        self.alpha = 0.01

        self.attacking_objectives = [ATTACKING_OBJECTIVE.get(name)(cfg, self).cuda() for name in cfg.ATTACK.OBJECTIVE.NAMES]


    @torch.enable_grad()
    def forward(self, clean_data, label):
        self.surrogate_model.train()
        self.surrogate_model.zero_grad()

        rand_init = True
        for objective in self.attacking_objectives:
            objective.hook_before_attack(self.surrogate_model, clean_data, label=label)
            rand_init &= objective.rand_init
        
        self.adversary = LinfPGDAttack(
            self.surrogate_model, loss_fn=self.loss_fn, eps=self.eps,
            nb_iter=40, eps_iter=self.alpha, rand_init=rand_init, clip_min=0.0, clip_max=1.0,
            targeted=False)

        image_final = self.adversary.perturb(clean_data, label).detach()
        
        for objective in self.attacking_objectives:
            objective.hook_after_attack(self.surrogate_model)
        
        self.surrogate_model.zero_grad()
        return image_final

    def loss_fn(self, x: torch.Tensor, y) -> torch.Tensor:
        loss = 0
        for objective in self.attacking_objectives:
            loss += objective.attack_objective(x, y)
        for objective in self.attacking_objectives:
            objective.hook_towards_loss(loss)
        return loss

    def return_results(self, probs, y):
        for objective in self.attacking_objectives:
            objective.hook_when_return_result(probs, y)
        return
