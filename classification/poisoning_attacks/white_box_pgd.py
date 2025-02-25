from .base import WhiteBoxModelAttackBase, softmax_entropy
from utils.registry import ATTACK_REGISTRY, ATTACKING_OBJECTIVE

import torch
import torch.nn
import torch.nn.functional as F
from advertorch.attacks import LinfPGDAttack
from copy import deepcopy
from functools import partial
from collections import OrderedDict


from .attacking_objective import *
from augmentations.transforms_cotta import Clip, ColorJitterPro, GaussianNoise
from torchvision import transforms

@ATTACK_REGISTRY.register()
class WhiteBoxPGDAttack(WhiteBoxModelAttackBase):
    def __init__(self, cfg, white_box_model):
        super().__init__(cfg, white_box_model)
        self.eps = 0.3
        self.alpha = 0.01

        self.attacking_objectives = [ATTACKING_OBJECTIVE.get(name)(cfg, self).cuda() for name in cfg.ATTACK.OBJECTIVE.NAMES]


    @torch.enable_grad()
    def forward(self, clean_data, label):
        self.inner_model = deepcopy(self.white_box_model)
        self.inner_model.train()
        self.inner_model.zero_grad()

        rand_init = True
        for objective in self.attacking_objectives:
            objective.hook_before_attack(self.inner_model, clean_data, label=label)
            rand_init &= objective.rand_init

        self.adversary = LinfPGDAttack(
            self.inner_model, loss_fn=self.loss_fn, eps=self.eps,
            nb_iter=40, eps_iter=self.alpha, rand_init=rand_init, clip_min=0., clip_max=1.0,
            targeted=False) # clip_min = -1 for pc; 0 for image
        
        image_final = self.adversary.perturb(clean_data, label).detach()

        for objective in self.attacking_objectives:
            objective.hook_after_attack(self.inner_model)
        
        self.inner_model.zero_grad()
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
