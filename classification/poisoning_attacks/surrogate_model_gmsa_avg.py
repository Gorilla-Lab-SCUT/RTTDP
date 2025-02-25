from .base import SurrogateModelAttackBase, softmax_entropy
from utils.registry import ATTACK_REGISTRY, ATTACKING_OBJECTIVE

import torch
import numpy as np

from copy import deepcopy
from collections import OrderedDict
from functools import partial

from .attacking_objective import *
from .attack_optimizer import GMSA


@ATTACK_REGISTRY.register()
class SurrogateModelGMSAAvgAttack(SurrogateModelAttackBase):
    def __init__(self, cfg, surrogate_model):
        super().__init__(cfg, surrogate_model)

        self.eps = 0.3
        self.alpha = 0.01

    @torch.enable_grad()
    def forward(self, clean_data, label):
        self.surrogate_model.train()
        self.surrogate_model.zero_grad()

        self.adversary = GMSA.ConfGMSAAVGAttack([self.surrogate_model], eps=self.eps, nb_iter=40,
                eps_iter=self.alpha, rand_init=True, clip_min=0., clip_max=1.,
                targeted=False, num_classes=10, elementwise_best=True, num_rand_init=5, batch_size=clean_data.shape[0])

        image_final = self.adversary.perturb(clean_data, label).detach()
        
        self.surrogate_model.zero_grad()
        return image_final
