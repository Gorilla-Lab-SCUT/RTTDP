from .base import SurrogateModelAttackBase, softmax_entropy
from utils.registry import ATTACK_REGISTRY, ATTACKING_OBJECTIVE

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from advertorch.attacks import LinfPGDAttack
from copy import deepcopy
from collections import OrderedDict
from functools import partial

from .attacking_objective import *

from autoattack import AutoAttack

from copy import deepcopy

@ATTACK_REGISTRY.register()
class SurrogateModelEstimateAttack(SurrogateModelAttackBase):
    def __init__(self, cfg, surrogate_model):
        super().__init__(cfg, surrogate_model)

        self.eps = 0.3  # for image
        self.alpha = 0.01

        params, _ = self.collect_learnable_params()
        if cfg.CORRUPTION.DATASET == "imagenet_c":
            if 'cotta' == cfg.MODEL.ADAPTATION or 'roid' == cfg.MODEL.ADAPTATION:
                self.inner_optimizer = optim.SGD(params, lr=0.00001)
            else:
                self.inner_optimizer = optim.SGD(params, lr=0.00025 * 2)
        else:
            self.inner_optimizer = optim.SGD(params, lr=1e-1, weight_decay=1e-3)

        self.attacking_objectives = [ATTACKING_OBJECTIVE.get(name)(cfg, self).cuda() for name in cfg.ATTACK.OBJECTIVE.NAMES]

        self.clean_data = None
        self.input_data = None

    @torch.enable_grad()
    def forward(self, clean_data, label):
        self.clean_data = clean_data.clone()
        self.surrogate_model.train()
        self.surrogate_model.zero_grad()
        
        rand_init = True
        for objective in self.attacking_objectives:
            objective.hook_before_attack(self.surrogate_model, clean_data, label=label)
            rand_init &= objective.rand_init
        
        self.adversary = LinfPGDAttack(
            self.surrogate_model, loss_fn=self.loss_fn, eps=self.eps,
            nb_iter=40, eps_iter=self.alpha, rand_init=rand_init, clip_min=0, clip_max=1.0,
            targeted=False)  # clip_min = -1 for pc; 0 for image

        image_final = self.adversary.perturb(clean_data, label).detach()
        self.input_data = image_final.clone()

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
        self.estimate_target_bias(probs)
        return

    @torch.enable_grad()
    def estimate_target_bias(self, target_prob):
        self.surrogate_model.train()
        for i in range(10):
            bs = self.cfg.TEST.BATCH_SIZE
            remaining_sample_idx = np.random.choice(self.clean_data.shape[0], bs - self.input_data.shape[0])
            batch_data = torch.cat([self.input_data, self.clean_data[remaining_sample_idx]], dim=0)

            surrogate_prediction = self.surrogate_model(batch_data)[:self.input_data.shape[0]]

            loss = (torch.nn.functional.kl_div(F.log_softmax(surrogate_prediction, dim=1), target_prob, reduction='batchmean') + torch.nn.functional.kl_div((target_prob + 1e-5).log(), surrogate_prediction.softmax(dim=1), reduction='batchmean')) / 2.
            if loss.item() < 1e-7:
                del loss
                self.inner_optimizer.zero_grad()
                self.inner_optimizer.step()
                break
            self.inner_optimizer.zero_grad()
            loss.backward()
            self.inner_optimizer.step()
        return
    
    def collect_learnable_params(self, ):
        params = []
        names = []

        if self.cfg.MODEL.ADAPTATION == 'cotta' and False:
            params = self.surrogate_model.parameters()
            names = ['all']
        else:
            for nm, m in self.surrogate_model.named_modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                    for np, p in m.named_parameters():
                        if np in ['weight', 'bias']:
                            params.append(p)
                            names.append(f"{nm}.{np}")
                
        return params, names
