import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import AttackingObjectiveBase
from utils.registry import ATTACKING_OBJECTIVE

from copy import deepcopy
from collections import OrderedDict
from torch_scatter import scatter_mean

@ATTACKING_OBJECTIVE.register()
class BLEAttackingObjective(AttackingObjectiveBase):
    def __init__(self, cfg, attack_algo):
        super().__init__(cfg, attack_algo)

        self.current_input = None

        if hasattr(attack_algo, 'white_box_model'):
            model = attack_algo.white_box_model
        else:
            model = attack_algo.surrogate_model
        class_num = [i for i in model.modules()][-1].out_features
        self.register_buffer('class_wise_momentum_prob', torch.ones(class_num, class_num) / class_num)
        self.momentum_coefficient = 0.8

    def attack_objective(self, x, y):
        with torch.no_grad():
            curr_prob_term = scatter_mean(x.softmax(1), y[:, None], dim=0, out=torch.zeros_like(self.class_wise_momentum_prob))
            new_ema_prob = self.class_wise_momentum_prob.clone()
            new_ema_prob[y.unique()] = self.momentum_coefficient * new_ema_prob[y.unique()] + (1 - self.momentum_coefficient) * curr_prob_term[y.unique()]
            
            new_ema_prob_select = new_ema_prob.clone()
            new_ema_prob_select[torch.diag(torch.ones(new_ema_prob_select.shape[0])).bool()] = 0.
            
            label_mapping = y.new_zeros(new_ema_prob_select.shape[0], dtype=torch.long)
            for i in range(new_ema_prob_select.shape[0]):
                biased_prob, biased_class = F.normalize(new_ema_prob_select, dim=-1, p=1).max(dim=-1)
                max_item = biased_prob.argmax(dim=-1)
                label_mapping[max_item] = biased_class[max_item]
                new_ema_prob_select[max_item, :] = 0.
                new_ema_prob_select[:, biased_class[max_item]] = 0.
        loss = -F.cross_entropy(x, label_mapping[y])
        self.current_prob = new_ema_prob.detach()
        return loss

    def hook_after_attack(self, attacked_model):
        self.class_wise_momentum_prob = self.current_prob.clone()
        return
