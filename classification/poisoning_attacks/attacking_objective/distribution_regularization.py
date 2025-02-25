import torch
import torch.nn as nn
from .base import AttackingObjectiveBase
from utils.registry import ATTACKING_OBJECTIVE
from functools import partial
from collections import OrderedDict


@ATTACKING_OBJECTIVE.register()
class Distribution_Regularization(AttackingObjectiveBase):
    def __init__(self, cfg, attack_algo):
        super().__init__(cfg, attack_algo)

        self.bn_stat = {}

        if hasattr(attack_algo, 'white_box_model'):
            model = attack_algo.white_box_model
        else:
            model = attack_algo.surrogate_model
        self.class_num = [i for i in model.modules()][-1].out_features

    def hook_before_attack(self, attacked_model, clean_data, **kwargs):
        self.config_network(attacked_model, attack_hook=True)

        with torch.no_grad():
            _ = attacked_model(clean_data)
        self.lambda_param = torch.zeros(len(self.bn_stat.keys()), requires_grad=True).cuda()
        self.updated_lambda_param = self.lambda_param.clone()
        return
    
    def hook_after_attack(self, attacked_model, **kwargs):
        self.config_network(attacked_model, attack_hook=False)
        return

    def attack_objective(self, x, y):
        with torch.no_grad():
            self.lambda_param.data.copy_(self.updated_lambda_param)

        loss = 0
        component_num = 0
        for idx, bn_name in enumerate(self.bn_stat):
            loss -= self.bn_stat[bn_name][-1] * self.lambda_param[idx]
            component_num += 1
       
        loss /= component_num

        return loss
    
    def hook_towards_loss(self, loss):
        with torch.no_grad():
            param_update_grad = torch.autograd.grad(loss, self.lambda_param, retain_graph=True)[0]
            if 'BLEAttackingObjective' in self.cfg.ATTACK.OBJECTIVE.NAMES:
                scale = self.cfg.ATTACK.OBJECTIVE.BLEAttackingObjective.UPDATE_STEP_SCALE
            elif 'NHEAttackingObjective' in self.cfg.ATTACK.OBJECTIVE.NAMES:
                scale = self.cfg.ATTACK.OBJECTIVE.NHEAttackingObjective.UPDATE_STEP_SCALE
            else:
                scale = self.cfg.ATTACK.UPDATE_STEP_SCALE
                # raise ValueError("Attack Objective is not implemented.")
            self.updated_lambda_param = self.lambda_param - param_update_grad * 0.001 * scale
    
    def config_network(self, model: torch.nn.Module, attack_hook: bool = True):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module._forward_pre_hooks = OrderedDict()
                if attack_hook:
                    module.register_forward_pre_hook(partial(self.bn_forward_hook, module_name=name))
                else:
                    self.bn_stat = {}
    

    def bn_forward_hook(self, module, input, module_name: str = ''):
        assert self.cfg.CORRUPTION.DATASET in ['cifar10_c', 'cifar100_c', 'imagenet_c', 'modelnet40_c'], 'NotImplementedError'
        data = input[0]
        if self.cfg.CORRUPTION.DATASET == 'cifar10_c':
            if 'block1' in module_name or 'block2' in module_name:
                # instance-wise
                mean = data.mean(dim=[2, 3])
                var = data.var(dim=[2, 3], unbiased=False)
                type_ = 'instance'
            else:
                # batch-wise
                mean = data.mean(dim=[0, 2, 3])
                var = data.var(dim=[0, 2, 3], unbiased=False)
                type_ = 'batch'
                return
        elif self.cfg.CORRUPTION.DATASET == 'cifar100_c':
            if 'stage_3' not in module_name:
                # instance-wise
                mean = data.mean(dim=[2, 3])
                var = data.var(dim=[2, 3], unbiased=False)
                type_ = 'instance'
            else:
                # batch-wise
                mean = data.mean(dim=[0, 2, 3])
                var = data.var(dim=[0, 2, 3], unbiased=False)
                type_ = 'batch'
                return
        elif self.cfg.CORRUPTION.DATASET == 'imagenet_c':
            if 'layer4' not in module_name:
                # instance-wise
                mean = data.mean(dim=[2, 3])
                var = data.var(dim=[2, 3], unbiased=False)
                type_ = 'instance'
            else:
                # batch-wise
                mean = data.mean(dim=[0, 2, 3])
                var = data.var(dim=[0, 2, 3], unbiased=False)
                type_ = 'batch'
                return
        elif self.cfg.CORRUPTION.DATASET == 'modelnet40_c':
            mean = data.mean(dim=[2, 3])
            var = data.var(dim=[2, 3], unbiased=False)
            type_ = 'instance'

        if module_name not in self.bn_stat:
            with torch.no_grad():
                self.bn_stat[module_name] = [torch.distributions.normal.Normal(mean, (var + module.eps).sqrt()), 0.]
        else:
            source_dist = torch.distributions.normal.Normal(mean, (var + module.eps).sqrt())
            loss = torch.distributions.kl.kl_divergence(self.bn_stat[module_name][0], source_dist).sum(dim=-1).sum()
            self.bn_stat[module_name][-1] = loss
        return

