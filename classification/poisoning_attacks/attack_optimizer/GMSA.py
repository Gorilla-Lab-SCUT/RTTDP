
from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
from .utils_torch import *


class LinfConfPGDAttack:
    """
    Confident PGD Attack with order=Linf

    :param loss_func: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: if the attack is targeted.
    """

    def __init__(
            self, model, eps=0.1, nb_iter=100,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False, num_classes=10, elementwise_best=False, num_rand_init=1):
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.targeted = targeted
        self.elementwise_best = elementwise_best
        self.model = model
        self.num_classes = num_classes
        self.num_rand_init = num_rand_init
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

        self.clip_min = clip_min
        self.clip_max = clip_max

    def get_loss(self, x, y, targeted=False, y_target=None):
        logits = self.model(x)
        if targeted:
            u = torch.arange(logits.shape[0])
            loss = -(logits[u, y] - logits[u, y_target]) 
        else:
            logits_sorted, ind_sorted = logits.sort(dim=1)
            ind = (ind_sorted[:, -1] == y).float()
            u = torch.arange(logits.shape[0])

            loss = -(logits[u, y] - logits_sorted[:, -2] * ind - logits_sorted[:, -1] * (
                1. - ind)) 

        return loss 

    def perturb_once(self, x, y, targeted=False, y_target=None):

        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta)
        delta.requires_grad_()

        if self.elementwise_best:
            loss = self.get_loss(x, y, targeted, y_target)
            worst_loss = loss.data.clone()
            worst_perb = delta.data.clone()

        if self.rand_init:
            delta.data.uniform_(-self.eps, self.eps)
            delta.data = (torch.clamp(x.data + delta.data, min=self.clip_min, max=self.clip_max) - x.data)

        for ii in range(self.nb_iter):
            adv_x = x + delta
            loss = self.get_loss(adv_x, y, targeted, y_target)

            if self.elementwise_best:
                cond = loss.data > worst_loss
                worst_loss[cond] = loss.data[cond]
                worst_perb[cond] = delta.data[cond]

            loss.mean().backward()
            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + grad_sign * self.eps_iter
            delta.data = torch.clamp(delta.data, min=-self.eps, max=self.eps)
            delta.data = torch.clamp(x.data + delta.data, min=self.clip_min, max=self.clip_max) - x.data

            delta.grad.data.zero_()

        if self.elementwise_best:
            adv_x = x + delta
            loss = self.get_loss(adv_x, y, targeted, y_target)
            cond = loss.data > worst_loss
            worst_loss[cond] = loss.data[cond]
            worst_perb[cond] = delta.data[cond]
        else:
            worst_perb = delta.data

        return worst_perb
    
    def get_error(self, x, y):
        with torch.no_grad():
            logits = self.model(x)
            loss = f7p_loss(logits, y, reduction='none')
        return loss
 
    def perturb(self, x, y):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
        :return: tensor containing perturbed inputs.
        """

        self.model.eval()

        x = x.detach().clone().cuda()
        y = y.detach().clone().cuda()
        
        worst_error = self.get_error(x, y)
        worst_perb = torch.zeros_like(x)

        for i in range(self.num_rand_init):
            curr_worst_perb = self.perturb_once(x, y, targeted=False)
            curr_error = self.get_error(x+curr_worst_perb, y)
            cond = curr_error.data > worst_error.data
            worst_error[cond] = curr_error[cond]
            worst_perb[cond] = curr_worst_perb[cond]

        for k in range(1, self.num_classes):
            y_target = (y + k) % self.num_classes
            curr_worst_perb = self.perturb_once(x, y, targeted=True, y_target=y_target)
            curr_error = self.get_error(x+curr_worst_perb, y)
            cond = curr_error.data > worst_error.data
            worst_error[cond] = curr_error[cond]
            worst_perb[cond] = curr_worst_perb[cond]

        return x + worst_perb

class ConfGMSAMINAttack:
    """
    GMSA-MIN Attack with order=Linf

    :param loss_func: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: if the attack is targeted.
    """

    def __init__(
            self, models, eps=0.1, nb_iter=100,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False, num_classes=10, elementwise_best=False, 
            num_rand_init=1, batch_size=64):

        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.targeted = targeted
        self.elementwise_best = elementwise_best
        self.models = models
        self.num_classes = num_classes
        self.num_rand_init = num_rand_init
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        self.batch_size = batch_size

        self.clip_min = clip_min
        self.clip_max = clip_max

    def get_loss(self, model, x, y, targeted=False, y_target=None):
        logits = model(x)
        if targeted:
            u = torch.arange(logits.shape[0])
            loss = -(logits[u, y] - logits[u, y_target]) 
        else:
            logits_sorted, ind_sorted = logits.sort(dim=1)
            ind = (ind_sorted[:, -1] == y).float()
            u = torch.arange(logits.shape[0])

            loss = -(logits[u, y] - logits_sorted[:, -2] * ind - logits_sorted[:, -1] * (
                1. - ind)) 
    
        return loss 

    def get_ensemble_loss(self, x, y, targeted=False, y_target=None):
        min_loss = None
        for model in self.models:
            curr_loss = self.get_loss(model, x, y, targeted, y_target)

            if min_loss is None:
                min_loss = curr_loss
            else:
                cond = curr_loss.data < min_loss.data
                min_loss[cond] = curr_loss[cond]

        return min_loss
    
    def restore_batchnorm(self, model):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight = nn.Parameter(m.ckpt_weight_bak, requires_grad=False)
                m.bias = nn.Parameter(m.ckpt_bias_bak, requires_grad=False)
                m.requires_grad_(False)

    def configure_batchnorm(self, model, start_idx, end_idx):
        """Configure model."""
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight = nn.Parameter(m.ckpt_weight_bak[start_idx:end_idx], requires_grad=False)
                m.bias = nn.Parameter(m.ckpt_bias_bak[start_idx:end_idx], requires_grad=False)
                m.requires_grad_(False)

    def add_model(self, model):
        self.models.append(model)
        self.save_batchnorm(model)

    def save_batchnorm(self, model):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.register_buffer("ckpt_weight_bak", m.weight)
                m.register_buffer("ckpt_bias_bak", m.bias)


    def perturb_once(self, x, y, targeted=False, y_target=None):
        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta)
        delta.requires_grad_()

        if self.elementwise_best:
            with torch.no_grad():
                loss = self.get_ensemble_loss(x, y, targeted, y_target)
            worst_loss = loss.data.clone()
            worst_perb = delta.data.clone()

        if self.rand_init:
            delta.data.uniform_(-self.eps, self.eps)
            delta.data = (torch.clamp(x.data + delta.data, min=self.clip_min, max=self.clip_max) - x.data)

        for ii in range(self.nb_iter*len(self.models)):
            adv_x = x + delta
            loss = self.get_ensemble_loss(adv_x, y, targeted, y_target)

            if self.elementwise_best:
                cond = loss.data > worst_loss
                worst_loss[cond] = loss.data[cond]
                worst_perb[cond] = delta.data[cond]

            loss.mean().backward()
            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + grad_sign * self.eps_iter
            delta.data = torch.clamp(delta.data, min=-self.eps, max=self.eps)
            delta.data = torch.clamp(x.data + delta.data, min=self.clip_min, max=self.clip_max) - x.data

            delta.grad.data.zero_()

        if self.elementwise_best:
            adv_x = x + delta
            with torch.no_grad():
                loss = self.get_ensemble_loss(adv_x, y, targeted, y_target)
            cond = loss.data > worst_loss
            worst_loss[cond] = loss.data[cond]
            worst_perb[cond] = delta.data[cond]
        else:
            worst_perb = delta.data
        
        return worst_perb

    def get_error(self, x, y):
        min_loss = None
        with torch.no_grad():
            for model in self.models:
                logits = model(x)
                curr_loss = f7p_loss(logits, y, reduction='none')

                if min_loss is None:
                    min_loss = curr_loss
                else:
                    cond = curr_loss.data < min_loss.data
                    min_loss[cond] = curr_loss[cond]

        return min_loss
    
    def perturb(self, x_test, y_test):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
        :return: tensor containing perturbed inputs.
        """
        for model in self.models:
            model.eval()

        x_test = x_test.detach().clone().cuda()
        y_test = y_test.detach().clone().cuda()

        adv_x_test = []
        n_batches = int(np.ceil(x_test.shape[0] / self.batch_size))
        for batch_idx in range(n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, x_test.shape[0])

            for k, model in enumerate(self.models):
                if k>0:
                    self.configure_batchnorm(model, start_idx, end_idx)

            x = x_test[start_idx:end_idx, :]
            y = y_test[start_idx:end_idx]
            
            worst_error = self.get_error(x, y)
            worst_perb = torch.zeros_like(x)

            for i in range(self.num_rand_init):
                curr_worst_perb = self.perturb_once(x, y, targeted=False)
                curr_error = self.get_error(x+curr_worst_perb, y)
                cond = curr_error.data > worst_error.data
                worst_error[cond] = curr_error[cond]
                worst_perb[cond] = curr_worst_perb[cond]

            for k in range(1, self.num_classes):
                y_target = (y + k) % self.num_classes
                curr_worst_perb = self.perturb_once(x, y, targeted=True, y_target=y_target)
                curr_error = self.get_error(x+curr_worst_perb, y)
                cond = curr_error.data > worst_error.data
                worst_error[cond] = curr_error[cond]
                worst_perb[cond] = curr_worst_perb[cond]

            adv_x_test.append(x + worst_perb)

        adv_x_test = torch.cat(adv_x_test, dim=0)
        for k, model in enumerate(self.models):
            if k>0:
                self.restore_batchnorm(model)

        return adv_x_test


class ConfGMSAAVGAttack:
    """
    GMSA-AVG Attack with order=Linf

    :param loss_func: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: if the attack is targeted.
    """

    def __init__(
            self, models, eps=0.1, nb_iter=100,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False, num_classes=10, elementwise_best=False, 
            num_rand_init=1, batch_size=64):

        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.targeted = targeted
        self.elementwise_best = elementwise_best
        self.models = models
        self.num_classes = num_classes
        self.num_rand_init = num_rand_init
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        self.batch_size = batch_size

        self.clip_min = clip_min
        self.clip_max = clip_max

    def get_loss(self, model, x, y, targeted=False, y_target=None):
        logits = model(x)
        if targeted:
            u = torch.arange(logits.shape[0])
            loss = -(logits[u, y] - logits[u, y_target])
        else:
            logits_sorted, ind_sorted = logits.sort(dim=1)
            ind = (ind_sorted[:, -1] == y).float()
            u = torch.arange(logits.shape[0])

            loss = -(logits[u, y] - logits_sorted[:, -2] * ind - logits_sorted[:, -1] * (
                1. - ind)) 

        return loss 

    def get_ensemble_loss(self, x, y, targeted=False, y_target=None, update=False):
        loss = 0.0
        for model in self.models:
            curr_loss = self.get_loss(model, x, y, targeted, y_target)

            if update:
                curr_loss.mean().backward()
            loss += curr_loss.data

        return loss

    def restore_batchnorm(self, model):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight = nn.Parameter(m.ckpt_weight_bak, requires_grad=False)
                m.bias = nn.Parameter(m.ckpt_bias_bak, requires_grad=False)
                m.requires_grad_(False)

    def configure_batchnorm(self, model, start_idx, end_idx):
        """Configure model."""
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight = nn.Parameter(m.ckpt_weight_bak[start_idx:end_idx], requires_grad=False)
                m.bias = nn.Parameter(m.ckpt_bias_bak[start_idx:end_idx], requires_grad=False)
                m.requires_grad_(False)

    def add_model(self, model):
        self.models.append(model)
        self.save_batchnorm(model)

    def save_batchnorm(self, model):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.register_buffer("ckpt_weight_bak", m.weight)
                m.register_buffer("ckpt_bias_bak", m.bias)

    def perturb_once(self, x, y, targeted=False, y_target=None):
        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta)
        delta.requires_grad_()

        if self.elementwise_best:
            with torch.no_grad():
                loss = self.get_ensemble_loss(x, y, targeted, y_target, update=False)
            worst_loss = loss.data.clone()
            worst_perb = delta.data.clone()

        if self.rand_init:
            delta.data.uniform_(-self.eps, self.eps)
            delta.data = (torch.clamp(x.data + delta.data, min=self.clip_min, max=self.clip_max) - x.data)

        for ii in range(self.nb_iter):
            adv_x = x + delta
            loss = self.get_ensemble_loss(adv_x, y, targeted, y_target, update=True)
            
            if self.elementwise_best:
                cond = loss.data > worst_loss
                worst_loss[cond] = loss.data[cond]
                worst_perb[cond] = delta.data[cond]

            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + grad_sign * self.eps_iter
            delta.data = torch.clamp(delta.data, min=-self.eps, max=self.eps)
            delta.data = torch.clamp(x.data + delta.data, min=self.clip_min, max=self.clip_max) - x.data

            delta.grad.data.zero_()

        if self.elementwise_best:
            adv_x = x + delta
            with torch.no_grad():
                loss = self.get_ensemble_loss(adv_x, y, targeted, y_target, update=False)
            cond = loss.data > worst_loss
            worst_loss[cond] = loss.data[cond]
            worst_perb[cond] = delta.data[cond]
        else:
            worst_perb = delta.data
        
        return worst_perb

    def get_error(self, x, y):
        loss = 0.0
        with torch.no_grad():
            for model in self.models:
                logits = model(x)
                curr_loss = f7p_loss(logits, y, reduction='none')
                loss += curr_loss.data

        return loss

    def perturb(self, x_test, y_test):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
        :return: tensor containing perturbed inputs.
        """
        for model in self.models:
            model.eval()

        x_test = x_test.detach().clone().cuda()
        y_test = y_test.detach().clone().cuda()

        adv_x_test = []
        n_batches = int(np.ceil(x_test.shape[0] / self.batch_size))
        for batch_idx in range(n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, x_test.shape[0])

            for k, model in enumerate(self.models):
                if k>0:
                    self.configure_batchnorm(model, start_idx, end_idx)

            x = x_test[start_idx:end_idx, :]
            y = y_test[start_idx:end_idx]
            
            worst_error = self.get_error(x, y)
            worst_perb = torch.zeros_like(x)

            for i in range(self.num_rand_init):
                curr_worst_perb = self.perturb_once(x, y, targeted=False)
                curr_error = self.get_error(x+curr_worst_perb, y)
                cond = curr_error.data > worst_error.data
                worst_error[cond] = curr_error[cond]
                worst_perb[cond] = curr_worst_perb[cond]

            for k in range(1, self.num_classes):
                y_target = (y + k) % self.num_classes
                curr_worst_perb = self.perturb_once(x, y, targeted=True, y_target=y_target)
                curr_error = self.get_error(x+curr_worst_perb, y)
                cond = curr_error.data > worst_error.data
                worst_error[cond] = curr_error[cond]
                worst_perb[cond] = curr_worst_perb[cond]

            adv_x_test.append(x + worst_perb)

        adv_x_test = torch.cat(adv_x_test, dim=0)
        for k, model in enumerate(self.models):
            if k>0:
                self.restore_batchnorm(model)
                
        return adv_x_test

class LinfPGDAttack:
    """
    PGD Attack with order=Linf

    :param loss_func: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: if the attack is targeted.
    """

    def __init__(
            self, model, eps=0.1, nb_iter=100,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False, num_classes=10, elementwise_best=False, num_rand_init=1):
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.targeted = targeted
        self.elementwise_best = elementwise_best
        self.model = model
        self.num_classes = num_classes
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        self.num_rand_init = num_rand_init

        self.clip_min = clip_min
        self.clip_max = clip_max

    def get_loss(self, x, y):
        outputs = self.model(x)
        loss = self.loss_func(outputs, y)
        if self.targeted:
            loss = -loss
        return loss 

    def perturb_once(self, x, y):

        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta)
        delta.requires_grad_()

        if self.elementwise_best:
            loss = self.get_loss(x, y)
            worst_loss = loss.data.clone()
            worst_perb = delta.data.clone()

        if self.rand_init:
            delta.data.uniform_(-self.eps, self.eps)
            delta.data = (torch.clamp(x.data + delta.data, min=self.clip_min, max=self.clip_max) - x.data)

        for ii in range(self.nb_iter):
            adv_x = x + delta
            loss = self.get_loss(adv_x, y)

            if self.elementwise_best:
                cond = loss.data > worst_loss
                worst_loss[cond] = loss.data[cond]
                worst_perb[cond] = delta.data[cond]

            loss.mean().backward()
            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + grad_sign * self.eps_iter
            delta.data = torch.clamp(delta.data, min=-self.eps, max=self.eps)
            delta.data = torch.clamp(x.data + delta.data, min=self.clip_min, max=self.clip_max) - x.data

            delta.grad.data.zero_()

        if self.elementwise_best:
            adv_x = x + delta
            loss = self.get_loss(adv_x, y)
            cond = loss.data > worst_loss
            worst_loss[cond] = loss.data[cond]
            worst_perb[cond] = delta.data[cond]
        else:
            worst_perb = delta.data

        return worst_perb

    def perturb(self, x, y):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
        :return: tensor containing perturbed inputs.
        """

        self.model.eval()

        x = x.detach().clone().cuda()
        y = y.detach().clone().cuda()

        worst_error = None
        worst_perb = None
        for i in range(self.num_rand_init):
            curr_worst_perb = self.perturb_once(x, y)
            with torch.no_grad():
                curr_error = self.get_loss(x+curr_worst_perb, y)
            if worst_error is None:
                worst_error = curr_error.data
                worst_perb = curr_worst_perb
            else:
                cond = curr_error.data > worst_error
                worst_error[cond] = curr_error.data[cond]
                worst_perb[cond] = curr_worst_perb[cond]

        return x + worst_perb