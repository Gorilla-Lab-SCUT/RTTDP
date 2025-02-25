import torch
import torch.nn as nn
from torch.utils.data import Sampler
from robustbench.loaders import CustomImageFolder


class AttackBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, drop_last, poisoning_ratio):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.poisoning_ratio = poisoning_ratio
        self.current_batch_idx = 0
        
        self.finish_preprocess=False
        self.preprocess()  # preprocess before multiprocessing

    def preprocess(self):
        re = [i for i in iter(self)]
        self.finish_preprocess=True
        return
    
    def __iter__(self):
        batch = []

        for idx in self.sampler:
            batch.append(idx)

            if self.finish_preprocess is False and self.poisoning_ratio != 0:
                mark_as_poisoning_batch = (self.current_batch_idx * self.poisoning_ratio) // 1 == (self.current_batch_idx * self.poisoning_ratio)
                if mark_as_poisoning_batch:
                    if isinstance(self.sampler.data_source, CustomImageFolder):
                        data_path = self.sampler.data_source.samples[idx][0]
                        data_path_split = data_path.split('/')
                        data_path_split[-4] = "poisoning_" + data_path_split[-4]
                        self.sampler.data_source.samples[idx] = ("/".join(data_path_split), self.sampler.data_source.samples[idx][1])
                    else:
                        self.sampler.data_source.samples[idx][-1] = 'poisoning_attack'
                    
                
            if len(batch) == self.batch_size:
                yield batch
                batch = []
                if self.finish_preprocess is False and self.poisoning_ratio != 0:
                    self.current_batch_idx += 1

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class NonUniformAttackBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, drop_last, poisoning_ratio):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.poisoning_ratio = poisoning_ratio
        self.current_batch_idx = 0
        
        self.finish_preprocess=False
        self.preprocess()  # preprocess before multiprocessing

    def preprocess(self):
        re = [i for i in iter(self)]
        self.finish_preprocess=True
        return
    
    def __iter__(self):
        batch = []

        for idx in self.sampler:
            batch.append(idx)

            if self.finish_preprocess is False and self.poisoning_ratio != 0:
                mark_as_poisoning_batch = self.current_batch_idx < len(self) * self.poisoning_ratio
                if mark_as_poisoning_batch:
                    if isinstance(self.sampler.data_source, CustomImageFolder):
                        data_path = self.sampler.data_source.samples[idx][0]
                        data_path_split = data_path.split('/')
                        data_path_split[-4] = "poisoning_" + data_path_split[-4]
                        self.sampler.data_source.samples[idx] = ("/".join(data_path_split), self.sampler.data_source.samples[idx][1])
                    else:
                        self.sampler.data_source.samples[idx][-1] = 'poisoning_attack'
                
            if len(batch) == self.batch_size:
                yield batch
                batch = []
                if self.finish_preprocess is False and self.poisoning_ratio != 0:
                    self.current_batch_idx += 1

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class AttackSampleSampler(Sampler):
    def __init__(self, data_source, poisoning_ratio):
        self.data_source = data_source

        self.poisoning_ratio = poisoning_ratio
        
        self.preprocess()  # preprocess before multiprocessing

    def preprocess(self):
        if self.poisoning_ratio != 0:
            for i in iter(self):
                mark_as_poisoning_batch = (i * self.poisoning_ratio) // 1 == (i * self.poisoning_ratio)
                if isinstance(self.sampler.data_source, CustomImageFolder):
                    data_path = self.sampler.data_source.samples[i][0]
                    data_path_split = data_path.split('/')
                    data_path_split[-4] = "poisoning_" + data_path_split[-4]
                    self.sampler.data_source.samples[i] = ("/".join(data_path_split), self.sampler.data_source.samples[i][1])
                else:
                    self.sampler.data_source.samples[i][-1] = 'poisoning_attack'
        return

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)