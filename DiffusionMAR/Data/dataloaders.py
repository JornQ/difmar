import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torchvision.transforms as transforms
import random

from Data.utils import get_img_paths, get_test_data_paths, unpair_img_paths

class IsalaUnpairedDataset(Dataset):
    def __init__(self, dataset_path, img_resized = 512, shuffle = True, clip_01 = True):
        input_paths, target_paths = get_img_paths(dataset_path)
        input_paths, target_paths = unpair_img_paths(input_paths, target_paths)
        
        self.image_size = (img_resized, img_resized)
        self.input_paths = input_paths
        self.target_paths = target_paths        
        self.clip_01 = clip_01
        self.shuffle = shuffle
        
        self.resize = transforms.Resize(img_resized, antialias=True)

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index):
        sample_paths = [self.input_paths[index], self.target_paths[index]]
        samples = []
        for sample_path in sample_paths:
            sample = None
            try:
                sample = Image.open(sample_path)
            except BaseException as e:
                print(sample_path)
            sample = np.asarray(sample, dtype=np.float32)
            sample = sample - 32768
            if self.clip_01:
                sample[sample < -1000] = -1000
                sample[sample > 3000] = 3000
            
            # sample = (sample + 1000)/4000
            sample = (sample - 1000)/2000
            
            sample = torch.Tensor(sample)
            sample = sample.unsqueeze(0)
            if not sample.size()[-1] == self.image_size[-1]:
                sample = self.resize(sample)
            samples.append(sample)
        return samples[0], samples[1]
    
    def get_loader(self, batch_size):
        return DataLoader(self, batch_size=batch_size, shuffle=self.shuffle)

class IsalaPairedDataset(Dataset):
    def __init__(self, dataset_path, img_resized = 512, shuffle = True, clip_01 = True, data_fraction = 1.0):
        input_paths, target_paths = get_img_paths(dataset_path)
        
        if data_fraction < 1:
            portion_size = int(len(input_paths) * data_fraction)
            
            instance = random.Random(2023)
            random_indices = instance.sample(range(len(input_paths)), portion_size)
            
            input_paths = [input_paths[i] for i in random_indices]
            target_paths = [target_paths[i] for i in random_indices]
        
        self.image_size = (img_resized, img_resized)
        self.input_paths = input_paths
        self.target_paths = target_paths        
        self.clip_01 = clip_01
        self.shuffle = shuffle
        
        self.resize = transforms.Resize(img_resized, antialias=True)

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index):
        sample_paths = [self.input_paths[index], self.target_paths[index]]
        samples = []
        for sample_path in sample_paths:
            sample = None
            try:
                sample = Image.open(sample_path)
            except BaseException as e:
                print(sample_path)
            sample = np.asarray(sample, dtype=np.float32)
            sample = sample - 32768
            if self.clip_01:
                sample[sample < -1000] = -1000
                sample[sample > 3000] = 3000
            
            # sample = (sample + 1000)/4000
            sample = (sample - 1000)/2000
            
            sample = torch.Tensor(sample)
            sample = sample.unsqueeze(0)
            if not sample.size()[-1] == self.image_size[-1]:
                sample = self.resize(sample)
            samples.append(sample)
        return samples[0], samples[1]
    
    def get_loader(self, batch_size):
        return DataLoader(self, batch_size=batch_size, shuffle=self.shuffle)
    
class IsalaPairedDatasetConditional(Dataset):
    def __init__(self, dataset_path, img_resized = 512, additional_condition = False, shuffle = True, clip_01 = True):
        input_paths, target_paths = get_img_paths(dataset_path)
        self.image_size = (img_resized, img_resized)
        self.input_paths = input_paths
        self.target_paths = target_paths        
        self.clip_01 = clip_01
        self.shuffle = shuffle
        self.additional_condition = additional_condition
   
        self.resize = transforms.Resize(img_resized, antialias=True)

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index):
        sample_paths = [self.input_paths[index], self.target_paths[index]]
        samples = []
        for sample_path in sample_paths:
            sample = None
            try:
                sample = Image.open(sample_path)
            except BaseException as e:
                print(sample_path)
            sample = np.asarray(sample, dtype=np.float32)
            sample = sample - 32768
            if self.clip_01:
                sample[sample < -1000] = -1000
                sample[sample > 3000] = 3000
            if self.additional_condition and len(samples) == 0:
                condition = sample.copy()
                fullcondition = sample.copy()
                condition[condition < -100] = -100
                condition[condition > 300] = 300
                fullcondition = (fullcondition + 1000)/4000
                condition = (condition + 1000)/4000
                fullcondition = torch.Tensor(fullcondition)
                condition = torch.Tensor(condition)
                fullcondition = fullcondition.unsqueeze(0)
                condition = condition.unsqueeze(0)
                if not condition.size()[-1] == self.image_size[-1]:
                    fullcondition = self.resize(fullcondition)
                    condition = self.resize(condition)
            sample = (sample + 1000)/4000
            sample = torch.Tensor(sample)
            sample = sample.unsqueeze(0)
            if not sample.size()[-1] == self.image_size[-1]:
                sample = self.resize(sample)
            samples.append(sample)
        if self.additional_condition:
            return samples[0], samples[1], condition, fullcondition
        return samples[0], samples[1]
    
    def get_loader(self, batch_size):
        return DataLoader(self, batch_size=batch_size, shuffle=self.shuffle)
    
class TestDataset(Dataset):
    def __init__(self, dataset_path, img_resized = 64, shuffle = True, clip_01 = True):
        input_paths, target_paths = get_test_data_paths(dataset_path)
        self.image_size = (img_resized, img_resized)
        self.input_paths = input_paths
        self.target_paths = target_paths        
        self.clip_01 = clip_01
        self.shuffle = shuffle
        
        #self.resize = transforms.Resize(img_resized, antialias=True)

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index):
        sample_paths = [self.input_paths[index], self.target_paths[index]]
        samples = []
        for sample_path in sample_paths:
            sample = None
            try:
                sample = np.load(sample_path)
            except BaseException as e:
                print(sample_path)
            sample = np.asarray(sample, dtype=np.float32)
            sample = torch.Tensor(sample)
            sample = sample.unsqueeze(0)
            samples.append(sample)
        return samples[0], samples[1]
    
    def get_loader(self, batch_size):
        return DataLoader(self, batch_size=batch_size, shuffle=self.shuffle)