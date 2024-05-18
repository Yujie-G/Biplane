import os
import re

import torch
import torch.utils.data as data
import torchvision
import numpy as np
from prefetch_generator import BackgroundGenerator
import h5py as h5

import exr
from utils import *
from config import *

class DataLoaderX(data.DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class TrainDataset_ubo2014_randomdirection(data.Dataset):

    def __init__(self, config: RepConfig, test_run: bool=False):
        super().__init__()
        self.config = config
        self.dataset_name = type(self).__name__
        self.dataloader = DataLoaderX(
        # self.dataloader = data.DataLoader(
            self,
            batch_size=self.config.batch_size,
            pin_memory=True,
            num_workers=config.num_workers,
            shuffle=True if not test_run else False,
            drop_last=False
        )
        self.test_run = test_run
        self.data_dir = os.path.join(config.root, config.data_root)
        self.query_perm = np.random.permutation(config.ubo2014_btf_size[2])[:config.num_query]
        self.all_data = {}
        if config.validate_only or config.compress_only:
            self.if_cache_in_memory = False
            print(f'train data skipped.' + ' ' * 25)
            return
        for m in config.train_materials:
            try:
                self.load_randomdirection_data(m)
                self.if_cache_in_memory = True
            except KeyboardInterrupt:
                print(f'train data "{m}" canceled.' + ' ' * 25)
                self.if_cache_in_memory = False

    def load_randomdirection_data(self, m):
        for u_index in range(self.config.train_uv_size[0]): ## dataloaders access integer u_index/v_indices, and return float u_index/v_index (0~1) in __getitem__
            for v_index in range(self.config.train_uv_size[1]):
                print(f'loading train data... "{m}" {u_index}.000_{v_index}.000', end='\r')
                with h5.File(os.path.join(
                            self.data_dir, m, 
                            f'{self.config.ubo2014_btf_size[2]}', 
                            f'{u_index * self.config.ubo2014_btf_uv_interval[0]}.000_{v_index * self.config.ubo2014_btf_uv_interval[1]}.000.h5'
                ),'r') as f:
                    self.all_data[f'{m}_{u_index}.000_{v_index}.000'] = f['data'][...].reshape(-1, 7)[self.query_perm]
        print(f'train data "{m}" finished. totally {len(self.all_data.keys())} files.' + ' ' * 25)

    def locate(self, num_query, array):
        num = num_query
        for i in range(len(array)):
            if num < array[i]:
                return i, num
            num -= array[i]
        raise ValueError(f'[{self.dataset_name}] locate out of boundary.')

    def __getitem__(self, index):

        material_index, brdf_index = self.locate(index, np.array(self.config.each_train_brdf_num) * self.config.num_query)
        material = self.config.train_materials[material_index]
        uv_index = brdf_index // self.config.num_query
        query_index = brdf_index % self.config.num_query
        u_index = uv_index // self.config.train_uv_size[1]
        v_index = uv_index % self.config.train_uv_size[1]
        
        if self.if_cache_in_memory:
            data = self.all_data[f'{material}_{u_index}.000_{v_index}.000'][query_index]
        else:
            with h5.File(os.path.join(self.data_dir, material, f'{self.config.ubo2014_btf_size[2]}', f'{u_index * self.config.ubo2014_btf_uv_interval[0]}.000_{v_index * self.config.ubo2014_btf_uv_interval[1]}.000.h5'), 'r') as f:
                data = f['data'][...].reshape(-1, 7)[self.query_perm[query_index]]
        
        view  = data[0:2]
        light = data[2:4]
        color = data[4:7]

        return index, view, light, u_index * 1.0 / self.config.train_uv_size[0], v_index * 1.0 / self.config.train_uv_size[1], color

    def __len__(self):
        return self.config.train_brdf_num * self.config.num_query

class ValidateDataset_ubo2014_randomdirection(data.Dataset):

    def __init__(self, config: RepConfig):
        super().__init__()
        self.config = config
        self.dataset_name = type(self).__name__
        self.dataloader = DataLoaderX(
        # self.dataloader = data.DataLoader(
            self,
            batch_size=config.batch_size,
            pin_memory=True,
            num_workers=config.num_workers,
            shuffle=True,
            drop_last=False
        )
        self.data_dir = os.path.join(config.root, config.data_root)
        self.query_perm = np.random.permutation(config.ubo2014_btf_size[2])[:config.num_query]
        self.all_data = {}
        for m in config.validate_materials:
            try:
                self.load_randomdirection_data(m)
                self.if_cache_in_memory = True
            except KeyboardInterrupt:
                print(f'validate data "{m}" canceled.' + ' ' * 25)
                self.if_cache_in_memory = False
                
    def load_randomdirection_data(self, m):
        for u_index in range(self.config.validate_uv_size[0]):
            for v_index in range(self.config.validate_uv_size[1]):
                print(f'loading validate data... "{m}" {u_index}.000_{v_index}.000', end='\r')
                with h5.File(os.path.join(
                            self.data_dir, m, 
                            f'{self.config.ubo2014_btf_size[2]}', 
                            f'{u_index * self.config.ubo2014_btf_uv_interval[0]}.000_{v_index * self.config.ubo2014_btf_uv_interval[1]}.000.h5'
                        ),'r') as f:
                    self.all_data[f'{m}_{u_index}.000_{v_index}.000'] = f['data'][...].reshape(-1, 7)[self.query_perm]            
        print(f'validate data "{m}" finished. totally {len(self.all_data.keys())} files.' + ' ' * 25)

    def locate(self, num_query, array):
        num = num_query
        for i in range(len(array)):
            if num < array[i]:
                return i, num
            num -= array[i]
        raise ValueError(f'[{self.dataset_name}] locate out of boundary.')

    def __getitem__(self, index):

        material_index, brdf_index = self.locate(index, np.array(self.config.each_validate_brdf_num) * self.config.num_query)
        material = self.config.validate_materials[material_index]
        uv_index = brdf_index // self.config.num_query
        query_index = brdf_index % self.config.num_query
        u_index = uv_index // self.config.validate_uv_size[1]
        v_index = uv_index % self.config.validate_uv_size[1]
        
        if self.if_cache_in_memory:
            data = self.all_data[f'{material}_{u_index}.000_{v_index}.000'][query_index]
        else:
            with h5.File(os.path.join(self.data_dir, material, f'{self.config.ubo2014_btf_size[2]}', f'{u_index * self.config.ubo2014_btf_uv_interval[0]}.000_{v_index * self.config.ubo2014_btf_uv_interval[1]}.000.h5'), 'r') as f:
                data = f['data'][...].reshape(-1, 7)[self.query_perm[query_index]]
        
        view  = data[0:2]
        light = data[2:4]
        color = data[4:7]

        return index, view, light, u_index * 1.0 / self.config.validate_uv_size[0], v_index * 1.0 / self.config.validate_uv_size[1], color

    def __len__(self):
        return self.config.validate_brdf_num * self.config.num_query


class TrainDataset_ubo2014_allrandom(data.Dataset):

    def __init__(self, config: RepConfig, test_run: bool=False):
        super().__init__()
        self.config = config
        self.dataset_name = type(self).__name__
        self.dataloader = DataLoaderX(
        # self.dataloader = data.DataLoader(
            self,
            batch_size=config.batch_size,
            pin_memory=True,
            num_workers=config.num_workers,
            shuffle=True if not test_run else False,
            drop_last=False
        )
        self.test_run = test_run
        self.data_dir = os.path.join(config.root, config.data_root)
        
    def locate(self, num_query, array):
        num = num_query
        for i in range(len(array)):
            if num < array[i]:
                return i, num
            num -= array[i]
        raise ValueError(f'[{self.dataset_name}] locate out of boundary.')

    def __getitem__(self, index):

        material_index, file_index = index // self.config.cache_file_shape[0], index % self.config.cache_file_shape[0]
        material = self.config.train_materials[material_index]
        
        f = exr.read(os.path.join(
                    self.data_dir, 
                    material, 
                    f'{self.config.train_uv_size[0]}x{self.config.train_uv_size[1]}'\
                    f'[{self.config.ubo2014_btf_uv_interval[0]}x{self.config.ubo2014_btf_uv_interval[1]}]_{self.config.num_query}in1', 
                    f'0_{file_index}.exr'
                ),
                channels=['0_wix', '1_wiy', '2_wox', '3_woy', '4_u', '5_v', '6_R', '7_G', '8_B']
            ).reshape(-1, 9) ## [400, 400, 9] -> [-1, 9]
        return material_index, f

    def __len__(self):
        return self.config.cache_file_shape[0] * len(self.config.train_materials)

class ValidateDataset_ubo2014_allrandom(data.Dataset):

    def __init__(self, config: RepConfig):
        super().__init__()
        self.config = config
        self.dataset_name = type(self).__name__
        self.dataloader = DataLoaderX(
        # self.dataloader = data.DataLoader(
            self,
            batch_size=config.batch_size,
            pin_memory=True,
            num_workers=config.num_workers,
            shuffle=True,
            drop_last=False
        )
        self.data_dir = os.path.join(config.root, config.data_root)
        self.query_perm = np.random.permutation(config.ubo2014_btf_size[2])[:config.num_query]
         
    def locate(self, num_query, array):
        num = num_query
        for i in range(len(array)):
            if num < array[i]:
                return i, num
            num -= array[i]
        raise ValueError(f'[{self.dataset_name}] locate out of boundary.')

    def __getitem__(self, index):

        material_index, file_index = index // self.config.cache_file_shape[0], index % self.config.cache_file_shape[0]
        material = self.config.train_materials[material_index]
        
        f = exr.read(os.path.join(
                    self.data_dir, 
                    material, 
                    f'{self.config.train_uv_size[0]}x{self.config.train_uv_size[1]}'\
                    f'[{self.config.ubo2014_btf_uv_interval[0]}x{self.config.ubo2014_btf_uv_interval[1]}]_{self.config.num_query}in1', 
                    f'0_{file_index}.exr'
                ),
                channels=['0_wix', '1_wiy', '2_wox', '3_woy', '4_u', '5_v', '6_R', '7_G', '8_B']
            ).reshape(-1, 9) ## [400, 400, 9] -> [-1, 9]
        return material_index, f

    def __len__(self):
        return self.config.cache_file_shape[0] * len(self.config.validate_materials)
