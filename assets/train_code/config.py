from datetime import datetime
import os
from pdb import set_trace as debug

import numpy as np

import model

## to avoid cuda fork() error
# from multiprocessing import set_start_method
# set_start_method('spawn')

class BaseConfig():

    def to_lines(self):
        lines = []
        lines.append('='*80)
        lines.append(f'[{self.__class__.__name__}]')
        lines.append('-'*80)
        if hasattr(self, "comments"):
            lines.append(f'{"comments":<30} {self.comments}')
        for k,v in vars(self).items():
            if k in ['data_root', 'network', 'trainer', 'save_root', 'decoder_params']:
                lines.append('')
            if k not in ['comments', 'other_info']:
                lines.append(f'{k:<30} {v}')
        lines.append('='*80)
        return lines
        
    def __repr__(self) -> str:
        string = ''
        for line in self.to_lines():
            string += line + '\n'
        return string
        
class RepConfig(BaseConfig):

    def __init__(self, device, global_random_seed):

        self.other_info = 'D6'
        self.start_time = datetime.now().strftime(r'%m%d_%H%M%S')
        self.random_seed = global_random_seed
        self.device = device
        self.root = '/test/repositories/mitsuba-pytorch-tensorNLB/'

        ## data config
        self.data_root = '/data/mitsuba-pytorch-tensorNLB/ubo2014_resampled_allrandom'
        self.train_dataset = 'TrainDataset_ubo2014_allrandom'
        self.validate_dataset = 'ValidateDataset_ubo2014_allrandom'
        self.ubo2014_btf_size = [400, 400, 22801] 
                ## this is the actual BTF data size. NOT all of them are necessarily sampled and saved into datafiles, may be just sparsely sampled by uv_intervals
        self.ubo2014_btf_uv_interval = [1, 1]
                ## this is the sparsely sampling interval for BTF data. only affects dataloaders. i.e., load data that numbered (i * interval)
        self.train_uv_size = self.validate_uv_size = [self.ubo2014_btf_size[0] // self.ubo2014_btf_uv_interval[0], self.ubo2014_btf_size[1] // self.ubo2014_btf_uv_interval[1]] 
                ## the equivalent BTF size used for training and validating. the limitation for all convertions in torch codes.
        self.num_query = 400 ## how many random queries are used in training and validating.
        self.cache_file_shape = [400, 400, 400, 9]
        self.train_materials = [
            *[f'carpet{i:02d}' for i in range(1, 13)],
            *[f'fabric{i:02d}' for i in range(1, 13)],
            *[f'felt{i:02d}' for i in range(1, 13)],
            *[f'leather{i:02d}' for i in range(1, 13)],
            *[f'stone{i:02d}' for i in range(1, 13)],
            *[f'wallpaper{i:02d}' for i in range(1, 13)],
            *[f'wood{i:02d}' for i in range(1, 13)],
        ]
        self.each_train_brdf_num = [self.train_uv_size[0] * self.train_uv_size[1] for _ in self.train_materials]
        self.validate_materials = []
        self.each_validate_brdf_num = [self.validate_uv_size[0] * self.validate_uv_size[1] for _ in self.validate_materials]
        self.train_brdf_num = sum(self.each_train_brdf_num)
        self.validate_brdf_num = sum(self.each_validate_brdf_num)

        ## model config
        self.network = 'Decoder'
        self.decom = 'DualTriPlane'
        self.query_size = 2
        self.latent_size = 12
        self.decom_H_reso = 20
        self.decom_R = 1
        
        ## training config
        self.trainer = 'Trainer'
        self.batch_size = 4
        self.num_workers = 8
        self.lr = [1e-3, 3e-4]
        self.lr_decay = 0.9
        self.start_radius = 0
        self.radius_half_life = 0
        self.max_epochs = 80
        self.decay_epoch = 1
        self.validate_epoch = 5
        self.print_freq = 1
        
        ## checkpoint config
        self.save_root = 'torch/saved_model'
        self.log_file = self.save_model = True
        self.validate_only = False
        self.compress_only = False
        self.validate_output = self.validate_only or self.compress_only or False
        self.checkpoint_path = ''
        self.continue_training = self.validate_only or self.compress_only or self.checkpoint_path 

        self.dummy_model = getattr(model, self.network)(self)
        self.decoder_params = sum(param.numel() for param in self.dummy_model.decoder.parameters())
        self.dummy_decom = getattr(model, self.decom)(self, self.train_uv_size, len(self.train_materials))
        self.decom_params = self.dummy_decom.get_param_count() if hasattr(self.dummy_decom, 'get_param_count') else None
        del self.dummy_model, self.dummy_decom
        
        self.gen_comment()

    def gen_comment(self):

        ## other info
        self.comments = self.other_info
        if self.validate_only:
            self.comments += '-validate_only'
            return
        if self.compress_only:
            self.comments += '-compress_only'
            return

        ## model name / structure
        self.comments += f'-{self.network}-{self.decom}'
        
        ## latent structure
        self.comments += f'-H{self.decom_H_reso}^2_L{self.latent_size}'
        
        ## training settings
        # self.comments += f'-bs{self.batch_size}'
        
        ## dataset
        self.comments += f'-{self.num_query}x{self.train_uv_size[0]}x{self.train_uv_size[1]}[{self.ubo2014_btf_uv_interval[0]}x{self.ubo2014_btf_uv_interval[1]}]'
        self.comments += f'_{len(self.train_materials)}BTF' + ('s' if len(self.train_materials) > 1 else '')
        # for i in range(len(self.train_materials)):
            # self.comments += f'_{self.train_materials[i]}'
        