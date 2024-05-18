from datetime import datetime
import os
import yaml

from lib.config.base_config import BaseConfig


## to avoid cuda fork() error
# from multiprocessing import set_start_method
# set_start_method('spawn')

class RepConfig(BaseConfig):
    def __init__(self, device, global_random_seed, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        self.device = device
        self.global_random_seed = global_random_seed

        self.other_info = config['other_info']
        self.start_time = datetime.now().strftime(r'%m%d_%H%M%S')
        self.root = config['root']


        ## data config
        self.data_root = config['data_root']
        self.train_dataset = config['train_dataset']
        self.btf_size = config['btf_size']
        self.cache_file_shape = config['cache_file_shape']
        self.train_materials = config['train_materials']

        ## model config
        self.reacitvate_trainale_layers = False
        self.network = config['network']
        self.decom = config['decom']
        self.adapter = config['adapter']
        self.offset = config['offset']
        self.normalmap = config['normalmap']
        self.reactivate_trainable_layers = config['reactivate_trainable_layers']
        self.query_size = config['query_size']
        self.greyscale = config['greyscale']
        self.latent_size = config['latent_size']
        self.decom_H_reso = config['decom_H_reso']

        ## training config
        self.trainer = config['trainer']
        self.batch_size = config['batch_size']
        self.random_drop_queries = config['random_drop_queries']
        self.num_workers = min(self.batch_size * 2, os.cpu_count())
        self.lr = config['lr']
        self.lr_decay_param = config['lr_decay_param']
        self.lr_decay_epoch = config['lr_decay_epoch']
        self.start_radius = config['start_radius']
        self.radius_decay = config['radius_decay']
        self.max_epochs = config['max_epochs']
        self.validate_epoch = config['validate_epoch']
        self.change_parameters_epoch = config['change_parameters_epoch']

        ## checkpoint config
        self.save_root = config['save_root']
        self.log_file = config['log_file']
        self.save_model = config['save_model']
        self.compress_only = config['compress_only']
        self.continue_training = config['continue_training']
        self.checkpoint_path = config['checkpoint_path']
        self.use_hxy_comb = config['use_hxy_comb']

        self.gen_comment()

    def gen_comment(self):
        ## other info
        self.comments = self.other_info
        if self.compress_only:
            self.comments = 'compress_only' + (f'-{self.other_info}' if self.other_info else '')
            ## data name
            if len(self.train_materials) < 3:
                self.comments += f'-{"_".join(self.train_materials)}'
            return
        ## model name / structure
        # self.comments += f'-{self.network}-{self.decom}'

        ## latent structure
        self.comments += f'-H{self.decom_H_reso}^2_L{self.latent_size[0]}+{self.latent_size[1]}'

        ## data name
        if len(self.train_materials) < 3:
            self.comments += f'-{"_".join(self.train_materials)}'

        ## training settings
        # self.comments += f'-bs{self.batch_size}'

        ## dataset
        self.comments += f'-{self.cache_file_shape[0]}x{len(self.train_materials)}BTF' + ('s' if len(self.train_materials) > 1 else '')

if __name__ == '__main__':
            
    # config = RepConfig('cuda:0', 0)
    config = RepConfig('cuda:0', 0, os.path.join('lib', 'config', 'train.yaml'))
    config.print_to_screen()

    