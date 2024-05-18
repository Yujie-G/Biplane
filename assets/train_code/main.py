import os, sys

import model
import trainer
import dataset
from utils import *
from config import *

def representation_main(device, global_random_seed, argv=None):
    
    config = RepConfig(device, global_random_seed)

    test_run = False
    if argv is not None and len(argv) > 1 and argv[1] == 'test_run':
        config.ubo2014_btf_size = [2, 2, 22801]
        config.train_uv_size = config.validate_uv_size = [2, 2] 
        config.cache_file_shape = [1, 40, 40, 9]

        config.validate_epoch = 1
        config.each_train_brdf_num = [config.train_uv_size[0] * config.train_uv_size[1] for _ in config.train_materials]
        config.each_validate_brdf_num = [config.validate_uv_size[0] * config.validate_uv_size[1] for _ in config.validate_materials]
        config.train_brdf_num = sum(config.each_train_brdf_num)
        config.validate_brdf_num = sum(config.each_validate_brdf_num)

        config.log_file = config.save_model = True
        config.comments = 'TEST_RUN-' + config.comments
        test_run = True
    
    ## training
    network = getattr(model, config.network)(config)
    decom = getattr(model, config.decom)(config, config.train_uv_size, len(config.train_materials), init='zeros')
    train_dataset = getattr(trainer, config.train_dataset)(config, test_run=test_run)
    # validate_dataset = getattr(dataset, config.validate_dataset)(config)
    getattr(trainer, config.trainer)(
        config, train_dataset, None, [network, decom]
    ).train()

if __name__ == '__main__':
        
    seed = set_global_random_seed()
    device = detect_device()

    representation_main(device, seed, sys.argv)