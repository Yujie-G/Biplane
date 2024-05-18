import sys
import os
import model
import trainer
import dataset
from utils import set_global_random_seed, detect_device
from config import RepConfig

def representation_main(device, global_random_seed, argv=None):
    
    config = None
    test_run = False
    if argv is not None and len(argv) > 1 :
        if argv[1] == 'test_run':
            config = RepConfig(device, global_random_seed, os.path.join('lib', 'config', 'test_run.yaml'))
            config.train_materials = config.train_materials[0:1]
            config.comments = 'TEST_RUN-' + config.comments
            test_run = True
        elif argv[1] == 'compress':
            config = RepConfig(device, global_random_seed, os.path.join('lib', 'config', 'compress.yaml'))
        elif argv[1] == 'acquisition':
            config = RepConfig(device, global_random_seed, os.path.join('lib', 'config', 'acquisiton.yaml'))
        elif argv[1] == 'train':
            config = RepConfig(device, global_random_seed, os.path.join('lib', 'config', 'train.yaml'))
    else:
        print('No config file specified, using default config file.')
        exit(0)
    
    ## training
    decom = getattr(model, config.decom)(config)
    network = getattr(model, config.network)(config)
    adapter = getattr(model, config.adapter)(config) if config.adapter is not None else None
    offset = getattr(model, config.offset)(config) if config.offset is not None else None
    normalmap = getattr(model, config.normalmap)(config) if config.normalmap is not None else None
    train_dataset = getattr(dataset, config.train_dataset)(config, test_run=test_run)
    getattr(trainer, config.trainer)(
        config, [train_dataset], [decom, network, adapter, offset, normalmap]
    ).train()


def compress_materials(device, global_random_seed):
    materials = ['ubo2014original_carpet01','ubo2014original_carpet02','ubo2014original_carpet03','ubo2014original_carpet04']
    for mat in materials:
        config = RepConfig(device, global_random_seed, os.path.join('lib', 'config', 'compress.yaml'))
        config.train_materials = config.train_materials.clear().append(mat)
        ## training
        decom = getattr(model, config.decom)(config)
        network = getattr(model, config.network)(config)
        adapter = getattr(model, config.adapter)(config) if config.adapter is not None else None
        offset = getattr(model, config.offset)(config) if config.offset is not None else None
        normalmap = getattr(model, config.normalmap)(config) if config.normalmap is not None else None
        train_dataset = getattr(dataset, config.train_dataset)(config, test_run=False)
        getattr(trainer, config.trainer)(
            config, [train_dataset], [decom, network, adapter, offset, normalmap]
        ).train()


if __name__ == '__main__':
        
    if sys.argv[-1] == 'debug':
        import debugpy
        debugpy.connect(('172.27.10.100', 5678))
        
    seed = set_global_random_seed(seed=None)
    device = detect_device()

    representation_main(device, seed, sys.argv)

