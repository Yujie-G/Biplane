import os
import glob
import argparse
import tqdm

from lib.utils import exr
import torch
from render_buffer import render
from utils import set_global_random_seed, detect_device, print

if __name__ == '__main__':

    set_global_random_seed()
    device = detect_device()

    parser = argparse.ArgumentParser()
    parser.add_argument('--buf_path', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--out_path', type=str, default=None)
    parser.add_argument('--mis', type=int, default=2)
    args = parser.parse_args()    

    print('sampling:', ['brdf', 'light', 'mis'][args.mis])
    checkpoint_path = args.ckpt_path
    print(checkpoint_path)
    if os.path.isdir(args.buf_path):
        buffer_file_list = sorted(list(glob.glob(os.path.join(args.buf_path, '*.exr'))))
    else:
        buffer_file_list = [args.buf_path]
    print('buffer num:', len(buffer_file_list))
    
    checkpoint = torch.load(checkpoint_path) ## dict
    config = checkpoint['config']
    decoder = checkpoint['decoder']
    decom = checkpoint['decom']
    if 'adapter' in checkpoint:
        adapter = checkpoint['adapter']
        print('adapter: ON')
    else:
        adapter = None
    if 'offset' in checkpoint:
        offset = checkpoint['offset']
        print('offset: ON')
    else:
        offset = None
    if 'normalmap' in checkpoint:
        normalmap = checkpoint['normalmap']
        print('normalmap: ON')
    else:
        normalmap = None
    print('checkpoint loaded.')

    res = 0
    out_path = args.out_path
    with tqdm.tqdm(buffer_file_list) as pbar:
        for i, file in enumerate(pbar):
            if args.mis in [1, 2]:
                res += render(
                    decoder, decom, adapter, offset, normalmap,
                    0, config, 
                    file, 'return', device, 
                    u=None, v=None, 
                    view=None, light=None,
                    output_brdf_value=False,
                    multiplier=1,
                    use_brdf_sampling=False,
                )
            if args.mis in [0, 2]:
                res += render(
                    decoder, decom, adapter, offset, normalmap,
                    0, config, 
                    file, 'return', device, 
                    u=None, v=None, 
                    view=None, light=None,
                    output_brdf_value=False,
                    multiplier=1,
                    use_brdf_sampling=True
                )
            exr.write(res / (i + 1), args.out_path)
            # break