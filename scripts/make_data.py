import os, glob, re, tqdm, sys

import numpy as np
import cv2

from lib.utils import exr

CHANNELS = ['0_dir.B', '0_dir.G', '0_dir.R', '1_uv.B', '1_uv.G', '1_uv.R', '2_rgb.B', '2_rgb.G', '2_rgb.R']


def my_print(print_func):
    from datetime import datetime
    import traceback, os
    def wrap(*args, **kwargs):
        i = -2
        call = traceback.extract_stack()[i]
        while call[2] in ('log', 'show'):
            i -= 1
            call = traceback.extract_stack()[i]
        print_func(f'\x1b[0;96;40m[{datetime.now().strftime("%H:%M:%S")} {os.path.relpath(call[0])}:{call[1]}]\x1b[0;37;49m ', end='')
        print_func(*args, **kwargs)
    return wrap
pr = print
print = my_print(print)


def make_data(folder, h, w, image_size):
    
    print(f'image shape: {h}x{w}, actual size: {image_size}')
    
    files = []
    i = 0
    while os.path.exists(os.path.join(folder, f'{i:04d}_rectify.exr')):
        files.append(os.path.join(folder, f'{i:04d}_rectify.exr'))
        i += 1
    # print(f'totally {len(files)} files:', '\n'.join(files), sep='\n')
    
    ## load camera position
    poses = np.loadtxt(os.path.join(folder, 'camera_pos.txt'), delimiter=',')
    # print(f'camera pos:\n {poses}')
    
    ## make uv matrix
    uv = np.stack(np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h)), -1) ## xy meshgrid (w, h) yields shape [h, w, 2].
    ##! do not change uv arrangement here, otherwise the rendering result will be different!
    # array([[[0.        , 0.        ],     h0 w0
    #         [0.00250627, 0.        ],     h1 w0
    #         [0.00501253, 0.        ],     h2 w0
    #         ...,
    #         [0.99498747, 0.        ],   
    #         [0.99749373, 0.        ],   
    #         [1.        , 0.        ]],  

    #        [[0.        , 0.00250627],     h0 w1   
    #         [0.00250627, 0.00250627],     h1 w1
    #         [0.00501253, 0.00250627],     h2 w1
    #         ...,
    xyz = np.concatenate([uv - 0.5, np.zeros([h, w, 1])], -1) * image_size

    print('assmbling')
    with tqdm.tqdm(files) as pbar:
        for i, file in enumerate(pbar):
            pos = poses.reshape(-1, 3)[i]
            pbar.set_description(os.path.basename(file))
            # img = cv2.imread(file, cv2.IMREAD_COLOR).astype(np.float32) / 255.0 ## BGR
            img = exr.read(file)

            ## calculate texture local frame
            wi = pos - xyz
            dist = np.sqrt(wi[..., 0] ** 2 + wi[..., 1] ** 2 + wi[..., 2] ** 2).reshape(h, w, 1)
            wi = wi / dist
            
            output = np.concatenate([wi[..., [2, 1, 0]], dist, uv[..., [1, 0]], img[..., [2, 1, 0]]], -1)
            exr.write(output, file.replace('_rectify.exr', '.exr'), channels=CHANNELS)

if __name__ == '__main__':
    root  = '/home/yujie/data/mat'
    mat = sys.argv[1]        
    res  = int(sys.argv[2]) if len(sys.argv) > 2 else 800 
    crop = int(sys.argv[3]) if len(sys.argv) > 3 else 300 
    board_size = 10.1 # cm
    
    folder = os.path.join(root, mat)
    make_data(folder, h=crop, w=crop, image_size=board_size * crop / res)