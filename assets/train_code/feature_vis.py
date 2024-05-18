import os
from pdb import set_trace as debug

import torch
import numpy as np

import exr


def write_blocks(tensor: np.ndarray, M, H, W, C, out_path):
    N = int(np.ceil(C // 3)) ## C channels divided into N images
    res = np.ones([M*H + M - 1, W*N + N - 1, 3], dtype=np.float32)

    ## compile all blocks
    for m in range(M):
        for n in range(N):

            c_start = n*3
            c_end = min((n+1) * 3, C)
            buf = tensor[m, :, :, c_start: c_end]

            ## fill non 3-channel blocks
            if buf.shape[-1] != 3:
                new_buf = np.ones([H, W, 3], dtype=np.float32)
                new_buf[:, :, :buf.shape[-1]] = buf
                buf = new_buf
                
            res[m*(H+1) : (m+1)*(H+1)-1, n*(W+1) : (n+1) * (W+1)-1] = buf
            
    exr.write(res, out_path)
    
    

checkpoint_path = '/test/repositories/mitsuba-pytorch-tensorNLB/torch/saved_model/#D6-Decoder-TriPlane-H14^2_L36-400x400x400[1x1]_12BTFs-1108_170853/epoch-78/epoch-78.pth'
save_dict = torch.load(checkpoint_path)
decom = save_dict['decom']
named_features = [i for i in decom.named_parameters()]

os.mkdir(checkpoint_path.replace('.pth', '_feature_vis'))
for k, v in named_features:
    write_blocks(v.detach().cpu().numpy(), *v.shape, checkpoint_path.replace('.pth', f'_feature_vis/{k}.exr'))