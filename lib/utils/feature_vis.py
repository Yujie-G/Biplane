import os

import torch
import numpy as np

from lib.utils import exr


def write_blocks(tensor: np.ndarray, out_path):
    """
    M, H, W, C = tensor.shape：将tensor的形状分别赋值给M、H、W和C。

    if isinstance(tensor, torch.Tensor): tensor = tensor.detach().cpu().numpy()：判断tensor是否为torch.Tensor类型，如果是，则将其转换为numpy数组。

    N = int(np.ceil(C // 3))：将C通道分成N组，每组3个通道，向上取整。

    res = np.ones([M*H + M - 1, W*N + N - 1, 3], dtype=np.float32)：创建一个形状为(MH + M - 1, WN + N - 1, 3)的三维数组res，元素类型为np.float32，初始值为1。

    for m in range(M):和for n in range(N):：遍历M和N，进行下面的操作。

    c_start = n*3和c_end = min((n+1) * 3, C)：计算每组的起始通道索引和结束通道索引。

    buf = tensor[m, :, :, c_start: c_end].transpose(1, 0, 2)：根据起始和结束通道索引，取出tensor中的对应子数组，并对维度进行转置。

    如果buf的最后一个维度不等于3，说明通道数不足3，需要进行填充。新建一个形状为(H, W, 3)的数组new_buf，将buf的值赋给new_buf前buf的最后一个维度。
    将buf赋值给res中对应位置。

    如果指定的输出路径的文件夹不为空字符串，则使用命令行创建该文件夹。
    将res保存为输出路径对应的exr文件。
    如果该脚本是主模块，则打印res的形状。

    """
    assert(len(tensor.shape) == 4)
    M, H, W, C = tensor.shape
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    H, W = W, H ## here swap H and W
    N = int(np.ceil(C // 3)) ## C channels divided into N images
    res = np.ones([M*H + M - 1, W*N + N - 1, 3], dtype=np.float32)

    ## compile all blocks
    for m in range(M):
        for n in range(N):

            c_start = n*3
            c_end = min((n+1) * 3, C)
            buf = tensor[m, :, :, c_start: c_end].transpose(1, 0, 2)

            ## fill non 3-channel blocks
            if buf.shape[-1] != 3:
                new_buf = np.ones([H, W, 3], dtype=np.float32)
                new_buf[:, :, :buf.shape[-1]] = buf
                buf = new_buf
                
            res[m*(H+1) : (m+1)*(H+1)-1, n*(W+1) : (n+1) * (W+1)-1] = buf
            
    if os.path.dirname(out_path) != '':
        os.system(f'mkdir -p {os.path.dirname(out_path)}')
    exr.write(res, out_path)
    if __name__ == '__main__':
        print('output shape:', res.shape)
    
def feature_vis(named_param_list, out_dir):
    ''' named_param_list: [i for i in xxx.named_parameters()] '''
    os.system(f'mkdir -p {out_dir}')
    for k, v in named_param_list:
        for h in range(0, v.shape[0], 32):        
            write_blocks(v[h: h+32].detach().cpu().numpy(), os.path.join(out_dir, f'{k}_{h}.exr'))
        
if __name__ == "__main__":
    checkpoint_path = (
'''
/test/repositories/mitsuba-pytorch-tensorNLB/torch/saved_model/#D6-Decoder-DualTriPlane-H20^2_L12-400x400x400[1x1]_84BTFs-1110_113917/epoch-30/epoch-30.pth
'''
    ).replace('\n', '').strip().replace(' ', '')
    save_dict = torch.load(checkpoint_path)
    decom = save_dict['decom']
    named_features = [i for i in decom.named_parameters()]

    os.system(f"mkdir -p {checkpoint_path.replace('.pth', '_feature_vis')}")
    feature_vis(named_features, checkpoint_path.replace('.pth', '_feature_vis'))