# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
# from mmcv.cnn import constant_init
# from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
# from mmcv.runner import load_checkpoint

from mmedit.models.backbones.sr_backbones.basicvsr_net import (
    ResidualBlocksWithInputConv, SPyNet)

spynet_pretrained = 'https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth'
spynet = SPyNet(pretrained=spynet_pretrained).to('cuda') # 5MB
def compute_flow(lqs):
    """Compute optical flow using SPyNet for feature alignment.

    Note that if the input is an mirror-extended sequence, 'flows_forward'
    is not needed, since it is equal to 'flows_backward.flip(1)'.

    Args:
        lqs (tensor): Input low quality (LQ) sequence with
            shape (n, t, c, h, w).

    Return:
        tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
            flows used for forward-time propagation (current to previous).
            'flows_backward' corresponds to the flows used for
            backward-time propagation (current to next).
    """

    n, t, c, h, w = lqs.size()
    lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
    lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

    flows_backward = spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)

    is_mirror_extended = True
    if is_mirror_extended:  # flows_forward = flows_backward.flip(1)
        flows_forward = None
    else:
        flows_forward = spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

    if False:
        flows_backward = flows_backward.cpu()
        flows_forward = flows_forward.cpu()

    return flows_forward, flows_backward


def vis_flow(optical_flow, save_path):
    # Move tensor to cpu and convert it to numpy array.
    optical_flow = optical_flow.cpu().data.numpy()

    normalized_optical_flow = np.zeros_like(optical_flow)
    for i in range(optical_flow.shape[0]):
        min_val = np.min(optical_flow[i, :, :])
        max_val = np.max(optical_flow[i, :, :])

        normalized_optical_flow[i, :, :] = (optical_flow[i, :, :] - min_val) / (max_val - min_val)

        # 使用线性变换,将数据从(0,1)映射到(-1,1)，即：2*(x-0.5)
        normalized_optical_flow[i, :, :] = 2.0*(normalized_optical_flow[i, :, :] - 0.5)
    
    # map the w*h matrix M to the RGB color space, M[i, j] = picture[i, j]'s color
    def vector_to_rgb(input_matrix, img_path):
        """
            input matrix: (2, n, n)
        """
        img = Image.open(img_path)
        img_data = np.array(img)
        width, height = img_data.shape[0], img_data.shape[1]
        # print(img_data.shape)
        rgb_data = np.zeros((3, input_matrix.shape[1], input_matrix.shape[2]))
        
        pos = input_matrix.transpose((1, 2, 0))
        pos = np.clip((pos + 1) * 0.5, 0, 1)
        pos[:, :, 0] *= width - 1
        pos[:, :, 1] *= height - 1
        pos = pos.astype(int)
        rgb_data = img_data[pos[:, :, 0], pos[:, :, 1], :3].astype(np.uint8)
        return rgb_data

    rgb_data = vector_to_rgb(normalized_optical_flow, 'assets/flow-field.png')

    # Save the figure
    plt.imshow(rgb_data)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    # Test the optical flow computation
    lqs = torch.randn(1, 5, 3, 128, 128) # shape (n, t, c, h, w).
    _, flows= compute_flow(lqs)
    print(flows.shape) # (n, t - 1, 2, h, w)
    vis_flow(flows[0, 0], 'flow_forward.png')
    # vis_flow(flows_backward[0, 0], 'flow_backward.png')
    print('Optical flow computation test passed!')
