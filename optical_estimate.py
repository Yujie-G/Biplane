# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmcv.runner import load_checkpoint

from mmedit.models.backbones.sr_backbones.basicvsr_net import (
    ResidualBlocksWithInputConv, SPyNet)

spynet_pretrained = 'https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth'
spynet = SPyNet(pretrained=spynet_pretrained) # 5MB
is_mirror_extended = None
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

    if is_mirror_extended:  # flows_forward = flows_backward.flip(1)
        flows_forward = None
    else:
        flows_forward = spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

    if False:
        flows_backward = flows_backward.cpu()
        flows_forward = flows_forward.cpu()

    return flows_forward, flows_backward


def vis_flow(optical_flow, save_path):
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    # Move tensor to cpu and convert it to numpy array.
    optical_flow = optical_flow.cpu().data.numpy()

    # Calculate magnitude and angle of 2D vectors
    magnitude, angle = cv2.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])

    # Normalize magnitude
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Encode HSV image
    hsv = np.zeros((optical_flow.shape[0], optical_flow.shape[1], 3), dtype=np.float32)
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = magnitude

    # Convert HSV to BGR (for visualization)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Save the figure
    plt.imshow(bgr)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)


if __name__=='__main__':
    pass