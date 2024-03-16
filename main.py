# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
import os.path as osp
import re
import argparse
from functools import reduce

# import mmcv
import numpy as np
import torch
from PIL import Image

from optical_estimate import compute_flow, vis_flow
# from mmedit.datasets.pipelines import Compose
# from mmedit.utils import modify_args

# from optical_estimate import compute_flow, vis_flow
# from mmedit.apis import init_model

# VIDEO_EXTENSIONS = ('.mp4', '.mov')

# def parse_args():
#     modify_args()
#     parser = argparse.ArgumentParser(description='Restoration demo')
#     parser.add_argument('config', help='test config file path')
#     parser.add_argument('checkpoint', help='checkpoint file')
#     parser.add_argument('input_dir', help='directory of the input video')
#     parser.add_argument('output_dir', help='directory of the output video')
#     parser.add_argument(
#         '--start-idx',
#         type=int,
#         default=0,
#         help='index corresponds to the first frame of the sequence')
#     parser.add_argument(
#         '--filename-tmpl',
#         default='{:08d}.png',
#         help='template of the file names')
#     parser.add_argument(
#         '--window-size',
#         type=int,
#         default=0,
#         help='window size if sliding-window framework is used')
#     parser.add_argument(
#         '--max-seq-len',
#         type=int,
#         default=None,
#         help='maximum sequence length if recurrent framework is used')
#     parser.add_argument('--device', type=int, default=0, help='CUDA device id')
#     args = parser.parse_args()
#     return args

# def pad_sequence(data, window_size):
#     padding = window_size // 2

#     data = torch.cat([
#         data[:, 1 + padding:1 + 2 * padding].flip(1), data,
#         data[:, -1 - 2 * padding:-1 - padding].flip(1)
#     ],
#                      dim=1)

#     return data


# def restoration_video_inference(model,
#                                 img_dir,
#                                 window_size,
#                                 start_idx,
#                                 filename_tmpl,
#                                 max_seq_len=None):
#     """Inference image with the model.

#     Args:
#         model (nn.Module): The loaded model.
#         img_dir (str): Directory of the input video.
#         window_size (int): The window size used in sliding-window framework.
#             This value should be set according to the settings of the network.
#             A value smaller than 0 means using recurrent framework.
#         start_idx (int): The index corresponds to the first frame in the
#             sequence.
#         filename_tmpl (str): Template for file name.
#         max_seq_len (int | None): The maximum sequence length that the model
#             processes. If the sequence length is larger than this number,
#             the sequence is split into multiple segments. If it is None,
#             the entire sequence is processed at once.

#     Returns:
#         Tensor: The predicted restoration result.
#     """

#     device = next(model.parameters()).device  # model device

#     # build the data pipeline
#     if model.cfg.get('demo_pipeline', None):
#         test_pipeline = model.cfg.demo_pipeline
#     elif model.cfg.get('test_pipeline', None):
#         test_pipeline = model.cfg.test_pipeline
#     else:
#         test_pipeline = model.cfg.val_pipeline

#     # check if the input is a video
#     file_extension = osp.splitext(img_dir)[1]
#     if file_extension in VIDEO_EXTENSIONS:
#         video_reader = mmcv.VideoReader(img_dir)
#         # load the images
#         data = dict(lq=[], lq_path=None, key=img_dir)
#         for frame in video_reader:
#             data['lq'].append(np.flip(frame, axis=2))

#         # remove the data loading pipeline
#         tmp_pipeline = []
#         for pipeline in test_pipeline:
#             if pipeline['type'] not in [
#                     'GenerateSegmentIndices', 'LoadImageFromFileList'
#             ]:
#                 tmp_pipeline.append(pipeline)
#         test_pipeline = tmp_pipeline
#     else:
#         # the first element in the pipeline must be 'GenerateSegmentIndices'
#         if test_pipeline[0]['type'] != 'GenerateSegmentIndices':
#             raise TypeError('The first element in the pipeline must be '
#                             f'"GenerateSegmentIndices", but got '
#                             f'"{test_pipeline[0]["type"]}".')

#         # specify start_idx and filename_tmpl
#         test_pipeline[0]['start_idx'] = start_idx
#         test_pipeline[0]['filename_tmpl'] = filename_tmpl

#         # prepare data
#         sequence_length = len(glob.glob(osp.join(img_dir, '*')))
#         img_dir_split = re.split(r'[\\/]', img_dir)
#         key = img_dir_split[-1]
#         lq_folder = reduce(osp.join, img_dir_split[:-1])
#         data = dict(
#             lq_path=lq_folder,
#             gt_path='',
#             key=key,
#             sequence_length=sequence_length)
#     # compose the pipeline
#     test_pipeline = Compose(test_pipeline)
#     data = test_pipeline(data)
#     data = data['lq'].unsqueeze(0)  # in cpu
#     print(data.shape)
#     return data



def convert_images_to_tensor(directory_path):
    # 获取文件夹中所有.jpg文件
    files = [f for f in os.listdir(directory_path) if f.endswith('.jpg')]
    files.sort() # 按文件名排序

    # 使用PIL加载第一张图片计算宽度和高度
    first_image = Image.open(os.path.join(directory_path, files[0]))
    # w, h = first_image.size
    w, h = 256, 256
    t = len(files)
    c = 3 # RGB通道

    # 创建存储张量，初始化为0
    tensor = torch.zeros(1, t, c, h, w).cuda()

    # 遍历文件夹中所有的图片
    for i, file_name in enumerate(files):
        file_path = os.path.join(directory_path, file_name)

        # 打开图像并转换为RGB
        image = Image.open(file_path).convert('RGB')

        # 将图像调整为指定大小
        image = image.resize((w, h), Image.ANTIALIAS)

        # 将PIL图像转换为PyTorch张量并添加到大张量中
        image_tensor = torch.from_numpy(np.array(image))
        image_tensor = image_tensor.permute(2, 0, 1).cuda()  # 调整通道位置
        tensor[0, i, :, :, :] = image_tensor
    return tensor



if __name__=='__main__':
    lqs = convert_images_to_tensor('/root/autodl-tmp/opticalflow-test') # shape: (1, T, C, H, W)

    print('lqs size: ', lqs.shape)
    _, flows= compute_flow(lqs)
    print(flows.shape) # (n, t - 1, 2, h, w)
    for i in range(58):
        vis_flow(flows[0, i], f'output/flow-vis/flow_forward{i}.png')