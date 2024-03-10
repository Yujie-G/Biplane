import cv2
import OpenEXR
import Imath
import os
import sys
from tqdm import tqdm
import numpy as np

exr_folder = sys.argv[1]
output_video = sys.argv[2] if len(sys.argv) > 2 else './video.mp4'
fps = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0


exr_files = [f for f in os.listdir(exr_folder) if f.endswith('.exr') and not f.endswith('_BAD.exr')]
exr_files.sort()

first_file = os.path.join(exr_folder, exr_files[0])
exr_file = OpenEXR.InputFile(first_file)
dw = exr_file.header()['dataWindow']
size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video, fourcc, fps, (size[0], size[1]))

for exr_file_name in tqdm(exr_files):
    exr_file_path = os.path.join(exr_folder, exr_file_name)
    exr_file = OpenEXR.InputFile(exr_file_path)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    dw = exr_file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    def read_channel(channel):
        return np.frombuffer(exr_file.channel(channel, pt), dtype=np.float32).reshape((size[1], size[0]))

    red = read_channel('R')
    green = read_channel('G')
    blue = read_channel('B')

    img_data = np.stack((red, green, blue), axis=-1)
    img_data = np.clip(img_data, 0, 1)  # 确保数据在合理范围内
    img_data = (img_data * 255).astype(np.uint8)
    img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
    video.write(img_data)

video.release()