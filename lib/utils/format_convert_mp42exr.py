from PIL import Image
import OpenEXR
import Imath
import numpy as np
import os
import sys
import shutil


def convert_jpg_to_exr(jpg_path, exr_path):
    # Open the JPEG file
    with Image.open(jpg_path) as img:
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Retrieve the image data
        red, green, blue = img.split()

        # Convert image data to float32
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        # red_str = red.tobytes("raw", "R", 0, -1)
        # green_str = green.tobytes("raw", "G", 0, -1)
        # blue_str = blue.tobytes("raw", "B", 0, -1)
        red_np = np.array(red).astype(np.float32) / 255.0
        green_np = np.array(green).astype(np.float32) / 255.0
        blue_np = np.array(blue).astype(np.float32) / 255.0

        # Convert NumPy arrays to byte strings in float32 format
        red_str = red_np.tobytes()
        green_str = green_np.tobytes()
        blue_str = blue_np.tobytes()

        # Create the EXR file
        exr = OpenEXR.OutputFile(exr_path, OpenEXR.Header(img.size[0], img.size[1]))
        exr.writePixels({'R': red_str, 'G': green_str, 'B': blue_str})
        exr.close()


def do_system(arg):
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)


if __name__ == '__main__':
    dir = '/home/yujie/data/mat'
    # read all jpg files in the folder
    mat = sys.argv[1]
    is_video = sys.argv[2] if len(sys.argv) > 2 else None
    fps = sys.argv[3] if len(sys.argv) > 3 else 2
    dir = os.path.join(dir, mat)
    sys.path.append(dir)
    if is_video == '--video_in':
        for file in os.listdir(dir):
            if file.endswith('.MOV') or file.endswith('.mov'):
                do_system(f"ffmpeg -i {os.path.join(dir, file)} -qscale:v 1 -qmin 1 -vf \"fps={fps}\" {dir}/%04d.jpg")

    for file in os.listdir(dir):
        if file.endswith('.JPG') or file.endswith('.jpg'):
            jpg_path = os.path.join(dir, file)
            if not os.path.exists(os.path.join(dir, 'exr')):
                os.mkdir(os.path.join(dir, 'exr'))
            exr_path = os.path.join(dir, 'exr', file[:-4] + '.exr')
            convert_jpg_to_exr(jpg_path, exr_path)
            print(f'>>> {file} done!')
