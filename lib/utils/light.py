import os
import sys

import exr
import numpy as np

PI = 3.1415926

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

root  = '/test/repositories/mitsuba-pytorch-tensorNLB/data/collocated/'
mat = sys.argv[1]
file = os.path.join(root, mat, '00.exr')
img = exr.read(file, channels=CHANNELS)
h, w, c = img.shape

location = tuple([[0, h // 2, h // 2, h - 1], [w // 2, 0, w - 1, w // 2]])

dist = img[location][..., 3]
grey_value = img[location][..., -3:].mean(-1)

print('dist', dist)
print('grey_value', grey_value)

## if the white paper albedo is 0.8
# 0.8 / pi = grey_value * dist ** 2 / intensity
intensity = (grey_value * dist ** 2 / 0.8 * PI).mean()
print(intensity)

np.savetxt(os.path.join(root, mat, 'light_intensity.txt'), np.array([intensity]))