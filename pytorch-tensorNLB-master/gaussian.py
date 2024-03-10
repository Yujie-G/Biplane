import os, sys
from pdb import set_trace as debug

import numpy as np

import exr

def gaussian_filter(img, stddev, mode='reflect'):
    alpha = -1.0 / (2.0 * stddev ** 2)
    radius = 4 * stddev
    taps = np.ceil(radius * 2).astype(np.int32)
    if taps % 2 != 1:
        taps -= 1

    def eval(x):
        return max(0.0, np.exp(alpha * x**2) - np.exp(alpha * radius**2))

    weights = []
    for i in range(taps):
        weights.append(eval(i - taps // 2))
    weights /= sum(weights)

    res = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(taps):
            res[i] += np.pad(img[i],((taps//2,taps//2), (0, 0)), mode=mode)[j:j+img.shape[1]] * weights[j]

    img = np.zeros_like(res)
    res = res.transpose(1, 0, 2)
    img = img.transpose(1, 0, 2)
    for i in range(img.shape[0]):
        for j in range(taps):
            img[i] += np.pad(res[i],((taps//2,taps//2), (0, 0)), mode=mode)[j:j+res.shape[1]] * weights[j]

    img = img.transpose(1, 0, 2)
    return img

if __name__ == '__main__':
    file_name = sys.argv[1]
    img = exr.read(file_name)

    if len(sys.argv) > 2:
        stddev = float(sys.argv[2])
    else:
        stddev = 0.5

    res = gaussian_filter(img, stddev)
    exr.write(res, file_name.replace('.exr', '_res.exr'))