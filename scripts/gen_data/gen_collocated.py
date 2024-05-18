import tqdm
import sys
import os
import glob
from pdb import set_trace as debug

import numpy as np

import exr

CHANNELS = ['0_dir.B', '0_dir.G', '0_dir.R', '1_uv.B', '1_uv.G', '1_uv.R', '2_rgb.B', '2_rgb.G', '2_rgb.R']
## wi.z, wi.y, wi.x, dist, uv.y, uv.x, B, G, R


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


def gen_data(
    save_path,
    img_path,
    cache_file_shape,
    intensity,
    output_rgb=True
):
    print(save_path)
    data = None
    files = []
    i = 0
    while os.path.exists(os.path.join(img_path, f'{i:04d}.exr')):
        files.append(os.path.join(img_path, f'{i:04d}.exr'))
        i += 1
    # print(f'totally {len(files)} files:', '\n'.join(files), sep='\n')
    print(f'totally {len(files)} files')
    
    with tqdm.tqdm(files, desc='loading images...') as pbar:
        for i, file_name in enumerate(pbar):
            
            raw = exr.read(file_name, channels=CHANNELS)[..., [2, 1, 2, 1, 5, 4, 8, 7, 6, 3]]
            dist = raw[..., -1:]
            img = raw[..., :-1]
            
            mask = np.where(np.logical_and(raw[..., -1] != -1, np.max(raw[..., -4:-1], axis=-1) < 1))
            valid_query = img[mask]
            valid_dist = dist[mask]
            
            ## brdf = albedo / pi = pixel_value * dist**2 / intensity. 
            valid_query[..., 6:9] *= valid_dist**2 / intensity ## here we get the brdf value
            ## just like we did in ubo2014_resampled_allrandom: eval(), the data to store is brdf/pi
            # valid_query[..., 6:9] /= np.pi 
            ## TEST: for a white diffuse paper, the brdf value should be albedo/pi ~= 0.8 / pi = 0.25, and the final data value should be 0.25 / pi = 0.08

            if data is None:
                data = valid_query
            else:
                data = np.concatenate([data, valid_query], axis=0)

    if data is None:
        raise RuntimeError("No data found in {}".format(img_path))
    print('total:', data.shape)
    print('mean:', np.mean(data, axis=0))
    print('max:', np.max(data, axis=0))

    np.random.shuffle(data)
    
    length = cache_file_shape[0] * cache_file_shape[1]
    with tqdm.tqdm(range(0, data.shape[0], length), desc='writing buffers...') as pbar:
        for i, start in enumerate(pbar):
            if start + length <= data.shape[0]:
                new_buffer = data[start: start + length]
            else:
                break
                new_buffer = -np.ones((length, 9), dtype=np.float32)
                new_buffer[:data.shape[0] - start] = data[start:]
            exr.write(
                new_buffer.reshape(*cache_file_shape), 
                os.path.join(save_path, f'0_{i}.exr'),
                channels=['0_wix', '1_wiy', '2_wox', '3_woy', '4_u', '5_v', '6_R', '7_G', '8_B']
            )
            if output_rgb:
                exr.write(
                    new_buffer.reshape(*cache_file_shape)[..., 6:], 
                    os.path.join(save_path, f'0_{i}_RGB.exr'),
                )
    

if __name__ == '__main__':

    np.set_printoptions(suppress=True, precision=4)

    ''' target data file shape '''
    cache_file_shape = [400, 400, 9]
    intensity = 3000 if len(sys.argv) == 2 else float(sys.argv[2]) ## here intensity seems to be 100x intensity in mitsuba

    material = sys.argv[1]
    # out_dir = f'/data/mitsuba-pytorch-tensorNLB/collocated_allrandom'
    out_dir = f'/home/yujie/data/Biplane_input'
    img_dir = f'/home/yujie/data/mat/{material}'
    # out_dir = f'/test/repositories/mitsuba-pytorch-tensorNLB/data/collocated_render'
    # img_dir = f'/test/repositories/mitsuba-pytorch-tensorNLB/mitsuba0.6-tensorNLB/scene/collocated/{material}'
    save_path = os.path.join(out_dir, material, f'{cache_file_shape[0]}x{cache_file_shape[1]}')
    if not os.path.exists(save_path):
        os.system('mkdir -p ' + save_path)

    gen_success = gen_data(
        save_path,
        img_dir,
        cache_file_shape,
        intensity=intensity,
        output_rgb=False
    )
    