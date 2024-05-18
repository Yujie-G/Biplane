import os, sys
from datetime import datetime

import tqdm
from mitsuba.core import Thread, EError, Spectrum, Vector, Properties, PluginManager
import numpy as np

import exr

logger = Thread.getThread().getLogger()
logger.setLogLevel(EError)

def getSpectrum(r, g=None, b=None):
    spe = Spectrum(1.0)
    spe[0] = r
    spe[1] = g if g is not None else r
    spe[2] = b if b is not None else r
    return spe

def getVector(x, y, z=None):
    texel_index_v = Vector(0.0)
    if z is None:
        z = np.sqrt(1 - x**2 - y**2)
    texel_index_v[0] = x
    texel_index_v[1] = y
    texel_index_v[2] = z
    return texel_index_v

def createUBO2014BSDF(path, available_texels_u, available_texels_v, interp_type):
    bsdfProps = Properties('ubo2014_bsdfList_rainer20_full_queries')
    bsdfProps['ubo_path'] = path
    bsdfProps['gray_scale'] = False
    bsdfProps['available_texels_u'] = available_texels_u
    bsdfProps['available_texels_v'] = available_texels_v
    bsdfProps['single_channel'] = False
    bsdfProps['interpType'] = interp_type
    pmgr = PluginManager.getInstance()
    return pmgr.createObject(bsdfProps)

def genData(
        save_dir, file_index, path, 
        available_texels_u, available_texels_v, 
        interp_type,
    ):
    
    if not os.path.exists(save_dir):
        os.system('mkdir -p ' + save_dir)
        
    ## check if already exists
    save_path = os.path.join(save_dir, f'{file_index:06d}.exr')
    # if os.path.exists(save_path):
    #     return False

    ## start generating data
    sample_count = available_texels_u * available_texels_v
    data = np.zeros([sample_count * 7], dtype=np.float32)

    ## make bsdfs
    bsdf = createUBO2014BSDF(path, available_texels_u, available_texels_v, interp_type)
    bsdf.configure()

    ## get bsdfList & assign & add to data array & free
    value = bsdf.bsdfList(sample_count, 7, datetime.now().microsecond, file_index)
    for j in range(data.size):
        data[j] = value[j]
    bsdf.deletebsdfList(value)
        
    ## save data
    data = data.reshape(available_texels_u, available_texels_v, 7)
    exr.write(data, save_path, channels='RGBCDEF')
    
if __name__ == '__main__':

    interp_type = 'bilinear'
    num_texels = [64, 64]
    num_queries = 22801
    btf_path = "/test/data/UBO2014/BTF_resampled/???_resampled_W400xH400_L151xV151.btf"
    materials = [
        # *[f'carpet{i:02d}' for i in range(1, 3)],
        # *[f'fabric{i:02d}' for i in range(1, 3)],
        # *[f'felt{i:02d}' for i in range(1, 3)],
        # *[f'leather{i:02d}' for i in range(1, 3)],
        # *[f'stone{i:02d}' for i in range(1, 3)],
        # *[f'wallpaper{i:02d}' for i in range(1, 3)],
        # *[f'wood{i:02d}' for i in range(1, 3)],
        # 'fabric04',
        'carpet07',
    ]

    for material in materials:
        save_dir = f'/test/repositories/mitsuba-pytorch-tensorNLB/rainer20/abrdf-conversion/generated_data_full_queries/{material}_{num_texels[0]}x{num_texels[1]}_{num_queries}'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        with tqdm.tqdm(range(num_queries), desc=material) as pbar:
            for i in pbar:
                gen_success = genData(
                    save_dir, i, btf_path.replace('???', material),
                    num_texels[0], num_texels[1],
                    interp_type,
                )