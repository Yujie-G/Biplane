import os
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

def createUBO2014BSDF(
        path, 
        available_texels_u, available_texels_v, 
        interval_u, interval_v, interp_type
    ):
    bsdfProps = Properties('ubo2014_bsdfList_allrandom')
    bsdfProps['ubo_path'] = path
    bsdfProps['gray_scale'] = False
    bsdfProps['single_channel'] = False
    bsdfProps['available_texels_u'] = available_texels_u
    bsdfProps['available_texels_v'] = available_texels_v
    bsdfProps['interval_u'] = interval_u
    bsdfProps['interval_v'] = interval_v
    bsdfProps['useTriangulation'] = True
    bsdfProps['interpType'] = interp_type
    pmgr = PluginManager.getInstance()
    return pmgr.createObject(bsdfProps)

def genData(
        save_path, sample_count, file_index, 
        path, 
        available_texels_u, available_texels_v, 
        interval_u, interval_v, interp_type,
        cache_file_shape, seed, highspp
    ):

    if not os.path.exists(save_path):
        os.system('mkdir -p ' + save_path)
        
    # check if already exists
    if os.path.exists(os.path.join(save_path, f'0_{file_index}.exr')):
        return False

    # start generating data
    data = np.zeros([sample_count*9], dtype=np.float32)

    # make bsdfs
    
    bsdf = createUBO2014BSDF(path, available_texels_u, available_texels_v, interval_u, interval_v, interp_type)
    bsdf.configure()

    # get bsdfList & assign & add to data array & free
    value = bsdf.bsdfList(sample_count, 9, seed, highspp) ## [400 x 400]
    for j in range(sample_count * 9):
        data[j] = value[j]
    bsdf.deletebsdfList(value)

    # save data
    exr.write(
        data.reshape(*cache_file_shape), 
        os.path.join(save_path, f'0_{file_index}.exr'),
        channels=['0_wix', '1_wiy', '2_wox', '3_woy', '4_u', '5_v', '6_R', '7_G', '8_B']
    )
    return True

if __name__ == '__main__':

    ''' normal data '''
    available_texels_u = 400
    available_texels_v = 400
    interval_u = 1
    interval_v = 1
    interp_type = 'bilinear'
    cache_file_shape = [400, 400, 9]
    total_query = 400
    sample_count = available_texels_u * available_texels_v 

    ''' test_run data '''
    # available_texels_u = 2
    # available_texels_v = 2
    # interval_u = 1
    # interval_v = 1
    # interp_type = 'bilinear'
    # cache_file_shape = [40, 40, 9]
    # total_query = 1
    # sample_count = available_texels_u * available_texels_v * 400
    
    btf_path = "/test/data/UBO2014/BTF_resampled/???_resampled_W400xH400_L151xV151.btf"
    materials = [
        *[f'carpet{i:02d}' for i in range(1, 13)],
        *[f'fabric{i:02d}' for i in range(1, 13)],
        *[f'felt{i:02d}' for i in range(1, 13)],
        *[f'leather{i:02d}' for i in range(1, 13)],
        *[f'stone{i:02d}' for i in range(1, 13)],
        *[f'wallpaper{i:02d}' for i in range(1, 13)],
        *[f'wood{i:02d}' for i in range(1, 13)],
    ]
    data_path = f'/opt/data/private/ubo2014/ubo2014_resampled_allrandom'


    for material in materials:
        save_path = os.path.join(data_path, material, f'{available_texels_u}x{available_texels_v}')

        with tqdm.tqdm(range(total_query), desc=material) as pbar:

            for i in pbar:
                
                gen_success = genData(
                    save_path,
                    sample_count,
                    i,
                    btf_path.replace('???', material),
                    available_texels_u, available_texels_v,
                    interval_u, interval_v,
                    interp_type,
                    cache_file_shape,
                    seed = datetime.now().microsecond,
                    highspp = 1
                )