import os, sys
import re
from time import sleep
from datetime import datetime
from pdb import set_trace as debug

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
        texel_index_u, texel_index_v, 
        u_interval, v_interval, interp_type
    ):
    bsdfProps = Properties('ubo2014_bsdfList_full_queries')
    bsdfProps['ubo_path'] = path
    bsdfProps['gray_scale'] = False
    bsdfProps['single_channel'] = False
    bsdfProps['available_texels_u'] = available_texels_u
    bsdfProps['available_texels_v'] = available_texels_v
    bsdfProps['texel_index_u'] = texel_index_u
    bsdfProps['texel_index_v'] = texel_index_v
    bsdfProps['u_interval'] = u_interval
    bsdfProps['v_interval'] = v_interval
    bsdfProps['interpType'] = interp_type
    pmgr = PluginManager.getInstance()
    return pmgr.createObject(bsdfProps)

def genData(
        save_path, sample_count, file_index, 
        path, 
        available_texels_u, available_texels_v, 
        texel_index_u, texel_index_v, 
        u_interval, v_interval, interp_type,
        dataset_name, material, seed, highspp
    ):

    if not os.path.exists(save_path):
        os.system('mkdir -p ' + save_path)
    output_path = os.path.join(save_path, f'{texel_index_u:03d}_{texel_index_v:03d}.exr')
        
    # check if already exists
    if os.path.exists(output_path):
        return False

    # start generating data
    data = np.zeros([sample_count*7], dtype=np.float32)
    sqrt_sample_count = int(np.sqrt(sample_count))

    # make bsdfs
    bsdf = createUBO2014BSDF(path, available_texels_u, available_texels_v, texel_index_u, texel_index_v, u_interval, v_interval, interp_type)
    bsdf.configure()

    # get bsdfList & assign & add to data array & free
    value = bsdf.bsdfList(sample_count, 7, seed, highspp)
    for j in range(sample_count * 7):
        data[j] = value[j]
    bsdf.deletebsdfList(value)

    # save data
    # size: [1, sample_count*7]
    exr.write(
        data.reshape(sqrt_sample_count, sqrt_sample_count, 7), 
        output_path,
        channels=['0_wix', '1_wiy', '2_wox', '3_woy', '4_R', '5_G', '6_B']
    )
    return True

def valiData(data_path, bsdf_index, out_path=None):
    with exr.File(data_path, 'r') as f:
        data = f['data'][...]
    data = data.reshape(1, -1, 7)
    sample_count = data.shape[1]
    sqrt_sample_count = int(np.sqrt(sample_count))

    data = data[bsdf_index, :, -3:]
    data_image = data.reshape(sqrt_sample_count, sqrt_sample_count, 3)

    if out_path is None:
        exr.write(data_image, data_path.replace('.exr', '.exr'))
    else:
        exr.write(data, out_path)

if __name__ == '__main__':

    available_texels_u = 20
    available_texels_v = 20
    interval_u = 20
    interval_v = 20
    interp_type = 'bilinear'
    
    sample_count = 151 ** 2
    dataset_name = 'ubo2014_resampled_randomdirection'

    btf_path = "/test/data/UBO2014/BTF_resampled/???_resampled_W400xH400_L151xV151.btf"
    materials = [
        # *[f'carpet{i:02d}' for i in range(1, 13)],
        # *[f'fabric{i:02d}' for i in range(1, 13)],
        # *[f'felt{i:02d}' for i in range(1, 13)],
        # *[f'leather{i:02d}' for i in range(1, 13)],
        # *[f'stone{i:02d}' for i in range(1, 13)],
        *[f'wallpaper{i:02d}' for i in range(1, 13)],
        *[f'wood{i:02d}' for i in range(1, 13)],
    ]
    data_path = f'/opt/data/private/ubo2014/ubo2014_full_queries'


    for material in materials:
        save_path = os.path.join(data_path, material, f'{sample_count}')
    
        with tqdm.tqdm(range(available_texels_u * available_texels_v)) as pbar:

            for texel_index_u in range(0, available_texels_u * interval_u, interval_u):
                for texel_index_v in range(0, available_texels_v * interval_v, interval_v):

                    file_index = texel_index_v * available_texels_u * interval_u + texel_index_u
                    
                    gen_success = genData(
                        save_path,
                        sample_count,
                        file_index,
                        btf_path.replace('???', material),
                        available_texels_u, available_texels_v,
                        texel_index_u, texel_index_v,
                        interval_u, interval_v,
                        interp_type,
                        dataset_name,
                        material,
                        seed = datetime.now().microsecond,
                        highspp = 1
                    )

                    if gen_success:
                        pbar.set_description(desc=f'{material} | {texel_index_u:3d} {texel_index_v:3d} [gen...]')
                    else:
                        pbar.set_description(desc=f'{material} | {texel_index_u:3d} {texel_index_v:3d} [cached]')
                    pbar.update(1)