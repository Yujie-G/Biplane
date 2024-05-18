import os, sys
import re
from time import sleep
from datetime import datetime
from pdb import set_trace as debug

import tqdm
from mitsuba.core import Thread, EError, Spectrum, Vector, Properties, PluginManager
import h5py as h5
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

def createRoughConductorBSDF(roughness):
    bsdfProps = Properties('roughconductor')
    bsdfProps['alpha'] = roughness
    bsdfProps['distribution'] = 'ggx'
    pmgr = PluginManager.getInstance()
    return pmgr.createObject(bsdfProps)

def createNormalmapBSDF():
    bsdfProps = Properties('normalmap')
    pmgr = PluginManager.getInstance()
    return pmgr.createObject(bsdfProps)

def createTexture(filename):
    texProps = Properties('bitmap')
    texProps['filename'] = filename
    texProps['gamma'] = 1.0
    pmgr = PluginManager.getInstance()
    return pmgr.createObject(texProps)

def genData(
        save_path, sample_count, file_index, 
        cache_file_shape, seed, highspp, output_rgb
    ):

    # start generating data
    data = np.zeros([sample_count*9], dtype=np.float32)

    # make bsdfs
    bsdf = {}
    
    bsdf['cond'] = createRoughConductorApproxBSDF(roughness=0.5)
    bsdf['cond'].configure()

    bitmap = createTexture('/test/repositories/mitsuba-pytorch-tensorNLB/mitsuba0.6-tensorNLB/scene/door/zhaoyun_normal.exr')
    bitmap.configure()
    
    bsdf['normalmap'] = createNormalmapBSDF()
    bsdf['normalmap'].addChild(bitmap)
    bsdf['normalmap'].addChild(bsdf['cond'])
    bsdf['normalmap'].configure()

    # get bsdfList & assign & add to data array & free
    value = bsdf['normalmap'].bsdfList(sample_count, 9, seed, highspp) ## [400 x 400]
    for j in range(sample_count * 9):
        data[j] = value[j]
    bsdf['multilayered'].deletebsdfList(value)

    # save data
    exr.write(
        data.reshape(*cache_file_shape), 
        os.path.join(save_path, f'0_{file_index}.exr'),
        channels=['0_wix', '1_wiy', '2_wox', '3_woy', '4_u', '5_v', '6_R', '7_G', '8_B']
    )
    if output_rgb:
        exr.write(data.reshape(*cache_file_shape)[..., -3:], os.path.join(save_path, f'0_{file_index}_RGB.exr'),)
    return True

if __name__ == '__main__':

    ''' normal data '''
    cache_file_shape = [100, 100, 9]
    total_query = 400
    sample_count = 100 * 100

    ''' test_run data '''
    # available_texels_u = 2
    # available_texels_v = 2
    # interval_u = 1
    # interval_v = 1
    # interp_type = 'bilinear'
    # cache_file_shape = [40, 40, 9]
    # total_query = 1
    # sample_count = available_texels_u * available_texels_v * 400
    
    tex_name = 'globe_100x100'
    data_path = f'/test/repositories/mitsuba-pytorch-tensorNLB/data/multilayered_allrandom'
    save_path = os.path.join(data_path, f'{tex_name}', f'{cache_file_shape[0]}x{cache_file_shape[1]}')
    if not os.path.exists(save_path):
        os.system('mkdir -p ' + save_path)

    with tqdm.tqdm(range(total_query), desc=f'{tex_name}') as pbar:

        for i in pbar:
            
            # check if already exists
            if os.path.exists(os.path.join(save_path, f'0_{i}.exr')):
                pbar.set_description(f'{tex_name} | {i} [cached]')
                continue
            
            pbar.set_description(f'{tex_name} | {i} [gen...]')
            gen_success = genData(
                save_path,
                sample_count,
                i,
                cache_file_shape,
                seed = datetime.now().microsecond,
                highspp = 128,
                output_rgb=True# if i < 3 else False
            )
            
            
                