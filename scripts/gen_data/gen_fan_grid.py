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

def createUBO2014BSDF(path, u_index, v_index):
    bsdfProps = Properties('ubo2014_bsdfList_fan_grid')
    bsdfProps['ubo_path'] = path
    bsdfProps['gray_scale'] = False
    bsdfProps['available_texels_u'] = 400
    bsdfProps['available_texels_v'] = 400
    bsdfProps['u_index'] = u_index
    bsdfProps['v_index'] = v_index
    bsdfProps['single_channel'] = False
    bsdfProps['interpType'] = 'bilinear'
    pmgr = PluginManager.getInstance()
    return pmgr.createObject(bsdfProps)

def createRoughDiffuseBSDF(i):
    bsdfProps = Properties('roughdiffuse_bsdfList_fan_grid')
    bsdfProps['distribution'] = 'ggx'
    if i == 0:
        bsdfProps['alpha'] = 0.3
        bsdfProps['reflectance'] = getSpectrum(0.49, 0.45, 0.26)
    elif i == 1:
        bsdfProps['alpha'] = 0.01
        bsdfProps['reflectance'] = getSpectrum(0.18, 0.12, 0.07)
    pmgr = PluginManager.getInstance()
    return pmgr.createObject(bsdfProps)

def createRoughConductorBSDF(i):
    bsdfProps = Properties('roughconductor_bsdfList_fan_grid')
    bsdfProps['distribution'] = 'ggx'
    if i == 0:
        bsdfProps['alpha'] = 0.3
        bsdfProps['specularReflectance'] = getSpectrum(0.49, 0.45, 0.26)
    if i == 1:
        bsdfProps['alpha'] = 0.01
        bsdfProps['specularReflectance'] = getSpectrum(0.18, 0.12, 0.07)
    pmgr = PluginManager.getInstance()
    return pmgr.createObject(bsdfProps)

def createRoughConductorApproxBSDF(i):
    bsdfProps = Properties('roughconductorApprox_bsdfList_fan_grid')
    bsdfProps['distribution'] = 'ggx'
    if i == 0:
        bsdfProps['alpha'] = 0.3
        bsdfProps['R0'] = getSpectrum(0.49, 0.45, 0.26)
    if i == 1:
        bsdfProps['alpha'] = 0.01
        bsdfProps['R0'] = getSpectrum(0.18, 0.12, 0.07)
    pmgr = PluginManager.getInstance()
    return pmgr.createObject(bsdfProps)

def createRoughPlasticBSDF():
    bsdfProps = Properties('roughplastic_bsdfList_fan_grid')
    bsdfProps['alpha'] = 0.001
    bsdfProps['distribution'] = 'ggx'
    bsdfProps['diffuseReflectance'] = getSpectrum(0.18, 0.12, 0.07)
    bsdfProps['specularReflectance'] = getSpectrum(1., 0.8, 0.3)
    pmgr = PluginManager.getInstance()
    return pmgr.createObject(bsdfProps)

def createTexture(filename):
    texProps = Properties('bitmap')
    texProps['filename'] = filename
    texProps['filterType'] = 'bilinear'
    texProps['gamma'] = 2.2
    pmgr = PluginManager.getInstance()
    return pmgr.createObject(texProps)
    
def genData(
        save_dir, file_index,
        path, u_index, v_index,
    ):
    
    if not os.path.exists(save_dir):
        os.system('mkdir -p ' + save_dir)
        
    ## check if already exists
    save_path = os.path.join(save_dir, f'{file_index}.npy')
    # if os.path.exists(save_path):
    #     return False

    ## start generating data
    sample_count = 25 ** 4
    data = np.zeros([sample_count * 7], dtype=np.float32)

    ## make bsdfs
    bsdf = createUBO2014BSDF(path, u_index, v_index)
    bsdf = createRoughDiffuseBSDF(i)
    # bsdf = createRoughConductorBSDF(file_index)
    # bsdf = createRoughConductorApproxBSDF(file_index)
    # bsdf = createRoughPlasticBSDF()
    bsdf.configure()
    # print(bsdf)

    ## get bsdfList & assign & add to data array & free
    value = bsdf.bsdfList(sample_count, 7, datetime.now().microsecond, 1)
    for j in range(data.size):
        data[j] = value[j]
    bsdf.deletebsdfList(value)
        
    ## save data
    data = data.reshape(sample_count, 7)
    exr.write(data[..., -3:].reshape(int(np.sqrt(sample_count)), int(np.sqrt(sample_count)), 3), save_path.replace('.npy', '.exr'))
    np.save(save_path, data)
    
if __name__ == '__main__':

    num_queries = 15**4
    materials = [
        *[f'carpet{i:02d}' for i in range(1, 3)],
        *[f'fabric{i:02d}' for i in range(1, 3)],
        *[f'felt{i:02d}' for i in range(1, 3)],
        *[f'leather{i:02d}' for i in range(1, 3)],
        *[f'stone{i:02d}' for i in range(1, 3)],
        *[f'wallpaper{i:02d}' for i in range(1, 3)],
        *[f'wood{i:02d}' for i in range(1, 3)],
        # 'fabric04',
        # 'silk_diffsue_manual',
        # 'carpet07',
    ]

    btf_path = "/test/data/UBO2014/BTF_resampled/???_resampled_W400xH400_L151xV151.btf"
    for material in materials:
        path = btf_path.replace('???', material)
        save_dir = f'/opt/data/private/ubo2014/BTF_resampled_texel_fan_grid/{material}/{num_queries}'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        with tqdm.tqdm(range(0, 20 * 20), desc=material) as pbar:
            for i in pbar:
                u = i // 20
                v = i % 20
                gen_success = genData(
                    save_dir, i, 
                    path, u, v,
                )