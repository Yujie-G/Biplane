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

def createRoughConductorApproxBSDF(roughness, R0):
    bsdfProps = Properties('roughconductorApprox')
    bsdfProps['alpha'] = roughness
    bsdfProps['specularReflectance'] = getSpectrum(1.0)
    if R0 is not None:
        bsdfProps['R0'] = getSpectrum(*R0)
    bsdfProps['distribution'] = 'ggx'
    pmgr = PluginManager.getInstance()
    return pmgr.createObject(bsdfProps)

def createRoughConductorBSDF(roughness, rgb):
    bsdfProps = Properties('roughconductor')
    bsdfProps['alpha'] = roughness
    bsdfProps['specularReflectance'] = getSpectrum(*rgb)
    bsdfProps['distribution'] = 'ggx'
    pmgr = PluginManager.getInstance()
    return pmgr.createObject(bsdfProps)

def createRoughDieletricBSDF(roughness, ior):
    bsdfProps = Properties('roughdielectric')
    bsdfProps['alpha'] = roughness
    bsdfProps['specularReflectance'] = getSpectrum(1.0)
    bsdfProps['specularTransmittance'] = getSpectrum(1.0)
    bsdfProps['intIOR'] = ior
    bsdfProps['distribution'] = 'ggx'
    pmgr = PluginManager.getInstance()
    return pmgr.createObject(bsdfProps)

def createDiffuseBSDF(reflectance):
    bsdfProps = Properties('diffuse')
    bsdfProps['reflectance'] = getSpectrum(*reflectance)
    pmgr = PluginManager.getInstance()
    return pmgr.createObject(bsdfProps)

def createMultilayeredBSDF(albedo, sigmaT, n1, n2):
    bsdfProps = Properties('multilayered_bsdfList_allrandom')
    bsdfProps['bidir'] = True
    bsdfProps["maxDepth"] = -1
    bsdfProps['nbLayers'] = 2
    bsdfProps['aniso_0'] = False
    bsdfProps['pdf'] = 'bidirStochTRT'
    bsdfProps['stochPdfDepth'] = 2
    bsdfProps['pdfRepetitive'] = 1
    bsdfProps['diffusePdf'] = 0.1
    bsdfProps['maxSurvivalProb'] = 1.0
    if albedo is not None:
        bsdfProps["albedo_0"] = getSpectrum(*albedo)
    bsdfProps["sigmaT_0"] = getSpectrum(sigmaT)
    bsdfProps["normal_0"] = getVector(*n1)
    bsdfProps["normal_1"] = getVector(*n2)
    pmgr = PluginManager.getInstance()
    return pmgr.createObject(bsdfProps)

def createPhase():
    phaseProps = Properties('hg')
    phaseProps['g'] = 0.0
    pmgr = PluginManager.getInstance()
    return pmgr.createObject(phaseProps)

def createTexture(filename):
    texProps = Properties('bitmap')
    texProps['warpMode'] = 'repeat'
    texProps['filename'] = filename
    texProps['filterType'] = 'nearest'
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
    
    ## roughness, ior
    bsdf['die'] = createRoughDieletricBSDF(roughness=0.04, ior=1.37)

    ## roughness, R0,
    bsdf['cond'] = createRoughConductorApproxBSDF(roughness=0.49, R0=[0.02, 0.84, 0.41])

    bsdf['die'].configure()
    bsdf['cond'].configure()

    bsdf['multilayered'] = createMultilayeredBSDF(albedo=None, sigmaT=1, n1=[0, 0], n2=[0, 0])
    albedo = createTexture('/test/repositories/mitsuba-pytorch-tensorNLB/mitsuba0.6-tensorNLB/scene/globe/100x100[4x4].exr')
    bsdf['multilayered'].addChild('albedo_tex_0', albedo)

    bsdf['multilayered'].addChild('surface_0',bsdf['die'])
    bsdf['multilayered'].addChild('surface_1',bsdf['cond'])

    phase = createPhase()
    bsdf['multilayered'].addChild('phase_0',phase)
    bsdf['multilayered'].configure()

    # get bsdfList & assign & add to data array & free
    value = bsdf['multilayered'].bsdfList(sample_count, 9, seed, highspp) ## [400 x 400]
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
            
            
                