import os, glob, time, tqdm
import shutil
import argparse

import torch
import torch.utils.data as data
from prefetch_generator import BackgroundGenerator
import warnings
warnings.filterwarnings("ignore", category=torch.serialization.SourceChangeWarning)
import numpy as np

from mitsuba.core import Thread, EError
from mitsuba.render import SceneHandler

from lib.utils import exr
from utils import wiwo_xyz_to_hd_thetaphi, thetaphi_to_xyz, xy_to_xyz, print, gaussian_filter

'''
    usage: render.py [-h] [--nocpu] [--nogpu] [--output OUTPUT] [--buffer_dir BUFFER_DIR] xml_path
'''

parser = argparse.ArgumentParser()
parser.add_argument('xml_path', type=str)
parser.add_argument('--nocpu', action='store_true', default=False)
parser.add_argument('--nogpu', action='store_true', default=False)
parser.add_argument('--output', type=str, default='')
parser.add_argument('--buffer_dir', type=str, default='')
parser.add_argument('--model_path', type=str)
args = parser.parse_args()

xml_path = args.xml_path
print('rendering :',xml_path)

logger = Thread.getThread().getLogger()
logger.setLogLevel(EError)
fileResolver = Thread.getThread().getFileResolver()
fileResolver.appendPath(os.path.sep.join(xml_path.split(os.path.sep)[:-1]))
scene = SceneHandler.loadScene(fileResolver.resolve(xml_path))

integration_type = scene.getIntegrator().getProperties()['integrationType'] if scene.getIntegrator().getProperties().hasProperty('integrationType') else 2
print(f"integration type: {['bsdf only', 'light only', 'MIS'][integration_type]}")
crop_size_vec2i = scene.getSensor().getFilm().getCropSize()
crop_offset_vec2i = scene.getSensor().getFilm().getCropOffset()
w_film, h_film = crop_size_vec2i.x, crop_size_vec2i.y
w_offset, h_offset = crop_offset_vec2i.x, crop_offset_vec2i.y
rfilter = 'box' if 'Box' in str(scene.getSensor().getFilm().getReconstructionFilter()) else 'gaussian'
spp = scene.getSensor().getSampler().getProperties()['sampleCount']
print(f'film: {h_film}x{w_film}, spp: {spp}, rfilter: {rfilter}')

decom_dict = {}
offset_network_dict = {}
adapter_dict = {}
uvscale = {}
uvoffset = {}
multiplier = {}
decoder = None
latent_size = None

## allow multiple bsdfs re-use a same checkpoint. use index = -N to specify Nth decom layer in main checkpoint
## then, the value stored in decom_dict will be only an integer
# main_ckpt = torch.load('/test/repositories/mitsuba-pytorch-tensorNLB/torch/saved_model/#D6-Decoder-DualTriPlane-H20^2_L12-400x400x400[1x1]_84BTFs-1110_113917/epoch-30/epoch-30.pth')
main_ckpt = torch.load('/lizixuan/Biplane/model/decoder1207-epoch-60.pth')
main_decom = main_ckpt['decom']
decoder = main_ckpt['decoder']
latent_size = 12

for i, shape in enumerate(scene.getShapes()):
    bsdf_props = shape.getBSDF().getProperties()
    if not bsdf_props.hasProperty('index'):
        continue

    if bsdf_props['index'] < 0:  ## if use dataset_checkpoint
        decom_dict[bsdf_props['index']] = -bsdf_props['index']
        offset_network_dict[bsdf_props['index']] = None
        # adapter_dict[bsdf_props['index']] = None
        adapter_dict[bsdf_props['index']] = torch.load('/root/autodl-tmp/torch/saved_model/#compress_only-offset_depth-ubo2014original_leather11-root-epoch-60-1208_091013/epoch-20/epoch-20.pth')['adapter']
    else:
        checkpoint = torch.load(bsdf_props['checkpointPath'])
        decom_dict[bsdf_props['index']] = checkpoint['decom']
        if 'offset' in checkpoint:
            offset_network_dict[bsdf_props['index']] = checkpoint['offset']
            # change offset_network
            print('using other offset...')
            offset_network_dict[bsdf_props['index']] = torch.load('/root/autodl-tmp/torch/saved_model/#compress_only-offset_depth-datagen_rockRoad_synthetic-scale5-tmp-epoch-30-0101_165107/epoch-50/epoch-50.pth')['offset']
        else:
            offset_network_dict[bsdf_props['index']] = None
        if 'adapter' in checkpoint:
            adapter_dict[bsdf_props['index']] = checkpoint['adapter']
        else:
            adapter_dict[bsdf_props['index']] = None

    uvscale[bsdf_props['index']] = (
        bsdf_props['uscale'] if bsdf_props.hasProperty('uscale') else 1.0,
        bsdf_props['vscale'] if bsdf_props.hasProperty('vscale') else 1.0,
    )
    uvoffset[bsdf_props['index']] = (
        bsdf_props['uoffset'] if bsdf_props.hasProperty('uoffset') else 0.0,
        bsdf_props['voffset'] if bsdf_props.hasProperty('voffset') else 0.0,
    )
    multiplier[bsdf_props['index']] = bsdf_props['multiplier'] if bsdf_props.hasProperty('multiplier') else 1.0

if decoder is None or latent_size is None:
    raise RuntimeError('no checkpoint loaded.')
print('decom_dict: ', decom_dict.keys())
print('offset_dict: ', offset_network_dict.values())
print('adapter_dict: ', adapter_dict.values())

# root = os.path.abspath(os.path.sep.join(xml_path.split(os.path.sep)[:-1]))
root = "/lizixuan/Biplane/torch/render"
model_path = args.model_path
file_name = xml_path.split(os.path.sep)[-1].replace('.xml', '')
buffer_dir = f'{root if not args.buffer_dir else args.buffer_dir}/buffers_{file_name}' 

############################################################
#  cpu part
############################################################

if not args.nocpu:
    if os.path.exists(buffer_dir):
        shutil.rmtree(buffer_dir, ignore_errors=True)
        time.sleep(1)
    os.system(f'mkdir -p {buffer_dir}')

    os.system(
    # print(
        f'mtsutil -q pt -q -i -o {os.path.join(buffer_dir, file_name+".exr")} -Dmodel_path={model_path} {xml_path}'
    )

if args.nogpu:
    exit(0)
    
############################################################
#  gpu part
############################################################

CHANNELS = ["0_wix", "1_wiy", "2_wox", "3_woy", "4_R", "5_G", "6_B", 
    "7_wix", "8_wiy", "9_wox", "a_woy", "b_R", "c_G", "d_B", 
    "e_isN", "f_u",   "g_v",   "h_ind", "i_dudx", "j_dvdx", "k_dudy", "l_dvdy"]

class DataLoaderX(data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class Dataset(data.Dataset):

    def __init__(self, file_list, batch_size):
        super(Dataset, self).__init__()
        if not len(file_list) > 0:
            raise ValueError("buffer file_list is empty.")
        self.file_list = file_list
        self.dataloader = DataLoaderX(
            self,
            batch_size = batch_size,
            shuffle = False, 
            drop_last = False
        )

    def __getitem__(self, file_index):
        return torch.tensor(exr.read(self.file_list[file_index], channels=CHANNELS))

    def __len__(self):
        return len(self.file_list)

file_list = glob.glob(os.path.join(buffer_dir, '*_*.exr'))
buffer_dataset = Dataset(file_list, batch_size=1)


with torch.no_grad():
    with tqdm.tqdm(buffer_dataset.dataloader, desc="GPU render") as pbar:
        final_output = np.zeros([h_film, w_film, 3], dtype=np.float32)
        avg_output = None
        for i, buf in enumerate(pbar):

            ## get all neural points
            buf = buf.reshape(h_film, w_film, 22).cuda()
            neural_pts_mask = torch.where(buf[:, :, 14] > 0)
            all_neural_pts = buf[neural_pts_mask] # size: [?, 22]

            ###################################################
            # Light sampling
            ###################################################

            ## deal with buffers -> wi wo h d u v illu cos
            all_wi, all_wo, all_illu, all_u, all_v = all_neural_pts[:, 0:2], all_neural_pts[:, 2:4], all_neural_pts[:, 4:7], all_neural_pts[:, 15], all_neural_pts[:, 16]
            all_wi, all_wo = map(xy_to_xyz, [all_wi, all_wo])
            all_cos_term = torch.clamp(all_wo[..., -1:], 0, 1)
            all_h, all_d = wiwo_xyz_to_hd_thetaphi(all_wi, all_wo) ## [?, 2]
            all_h, all_d = thetaphi_to_xyz(all_h)[:, :2], thetaphi_to_xyz(all_d)[:, :2] ## [?, 2]
            
            ## for each bsdf, forward their decom() to get their latents
            all_output = torch.zeros([all_neural_pts.shape[0], 3]).cuda()
            for bsdf_index, decom in decom_dict.items():
                
                ## deal with re-use checkpoint
                if isinstance(decom, int):
                    material_index = decom
                    decom = main_decom
                else:
                    material_index = 0
                
                bsdf_mask = torch.where(all_neural_pts[..., 17] == bsdf_index)
                wi, illu, u, v, cos_term, h, d = map(lambda x: x[bsdf_mask], [all_wi, all_illu, all_u, all_v, all_cos_term, all_h, all_d])
                
                u = (u * uvscale[bsdf_index][0] + uvoffset[bsdf_index][0]) % 1.0
                v = (v * uvscale[bsdf_index][1] + uvoffset[bsdf_index][1]) % 1.0
                if offset_network_dict[bsdf_index] is not None:
                    new_u, new_v, _ = offset_network_dict[bsdf_index](0, u, v, wi)
                else:
                    new_u, new_v = u, v
                latent = decom(material_index, h, new_u, new_v)
                output = decoder(torch.cat([d, latent], dim=-1))
                if adapter_dict[bsdf_index] is not None:
                    output = adapter_dict[bsdf_index](output, 0, new_u, new_v, radius=0)
                output = output * cos_term * illu * multiplier[bsdf_index]
                output = torch.nan_to_num(output, nan=0.0) # [yujie] fix nan issue
                all_output[bsdf_mask] = output.reshape(-1, 3)
            
            ## assemble output points back into image
            tmp = buf[:, :, 4:7]
            tmp[neural_pts_mask] = all_output
            all_output = tmp.cpu().numpy()
            final_output += all_output
             
            ###################################################
            # BSDF sampling
            ###################################################

            ## deal with buffers -> wi wo h d u v illu cos
            all_wi, all_wo, all_illu, all_u, all_v = all_neural_pts[:, 7:9], all_neural_pts[:, 9:11], all_neural_pts[:, 11:14], all_neural_pts[:, 15], all_neural_pts[:, 16]
            all_wi, all_wo = map(xy_to_xyz, [all_wi, all_wo])
            all_cos_term = torch.clamp(all_wo[..., -1:], 0, 1)
            all_h, all_d = wiwo_xyz_to_hd_thetaphi(all_wi, all_wo) ## [?, 2]
            all_h, all_d = thetaphi_to_xyz(all_h)[:, :2], thetaphi_to_xyz(all_d)[:, :2] ## [?, 2]
            
            ## for each bsdf, forward their decom() to get their latents
            all_output = torch.zeros([all_neural_pts.shape[0], 3]).cuda()
            for bsdf_index, decom in decom_dict.items():

                ## deal with re-use checkpoint
                if isinstance(decom, int):
                    material_index = decom
                    decom = main_decom
                else:
                    material_index = 0
                    
                bsdf_mask = torch.where(all_neural_pts[..., 17] == bsdf_index)
                wi, illu, u, v, cos_term, h, d = map(lambda x: x[bsdf_mask], [all_wi, all_illu, all_u, all_v, all_cos_term, all_h, all_d])
                
                u = (u * uvscale[bsdf_index][0] + uvoffset[bsdf_index][0]) % 1.0
                v = (v * uvscale[bsdf_index][1] + uvoffset[bsdf_index][1]) % 1.0
                if offset_network_dict[bsdf_index] is not None:
                    new_u, new_v, _ = offset_network_dict[bsdf_index](0, u, v, wi)
                else:
                    new_u, new_v = u, v
                latent = decom(material_index, h, new_u, new_v)
                output = decoder(torch.cat([d, latent], dim=-1))
                if adapter_dict[bsdf_index] is not None:
                    output = adapter_dict[bsdf_index](output, 0, new_u, new_v, radius=0)
                output = output * cos_term * illu * multiplier[bsdf_index]
                output = torch.nan_to_num(output, nan=0.0)  # [yujie] fix nan issue
                all_output[bsdf_mask] = output.reshape(-1, 3)
            
            ## assemble output points back into image
            tmp = buf[:, :, 11:14]
            tmp[neural_pts_mask] = all_output
            all_output = tmp.cpu().numpy()
            final_output += all_output
             
            #####################################################
            
            avg_output = final_output / (i+1)
            exr.write(avg_output, os.path.join(root, '{}_tmp.exr').format(file_name))
        
    if avg_output is None:
        raise RuntimeError('no GPU outputs.')

    if rfilter == 'gaussian':
        output_name = file_name + '_gaussian.exr' if not args.output else args.output
        exr.write(gaussian_filter(avg_output, 0.5), os.path.join(root, output_name))
    else:
        output_name = file_name + '_box.exr' if not args.output else args.output
        exr.write(avg_output, os.path.join(root, output_name))