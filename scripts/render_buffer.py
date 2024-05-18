import os
from typing import List
import argparse

from lib.utils import exr
# import model ## should be imported before config.py
import torch
from utils import wiwo_xyz_to_hd_thetaphi, thetaphi_to_xyz, xy_to_xyz, set_global_random_seed, detect_device, print
from config import RepConfig

CHANNELS = ["0_wix", "1_wiy", "2_wox", "3_woy", "4_R", "5_G", "6_B", 
    "7_wix", "8_wiy", "9_wox", "a_woy", "b_R", "c_G", "d_B", 
    "e_isN", "f_u",   "g_v",   "h_ind", "i_dudx", "j_dvdx", "k_dudy", "l_dvdy"]

def render(
    decoder, decom, adapter, offset, normalmap,
    material_index, config: RepConfig, 
    buf_path, out_path, device, 
    u :float=None, v: float=None, 
    view :List[float]=None, light :List[float]=None,
    output_brdf_value=False, multiplier=1, use_brdf_sampling=False):

    ''' load buffer '''

    buf = exr.read(buf_path, channels=CHANNELS)
    buf = torch.tensor(buf, dtype=torch.float32, device=device)
    mask = torch.where(buf[:, :, 14] > 0)
    pts = buf[mask] ##shape (-1, 22)
    
    if not use_brdf_sampling:
        wi, wo, illu, uv = pts[:, 0:2], pts[:, 2:4], pts[:, 4:7], pts[:, 15:17]
    else:
        wi, wo, illu, uv = pts[:, 7:9], pts[:, 9:11], pts[:, 11:14], pts[:, 15:17]
        
    cos_term = torch.sqrt(torch.clamp(1 - wo[:, 0:1] ** 2 - wo[:, 1:2] ** 2, 0, 1))
    if output_brdf_value:
        illu = torch.ones_like(illu)
        out_path = out_path.replace('.exr', '_brdfval.exr')
    if u is not None and v is not None:
        assert u < 1 and v < 1, "[render_buffer.py: render] uv location must be in [0, 1)"
        uv[:, 0] = u
        uv[:, 1] = v
        out_path = out_path.replace('.exr', f'_u{u}_v{v}.exr')
    if view is not None:
        wi = torch.tensor([*view[:2]], dtype=wi.dtype, device=wi.device).reshape(1, 2).expand_as(wi)
    if light is not None:
        wo = torch.tensor([*light[:2]], dtype=wo.dtype, device=wo.device).reshape(1, 2).expand_as(wo)
    ## wi.shape, wo.shape, illu.shape, uv.shape, cos_term.shape: [n, 2/2/3/2/1]
    
    ##! uv-scale, and then uv-offset. at last, pick n texels out of 400
    # uv = (uv * 4 + 0.0) % 1
    # uv = 1 - uv ## check the flipNormal option in rendering
    
    if __name__ == "__main__":
        print('buffer loaded', buf_path)

    ''' render '''
    n = pts.shape[0]
    batch_size = min(n, 262144)
    with torch.no_grad():

        wi, wo = map(xy_to_xyz, [wi, wo])
        h, d = wiwo_xyz_to_hd_thetaphi(wi, wo) ## [512, 512, 2]
        h, d = thetaphi_to_xyz(h)[:, :2], thetaphi_to_xyz(d)[:, :2] ## [512, 512, 2]
        
        if __name__ == "__main__":
            print('start rendering.')
        final_output = None
        for start in range(0, n, batch_size):
            
            end = start + batch_size
            
            if normalmap is not None:
                new_h = normalmap(material_index, h[start:end], uv[start:end, 0], uv[start:end, 1])
            else:
                new_h = h[start:end]
            if offset is not None:
                new_u, new_v, off = offset(material_index, uv[start:end, 0], uv[start:end, 1], wi[start:end])
                # exr.write(wi.detach().cpu().numpy().reshape(512, 512, 3), 'wi.exr')
                # exr.write(torch.cat([off, torch.zeros_like(off[..., :1])], -1).detach().cpu().numpy().reshape(512, 512, 3), 'offset.exr')
            else:
                new_u, new_v, off = uv[start:end, 0], uv[start:end, 1], None
            latent = decom(material_index, new_h, new_u, new_v) ## size: [bs, latent_size]
            output = decoder(torch.cat([d[start:end,], latent], dim=-1)) ## size: [bs, 4+latent_size] -> [bs, 3]
            if adapter is not None: ## [uv, 3, 3]
                output = adapter(output, material_index, new_u, new_v, radius=0)
            output = output # * cos_term[start:end,] * illu[start:end, :]
            output = output * multiplier ## the training dataset is divided by pi. so for validation, we need to *pi
            # output = torch.exp(torch.exp(torch.exp(output) - 1.0) - 1.0) - 1.0
            
            # output = torch.cat([off, torch.zeros_like(off[..., :1])], -1)

            if final_output is None:
                final_output = output.reshape(-1, 3)
            else:
                final_output = torch.cat([final_output, output.reshape(-1, 3)], dim=0)

        tmp = buf[:, :, 4:7] if not use_brdf_sampling else buf[:, :, 11:14]
        tmp[mask] = final_output
        final_output = tmp.cpu().numpy()

        if out_path == 'return':
            return final_output
        
        exr.write(final_output, out_path)
        if __name__ == "__main__":
            print('saved into', out_path)
    
        
if __name__ == '__main__':

    set_global_random_seed()
    device = detect_device()

    parser = argparse.ArgumentParser()
    parser.add_argument('--buf_path', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--out_path', type=str, default=None)
    parser.add_argument('--use_brdf_sampling', action='store_true', default=False)
    args = parser.parse_args()    

    if args.ckpt_path is None:
        checkpoint_path = ('''
/lizixuan/Biplane/torch/saved_model/#compress_only-offset_depth-ubo2014original_leather11-Biplane-decoder1207-epoch-60-0515_094424/epoch-50/epoch-50.pth
        ''').replace('\n', '').strip().replace(' ', '')
    else:
        checkpoint_path = args.ckpt_path
    print(checkpoint_path)
    if args.buf_path is None:
        buffer_file_list = [
                    'buffer/buffer_plane_collocated.exr',
                    'buffer/buffer_plane_parallax.exr',
                    'buffer/buffer_plane_point_persp.exr'
                    # 'buffer_sphere_dir_orth.exr',
                ]
    else:
        buffer_file_list = [args.buf_path]
    
    checkpoint = torch.load(checkpoint_path) ## dict
    config = checkpoint['config']
    decoder = checkpoint['decoder']
    decom = checkpoint['decom']
    if 'adapter' in checkpoint:
        adapter = checkpoint['adapter']
        print('adapter: ON')
    else:
        adapter = None
        print('adapter: OFF')
    if 'offset' in checkpoint:
        offset = checkpoint['offset']
        if not hasattr(offset, 'output_channels'):
            offset.output_channels = 2
        print('offset: ON')
    else:
        offset = None
        print('offset: OFF')
    if 'normalmap' in checkpoint:
        normalmap = checkpoint['normalmap']
        print('normalmap: ON')
    else:
        normalmap = None
        print('normalmap: OFF')
    print('checkpoint loaded.')

    for file in buffer_file_list:
        if args.out_path is None:
            # out_path = os.path.join(os.path.dirname(checkpoint_path), os.path.basename(file).replace('buffer_', ''))
            out_path = os.path.join('/root/autodl-tmp/torch/render', os.path.basename(file).replace('buffer_', ''))
        else:
            out_path = args.out_path
        render(
            decoder, decom, adapter, offset, normalmap,
            0, config, 
            file, out_path, device, 
            u=None, v=None, 
            view=None, light=None,
            output_brdf_value=False,
            multiplier=1,
            use_brdf_sampling=args.use_brdf_sampling
        )