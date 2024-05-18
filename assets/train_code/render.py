import sys, os
from functools import partial

import exr
import model ## should be imported before config.py
from utils import *
from config import *

CHANNELS = ["0_wix", "1_wiy", "2_wox", "3_woy", "4_R", "5_G", "6_B", 
    "7_wix", "8_wiy", "9_wox", "a_woy", "b_R", "c_G", "d_B", 
    "e_isN", "f_u",   "g_v",   "h_ind", "i_dudx", "j_dvdx", "k_dudy", "l_dvdy"]

def render(decoder, decom, material_index, config: RepConfig, buf_path, out_path, device, output_brdf_value=False):

    ''' load buffer '''

    buf = exr.read(buf_path, channels=CHANNELS)
    buf = torch.tensor(buf, dtype=torch.float32, device=device)
    
    wi, wo, illu, isNeural, uv = buf[..., 0:2], buf[..., 2:4], buf[..., 4:7], buf[..., 14:15], buf[..., 15:17]
    cos_term = torch.sqrt(torch.clamp(1 - wo[..., 0:1] ** 2 - wo[..., 1:2] ** 2, 0, 1))
    cos_term[isNeural <= 0] = 0
    if output_brdf_value:
        illu = np.ones_like(illu)
    H, W, _ = wi.shape
    ## wi.shape, wo.shape, illu.shape, uv.shape, cos_term.shape: [H, W, 2/2/3/2/1]
    
    ##! uv-scale, and then uv-offset
    # uv = (uv * 16.0 + 0.5) % 1.0
    
    if __name__ == "__main__":
        print('buffer loaded', buf_path)

    ''' render '''

    with torch.no_grad():

        wi = torch.cat([wi[..., 0:1], wi[..., 1:2], torch.sqrt(torch.clamp(1 - wi[..., 0:1] ** 2 - wi[..., 1:2] ** 2, 0, 1))], dim=-1)
        wo = torch.cat([wo[..., 0:1], wo[..., 1:2], torch.sqrt(torch.clamp(1 - wo[..., 0:1] ** 2 - wo[..., 1:2] ** 2, 0, 1))], dim=-1)
        h, d = wiwo_xyz_to_hd_thetaphi(wi, wo) ## [512, 512, 2]
        h, d = thetaphi_to_xyz(h)[..., :2], thetaphi_to_xyz(d)[..., :2] ## [512, 512, 2]
        # normalize_offset = torch.tensor([0.25, 1], device=device) * np.pi
        # d = d - normalize_offset
        
        if __name__ == "__main__":
            print('start rendering.')
        final_output = None
        for i in range(H):
            
            latent = decom(config.decom_R, material_index, h[i], uv[i, :, 0], uv[i, :, 1]) ## size: [bs, latent_size]
            output = decoder(torch.cat([d[i], latent], dim=-1)) * cos_term[i] * illu[i, :, 0:1] ## size: [bs, 4+latent_size] -> [bs, 1]
            output1 = output.detach().cpu().numpy()
            # latent = decom(config.decom_R, wi[i], uv[i, :, 0], uv[i, :, 1], 1) ## size: [bs, latent_size]
            # output = decoder(torch.cat([wo[i], latent], dim=-1)) # * cos_term[i] * illu[i, :, 1:2] ## size: [bs, 4+latent_size] -> [bs, 1]
            # output2 = output.detach().cpu().numpy()
            # latent = decom(config.decom_R, wi[i], uv[i, :, 0], uv[i, :, 1], 2) ## size: [bs, latent_size]
            # output = decoder(torch.cat([wo[i], latent], dim=-1)) # * cos_term[i] * illu[i, :, 2:3] ## size: [bs, 4+latent_size] -> [bs, 1]
            # output3 = output.detach().cpu().numpy()

            if final_output is None:
                final_output = output1.reshape(1, W, 3)
            else:
                final_output = np.concatenate([final_output, output1.reshape(1, W, 3)], axis=0)
            # if final_output is None:
            #     final_output = np.concatenate([output1, output2, output3], axis=-1).reshape(1, W, 3)
            # else:
            #     final_output = np.concatenate([final_output, np.concatenate([output1, output2, output3], axis=-1).reshape(1, W, 3)], axis=0)

        exr.write(final_output, out_path)
        if __name__ == "__main__":
            print('saved into', out_path)
    
        
if __name__ == '__main__':

    set_global_random_seed()
    device = detect_device()
    
    checkpoint_path = (
        '/test/repositories/mitsuba-pytorch-tensorNLB/'
        '''
        torch/saved_model/#D6-uv_floor-hd_xyz-Decoder-LatentsTensor-H14^2_L32-400x25x25[16x16]_wallpaper04-1102_204400/epoch-80/epoch-80.pth
        '''
    ).replace('\n', '').strip().replace(' ', '')
    print(checkpoint_path)
    checkpoint = torch.load(checkpoint_path) ## dict
    config = checkpoint['config']
    decoder = checkpoint['decoder']
    decom = checkpoint['decom']

    print('checkpoint loaded.')
    
    render(decoder, decom, 0, config, './buffer_plane_dirlight_orth.exr', os.path.join(os.path.dirname(checkpoint_path), 'plane_dirlight_orth.exr'), device)
    render(decoder, decom, 0, config, './buffer_sphere_dirlight_orth.exr', os.path.join(os.path.dirname(checkpoint_path), 'sphere_dirlight_orth.exr'), device)
    render(decoder, decom, 0, config, './buffer_plane_pointlight_perspec.exr', os.path.join(os.path.dirname(checkpoint_path), 'plane_pointlight_perspec.exr'), device)
    render(decoder, decom, 0, config, './buffer_sphere_pointlight_perspec.exr', os.path.join(os.path.dirname(checkpoint_path), 'sphere_pointlight_perspec.exr'), device)