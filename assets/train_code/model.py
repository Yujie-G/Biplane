from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from utils import *
from config import *

class Decoder(nn.Module):

    def __init__(self, config: RepConfig):
        super().__init__()
        self.network_name = type(self).__name__
        self.config = config
        self.relu = nn.ReLU()
        self.init_decoder()

    def init_decoder(self):
        channels = 256
        self.decoder = nn.Sequential(
            self.fcrelu(self.config.query_size + self.config.latent_size, channels),
            fc_residual_block(channels, channels),
            fc_residual_block(channels, channels),
            nn.Linear(channels, 3),
            nn.Sigmoid(),
        )

    def fcrelu(self, in_dims, out_dims):
        return nn.Sequential(
            nn.Linear(in_dims, out_dims),
            nn.ReLU(),
        )

    def forward(self, x): ## x size: [bs, query_per_step, wiwo + latent_size]
        return self.decoder(x)


class DualTriPlane(nn.Module):
    
    def __init__(self, config: RepConfig, uv_size: List[int], M: int, init='randn'):
        super().__init__()
        self.network_name = type(self).__name__
        self.config = config
        self.R = config.decom_R
        self.M = M
        self.Hx = self.Hy = config.decom_H_reso
        self.H = self.Hx * self.Hy
        self.L = config.latent_size // 2
        assert (self.L * 2 == config.latent_size)
        self.U = uv_size[0]
        self.V = uv_size[1]
        self.Fxy = nn.Parameter(getattr(torch, init)(self.M, self.Hx, self.Hy, self.L, requires_grad=True, device=config.device, dtype=torch.float32))
        self.Fuv = nn.Parameter(getattr(torch, init)(self.M, self.U,  self.V,  self.L, requires_grad=True, device=config.device, dtype=torch.float32))
        
    def get_param_count(self):
        return (f'{self.M} x ({self.Hx} x {self.Hy} + {self.U} x {self.V}) x {self.L} = '
                f'{self.M * (self.Hx * self.Hy + self.U * self.V) * self.L} ')
    
    def get_hxy_indices(self, h):
        ind_hx = (h[..., 0] + 1) / 2 * self.Hx
        ind_hy = (h[..., 1] + 1) / 2 * self.Hy
        ind_hx[ind_hx == self.Hx] = self.Hx - 1
        ind_hy[ind_hy == self.Hy] = self.Hy - 1
        return ind_hx, ind_hy ## float        

    def get_uv_indices(self, u, v): ## uv in [0, 1]
        ind_u = u * self.U
        ind_v = v * self.V
        ind_u[ind_u == self.U] = self.U - 1
        ind_v[ind_v == self.V] = self.V - 1
        return ind_u, ind_v ## float

    def bilinear_interp(self, x, m, i, j, I, J):
        i1, j1 = map(lambda x: torch.floor(x).long(), (i, j))
        i2, j2 = (i1 + 1) % I, (j1 + 1) % J
        ir, jr = map(lambda x: x.float().unsqueeze(-1), (i - i1, j - j1))
        return (x[m, i1, j1, :] * (1 - ir) + x[m, i2, j1, :] * ir) * (1 - jr) + (x[m, i1, j2, :] * (1 - ir) + x[m, i2, j2, :] * ir) * jr
    
    def blur(self, m, radius):
        if radius != 0:
            raise ValueError("[TriPlane: blur] blurring not supported yet!")
        return m
    
    def forward(self, r, m, h, u, v, radius=0):
        ## h: [bs, 2], u: [bs, 1], v: [bs, 1]
        ind_hx, ind_hy = self.get_hxy_indices(h)
        ind_u, ind_v = self.get_uv_indices(u, v)

        latent = torch.cat([
            self.bilinear_interp(self.Fxy, m, ind_hx, ind_hy, self.Hx, self.Hy),
            self.bilinear_interp(self.Fuv, m, ind_u,  ind_v,  self.U , self.V ),
        ], dim=-1)
        return latent


class TriPlane(nn.Module):
    
    def __init__(self, config: RepConfig, uv_size: List[int], M: int, init='randn'):
        super().__init__()
        self.network_name = type(self).__name__
        self.config = config
        self.R = config.decom_R
        self.M = M
        self.Hx = self.Hy = config.decom_H_reso
        self.H = self.Hx * self.Hy
        self.L = config.latent_size // 6
        assert (self.L * 6 == config.latent_size)
        self.U = uv_size[0]
        self.V = uv_size[1]
        self.Fxy = nn.Parameter(getattr(torch, init)(self.M, self.Hx, self.Hy, self.L, requires_grad=True, device=config.device, dtype=torch.float32))
        self.Fxu = nn.Parameter(getattr(torch, init)(self.M, self.Hx, self.U,  self.L, requires_grad=True, device=config.device, dtype=torch.float32))
        self.Fxv = nn.Parameter(getattr(torch, init)(self.M, self.Hx, self.V,  self.L, requires_grad=True, device=config.device, dtype=torch.float32))
        self.Fyu = nn.Parameter(getattr(torch, init)(self.M, self.Hy, self.U,  self.L, requires_grad=True, device=config.device, dtype=torch.float32))
        self.Fyv = nn.Parameter(getattr(torch, init)(self.M, self.Hy, self.V,  self.L, requires_grad=True, device=config.device, dtype=torch.float32))
        self.Fuv = nn.Parameter(getattr(torch, init)(self.M, self.U,  self.V,  self.L, requires_grad=True, device=config.device, dtype=torch.float32))
        
    def get_param_count(self):
        return (f'[{self.M} x {self.Hx} x {self.Hy} x {self.U} x {self.V} x {self.L}] = '
                f'{self.M * (self.Hx * (self.Hy + self.U + self.V) + self.Hy * (self.U + self.V) + self.U * self.V) * self.L} ')
    
    def get_hxy_indices(self, h):
        ind_hx = (h[..., 0] + 1) / 2 * self.Hx
        ind_hy = (h[..., 1] + 1) / 2 * self.Hy
        ind_hx[ind_hx == self.Hx] = self.Hx - 1
        ind_hy[ind_hy == self.Hy] = self.Hy - 1
        return ind_hx, ind_hy ## float        

    def get_uv_indices(self, u, v): ## uv in [0, 1]
        ind_u = u * self.U
        ind_v = v * self.V
        ind_u[ind_u == self.U] = self.U - 1
        ind_v[ind_v == self.V] = self.V - 1
        return ind_u, ind_v ## float

    def bilinear_interp(self, x, m, i, j, I, J):
        i1, j1 = map(lambda x: torch.floor(x).long(), (i, j))
        i2, j2 = (i1 + 1) % I, (j1 + 1) % J
        ir, jr = map(lambda x: x.float().unsqueeze(-1), (i - i1, j - j1))
        return (x[m, i1, j1, :] * (1 - ir) + x[m, i2, j1, :] * ir) * (1 - jr) + (x[m, i1, j2, :] * (1 - ir) + x[m, i2, j2, :] * ir) * jr
    
    def blur(self, m, radius):
        if radius != 0:
            raise ValueError("[TriPlane: blur] blurring not supported yet!")
        return m
    
    def forward(self, r, m, h, u, v, radius=0):
        ## h: [bs, 2], u: [bs, 1], v: [bs, 1]
        ind_hx, ind_hy = self.get_hxy_indices(h)
        ind_u, ind_v = self.get_uv_indices(u, v)

        latent = torch.cat([
            self.bilinear_interp(self.Fxy, m, ind_hx, ind_hy, self.Hx, self.Hy),
            self.bilinear_interp(self.Fxu, m, ind_hx, ind_u,  self.Hx, self.U ),
            self.bilinear_interp(self.Fxv, m, ind_hx, ind_v,  self.Hx, self.V ),
            self.bilinear_interp(self.Fyu, m, ind_hy, ind_u,  self.Hy, self.U ),
            self.bilinear_interp(self.Fyv, m, ind_hy, ind_v,  self.Hy, self.V ),
            self.bilinear_interp(self.Fuv, m, ind_u,  ind_v,  self.U , self.V ),
        ], dim=-1)
        return latent


class LatentsTensor(nn.Module):
    
    def __init__(self, config: RepConfig, uv_size: List[int], init='zeros'):
        ''' self.R: rank, self.H: half vector num, self.L: latent size, self.B: BRDF num '''
        super().__init__()
        self.network_name = type(self).__name__
        self.config = config
        self.R = config.decom_R
        self.Hx = self.Hy = config.decom_H_reso
        self.L = config.latent_size
        self.U = uv_size[0]
        self.V = uv_size[1]
        self.uv_size = uv_size
        self.latents = nn.Parameter(getattr(torch, init)(self.Hx, self.Hy, self.U, self.V, self.L, requires_grad=True, device=config.device, dtype=torch.float32))

    def get_param_count(self):
        return f'{self.Hx} x {self.Hy} x {self.U * self.V} x {self.L} = {self.Hx * self.Hy * self.U * self.V * self.L}'

    def get_hxy_indices(self, h): ## h in [-1, 1] x [-1, 1]
        ind_hx = (h[..., 0] + 1) / 2 * self.Hx
        ind_hy = (h[..., 1] + 1) / 2 * self.Hy
        ind_hx[ind_hx == self.Hx] = self.Hx - 1
        ind_hy[ind_hy == self.Hy] = self.Hy - 1
        return ind_hx, ind_hy ## float
    
    def get_uv_indices(self, u, v): ## uv in [0, 1]
        ind_u = u * self.U
        ind_v = v * self.V
        ind_u[ind_u == self.U] = self.U - 1
        ind_v[ind_v == self.V] = self.V - 1
        return ind_u, ind_v ## float
    
    def quadro_interp(self, m, i, j, k, l):
        i1, j1, k1, l1 = map(lambda x: torch.floor(x).long(), (i, j, k, l))
        i2, j2, k2, l2 = (i1 + 1) % self.Hx, (j1 + 1) % self.Hy, (k1 + 1) % self.U, (l1 + 1) % self.V
        ir, jr, kr, lr = map(lambda x: x.float().unsqueeze(-1), (i - i1, j - j1, k - k1, l - l1))
        
        ## interpolation
        m00 = (m[i1, j1, k1, l1] * (1 - lr) + m[i1, j1, k1, l2] * lr) * (1 - kr) + (m[i1, j1, k2, l1] * (1 - lr) + m[i1, j1, k2, l2] * lr) * kr
        m01 = (m[i1, j2, k1, l1] * (1 - lr) + m[i1, j2, k1, l2] * lr) * (1 - kr) + (m[i1, j2, k2, l1] * (1 - lr) + m[i1, j2, k2, l2] * lr) * kr
        m10 = (m[i2, j1, k1, l1] * (1 - lr) + m[i2, j1, k1, l2] * lr) * (1 - kr) + (m[i2, j1, k2, l1] * (1 - lr) + m[i2, j1, k2, l2] * lr) * kr
        m11 = (m[i2, j2, k1, l1] * (1 - lr) + m[i2, j2, k1, l2] * lr) * (1 - kr) + (m[i2, j2, k2, l1] * (1 - lr) + m[i2, j2, k2, l2] * lr) * kr
        return (m00 * (1 - jr) + m01 * jr) * (1 - ir) + (m10 * (1 - jr) + m11 * jr) * ir
    
    def blur(self, m, radius):
        ## m: [Hx, Hy, U, V, L]
        r = int(np.ceil(radius))
        if r < 1:
            return m
        sigma = radius / 3.0
        x_grid = np.linspace(-r, r, r * 2 + 1)
        gaussian_weight = np.exp( - x_grid * x_grid / (2 * sigma ** 2) )
        gaussian_weight /= gaussian_weight.sum()
        kernel = torch.tensor(gaussian_weight).reshape(1, 1, r * 2 + 1).float().to(self.config.device)
        
        ## do 1d-blurring on each channel
        output = F.pad(m.permute(1, 2, 3, 4, 0).reshape(-1, 1, self.Hx), mode='replicate', pad=(r, r))
        output = F.conv1d(output, kernel).reshape(self.Hy, self.U, self.V, self.L, self.Hx)
        output = F.pad(output.permute(1, 2, 3, 4, 0).reshape(-1, 1, self.Hy), mode='replicate', pad=(r, r))
        output = F.conv1d(output, kernel).reshape(self.U, self.V, self.L, self.Hx, self.Hy)
        output = output.permute(3, 4, 0, 1, 2)
        # output = F.pad(output.permute(1, 2, 3, 4, 0).reshape(-1, 1, self.U), mode='replicate', pad=(r, r))
        # output = F.conv1d(output, kernel).reshape(self.V, self.L, self.Hx, self.Hy, self.U)
        # output = F.pad(output.permute(1, 2, 3, 4, 0).reshape(-1, 1, self.V), mode='replicate', pad=(r, r))
        # output = F.conv1d(output, kernel).reshape(self.L, self.Hx, self.Hy, self.U, self.V)
        # output = output.permute(1, 2, 3, 4, 0)
        
        return output
    
    def forward(self, r, h, u, v, radius=0): ## input float u/v (0~1). indices are integers (0, U/V)
        ## h: [bs, 2], u: [bs, 1], v: [bs, 1]
        ind_hx, ind_hy = self.get_hxy_indices(h)
        ind_u, ind_v = self.get_uv_indices(u, v)

        ind_u = torch.floor(ind_u)
        ind_v = torch.floor(ind_v)

        ## quadro interpolation
        latent = self.quadro_interp(self.blur(self.latents, radius=radius), ind_hx, ind_hy, ind_u, ind_v)
        return latent
        
        
class NLB(nn.Module):
        
    def __init__(self, config: RepConfig, uv_size: List[int], init='randn'):
        ''' self.R: rank, self.H: half vector num, self.L: latent size, self.B: BRDF num '''
        super().__init__()
        self.network_name = type(self).__name__
        self.config = config
        self.L = config.latent_size
        self.U = uv_size[0]
        self.V = uv_size[1]
        self.uv_size = uv_size
        self.latents = nn.Parameter(getattr(torch, init)(self.U, self.V, self.L, requires_grad=True, device=config.device))

    def get_param_count(self):
        return f'{self.U * self.V} x {self.L} = {self.U * self.V * self.L}'

    def get_uv_indices(self, u, v): ## uv in [0, 1]
        ind_u = u * self.U
        ind_v = v * self.V
        ind_u[ind_u == self.U] = self.U - 1
        ind_v[ind_v == self.V] = self.V - 1
        return ind_u, ind_v ## float
    
    def forward(self, r, h, u, v): ## input float u/v (0~1). indices are integers (0, U/V)
        ## h: [bs, 2], u: [bs, 1], v: [bs, 1]
        ind_u, ind_v = self.get_uv_indices(u, v)
        ind_u = torch.floor(ind_u).long()
        ind_v = torch.floor(ind_v).long()
        return torch.cat([h, self.latents[ind_u, ind_v]], dim=-1)