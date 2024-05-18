import os
from time import perf_counter as clock
from typing import List

import tqdm
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

from utils import *
from config import *
from dataset import *
from render import render

class BaseTrainer:

    def __init__(self, config: RepConfig, train_dataset: torch.utils.data.Dataset, validate_dataset: torch.utils.data.Dataset, models: List[nn.Module] or nn.Module):
        self.config = config
        self.trainer_name = type(self).__name__
        self.train_dataset = train_dataset
        self.validate_dataset = validate_dataset
        if isinstance(models, list):
            self.network = models[0]
            self.decom = models[1]
        elif isinstance(model, nn.Module):
            self.network = models
        else:
            raise TypeError('[Trainer] argument "models" could only be a list or nn.Module, got', type(models))
        self.network.apply(inplace_relu)
        self.network = self.network.to(self.config.device)
        self.init_checkpoint()
        self.init_log()

    def init_log(self):
        log_path = os.path.join(
            self.config.root,
            self.config.save_root,
            self.start_time,
        )
        self.log_file = os.path.join(log_path, 'log.txt')
        if self.config.log_file:
            if not os.path.exists(log_path):
                os.system(f'mkdir -p {log_path}')
            os.system(f'mkdir -p {log_path}/code')
            os.system(f'cp -a {os.path.dirname(os.path.realpath(__file__))}/*.py {log_path}/code')
        self.log(self.config.to_lines())
    
    def log(self, line='', newline='', endline='\n', output='both'):
        '''
            log file helper function
        '''

        if not output in ['both', 'console', 'file']:
            raise ValueError('[Trainer: log] Unknown output direction.')

        if isinstance(line, list):
            if output in ['both', 'console']:
                for l in line:
                    print(newline, l, end=endline)
            if self.config.log_file and output in ['both', 'file']:
                with open(self.log_file, 'ab') as f:
                    np.savetxt(f, line, fmt='%s')
        elif isinstance(line, str):
            if output in ['both', 'console']:
                print(newline + line, end=endline)
            if self.config.log_file and output in ['both', 'file']:
                with open(self.log_file, 'ab') as f:
                    np.savetxt(f, [line], fmt='%s')
        else:
            if output in ['both', 'console']:
                print(f'{line}')
            if self.config.log_file and output in ['both', 'file']:
                with open(self.log_file, 'ab') as f:
                    np.savetxt(f, [line], fmt='%s')

    def show(self, obj, length=100):
        if isinstance(obj, nn.Module):
            for name, params in obj.named_parameters():
                if params.grad is not None:
                    line = f'>>  module {name:<20}\tgrad: {params.grad.mean():.1e} +- ({params.grad.std():.1e})\t'\
                        f'param: {params.detach().cpu().numpy().mean():.1e} +- ({params.detach().cpu().numpy().std():.1e})'
                    self.log(line, output='console')
                else:
                    line = f'>>  module {name:<20}\t\t\t\t\t'\
                        f'param: {params.detach().cpu().numpy().mean():.1e} +- ({params.detach().cpu().numpy().std():.1e})'
                    self.log(line, output='console')
        elif isinstance(obj, torch.Tensor):
            for i in range(min(obj.shape[0], length)):
                line = f'>>  tensor {i}\t{obj[i].detach().cpu().numpy().mean():.1e} +- ({obj[i].detach().cpu().numpy().std():.1e})'
                self.log(line, output='console')

    def set_lr(self, lr, index):
        self.optimizer.param_groups[index]['lr'] = lr
        self.log(f'lr {index} changed -> {lr}')

    def update_lr(self, scale, optimizer=None, eps=1e-8):
        if optimizer is None:
            optimizer = self.optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = (param_group['lr'] - eps) * scale + eps

    def draw_loss_error_curves(self, train_loss_curve, validate_error_curve, epoch):
        '''
        draw loss and error curves

        Args:
            train_loss_curve, N curves (list of np.array, [epoch+1, N])
            validate_error_curve, N curves (list of np.array, [epoch+1, N])

        '''
        train_loss_curve = np.array(train_loss_curve)
        validate_error_curve = np.array(validate_error_curve)

        x0 = np.arange(0, epoch)
        plt.clf()
        for i in range(train_loss_curve.shape[-1]):
            plt.plot(x0, [float(format(x, '.2g')) for x in train_loss_curve[:, i].reshape(-1).tolist()], label=f'train_loss_{i}')
        x1 = np.arange(self.config.validate_epoch-1, epoch, self.config.validate_epoch)
        if self.config.validate_epoch > 0 and validate_error_curve.shape[0] == x1.size:
            for i in range(validate_error_curve.shape[-1]):
                plt.plot(x1, [float(format(x, '.2g')) for x in validate_error_curve[:, i].reshape(-1).tolist()], label=f'validate_error_{i}')
        plt.xlabel('epoch')
        plt.ylabel('loss / error')
        title = self.start_time
        if len(self.start_time) > 50:
            break_point = (len(title.split('-')) + 1) // 2
            title = '-'.join(title.split('-')[:break_point]) + '\n-' + '-'.join(title.split('-')[-break_point:])
        plt.title(title)
        plt.grid(True)
        plt.legend()
        max_y = max(train_loss_curve.max(), validate_error_curve.max() if len(validate_error_curve) > 0 else 0)
        plt.yticks(np.arange(0, max_y, np.ceil(max_y) / 10))
        plt.savefig(os.path.join(self.config.root, self.config.save_root, self.start_time, 'loss.png'))
        plt.yscale('log')
        plt.savefig(os.path.join(self.config.root, self.config.save_root, self.start_time, 'loss_log.png'))

    
class Trainer(BaseTrainer):

    def __init__(self, config: RepConfig, train_dataset: torch.utils.data.Dataset, validate_dataset: torch.utils.data.Dataset, models: List[nn.Module] or nn.Module):
        super().__init__(config, train_dataset, validate_dataset, models)
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.smooth_l1loss = nn.SmoothL1Loss()

    def criterion(self, pred, gt):
        rgb_loss = self.mae_loss(pred, gt)
        return rgb_loss, (rgb_loss.item(),)

    def get_optimizer(self):
        return torch.optim.AdamW([
            {
                'params': self.decom.parameters(), 
                'lr': self.config.lr[0], 
            },
            {
                'params': self.network.parameters(), 
                'lr': self.config.lr[1],
            }
        ])

    def init_checkpoint(self):
        self.save_epoch = 0
        self.start_time = self.config.start_time
        self.radius = self.start_radius = self.config.start_radius
        self.batch_num_per_epoch = self.train_dataset.__len__() // self.config.batch_size

        if self.config.continue_training:
            self.start_time = self.config.checkpoint_path.split('/')[-3].split('-')[-1] + '-' + self.config.checkpoint_path.split('/')[-1].replace('.pth', '') + '-' + self.start_time
        if self.config.comments != '':
            self.start_time = self.config.comments + '-' + self.start_time

        self.optimizer = self.get_optimizer()
        if self.config.continue_training:
            if self.config.checkpoint_path.startswith('/'):
                save_dict = torch.load(self.config.checkpoint_path)
            else:
                save_dict = torch.load(os.path.join(self.config.root, self.config.save_root, self.config.checkpoint_path))
            self.optimizer = save_dict['optimizer']
            optimizer_to(self.optimizer, self.config.device)
            if '.state_dict' in self.config.checkpoint_path:
                self.network.decoder.load_state_dict(save_dict['decoder']())
                self.decom = save_dict['decom']()
            else:
                self.network.decoder = save_dict['decoder']
                self.decom = save_dict['decom']
            self.start_radius = save_dict['radius']
            self.save_epoch = save_dict['epoch'] + 1
            self.network.to(self.config.device) ## sometimes the saved models are in different devices with the current work

    def save_model(self, network, decom, epoch, save_name, use_state_dict=False):
        path = os.path.join(network._config.root, network._config.save_root, save_name)
        save_dir = f'epoch-{epoch}'
        model_name = f'{save_dir}{".state_dict" if use_state_dict else ""}.pth'

        if not os.path.exists(path):
            os.system(f"mkdir -p {path}")
        if not os.path.exists(os.path.join(path, save_dir)):
            os.system(f"mkdir -p {os.path.join(path, save_dir)}")

        decoder_save_obj = network.decoder.state_dict if use_state_dict else network.decoder
        if decom is not None:
            decom_save_obj = decom.state_dict if use_state_dict else decom
        else:
            decom_save_obj = None

        save_dict = {
            'decoder': decoder_save_obj,
            'decom': decom_save_obj,
            'config': self.config,
            'optimizer': self.optimizer,
            'radius': self.radius,
            'epoch': epoch
        }
        
        torch.save(save_dict, os.path.join(path, save_dir, model_name))
        self.log(f'checkpoint {epoch} saved into {path}/{save_dir}/{model_name}')

    def get_radius(self, epoch):
        if self.config.radius_half_life == 0 or self.config.radius == 0:
            return 0
        self.radius = self.config.start_radius * (2.0 ** ( -(epoch-1) / self.config.radius_half_life))
        return self.radius

    def train(self):
        
        assert(isinstance(self.network, nn.Module))
        self.network.train()
        train_loss_curve = []
        validate_error_curve = []

        epoch = -1
        while True:
            epoch += 1
            epoch_loss = []

            pbar = tqdm.tqdm(self.train_dataset.dataloader)
            pbar.set_description(f'epoch: {epoch+1} / {self.config.max_epochs}')
            avg_loss = []
            for batch_index, (material_index, data) in enumerate(pbar):

                if self.config.validate_only or self.config.compress_only:
                    self.log('validate/compress only mode.', output='console')
                    break
                
                ## initialize latent vectors for these brdfs
                data = data.to(self.config.device)
                material_index = material_index.unsqueeze(-1).expand(-1, data.shape[1])
                view, light, u, v, color = data[..., 0:2], data[..., 2:4], data[..., 4], data[..., 5], data[..., 6:9]
                        
                ## process angles 
                wi, wo = map(xy_to_xyz, [view, light])
                cos = wo[..., -1:]
                h, d = wiwo_xyz_to_hd_thetaphi(wi, wo) ## [bs, 2,], [bs, 2]. phis are in (-pi, pi)
                h, d = thetaphi_to_xyz(h)[..., :2], thetaphi_to_xyz(d)[..., :2] ## [bs, 2], [bs, 2], [-1, 1]x[-1, 1]
                
                ## train this batch
                latent = self.decom(self.config.decom_R, material_index, h, u, v, radius=self.get_radius(epoch+1)) ## size: [bs, latent_size]
                output = self.network(torch.cat([d, latent], dim=-1)) ## size: [bs, 4+latent_size] -> [bs, 1]

                loss, division = self.criterion(output * cos, color)
                avg_loss.append(division)

                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()

                if batch_index % self.config.print_freq == 0:
                    line = (
                        f'ep {epoch+1}, {batch_index}/{self.batch_num_per_epoch}, '
                        f'loss: {", ".join([f"{l:.1e}" for l in np.mean(avg_loss, 0)])}'
                    )
                    self.log(line, newline='\r', endline='', output='file')
                    pbar.set_postfix(loss=f'{", ".join([f"{l:.1e}" for l in np.mean(avg_loss, 0)])}', radius=f'{int(self.get_radius(epoch+1))}')
                    epoch_loss.append(np.mean(avg_loss, 0))
                    avg_loss = []
                    
                if os.path.exists('DEBUG.bat'):
                    print('DEBUG switch triggered.')
                    torch.cuda.empty_cache()
                    del loss, output # prevent OOM error
                    debug()
                    
            ## end of an epoch
            torch.cuda.empty_cache()
            self.log(output='file')  ## every epoch starts a new line

            if epoch % self.config.validate_epoch == self.config.validate_epoch - 1 or \
                self.config.validate_only or self.config.compress_only or \
                self.config.validate_epoch == -1 and epoch == self.config.max_epochs - 1:
                # self.show(self.latent, 6)

                out_dir = os.path.join(self.config.root, self.config.save_root, self.start_time, f'epoch-{epoch+1}')

                if self.config.save_model:
                    os.system('mkdir -p {}'.format(out_dir))
                
                if not (self.config.validate_only or self.config.compress_only) and self.config.log_file:
                    with tqdm.tqdm(self.config.train_materials) as pbar:
                        for i, m in enumerate(pbar):
                            pbar.set_description(f'rendering {m}:')
                            render(self.network, self.decom, i, self.config, './buffer_plane_point_persp.exr', os.path.join(out_dir, f'plane_train_{m}.exr'), self.config.device)
                            render(self.network, self.decom, i, self.config, './buffer_sphere_dir_orth.exr', os.path.join(out_dir, f'sphere_train_{m}.exr'), self.config.device)
                            # render(self.network, self.decom, i, self.config, './buffer_sphere_dir_orth.exr', os.path.join(out_dir, f'plane_train_BRDFval_{m}.exr'), self.config.device, output_brdf_value=True)

                # if self.validate_dataset is not None and not self.config.compress_only:
                if self.config.validate_only:
                    validate_error_curve.append(self.validate(out_dir))
                    
                if self.config.save_model and not (self.config.validate_only or self.config.compress_only):
                    self.save_model(self.network, self.decom, epoch+1, save_name=self.start_time)
            
            if self.config.validate_only or self.config.compress_only:
                break
                
            if epoch == self.config.max_epochs - 1:
                break

            if epoch % self.config.decay_epoch == self.config.decay_epoch - 1 and self.config.lr_decay:
                self.update_lr(self.config.lr_decay, eps=1e-7)
                self.log(f'lr -> {self.optimizer.param_groups[0]["lr"]:.1e}, {self.optimizer.param_groups[1]["lr"]:.1e}')

            pbar.close()
            train_loss_curve.append(np.mean(epoch_loss, 0))
            if self.config.save_model and not self.config.validate_only:
                self.draw_loss_error_curves(train_loss_curve, validate_error_curve, epoch+1)

            self.log(output='file')
            
        ## end of training
        if self.config.log_file:
            os.system(f'mv {os.path.join(self.config.root, self.config.save_root, self.start_time)} {os.path.join(self.config.root, self.config.save_root, "#" + self.start_time)} ')

    def validate(self, out_dir):

        assert(isinstance(self.network, nn.Module))
        self.network.eval()

        decom = getattr(model, self.config.decom)(self.config, self.config.validate_uv_size, init='zeros')
        target_lr = self.optimizer.param_groups[0]['lr']
        total_epochs = 3
        start_amptitude = 1.5
        decay_ratio = (target_lr / self.config.lr[0] * 1.5) ** (1 / total_epochs)

        optimizer = torch.optim.AdamW([
            {
                'params': decom.parameters(), 
                'lr': self.config.lr[0] * start_amptitude, 
            },
        ])

        validate_loss = []
        for epoch in range(total_epochs):
            pbar = tqdm.tqdm(self.validate_dataset.dataloader, desc=f'validate ep{epoch}/{total_epochs}')
            avg_loss = []
            for batch_index, (material_index, data) in enumerate(pbar):
                
                ## initialize latent vectors for these brdfs
                data = data.to(self.config.device)
                material_index = material_index.unsqueeze(-1).expand(-1, data.shape[1])
                view, light, u, v, color = data[..., 0:2], data[..., 2:4], data[..., 4], data[..., 5], data[..., 6:9]
       
                ## process angles 
                wi, wo = map(xy_to_xyz, [view, light])
                cos = wo[..., -1:]
                h, d = wiwo_xyz_to_hd_thetaphi(wi, wo) ## [bs, 2,], [bs, 2]. phis are in (-pi, pi)
                h, d = thetaphi_to_xyz(h)[..., :2], thetaphi_to_xyz(d)[..., :2] ## [bs, 2], [bs, 2], [-1, 1]x[-1, 1]
                
                ## train this batch
                latent = decom(self.config.decom_R, material_index, h, u, v, radius=0) ## size: [bs, latent_size]
                output = self.network(torch.cat([d, latent], dim=-1)) ## size: [bs, 4+latent_size] -> [bs, 1]

                loss, division = self.criterion(output * cos, color)
                avg_loss.append(division)

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                if batch_index % self.config.print_freq == 0:
                    line = (
                        f'validate loss: {", ".join([f"{l:.1e}" for l in np.mean(avg_loss, 0)])}'
                    )
                    self.log(line, newline='\r', endline='', output='file')
                    pbar.set_postfix(loss=f'{", ".join([f"{l:.1e}" for l in np.mean(avg_loss, 0)])}')
                    validate_loss.append(np.mean(avg_loss, 0))
                    avg_loss = []

                if os.path.exists('DEBUG.bat'):
                    print('DEBUG switch triggered.')
                    torch.cuda.empty_cache()
                    del loss, output # prevent OOM error
                    debug()
                    

            # end of an epoch
            self.update_lr(decay_ratio, optimizer=optimizer)
            
            # del loss, output
            pbar.close()
        self.network.train()
        
        ## render
        if self.config.log_file and self.config.validate_output:
            with tqdm.tqdm(self.config.validate_materials) as pbar:
                for i, m in enumerate(pbar):
                    render(self.network, self.decom, i, self.config, './buffer_plane_point_persp.exr', os.path.join(out_dir, f'plane_validate_{m}.exr'), self.config.device)
                    render(self.network, self.decom, i, self.config, './buffer_sphere_dir_orth.exr', os.path.join(out_dir, f'sphere_validate_{m}.exr'), self.config.device)
                    # render(self.network, self.decom, i, self.config, './buffer_sphere_dir_orth.exr', os.path.join(out_dir, f'plane_validate_BRDFval_{m}.exr'), self.config.device, output_brdf_value=True)

        return np.mean(validate_loss, 0)
