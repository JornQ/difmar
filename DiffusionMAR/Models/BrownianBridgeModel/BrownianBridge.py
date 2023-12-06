# import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.autonotebook import tqdm
import numpy as np

from kornia.losses import ssim_loss

from Models.BrownianBridgeModel.util import extract

from Models.BrownianBridgeModel.UNet import UNetModel

class BrownianBridgeModel(nn.Module):
    def __init__(self, model_config, device):
        super().__init__()
        self.model_config = model_config
        
        self.model_params = self.model_config.BB.params
        
        self.mt_type = self.model_params.mt_type # 'linear' or 'sin'
        self.num_timesteps = self.model_params.num_timesteps # T
        self.max_var = self.model_params.max_var # s
        
        self.m_t, self.var_t = self.time_variance_schedule(device)
        
        self.loss_type = self.model_params.loss_type
        self.denoise_fn = UNetModel(**vars(self.model_params.UNetParams))
        
        self.sample_type = self.model_params.sample_type
        self.sample_steps = self.model_params.sample_steps
        self.sample_schedule = self.get_sample_schedule()
        
    def get_sample_schedule(self):
        if self.sample_type == 'linear':
            # midsteps = torch.arange(self.num_timesteps - 1, 1, step=-((self.num_timesteps - 1) / (self.sample_steps - 2))).long()
            # schedule = torch.cat((midsteps, torch.Tensor([1, 0]).long()), dim=0)
            
            schedule = torch.linspace(self.num_timesteps-1, 0, self.sample_steps).int()        
        elif self.sample_type == 'cosine':
            schedule = np.linspace(start=0, stop=self.num_timesteps, num=self.sample_steps + 1)
            schedule = (np.cos(schedule / self.num_timesteps * np.pi) + 1.) / 2. * self.num_timesteps
            schedule = torch.from_numpy(schedule)
        return schedule
    
    def set_sample_schedule(self, sample_steps):
        self.sample_steps = sample_steps
        self.sample_schedule = self.get_sample_schedule()
    
    def time_variance_schedule(self, device):
        T = self.num_timesteps
        
        if self.mt_type == "linear":
            m_min, m_max = 0.0, 1.0
            m_t = np.linspace(m_min, m_max, T)
        elif self.mt_type == "sin":
            m_t = 1.0075 ** np.linspace(0, T, T)
            m_t = m_t / m_t[-1]
            m_t[-1] = 0.999
        else:
            raise NotImplementedError
            
        variance_t = 2. * (m_t - m_t ** 2) * self.max_var
        
        return torch.as_tensor(m_t, device=device, dtype=torch.float32), torch.as_tensor(variance_t, device=device, dtype=torch.float32)
    
    def q_sample(self, x0, y, t, noise=None):
        if noise == None:
            noise = torch.randn_like(x0)
        
        m_t = extract(self.m_t, t, x0.shape)
        var_t = extract(self.var_t, t, x0.shape)
        sigma_t = torch.sqrt(var_t)
        
        x_t = (1. - m_t) * x0 + m_t * y + sigma_t * noise
                
        return x_t
    
    def forward(self, x0, y):
        # x0 = torch.unsqueeze(x0, 1)
        # y = torch.unsqueeze(y, 1)
        b, c, h, w, device = *x0.shape, x0.device
        t = torch.randint(1, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x0, y, t)
    
    def p_losses(self, x0, y, t):
        b, c, h, w, device = *x0.shape, x0.device 
        
        x_t = self.q_sample(x0, y, t)
        x_t = x_t.to(device=device)
        
        dif = x0 - x_t
        
        dif_prediction = self.denoise_fn(x_t, timesteps=t)
        
        if self.loss_type == 'l1':
            loss = (dif - dif_prediction).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(dif, dif_prediction)
        elif self.loss_type == 'ssim':
            loss = ssim_loss(x0, x_t + dif_prediction, 11)
        else:
            raise NotImplementedError()
   
        return loss.float()
        
    @torch.no_grad()
    def p_sample(self, x_t, y, i, clip_denoised=False):
        b, *_, device = *x_t.shape, x_t.device
        
        t = torch.full((x_t.shape[0],), self.sample_schedule[i], device=device, dtype=torch.long)
        dif_prediction = self.denoise_fn(x_t, timesteps = t)
        x0_recon = x_t + dif_prediction
        if clip_denoised:
            x0_recon = x0_recon.clamp_(0., 1.)
        
        if self.sample_schedule[i] == 0:
            return x0_recon, x0_recon
        elif i == 0:
            n_t = torch.full((x_t.shape[0],), self.sample_schedule[i+1], device=x_t.device, dtype=torch.long)
            
            m_t = extract(self.m_t, t, x_t.shape)
            m_nt = extract(self.m_t, n_t, x_t.shape)
            var_t = extract(self.var_t, t, x_t.shape)
            var_nt = extract(self.var_t, n_t, x_t.shape)
            
            sigma2_t = var_nt
            sigma_t = torch.sqrt(sigma2_t)

            noise = torch.randn_like(x_t)
            x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y
            return x_tminus_mean + sigma_t * noise, x0_recon
        else:
            n_t = torch.full((x_t.shape[0],), self.sample_schedule[i+1], device=x_t.device, dtype=torch.long)
            
            m_t = extract(self.m_t, t, x_t.shape)
            m_nt = extract(self.m_t, n_t, x_t.shape)
            var_t = extract(self.var_t, t, x_t.shape)
            var_nt = extract(self.var_t, n_t, x_t.shape)
            
            #sigma2_t = var_nt
            sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt) ** 2) * var_nt / var_t
            sigma_t = torch.sqrt(sigma2_t)
            
            noise = torch.randn_like(x_t)
            x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y + torch.sqrt((var_nt - sigma2_t) / var_t) * (x_t - (1. - m_t) * x0_recon - m_t * y)
            return x_tminus_mean + sigma_t * noise, x0_recon
           
    
    @torch.no_grad()
    def p_sample_loop(self, y, clip_denoised=False):
        imgs, one_step_imgs = [y], []
        for i in tqdm(range(len(self.sample_schedule))):
            img, x0_recon = self.p_sample(x_t = imgs[-1], y = y, i = i, clip_denoised = clip_denoised)
            imgs.append(img)
            one_step_imgs.append(x0_recon)
        return imgs, one_step_imgs
    
    
        
        
        
        
        
        
        
        