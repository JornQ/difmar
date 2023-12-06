import os
import yaml

import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from torch.cuda.amp import GradScaler, autocast
from torchvision.utils import save_image

from Data.dataloaders import IsalaPairedDataset
from Models.BrownianBridgeModel.UNet import UNetModel
#from generative.networks.nets.diffusion_model_unet import DiffusionModelUNet
from generative.networks.schedulers.ddpm import DDPMScheduler
from generative.networks.schedulers.ddim import DDIMScheduler
from utils import namespace2dict, dict2namespace, create_gif

def load_config(config_path):
    with open(config_path, 'r') as f:
        dict_config = yaml.load(f, Loader=yaml.FullLoader)
    n_config = dict2namespace(dict_config)
    
    if n_config.model.test_unet_exclusive:
        n_config.model.use_condition = False
        
    if n_config.model.use_condition:
        n_config.model.DDPM.UNetParams.in_channels = 2
    else:
        n_config.model.DDPM.UNetParams.in_channels = 1
    
    return n_config, namespace2dict(n_config) 

def load_model(device, run, epoch = 0):
    results_path = "/home/s1736191/myjupyter/DiffusionMAR/Results"
    run = str(run)
    results_path = os.path.join(results_path, run)

    config_file = "config.yaml"
    config_path = os.path.join(results_path, config_file)

    n_config, _ = load_config(config_path)

    unet_config = n_config.model.DDPM.UNetParams
    #epochs = n_config.trainingparams.epochs
    test_unet_exclusive = n_config.model.test_unet_exclusive
    use_condition = n_config.model.use_condition
    loss_type = n_config.model.DDPM.params.loss_type
    
    DEVICE = device
    
    checkpoints_folder = "Checkpoints"
    checkpoints_path = os.path.join(results_path, checkpoints_folder)
    
    for epoch_file in os.listdir(checkpoints_path):
        if '_' + str(epoch-1) + '.' in epoch_file:
            last_epoch_file = epoch_file
            break
    
    # Load unet model
    model = UNetModel(**vars(unet_config)).to(device=DEVICE)
    
    model.load_state_dict(torch.load(os.path.join(checkpoints_path, last_epoch_file), map_location=DEVICE)['state_dict'])
    model.eval()
    return model

def load_data():
    DATA_PATH_VAL = "/home/s1736191/myjupyter/DiffusionMAR/Data/Isala_2/val"
    val_dataset = IsalaPairedDataset(DATA_PATH_VAL, 256, True)
    val_loader = val_dataset.get_loader(1)
    return val_dataset, val_loader

def ini_save_path(folder, run_name):
    N = 100
    
    while True: 
        if run_name + '-example_' + str(N) in os.listdir(folder):
            N += 1
            continue
        save_path = os.path.join(folder, run_name + '-example_' + str(N))
        break
    os.makedirs(save_path)    
    return save_path

def to01(sample):
    return (sample - torch.min(sample))/(torch.max(sample) - torch.min(sample)) 

def toHU(sample):
    return sample * 2000 + 1000

def save_loop(inp, tar, prds, chains, save_path):
    np.save(os.path.join(save_path, 'input.npy'), inp.cpu().detach().numpy())
    np.save(os.path.join(save_path, 'target.npy'), tar.cpu().detach().numpy())
    for i, prd in enumerate(prds):
        name = 'prediction_' + str(i) + '.npy'
        np.save(os.path.join(save_path, name), prd)
    for k, chain in enumerate(chains):
        name = 'chain_'
        if k % 2 == 0:
            name = name + 'diffusion_'
        else:
            name = name + 'onestep_'
        name_np = name + str((k)//2) + '.npy'
        np.save(os.path.join(save_path, name_np), chain)
        
def loop(device, model, scheduler, input_img, number_of_loops):
    chains = []
    predictions = []
    
    input_img = input_img.unsqueeze(0)
    for k in range(number_of_loops):
        noise = torch.randn_like(input_img).to(device=device)
        current_img = noise
        combined = torch.cat(
            (noise, input_img), dim=1
        ).to(device=device)
        
        chain_diffusion = torch.zeros(current_img.shape)
        chain_onestep = torch.zeros(current_img.shape)
        for t in scheduler.timesteps:
            with autocast(enabled=False):
                with torch.no_grad():
                    # if prediction_type is 'sample' (always True, other options are not implemented)
                    model_output = model(combined, timesteps = torch.Tensor((t,)).to(device=device))
                    
                    current_img, original_img = scheduler.step(
                        model_output, t, current_img
                    )
                    combined = torch.cat(
                        (current_img, input_img), dim=1
                    )
                    if True or (t+1)%100 == 0 or t == 0:
                        #print('t: ' + str(t) + '/1000') 
                        chain_diffusion = torch.cat((chain_diffusion, current_img.cpu()), dim=-1)
                        chain_onestep = torch.cat((chain_onestep, original_img.cpu()), dim=-1)                    
        predictions.append(original_img.cpu().squeeze().detach().numpy())
        chains.append(chain_diffusion[:,...,256:])
        chains.append(chain_onestep[:,...,256:])
    return chains, predictions

def load_scheduler(sch_type, var_type, inf_steps = 1000):
    if sch_type == 'DDPM':
        scheduler = DDPMScheduler(num_train_timesteps = 1000, schedule = var_type, clip_sample = True)
    elif sch_type == 'DDIM':
        scheduler = DDIMScheduler(num_train_timesteps = 1000, schedule = var_type, clip_sample = True)
        scheduler.set_timesteps(num_inference_steps = inf_steps)
    else:
        raise NotImplementedError()
    return scheduler

def main():
    # specs
    number_of_input_examples = 5
    number_of_loops = 10
    DEVICE = torch.device('cuda')
    # , 'DDPM_S'
    runs = {
        #'cosine': ['10'],
        'linear_beta': ['DDPM_lin', 'DDPM10p', 'DDPM1p']
           }
    get_epoch_n = [15, 75, 150]
    
    #schedule_types = ['DDPM', 'DDIM']
    schedule_type = 'DDPM'
    prediction_type = 'sample'
    inf_steps = 50
    
    # save
    save_location = "/home/s1736191/myjupyter/DiffusionMAR/eval_images/"
    
    # load data
    _, val_loader = load_data()
    
    # load models, schedulers
    models = {}
    schedulers = {}
    for variance_type in runs.keys():
        schedulers.update({variance_type: load_scheduler(schedule_type, variance_type, inf_steps)})
        
        loaded_models = []
        for i, run in enumerate(runs[variance_type]):
            loaded_models.append(load_model(DEVICE, run, get_epoch_n[i]))
        models.update({variance_type: loaded_models})
    
    # eval
    for batch_idx, (input_samples, target_samples) in enumerate(val_loader):
        input_sample = input_samples[0].to(device=DEVICE)
        target_sample = target_samples[0].to(device=DEVICE)
        
        for variance_type in runs.keys():
            for i, model in enumerate(models[variance_type]):
                chains, predictions = loop(DEVICE, model, schedulers[variance_type], input_sample, number_of_loops)
                save_path = ini_save_path(save_location, runs[variance_type][i])
                save_loop(input_sample.squeeze(), target_sample.squeeze(), predictions, chains, save_path)  
        
        if batch_idx == number_of_input_examples - 1:
            break

if __name__ == "__main__":
    main()
