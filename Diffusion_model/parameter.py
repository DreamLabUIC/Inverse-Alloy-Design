import torch
import pandas as pd
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

torch.cuda.empty_cache()
torch.manual_seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_input = None
            return
           
    def next(self):
        data = self.next_data
        self.preload()
        return data

# Background prefetch-enabled DataLoader
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
    

def get_config_value(config, key, default_value):
    return config[key] if key in config else default_value
    

latent_dim = 15

z_dim = latent_dim


epochs = 200
dropout = 0.
valid_size = 0.05
batch_size = 32
test_batch_size = batch_size

torch.backends.cudnn.benchmark = True

ResultsFolder = 'files/results'
savedModelFolder = 'files/Models'
load_pretrained_model = False

c_lr = 1e-3
learningRate = 5e-4

param_Dim = 14
add_noise = False
recon_criterion = nn.MSELoss(reduction = 'sum')

Comp_vec_dim = 70

# Denoinsing Parameters 
lr = 1e-3
latent_dim = 15
epochs_denoise = 500
timesteps = 100
hidden_dim_denoise = 128
n_layers_denoise = 3
train_denoiser = True
n_properties = 3
dim_condition = 128

