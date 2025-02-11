import numpy as np
import torch
from .utils import normalize_params, normalize_cls

class CMBDataLoader:
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        
    def load_data(self, path):
        x_train, y_train, x_val, y_val, delta_y = read_data(path)
        return (
            data_utils.TensorDataset(x_train, y_train),
            data_utils.TensorDataset(x_val, y_val),
            delta_y
        )
    
    def get_loaders(self, dataset):
        train_loader = DataLoader(dataset[0], batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(dataset[1], batch_size=self.batch_size)
        return train_loader, val_loader