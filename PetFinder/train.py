import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import PetFinderRegressor
from trainer import Trainer
from dataloader import get_loaders


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)
    
    p.add_argument('--train_ratio', type=float, default=.8)

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--verbose', type=int, default=2)

    config = p.parse_args()

    return config


def main(config):
    # Set device based on user defined configuration.
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device(f'cuda:{config.gpu_id}')

    train_loader, valid_loader = get_loaders(config)

    print(f'Train: {len(train_loader.dataset)}')
    print(f'Valid: {len(valid_loader.dataset)}')

    model = PetFinderRegressor().to(device)
    optimizer = optim.Adam(model.parameters())
    crit = nn.MSELoss()

    trainer = Trainer(config)
    trainer.train(model, crit, optimizer, train_loader, valid_loader)


if __name__ == '__main__':
    config = define_argparser()
    main(config)