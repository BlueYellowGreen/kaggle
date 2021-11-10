import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import numpy as np


class PetFinderDataset(Dataset):
    
    def __init__(self, img_path, feature, pawpularity):
        self.img_path = img_path
        self.feature = feature
        self.pawpularity = pawpularity
        
    def __len__(self):
        return self.img_path.shape[0]
    
    def __getitem__(self, idx):
        image = Image.open(self.img_path[idx])
        image = image.resize((224, 224))                      # IMG_SIZE
        image = np.array(image) / 255
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)                       # (224, 224, 3)
        image = image.view(3, image.size(-2), image.size(-1)) # (3, 224, 224)
        
        feature = torch.from_numpy(self.feature.iloc[idx].values)
        
        pawpularity = torch.from_numpy(self.pawpularity.values)
        
        return image, feature, pawpularity


def load_dataset():
    import pandas as pd

    train_df = pd.read_csv('data/train.csv')
    train_df['Id'] = 'data/train/' + train_df['Id'].astype(str) + '.jpg'
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    
    return train_df


def get_loaders(config):
    train_df = load_dataset()

    train_cnt = int(train_df.shape[0] * config.train_ratio)

    image_train = train_df['Id'].iloc[:train_cnt]
    image_valid = train_df['Id'].iloc[train_cnt:].reset_index(drop=True)

    feature_train = train_df[train_df.columns[1:-1]].iloc[:train_cnt]
    feature_valid = train_df[train_df.columns[1:-1]].iloc[train_cnt:].reset_index(drop=True)

    pawpularity_train = train_df['Pawpularity'].iloc[:train_cnt]
    pawpularity_valid = train_df['Pawpularity'].iloc[train_cnt:].reset_index(drop=True)

    train_loader = DataLoader(
        dataset=PetFinderDataset(image_train, feature_train, pawpularity_train),
        batch_size=config.batch_size,
        shuffle=True
    )
    valid_loader = DataLoader(
        dataset=PetFinderDataset(image_valid, feature_valid, pawpularity_valid),
        batch_size=config.batch_size,
        shuffle=True
    )

    return train_loader, valid_loader