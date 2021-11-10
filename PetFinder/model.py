import torch
import torch.nn as nn


class ConvolutionBlock(nn.Module):
    
    def __init__(self, in_channel, out_channel):
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, (3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, (3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel)
        )
    
    def forward(self, x):
        y = self.layers(x)
        
        return y


class PetFinderRegressor(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.blocks = nn.Sequential(
            ConvolutionBlock(3, 16),    # 16,112,112
            ConvolutionBlock(16, 32),   # 32, 56, 56
            ConvolutionBlock(32, 64),   # 64, 28, 28
            ConvolutionBlock(64, 128),  # 128, 14, 14
            ConvolutionBlock(128, 256), # 256, 7, 7
            ConvolutionBlock(256, 512), # 512, 4, 4
            ConvolutionBlock(512, 512), # 512, 2, 2
            ConvolutionBlock(512, 512), # 512, 1, 1
        )
        self.layers = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.BatchNorm1d(10)
        )
        self.layers_2 = nn.Sequential(
            nn.Linear(22, 10),
            nn.ReLU(),
            nn.BatchNorm1d(10),
            nn.Linear(10, 1),
            nn.ReLU()
        )
        
    def forward(self, image, feature):
        # image        =>  (batch, 3, 224, 224)
        # feature      =>  (batch, 12)
        # pawpularity  =>  (batch, 1)

        image = self.blocks(image) # (batch, 512, 1, 1)
        image = image.squeeze()    # (batch, 512)

        image = self.layers(image) # (batch, 10)

        image_with_feature = torch.cat((image, feature), dim=1) # (batch, 22)
        image_with_feature = self.layers_2(image_with_feature)  # (batch, 1)

        
        return image_with_feature