from torch import nn
import torch
import torch.nn.functional as F
import pdb
import numpy as np
class LocalNetwork(nn.Module):
    def __init__(self, channel,height,width):
        super(LocalNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=channel * height * width,
                      out_features=20),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(in_features=20, out_features=6),
            nn.Tanh(),
        )
        bias = torch.from_numpy(np.array([1, 0, 0, 0, 1, 0]))

        nn.init.constant_(self.fc[3].weight, 0)
        self.fc[3].bias.data.copy_(bias)
        self.channel = channel
        self.height = height
        self.width = width
    def forward(self, img):
        '''
        :param img: (b, c, h, w)
        :return: (b, c, h, w)
        '''
        batch_size = img.size(0)

        theta = self.fc(img.view(batch_size, -1)).view(batch_size, 2, 3)

        grid = F.affine_grid(theta, torch.Size((batch_size, self.channel, self.height, self.width)))
        img_transform = F.grid_sample(img, grid)

        return img_transform
    
class CNNLocalNetwork(nn.Module):
    def __init__(self, channel,height,width): #35,64,64
        super(CNNLocalNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=20, kernel_size=3,
                              padding=1),
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv2d(in_channels=20, out_channels=6, kernel_size=3,
                              padding=1),
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Tanh(),
        )
        # bias = torch.from_numpy(np.array([1, 0, 0, 0, 1, 0]))

        # nn.init.constant_(self.fc[3].weight, 0)
        # self.fc[3].bias.data.copy_(bias)
        self.channel = channel
        self.height = height
        self.width = width
    def forward(self, img):
        '''
        :param img: (b, c, h, w)
        :return: (b, c, h, w)
        '''
        batch_size = img.size(0)

        theta = self.fc(img).mean(-1).mean(-1).view(batch_size, 2, 3)
        grid = F.affine_grid(theta, torch.Size((batch_size, self.channel, self.height, self.width)))
        img_transform = F.grid_sample(img, grid)

        return img_transform
    
if __name__ == '__main__':
    net = LocalNetwork(1,64,64)
    x = torch.randn(1, 1, 64, 64)
    print(net(x).shape)
    