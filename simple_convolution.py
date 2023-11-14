import torch
from torch import nn
from torch.nn import functional as F


class SimpleConvolutionalNetwork(nn.Module):
    def __init__(self):
        super(SimpleConvolutionalNetwork, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 4, kernel_size=1, stride=1)

    def forward(self, img):
        x = self.conv1(img)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        return torch.argmax(x, dim=1).float()


