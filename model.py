import torch.nn.functional as F
import torch.nn as nn
import torch

dropout_value = 0.05


class depthwise_separable_conv(nn.Module):
 def __init__(self, nin, kernels_per_layer, nout): 
   super(depthwise_separable_conv, self).__init__() 
   self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin, bias=False) 
   self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1, bias=False) 
  
 def forward(self, x): 
   out = self.depthwise(x) 
   out = self.pointwise(out) 
   return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.block1 = nn.Sequential(
            depthwise_separable_conv(3, 1, 16),
            depthwise_separable_conv(16, 1, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )

        self.transblock1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=2, dilation=2, bias=False),
            nn.Conv2d(32, 32, 1, bias=False)
        )

        self.block2 = nn.Sequential(
            depthwise_separable_conv(32, 1, 32), 
            depthwise_separable_conv(32, 1, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )

        self.transblock2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, dilation=2, bias=False),
            nn.Conv2d(64, 64, 1, bias=False)
        )

        self.block3 = nn.Sequential(
            depthwise_separable_conv(64, 1, 64),
            depthwise_separable_conv(64, 1, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.transblock3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, dilation=2, bias=False),
            nn.Conv2d(128, 128, 1, bias=False)
        )

        self.block4 = nn.Sequential(
            nn.AvgPool2d(1),
            nn.Conv2d(128, 64, 1, bias=False), 
            nn.Conv2d(64, 10, 1, bias=False) 
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.transblock1(x)

        x = self.block2(x)
        x = self.transblock2(x)

        x = self.block3(x)
        x = self.transblock3(x)

        x = self.block4(x)
        x = x.view(-1, 10)
        x = F.log_softmax(x, dim=-1)
        return x