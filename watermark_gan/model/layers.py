import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

class LayerNorm(nn.Module):
    def __init__(self, nFeatures):
        super().__init__()
        self.layer = nn.LayerNorm(nFeatures)
    
    def forward(self, x):
        return self.layer(x.transpose(1,3)).transpose(1,3) # layer norm over channel dimension
    
class ConvNextBlock(nn.Module):
    def __init__(self,
                channels, 
                hidden_channels, 
                kernel_size=7, 
                channel_factor = 4,
                bias = False
                ):
        super().__init__()
        
        self.layer_in = nn.Conv2d(channels, 
                                hidden_channels, 
                                kernel_size, 
                                stride=1, 
                                groups=1,
                                padding="same",
                                bias=bias)
        
        self.layer_norm = LayerNorm(hidden_channels)
        
        self.layers = nn.Sequential(
            nn.Conv2d(  hidden_channels, 
                        hidden_channels*channel_factor, 
                        kernel_size=1, 
                        stride=1, 
                        groups=1,
                        padding="same",
                        bias=bias),
            nn.GELU(),
            nn.Conv2d(  hidden_channels*channel_factor, 
                        channels, 
                        kernel_size=1, 
                        stride=1, 
                        groups=1,
                        padding="same",
                        bias=bias),
        )
        
    def forward(self, x):
        x_in = x
        x = self.layer_in(x)
        x = self.layer_norm(x) # layer norm over channel dimension
        return x_in + self.layers(x)
    
class ConvBlock(nn.Module):
    def __init__(self,
                in_channels, 
                out_channels, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                apply_batchnorm = True,
                apply_spectral_norm = False,
                activation = nn.GELU(),
                **kwargs
                ):
        super().__init__()
        bias = False if apply_batchnorm else True
        
        layer = nn.Conv2d(in_channels, 
                        out_channels, 
                        kernel_size, 
                        stride=stride, 
                        padding=padding,
                        bias=bias,
                        **kwargs)

        if apply_spectral_norm:
            layer = spectral_norm(layer)
    
        self.layers = nn.Sequential(layer)

        if apply_batchnorm:
            self.layers.add_module("BN", nn.BatchNorm2d(out_channels))

        self.layers.add_module("NL", activation)

    def forward(self, x):
        return self.layers(x)
    
class UpSampleBlock(nn.Module):
    def __init__(self,
                in_channels, 
                out_channels, 
                kernel_size, 
                apply_dropout = False,
                apply_batchnorm = True,
                activation = nn.GELU()):
        super().__init__()
        bias = False if apply_batchnorm else True

        self.layers = nn.Sequential(
            nn.ConvTranspose2d( in_channels, 
                                out_channels, 
                                kernel_size, 
                                stride=2, 
                                groups=1,
                                padding=1,
                                output_padding=1,
                                bias=bias))

        if apply_batchnorm:
            self.layers.add_module("BN",nn.BatchNorm2d(out_channels))
        
        if apply_dropout:
            self.layers.add_module("DR",nn.Dropout(0.5))

        self.layers.add_module("NL", activation)

    def forward(self, x):
        return self.layers(x)

class ResidualBlock(nn.Module):
    def __init__(self, 
                channels, 
                kernel_size = 3,
                survival_rate = 0.8,
                activation=nn.ReLU(inplace=True)):
        super().__init__()
        assert survival_rate>0.0
        self.survival_rate = survival_rate
        
        self.layers = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, activation=activation),
            ConvBlock(channels, channels, kernel_size=3, activation=activation)
        )
        
    def forward(self, x):
        return self.layers(x) + x #self.stochastic_depth(x)
    
    def stochastic_depth(self, x):
        if not self.training:
            return x
        
        switch = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_rate
        return x/self.survival_rate * switch
        