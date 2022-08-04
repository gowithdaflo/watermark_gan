import torch
import torch.nn as nn
from .layers import ConvBlock, UpSampleBlock, ConvNextBlock, ResidualBlock, MaskBlock
from .initialization import he_orthogonal_init

def weights_init(m):
    classname = m.__class__.__name__
    if not hasattr(m, "weight"):
        return
    if classname.find('Conv') != -1:
        he_orthogonal_init(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0) 
    elif classname.find('LayerNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0) 

class Generator(nn.Module):
    def __init__(self, nOutputChannels, features=64):
        super().__init__()
        
        self.in_channels = 3
        in_channels = self.in_channels
        nResidualBlocks = 9
        
        feature_list = [features,2*features,4*features,8*features,8*features]
        
        self.down_blocks = nn.ModuleList( [ConvBlock(in_channels,
                                                    feature_list[0], 
                                                    kernel_size=7,
                                                    padding="same",
                                                    stride=1,
                                                    apply_batchnorm=True)] )
        in_channels = feature_list[0]
        for i in range(1,len(feature_list)):
            out_channels = feature_list[i]
            self.down_blocks.add_module(f"down_{i}", ConvBlock(in_channels,
                                                            out_channels, 
                                                            kernel_size=3,
                                                            stride=2,
                                                            apply_batchnorm=True))
            in_channels = out_channels
        
        mask_in_channels = in_channels
            
        self.res_blocks = nn.Sequential(*[ResidualBlock(in_channels) for _ in range(nResidualBlocks)])
            
        self.up_blocks = nn.ModuleList( [] )
        feature_list_reverse = reversed(feature_list[:-1])
        for i, out_channels in enumerate(feature_list_reverse):
            self.up_blocks.add_module(f"up_{i}", UpSampleBlock(in_channels,
                                                                out_channels, 
                                                                kernel_size=3,
                                                                apply_dropout=False,
                                                                apply_batchnorm=True))
            in_channels = 2*out_channels
            
        self.last = nn.Sequential( 
            ConvBlock( in_channels, features, kernel_size=3, padding="same", apply_batchnorm=False),
            ConvBlock( features, features, kernel_size=3, padding="same", apply_batchnorm=False),
            ConvBlock( features, nOutputChannels, kernel_size=7, padding="same", padding_mode="reflect", apply_batchnorm=False, activation=nn.Tanh()),
        )
        
        self.mask_blocks = nn.ModuleList( [] )
        nMaskBlocks = len(feature_list)-1
        feature_list_reverse = reversed(feature_list[:-1])
        for i, out_channels in enumerate(feature_list_reverse):
            self.mask_blocks.add_module(f"mask_{i}", MaskBlock(mask_in_channels, 
                                                            out_channels,
                                                            apply_batchnorm=True if i<nMaskBlocks else False,
                                                            activation=nn.GELU() if i<nMaskBlocks else nn.Identity()))
            mask_in_channels = 2 * out_channels
        # (None, 2*64, 128, 128)
        self.last_mask = nn.Sequential( 
            ConvBlock( mask_in_channels, features, kernel_size=3, padding="same", apply_batchnorm=False),
            ConvBlock( features, features, kernel_size=3, padding="same", apply_batchnorm=False),
            ConvBlock( features, 1, kernel_size=7, padding="same", padding_mode="reflect", apply_batchnorm=False, activation=nn.Identity()),
        )

        # Initialize the weights
        self.apply(weights_init)
        
    def forward(self, x):
        # Downsampling 
        skips = []
        for down in self.down_blocks:
            x = down(x)
            skips.append(x)
            
        x_mask = x
        skips_rev_mask = reversed(skips[:-1])
        
        x = x + self.res_blocks(x)
        skips_rev = reversed(skips[:-1])

        # Upsampling image generator
        for up, skip in zip(self.up_blocks, skips_rev):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
        x = self.last(x)
        
        
        # Mask detection
        for mask, skip in zip(self.mask_blocks, skips_rev_mask):
            x_mask = mask(x_mask)
            x_mask = torch.cat([x_mask, skip], dim=1)
        x_mask = self.last_mask(x_mask)
        
        return x, x_mask
    
    def predict(self, x):
        x_gen, x_mask = self.forward(x)
        x_mask = torch.sigmoid(x_mask)
        return x * (1-x_mask) + x_gen * x_mask 
    
    
class Discriminator(nn.Module):
    def __init__(self, channelsImage, channelsTarget):
        super().__init__()
        
        in_channels = channelsImage+channelsTarget
        nChannels = [64, 128, 256, 512]
        self.layers = nn.Sequential()
        # (None, 256, 256, 6)
        for i in range(len(nChannels)):
            out_channels = nChannels[i]
            self.layers.add_module(f"down_{i}", ConvBlock(in_channels,
                                                        out_channels,
                                                        stride=2 if i>0 else 1, 
                                                        kernel_size=3,
                                                        apply_batchnorm=False,
                                                        apply_spectral_norm=True,
                                                        activation=nn.GELU()))
            in_channels = out_channels
        # (None, 256, 32, 32)
        self.layers.add_module("conv", nn.Conv2d(nChannels[-1],1,kernel_size=3, padding=1, padding_mode="reflect"))
        # (None, 32, 32, 1)
        
        self.apply(weights_init)
        
    def forward(self, image, target):
        x = torch.cat([image, target], dim=1)
        x = self.layers(x)
        return x