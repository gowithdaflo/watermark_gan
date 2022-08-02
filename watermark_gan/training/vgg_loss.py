import torch.nn as nn
import torchvision

class VGG(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()    
        
        model = torchvision.models.vgg19(pretrained=True)
        self.slices = nn.ModuleList([])
        
        block = nn.Sequential()
        count = 0
        for i, f in enumerate(model.features):
            if isinstance(f, nn.MaxPool2d):
                self.slices.append(block)
                
                block = nn.Sequential(f)
                count += 1
            else:
                block.add_module(str(i), f)
                
        # freeze the model
        for param in self.parameters():
            param.requires_grad = False
        
        for i in range(len(self.slices)):
            self.slices[i] = self.slices[i].to(device)
        
    def forward(self, x):
        out = []
        for slice in self.slices:
            x = slice(x)
            out.append(x)
        # torch.Size([None, 64,  256, 256])
        # torch.Size([None, 128, 128, 128])
        # torch.Size([None, 256, 64,  64])
        # torch.Size([None, 512, 32,  32])
        # torch.Size([None, 512, 16,  16])
        return out
        
class VGGLoss(nn.Module):
    def __init__(self, criterion=nn.L1Loss(), device="cpu"):
        super().__init__()    
        self.vgg = VGG(device=device)
        self.criterion = criterion
        self.factors = [1/2**5, 1/2**4, 1/2**3, 1/2**2, 1]
        
    def forward(self, x_gen, y_tar):
        x_vgg, y_vgg = self.vgg((x_gen+1)/2), self.vgg((y_tar+1)/2)
        loss = 0
        for i, f in enumerate(self.factors):
            loss += f * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
        
        