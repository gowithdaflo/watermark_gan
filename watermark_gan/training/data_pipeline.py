import random
from PIL import Image, ImageDraw, ImageFont

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import os
import string
import torch
from ..utils.utils import uint8_to_float, find_files
import logging

FONTS = [
    "calibri.ttf", "arial.ttf", "Candara.ttf", "consola.ttf", "constan.ttf", "simsunb.ttf", "sylfaen.ttf", "tahoma.ttf", 
    "verdana.ttf", "himalaya.ttf", "impact.ttf", "georgia.ttf", "corbel.ttf", "ebrima.ttf", "Gabriola.ttf", "gadugi.ttf", 
    "seguisb.ttf", "segoeuisl.ttf", "pala.ttf", "ariali.ttf", "arialbd.ttf", "arialbi.ttf", "ariblk.ttf", "bahnschrift.ttf", 
    "calibril.ttf", "calibrili.ttf", "calibrii.ttf", "calibrib.ttf", "calibriz.ttf", "cambriai.ttf", "cambriab.ttf", 
    "Candaral.ttf", "Candarali.ttf", "Candarai.ttf", "Candarab.ttf", "Candaraz.ttf", "comic.ttf", "comici.ttf", "comicbd.ttf", 
    "comicz.ttf", "consolai.ttf", "consolab.ttf", "consolaz.ttf", "constani.ttf", "constanb.ttf", "constanz.ttf", "corbell.ttf", 
    "corbelli.ttf", "corbeli.ttf", "corbelb.ttf", "corbelz.ttf", "cour.ttf", "couri.ttf", "cambriaz.ttf", "seguibl.ttf", 
    "courbd.ttf", "courbi.ttf", "ebrimabd.ttf", "framd.ttf", "framdit.ttf", "gadugib.ttf", "georgiai.ttf", "georgiab.ttf", 
    "Inkfree.ttf", "javatext.ttf", "lucon.ttf", "l_10646.ttf", "malgun.ttf", "malgunbd.ttf", "malgunsl.ttf", "ntailu.ttf", 
    "ntailub.ttf", "phagspa.ttf", "phagspab.ttf", "micross.ttf", "taile.ttf", "taileb.ttf", "msyi.ttf", "monbaiti.ttf", 
    "mvboli.ttf", "mmrtext.ttf", "mmrtextb.ttf", "Nirmala.ttf", "palai.ttf", "seguibli.ttf", "seguihis.ttf", "trebucit.ttf", 
    "palab.ttf", "palabi.ttf", "segoepr.ttf", "segoeprb.ttf", "segoesc.ttf", "segoescb.ttf", "segoeuil.ttf", "georgiaz.ttf",
    "seguili.ttf", "seguisli.ttf", "segoeui.ttf", "segoeuii.ttf", "seguisbi.ttf", "segoeuib.ttf", "segoeuiz.ttf", 
    "seguiemj.ttf","seguisym.ttf", "tahomabd.ttf", "times.ttf", "timesi.ttf", "timesbd.ttf", "timesbi.ttf", "trebuc.ttf",
    "trebucbd.ttf", "trebucbi.ttf", "verdanai.ttf", "verdanab.ttf", "verdanaz.ttf"]


class WatermarkDataset(Dataset):
    def __init__(self, dir_path, transform=None, device="cpu", model_stride=8, model_rec_field=(38,38)):
        """
        Args:
            dir_path (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__()
        self.image_paths = self.find_images(dir_path)
        self.transform = transform
        self.font_ratio_range = (5,40)
        self.device = device
        
        padding = (int((model_rec_field[0]-model_stride)/2), int((model_rec_field[1]-model_stride)/2))
        self.recep_avgpool = torch.nn.AvgPool2d(model_rec_field,model_stride,padding)
        self._fonts = FONTS
        
    def load_fonts(self, path):
        self._fonts = find_files(path, pattern="*.ttf", recursive=True)
    
    def __len__(self):
        return len(self.image_paths)
    
    def find_images(self, path, recursive=True):
        images = []
        
        files = find_files(path, pattern="*.*", recursive=recursive)
        logging.debug(f"Found {len(files)} files.")
        
        valid_images = [".jpg",".jpeg",".gif",".png",".tga"]
        for f in files:
            ext = os.path.splitext(f)[-1]
            if ext.lower() not in valid_images:
                continue
            images.append(f)
            
        logging.debug(f"Found {len(images)} images.")
        return images

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = int(idx)
        
        sample = {}
        
        path = self.image_paths[idx]
        #Opening Image & Creating New Text Layer
        with Image.open(path) as img:
            img = img.convert("RGBA")
            img = self._random_crop_image(img)
            case = random.randint(0,1)

            if case == 0:
                watermark = self._gen_text_simple(img)
            else:
                watermark = self._gen_text_pattern(img)
            
            #Combining both layers and saving new image
            sample["inputs"] = np.asarray(Image.alpha_composite(img, watermark)) 
            sample["inputs"] = uint8_to_float(sample["inputs"][...,:3])
            sample["targets"] = np.asarray(img)
            sample["targets"] = uint8_to_float(sample["targets"][...,:3])
            sample["mask"] = np.asarray(watermark)[...,-1] > 0
            
        sample["inputs"] = torch.FloatTensor(sample["inputs"], device=self.device).permute(2,0,1)
        sample["targets"] = torch.FloatTensor(sample["targets"], device=self.device).permute(2,0,1)
        sample["mask"] = torch.BoolTensor(sample["mask"], device=self.device).unsqueeze(0)

        if self.transform:
            sample = self.transform(sample)
            
        # predict if more than 2% of pixels are true of fake
        # sample["recep_mask"] = (self.recep_avgpool(sample["mask"].clone().detach().type(torch.float32)) <= 0.02).type(torch.float32)
                
        return sample
    
    @staticmethod
    def _random_text():
        nLetters = random.randint(5,20)
        return "".join( random.choices(string.ascii_letters + ".@", k=nLetters) )
    
    def _random_font(self, img_height):
        font_size = int(img_height/random.randint(*self.font_ratio_range)) # in pixels
        font_type = np.random.choice(self._fonts)
        return ImageFont.truetype(font_type, font_size) 
    
    def _random_opactity(self, min_value = 80, max_value = 150):
        return random.randint(min_value, max_value)
    
    def _gen_text_simple(self, img: Image.Image):
        width, height = img.size 
        watermark_img = Image.new('RGBA', img.size, (0,0,0,0))
        draw = ImageDraw.Draw(watermark_img)

        # Creating Text
        text = self._random_text()
        font = self._random_font(height)
        opacity = self._random_opactity()
        pos = (random.randint(0,int(0.7*width)), random.randint(0,int(0.7*height)))
        
        draw.text(pos, text, fill=(255,255,255, opacity), font=font)
        
        return watermark_img

    def _gen_text_pattern(self, img: Image.Image):
        width, height = img.size 
        watermark_img = Image.new('RGBA', img.size, (0,0,0,0))
        draw = ImageDraw.Draw(watermark_img)

        # Creating Text
        text = self._random_text()
        font = self._random_font(height)
        opacity = self._random_opactity()
        start_pos = (random.randint(0,int(0.2*width)), random.randint(0,int(0.2*height)))
        
        # random positions        
        positions = [start_pos]
        y = start_pos[1]
        for _ in range(10):
            y += height * (random.random() * 0.1 + 0.1)
            x = start_pos[0]
            for _ in range(10):
                x += len(text)*font.size * (random.random()*0.4+0.6) 
                positions.append((int(x),int(y)))
        
        # draw text
        for pos in positions:
            draw.text(pos, text, fill=(255,255,255, opacity), font=font)
            
        # rotate text up to at most 45°
        angle = random.random() * 45
        watermark_img = watermark_img.rotate(angle, resample=Image.BILINEAR)
        
        return watermark_img
    
    @staticmethod
    def _random_crop_image(image: Image.Image):
        width, height = image.size
        
        wh = min(height, width)
        new_wh = 2**int(np.log2(wh))
        new_w, new_h = new_wh, new_wh
        
        upper = random.randint(0, height - new_h)
        left = random.randint(0, width - new_w)
        
        image = image.crop((left,upper, left+new_w,upper+new_w))

        return image
    
def create_test_image(img_path, pos, text, opacity, font_type, font_size):
    with Image.open(img_path) as img:
        img = img.convert("RGBA")
        
        width, height = img.size
        new_w, new_h = 2**int(np.log2(width)), 2**int(np.log2(height))
        upper, left = 0, 0
        img = img.crop((left,upper, left+new_w,upper+new_h))

        watermark = Image.new('RGBA', img.size, (0,0,0,0))
        draw = ImageDraw.Draw(watermark)
        # Creating Text
        draw.text(pos, text, fill=(255,255,255, opacity), font=ImageFont.truetype(font_type, font_size))
        
        output = np.asarray(Image.alpha_composite(img, watermark)) 
        output = uint8_to_float(output[...,:3])
        return torch.FloatTensor(output).permute(2,0,1)
    
    
    # def _gen_image(self, img):
    #     watermark_image_path = "D:/Programmieren/gans/datasets/watermarks/Download.png"
    #     watermark = Image.open(watermark_image_path).convert("RGBA")
    #     transparent = Image.new('RGBA', (width, height), (0,0,0,0))
    #     transparent.paste(img, (0,0))
    #     position = (10,10)
    #     transparent.paste(watermark.convert("L"), position, mask=watermark.point( lambda p: 255 if p < 230 else 0 ).convert("1") )
    #     transparent.putalpha("opacity")

    #     return img
    
# class RandomCrop(torch.nn.Module):
#     """
#     Random crop the image in square whose shape is multiples of 2
#     """
#     def __init__(self):
#         super().__init__()

#     def __call__(self, sample):
#         image = sample['inputs']
#         _, height, width = image.shape
        
#         wh = min(height, width)
#         new_wh = 2**int(np.log2(wh))
#         new_w, new_h = new_wh, new_wh
        
#         top = random.randint(0, height - new_h)
#         left = random.randint(0, width - new_w)

#         for key in sample.keys():
#             sample[key] = sample[key][:,top: top + new_h, left: left + new_w]

#         return sample

class Scale(torch.nn.Module):
    """
    """
    def __init__(self, output_shape):
        super().__init__()
        self.output_shape = output_shape

    def __call__(self, sample):
        image = sample['inputs']
        _, _, wh = image.shape
        
        if self.output_shape == wh:
            return sample
        
        for key in sample.keys():
            interpolation = transforms.InterpolationMode.BILINEAR if key != "mask" else transforms.InterpolationMode.NEAREST
            sample[key] = transforms.functional.resize(sample[key], 
                                            size=(self.output_shape, self.output_shape), 
                                            interpolation = interpolation)

        return sample

