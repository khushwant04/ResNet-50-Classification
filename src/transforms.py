from logging import getLogger

from PIL import ImageFilter

import torch 
import torchvision.transforms as transforms

logger = getLogger()


def make_transforms(
    color_jitter=1.0,
    horizontal_flip = False,
    color_distortion = False,
    normalization = ((0.4143, 0.4072, 0.3410),
                     (0.1671, 0.1613, 0.1685))
    ):
    logger.info("making animals data transforms")
    
    def get_color_distortion(s=1.0):
        color_jitter = transforms.ColorJitter(0.5*s,0.5*s,0.5*s,0.1*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter],p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray
        ])
        
        return color_distort
    
    transform_list = []
    if horizontal_flip:
        transform_list += [transforms.RandomHorizontalFlip()]
    if color_distortion:
        transform_list += [get_color_distortion(s=color_jitter)] 
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize(mean=normalization[0],std=normalization[1])]
    transform = transforms.Compose(transform_list)
    return transform    
        
        
    
    
        
        
        
  
        
        
        
        
        
        
        
        
        
        
