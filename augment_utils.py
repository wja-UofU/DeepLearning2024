import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms 
from torchvision.transforms import v2

#based on : https://pytorch.org/vision/0.19/transforms.html

def augment_dataset(x,transform_specs):
        x = torch.tensor(x,dtype=torch.float32)
        if x.shape[1] != 3: 
            x = x.permute(0, 3, 1, 2)
        # x_images_PIL = transforms.ToPILImage()(x)
        if transform_specs =='randomFlip_Hoz':
              transform  = v2.Compose([v2.RandomHorizontalFlip(p=0.5)])
            # transform = v2.Compose([v2.RandomHorizontalFlip(p=0.5),v2.RandomVerticalFlip(p=0.5),v2.RandomRotation(degrees=(0,360))])
        elif transform_specs == 'randomFlip_Vert':
              transform  = v2.Compose([v2.RandomVerticalFlip(p=0.5)])
        elif transform_specs == 'randomRotation':
              transform = v2.Compose([v2.RandomRotation(degrees=(0,360))])
        elif transform_specs == 'randomResizeCrop':
              transform  = v2.Compose([v2.RandomResizedCrop(size=(256,256),scale=(0.8,1.0))])
        elif transform_specs =='affine':
              transform = v2.Compose([v2.RandomAffine(degrees=0,translate=[0.2,0.2])])
        elif transform_specs =='gaussian':
              transform = v2.Compose([v2.GaussianBlur(kernel_size=(3,3),sigma=(0.1,2.0))])
        elif transform_specs =='best':
              transform1 = v2.Compose([v2.RandomRotation(degrees=(0,360))])
            #   transform2 = v2.Compose([v2.RandomAffine(degrees=0,translate=[0.2,0.2])])
              transform3 = v2.Compose([v2.GaussianBlur(kernel_size=(3,3),sigma=(0.1,2.0))])
              x1 = transform1(x)
            #   x2 = transform2(x)
              x3 = transform3(x)
              x_out = torch.cat([x1, x3], dim=0) 
        elif transform_specs =='paper':
                transform = v2.Compose([v2.RandomHorizontalFlip(p=0.5),v2.RandomVerticalFlip(p=0.5),v2.RandomRotation(degrees=(0,360))])
        if transform_specs != 'best':
            x_out = transform(x)
      
            
        return x_out

