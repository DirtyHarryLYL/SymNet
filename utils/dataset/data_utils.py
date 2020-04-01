import torch
import torchvision.transforms as transforms
from PIL import Image
import scipy.io as sio
import os.path as osp
import numpy as np



class ImageLoader:
    def __init__(self, root):
        self.img_dir = root

    def __call__(self, img):
        str_types = [str]
        try:
            str_types.append(unicode)
        except NameError:
            pass

        if type(img) in str_types:
            f = '%s/%s'%(self.img_dir, img)
            img = Image.open(f).convert('RGB')
        elif type(img) in [list, tuple]:
            f = '%s/%s'%(self.img_dir, img[0])
            x,y,w,h = img[1:]  # bbox
            img = Image.open(f).convert('RGB')
            img = img.crop((x, y, x+w, y+h))
        else:
            raise NotImplementedError(str(type(img)))
        return img


def imagenet_transform(phase, transform_type):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if transform_type == 'normal':
        if phase=='train':
            transform = transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])
        elif phase in ['test', 'val']:
            transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])
    elif transform_type == 'fixed':
        transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])
    else:
        raise NotImplementedError("transform_type %s"%transform_type)

    return transform


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]