import torch
import torch.nn as nn
import os
from model import *
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from scipy import io
import time
import torchvision.transforms.functional as F
from torchvision.io import read_image
import scipy.stats as stats

"""
Code for inferring image MOS based on model loading weights
Run visual.py and replace the path of image
"""


checkopint_path = '/home/long/IQA/MixerCaps/log/checkpoint/koniq/1/ckpt.pt'
model = nn.DataParallel(MixerCaps()).cuda()
model.load_state_dict(torch.load(checkopint_path))
model.eval()

transf = transforms.Compose([
    torchvision.transforms.Resize((512, 384)),
    #torchvision.transforms.RandomCrop(size=224),
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

#image = '/home/weiyijie/dataset/database/blur_dataset/out_of_focus0303.jpg'

img = Image.open(image).convert('RGB')
#     img = Image.open(image + 'img' + str(i) + '.bmp').convert('RGB')
img = transf(img).unsqueeze(0).cuda()
out = model(img)
print(out)
