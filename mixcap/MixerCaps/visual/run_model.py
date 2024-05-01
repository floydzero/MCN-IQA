import torch
import torch.nn as nn
import os
from model import *
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

def load_model(checkopint_path):
    model = nn.DataParallel(MixerCaps().cuda())
    model.load_state_dict(torch.load(checkopint_path))
    model.eval()
    return model

def read_img(img_path):
    img = Image.open(img_path).convert('RGB')
    transf = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    img = transf(img).unsqueeze(0).cuda()
    return img


