import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from matplotlib import cm
 
#The path of the image to be visualized

img_161 = Image.open('img/161.bmp').convert('RGB')
v_161 = torch.load('./161_v.tensor').norm(dim=-1).squeeze().cpu().detach().numpy()
max_161 = v_161[6].reshape(14, 14)
min_161 = v_161[4].reshape(14, 14)

img_156 = Image.open('img/156.bmp').convert('RGB')
v_156 = torch.load('./156_v.tensor').norm(dim=-1).squeeze().cpu().detach().numpy()
max_156 = v_156[6].reshape(14, 14)
min_156 = v_156[4].reshape(14, 14)

img_471 = Image.open('img/471.bmp').convert('RGB')
v_471 = torch.load('./471_v.tensor').norm(dim=-1).squeeze().cpu().detach().numpy()
max_471 = v_471[6].reshape(14, 14)
min_471 = v_471[1].reshape(14, 14)

img_1 = Image.open('img/1.png').convert('RGB')
v_1 = torch.load('./1_v.tensor').norm(dim=-1).squeeze().cpu().detach().numpy()
max_1 = v_1[6].reshape(14, 14)
min_1 = v_1[1].reshape(14, 14)

img_2 = Image.open('img/2.png').convert('RGB')
v_2 = torch.load('./2_v.tensor').norm(dim=-1).squeeze().cpu().detach().numpy()
max_2 = v_2[6].reshape(14, 14)
min_2 = v_2[3].reshape(14, 14)

"""
Visualization code using another color (jet) of the heatmaps.
"""
# plt.subplot(4, 4, 1)
# e = plt.imshow(sim_30, cm.jet,interpolation='bilinear')
# plt.xticks([])  
# plt.yticks([])

plt.subplot(3, 5, 1)
plt.text(0, 80, '0.7920', fontsize=6) # spaq
plt.imshow(img_471)

plt.subplot(3, 5, 2)
plt.text(0, 80, '0.4684', fontsize=6) # spaq
plt.imshow(img_156)

plt.subplot(3, 5, 3)
plt.text(0, 80, '0.3449', fontsize=6) # spaq
plt.imshow(img_161)

plt.subplot(3, 5, 4)
plt.text(0, 80, '0.5550', fontsize=6) # spaq
plt.imshow(img_1)

plt.subplot(3, 5, 5)
plt.text(0, 80, '0.8687', fontsize=6) # spaq
plt.imshow(img_2)

plt.subplot(3, 5, 6)
a = plt.imshow(max_471, cmap='viridis')

plt.subplot(3, 5, 7)
b = plt.imshow(max_156, cmap='viridis')

plt.subplot(3, 5, 8)
c = plt.imshow(max_161, cmap='viridis')

plt.subplot(3, 5, 9)
c = plt.imshow(max_1, cmap='viridis')

plt.subplot(3, 5, 10)
c = plt.imshow(max_2, cmap='viridis')

plt.subplot(3, 5, 11)
d = plt.imshow(min_471, cmap='viridis')

plt.subplot(3, 5, 12)
e = plt.imshow(min_156, cmap='viridis')

plt.subplot(3, 5, 13)
f = plt.imshow(min_161, cmap='viridis')

plt.subplot(3, 5, 14)
c = plt.imshow(min_1, cmap='viridis')

plt.subplot(3, 5, 15)
c = plt.imshow(min_2, cmap='viridis')


plt.savefig('files/pretrained_on_spaq_with_16_caps.png', dpi = 300)
plt.close()
