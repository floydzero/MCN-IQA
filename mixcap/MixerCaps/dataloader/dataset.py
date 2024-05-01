import torch
import torchvision
import scipy.io
import cv2
import random
import os
import numpy as np
from PIL import Image
import csv
from config import cfg
import pandas as pd
import xlrd

# determine if using map into [-1, 1]
# def map(data,MIN=-1,MAX=1):
#     d_min = np.max(data)
#     d_max = np.min(data)
#     return MIN +(MAX-MIN)/(d_max-d_min) * (data - d_min)

def map(data, MIN=-1, MAX=1):
    return data

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def getFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename


def getDistortionTypeFileName(path, num):
    filename = []
    index = 1
    for i in range(0, num):
        name = '%s%s%s' % ('img', str(index), '.bmp')
        filename.append(os.path.join(path, name))
        index = index + 1
    return filename


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def getTIDFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if suffix.find(os.path.splitext(i)[1]) != -1:
            filename.append(i[1:3])
    return filename


# LIVE with classification
class LIVEFolder(torch.utils.data.Dataset):
    def __init__(self, root, index, transform, patch_num):

        classlabel = []

        refpath = os.path.join(root, 'refimgs')
        refname = getFileName(refpath, '.bmp')

        jp2kroot = os.path.join(root, 'jp2k')
        jp2kname = self.getDistortionTypeFileName(jp2kroot, 227)
        for _ in range(len(jp2kname)):
            classlabel.append(0)

        jpegroot = os.path.join(root, 'jpeg')
        jpegname = self.getDistortionTypeFileName(jpegroot, 233)
        for _ in range(len(jpegname)):
            classlabel.append(1)

        wnroot = os.path.join(root, 'wn')
        wnname = self.getDistortionTypeFileName(wnroot, 174)
        for _ in range(len(wnname)):
            classlabel.append(2)

        gblurroot = os.path.join(root, 'gblur')
        gblurname = self.getDistortionTypeFileName(gblurroot, 174)
        for _ in range(len(gblurname)):
            classlabel.append(3)

        fastfadingroot = os.path.join(root, 'fastfading')
        fastfadingname = self.getDistortionTypeFileName(fastfadingroot, 174)
        for _ in range(len(fastfadingname)):
            classlabel.append(4)

        imgpath = jp2kname + jpegname + wnname + gblurname + fastfadingname

        dmos = scipy.io.loadmat(os.path.join(root, 'dmos_realigned.mat'))
        labels = dmos['dmos_new'].astype(np.float32)

        labels = map(labels)

        orgs = dmos['orgs']
        refnames_all = scipy.io.loadmat(os.path.join(root, 'refnames_all.mat'))
        refnames_all = refnames_all['refnames_all']

        sample = []

        for i in range(0, len(index)):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = train_sel * ~orgs.astype(np.bool_)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[1].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    # sample.append((imgpath[item], labels[0][item], classlabel[item]))
                    sample.append((imgpath[item], labels[0][item]))
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        path, target, classlabel = self.samples[index]
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return [sample, target, classlabel]

    def __len__(self):
        length = len(self.samples)
        return length

    def getDistortionTypeFileName(self, path, num):
        filename = []
        index = 1
        for i in range(0, num):
            name = '%s%s%s' % ('img', str(index), '.bmp')
            filename.append(os.path.join(path, name))
            index = index + 1
        return filename

# LIVEC
class LIVEChallengeFolder(torch.utils.data.Dataset):

    def __init__(self, root, index, transform, patch_num):

        imgpath = scipy.io.loadmat(os.path.join(root, 'Data', 'AllImages_release.mat'))
        imgpath = imgpath['AllImages_release']
        imgpath = imgpath[7:1169]
        mos = scipy.io.loadmat(os.path.join(root, 'Data', 'AllMOS_release.mat'))
        labels = mos['AllMOS_release'].astype(np.float32)
        labels = labels[0][7:1169]

        labels = map(labels)

        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                # sample.append((os.path.join(root, 'Images', imgpath[item][0][0]), labels[item]))
                sample.append((os.path.join(root, 'Images', imgpath[item][0][0])))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


# Koniq-10k
class Koniq_10kFolder(torch.utils.data.Dataset):

    def __init__(self, root, index, transform, patch_num):
        imgname = []
        mos_all = []
        csv_file = os.path.join(root, 'koniq10k_scores_and_distributions.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['image_name'])
                mos = np.array(float(row['MOS_zscore'])).astype(np.float32)
                mos_all.append(mos)

        mos_all = map(np.array(mos_all))

        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join(root, '1024x768', imgname[item]), mos_all[item]))
                # sample.append((os.path.join(root, '512x384', imgname[item]), mos_all[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


# tid2013
class TID2013Folder(torch.utils.data.Dataset):
    def __init__(self, root, index, transform, patch_num):
        refpath = os.path.join(root, 'reference_images')
        refname = getTIDFileName(refpath,'.bmp.BMP')
        txtpath = os.path.join(root, 'mos_with_names.txt')
        fh = open(txtpath, 'r')
        imgnames = []
        target = []
        class_label = []
        refnames_all = []
        for line in fh:
            line = line.split('\n')
            words = line[0].split()
            class_label.append(words[1][4:6])
            imgnames.append(words[1])
            target.append(words[0])
            ref_temp = words[1].split("_")
            refnames_all.append(ref_temp[0][1:])
        labels = np.array(target).astype(np.float32)
        class_label = np.array(class_label).astype(np.int32) - 1
        class_label = class_label.astype(np.long)
        refnames_all = np.array(refnames_all)
        sample = []

        labels = map(labels)

        for i, item in enumerate(index):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    # sample.append((os.path.join(root, 'distorted_images', imgnames[item]), labels[item], class_label[item]))
                    sample.append((os.path.join(root, 'distorted_images', imgnames[item]), labels[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        path, target, class_label = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return [sample, target, class_label]

    def __len__(self):
        length = len(self.samples)
        return length


# Kadid-10k
class Kadid_10kFolder(torch.utils.data.Dataset):

    def __init__(self, root, index, transform, patch_num):
        csv_file = os.path.join(root, 'dmos.csv')
        df = pd.read_csv(csv_file)
        imgnames = df['dist_img'].tolist()
        classlabels = []
        for img in imgnames:
            classlabels.append(int(img[4: 6]) - 1)
        classlabels = np.array(classlabels).astype(np.long)
        labels = np.array(df['dmos']).astype(np.float32)
        refname = np.unique(np.array(df['ref_img']))
        refnames_all = np.array(df['ref_img'])
        sample = []
        for i, item in enumerate(index):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    # sample.append((os.path.join(root, 'images', imgnames[item]), labels[item], classlabels[item]))
                    sample.append((os.path.join(root, 'images', imgnames[item]), labels[item]))

        self.samples = sample
        self.transform = transform

        labels = map(labels)

    def __getitem__(self, index):
        path, target, class_label = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)

        return [sample, target, class_label]

    def __len__(self):
        length = len(self.samples)
        return length


# SPAQ
class SPAQ_Folder(torch.utils.data.Dataset):

    def __init__(self, root, index, transform, patch_num):
        imgname = []
        mos_all = []
        csv_file = os.path.join(root, 'Annotations/MOS and Image attribute scores.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['Image name'])
                mos = np.array(float(row['MOS'])).astype(np.float32)
                mos_all.append(mos)

        mos_all = map(np.array(mos_all))

        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                # sample.append((os.path.join(root, 'TestImage', imgname[item]), mos_all[item]))
                sample.append((os.path.join(root, 'result', imgname[item]), mos_all[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length

