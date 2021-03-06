import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from model_unet import UNet
import os
import cv2
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
from dataset import Dataset


def my32channels2rgb(img):
    matColered = np.array([[64, 192, 0, 0, 128, 64, 64, 192, 192, 64, 128, 192, 128, 192, 128, 64, 64, 128, 128, 0, 192,
                            128, 64, 0, 0, 192, 128, 192, 64, 192, 0, 64],
                           [128, 0, 128, 128, 0, 0, 0, 128, 192, 64, 0, 0, 128, 0, 64, 192, 64, 64, 128, 0, 128, 128,
                            128, 0, 64, 64, 128, 128, 0, 192, 0, 192],
                           [64, 128, 192, 64, 0, 128, 192, 64, 128, 128, 192, 64, 64, 192, 64, 128, 0, 128, 192, 192,
                            128, 128, 192, 64, 64, 128, 0, 192, 64, 0, 0, 0]])

    [m, n] = np.shape(img)
    outImg = np.zeros([m, n, 3])
    for i in range(m):
        for j in range(n):
            for k in range(3):
                outImg[i, j, k] = matColered[k, img[i, j].astype(np.integer)] / 255

    return outImg

def conveter_32channels(img):
    matColered = np.array([[64, 192, 0, 0, 128, 64, 64, 192, 192, 64, 128, 192, 128, 192, 128, 64, 64, 128, 128, 0, 192,
                            128, 64, 0, 0, 192, 128, 192, 64, 192, 0, 64],
                           [128, 0, 128, 128, 0, 0, 0, 128, 192, 64, 0, 0, 128, 0, 64, 192, 64, 64, 128, 0, 128, 128,
                            128, 0, 64, 64, 128, 128, 0, 192, 0, 192],
                           [64, 128, 192, 64, 0, 128, 192, 64, 128, 128, 192, 64, 64, 192, 64, 128, 0, 128, 192, 192,
                            128, 128, 192, 64, 64, 128, 0, 192, 64, 0, 0, 0]])
    [m, n, k] = np.shape(img)
    outImg = np.zeros([m, n, 32])
    for i in range(32):
        mymap = np.ones([m, n, k])
        for j in range(3):
            mymap[:, :, j] = matColered[j, i] * mymap[:, :, j]
        outtmp = (np.ones([m, n, k]) - np.ceil(np.abs(img - mymap) / 255))
        outImg[:, :, i] = (i) * (outtmp[:, :, 0] * outtmp[:, :, 1] * outtmp[:, :, 2])
    return outImg

dir_inp = '/home/soroush/codes/test/camvid-master/701_StillsRaw_full/'
dir_lbl = '/home/soroush/codes/test/camvid-master/LabeledApproved_full/'

saved_model_path = '/content/drive/My Drive/unet_adam.pth'

device = 'cpu'
model_unet = UNet().to(device)

# model_unet.load_state_dict(torch.load(saved_model_path))
image_dataset = Dataset(dir_inp, dir_lbl)
subset = Subset(image_dataset, [0])

data_loader = DataLoader(subset, batch_size=1, num_workers=0, shuffle=False)

for image, lbl_img in data_loader:
    plt.figure()
    plt.imshow(image.cpu().data.numpy()[0].swapaxes(0,1).swapaxes(1,2))
    plt.show()

    img_output = model_unet(image)
    img_output_np = img_output.cpu().data.numpy()[0]
    img_output_np_max = np.argmax(img_output_np, axis=0)

    plt.figure()
    plt.imshow(my32channels2rgb(img_output_np_max))
    plt.show()
    plt.figure()
    plt.imshow(my32channels2rgb(lbl_img.cpu().data.numpy()[0]))
    plt.show()
