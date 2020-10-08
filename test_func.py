import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from model_unet import UNet
import os
import cv2


def my32channels2rgb(img):
    matColered = np.array([[64, 192, 0, 0, 128, 64, 64, 192, 192, 64, 128, 192, 128, 192, 128, 64, 64, 128, 128, 0, 192,
                            128, 64, 0, 0, 192, 128, 192, 64, 192, 0, 64],
                           [128, 0, 128, 128, 0, 0, 0, 128, 192, 64, 0, 0, 128, 0, 64, 192, 64, 64, 128, 0, 128, 128,
                            128, 0, 64, 64, 128, 128, 0, 192, 0, 192],
                           [64, 128, 192, 64, 0, 128, 192, 64, 128, 128, 192, 64, 64, 192, 64, 128, 0, 128, 192, 192,
                            128, 128, 192, 64, 64, 128, 0, 192, 64, 0, 0, 0]])
    print(matColered.shape)
    [m, n] = np.shape(img)
    outImg = np.zeros([m, n, 3]);
    for i in range(m):
        for j in range(n):
            for k in range(3):
                outImg[i, j, k] = matColered[k, img[i, j].astype(np.integer)] / 255
                # print(outImg[i,j,:])
                # print(matColered[:,3].shape)

    return outImg

def conveter_32channels(img):
    matColered = np.array([[64, 192, 0, 0, 128, 64, 64, 192, 192, 64, 128, 192, 128, 192, 128, 64, 64, 128, 128, 0, 192,
                            128, 64, 0, 0, 192, 128, 192, 64, 192, 0, 64],
                           [128, 0, 128, 128, 0, 0, 0, 128, 192, 64, 0, 0, 128, 0, 64, 192, 64, 64, 128, 0, 128, 128,
                            128, 0, 64, 64, 128, 128, 0, 192, 0, 192],
                           [64, 128, 192, 64, 0, 128, 192, 64, 128, 128, 192, 64, 64, 192, 64, 128, 0, 128, 192, 192,
                            128, 128, 192, 64, 64, 128, 0, 192, 64, 0, 0, 0]])
    [m, n, k] = np.shape(img)
    # print(np.shape(img))
    outImg = np.zeros([m, n, 32])
    # print(np.shape(outImg))
    for i in range(32):
        mymap = np.ones([m, n, k])
        for j in range(3):
            mymap[:, :, j] = matColered[j, i] * mymap[:, :, j]
            # mymap[:][:][j] = 1*mymap[:][:][j]
        outtmp = (np.ones([m, n, k]) - np.ceil(np.abs(img - mymap) / 255))
        # print(np.shape(outtmp))
        outImg[:, :, i] = (i) * (outtmp[:, :, 0] * outtmp[:, :, 1] * outtmp[:, :, 2])
    return outImg

dir_inp = '/content/camvid-master/701_StillsRaw_full/'
dir_lbl = '/content/camvid-master/LabeledApproved_full/'
saved_model_path = '/content/drive/My Drive/unet_adam.pth'

device = 'cpu'
model_unet = UNet().to(device)

model_unet.load_state_dict(torch.load(saved_model_path))
image_dataset = Dataset(dir_inp, dir_inp)

data_loader = DataLoader(image_dataset, batch_size=1, shuffle=True)

image, lbl_img = data_loader[0]

plt.figure()
plt.imshow(image.cpu().data.numpy())
plt.show()

img_output = model_unet(image)
img_output_np = img_output.cpu().data.numpy()
img_output_np_max = np.argmax(img_output_np[200], axis=0)
# print(img_output_np_max)
plt.figure()
plt.imshow(my32channels2rgb(img_output_np_max))
plt.show()
target_tmp = conveter_32channels(lbl_img[0])
target_tmp2 = np.zeros([388, 388])
for j in range(32):
    target_tmp2 += target_tmp[:, :, j]
plt.figure()
plt.imshow(my32channels2rgb(target_tmp2))
plt.show()
target_tmp = conveter_32channels(lbl_img[0])
target_tmp2 = np.zeros([388, 388])
for j in range(32):
    target_tmp2 += target_tmp[:, :, j]
plt.figure()
plt.imshow(target_tmp2)
plt.show()