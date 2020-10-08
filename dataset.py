import os
import numpy as np
import torch
from PIL import Image
from skimage import io, transform, color
import cv2
from torch.utils.data import DataLoader, IterableDataset, Dataset
import scipy.io
import random
from zipfile import ZipFile

with ZipFile('/content/drive/My Drive/camvid-master.zip', 'r') as zip_file:
    zip_file.extractall()

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




class Dataset(Dataset):
    def __init__(self, dataset_image_dir, dataset_lbl_dir):
        self.dataset_image_dir = dataset_image_dir
        self.dataset_lbl_dir = dataset_lbl_dir
        self.list_files = [file_name for file_name in os.listdir(self.dataset_image_dir) if file_name.endswith(".png")]
        self.labels = [file_name[0:-4] for file_name in self.list_files]
        self.transform = transform

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):
        inp_img_tmp = cv2.imread(self.dataset_image_dir + self.list_files[idx])
        inp_img_tmp = cv2.cvtColor(inp_img_tmp, cv2.COLOR_BGR2RGB)
        # plt.figure()
        # plt.imshow(inp_img_tmp)
        # plt.show()
        inp_img_tmp = cv2.resize(inp_img_tmp, (388, 388))
        part1 = np.fliplr(inp_img_tmp)
        imgp1 = np.concatenate((part1[:, 388 - 92 - 1:-1], inp_img_tmp, part1[:, 0:92]), axis=1)
        part2 = np.flipud(imgp1)
        inp_img_tmp = np.concatenate((part2[388 - 92 - 1:-1, :], imgp1, part2[0:92, :]))
        images = inp_img_tmp.astype(np.float32)
        # print(images)
        for j in range(3):
            images[:, :, j] = images[:, :, j] / np.max(images[:, :, j])
        # print(np.max(lbl_img[i]))
        img_input = torch.from_numpy(np.array([[images[:, :, 0], images[:, :, 1], images[:, :, 2]]]))

        lbl_img_tmp = cv2.imread(self.dataset_lbl_dir + self.labels[idx] + '_L.png')
        lbl_img_tmp = cv2.cvtColor(lbl_img_tmp, cv2.COLOR_BGR2RGB)
        lbl_img = torch.from_numpy(np.sum(conveter_32channels(cv2.resize(lbl_img_tmp, (388, 388))), 2))

        return img_input, lbl_img
