from zipfile import ZipFile
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd import Variable
from model_unet import UNet
from torch.utils.data.dataloader import DataLoader
from dataset import Dataset


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dir_inp = '/home/soroush/codes/test/camvid-master/701_StillsRaw_full/'
dir_lbl = '/home/soroush/codes/test/camvid-master/LabeledApproved_full/'

image_dataset = Dataset(dir_inp, dir_lbl)
saved_model_path = '/home/soroush/codes/test/unet_adam.pth'
data_loader = DataLoader(image_dataset, batch_size=80, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_unet = UNet().to(device)
criterion = nn.CrossEntropyLoss().to(device)
learning_rate = 0.0001
optimizer = optim.Adam(model_unet.parameters(), lr=learning_rate)
num_epochs = 100

# model_unet.load_state_dict(torch.load(PATH))

for epoch in range(num_epochs):
    print(epoch)
    train_epoch_loss = []
    loss_tot = 0
    for batch_data, batch_label in data_loader:

        batch_data = batch_data.to(device)
        batch_label = batch_label.to(device)

        optimizer.zero_grad()
        img_output = model_unet(batch_data)

        loss = criterion(img_output, batch_label.to(dtype=torch.long))

        loss.backward()
        optimizer.step()

        loss_tot += loss
        train_epoch_loss.append(loss_tot)

    torch.save(model_unet.state_dict(), saved_model_path)


