import streamlit as st
import torch
import torch.nn as nn
import torch.utils.data
import random
import torch.nn.parallel
import torch.utils.data
import torchvision
import numpy as np
import pandas as pd
from PIL import Image

ngpu = 0


def rand_pic(n):
    str_n = str(n)
    return str_n


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


def loss_chart():
    dict = torch.load("C:/Users/orita/Documents/DCGAN weights/DCGAN weights.pth", map_location=torch.device('cpu'))
    D_losses = np.array(dict.get('D_losses'))
    G_losses = np.array(dict.get('G_losses'))
    losses = pd.DataFrame()
    losses["D loss"] = D_losses
    losses["G loss"] = G_losses
    st.line_chart(losses, 1000, 400)


header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()
generating = st.container()

with header:
    st.title("Welcome to my website")
    st.text("In this page i'll show you my generated Anime picture")

with dataset:
    st.header("Anime dataset")
    st.text("Here is an example from the dataset!")
    n = random.randint(1, 21551)
    path = 'C:/Users/orita/Documents/dataFinalProject/data/' + rand_pic(n) + '.png'
    image = Image.open(path)
    st.image(image)

with model_training:
    st.header("Training info")
    if st.button('chart'):
        loss_chart()
    else:
        st.write("click to open loss chart")

with generating:
    st.header("Generated photo")
    device = torch.device("cpu")
    netG = Generator(ngpu).to(device)
    dict = torch.load("C:/Users/orita/Documents/DCGAN weights/DCGAN weights.pth", map_location=torch.device('cpu'))
    netG.load_state_dict(dict.get('netG'))
    noise = torch.randn(1, 100, 1, 1, device=device)
    fake = netG(noise)
    torchvision.utils.save_image(fake, "C:/Users/orita/Documents/generatedImages/" "temp_img.png", normalize=True)
    fake_image = Image.open("C:/Users/orita/Documents/generatedImages/temp_img.png")
    st.image(fake_image, width=300)
