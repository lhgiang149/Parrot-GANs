from __future__ import print_function

import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from model import Generator,Discriminator
from utils import weights_init, label_init

# Get the maximum number of workers
import multiprocessing
max_workers = multiprocessing.cpu_count()-2 if multiprocessing.cpu_count() > 2 else 1

# Set random seed
manualSeed = 1234
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# path of data
data_path = "./../imgs/"

# number of workers in cpu
workers = max_workers

# Just batch size
batch_size = 128

# size of image, change later
image_size = 64

# number of channels in one image
nc = 3

# Size of latent vector
nz = 100

# Size of feature maps in generator 
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Num epochs
num_epochs = 5

# Beta1 for Adam
beta1 = 0.5

# Number of GPUs
ngpu = 1

# load data and resize image
dataset = dset.ImageFolder(root=data_path, 
                            transform=transforms.Compose([
                                transforms.Resize((image_size, image_size)),
                                # transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                            ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = workers)

# choose cuda
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# generate G_model
G = Generator(latent_depth = nz, feature_depth = ngf).to(device)

# generate D_model
D = Discriminator(feature_depth = ndf).to(device)

if (device.type == 'cuda') and (ngpu > 1):
    G = nn.DataParallel(G, list(range(ngpu)))
    D = nn.DataParallel(D, list(range(ngpu)))

G.apply(weights_init)
D.apply(weights_init)

criterion = nn.BCELoss()

fixed_noise = torch.randn(64, nz, 1, 1, device=device)

real_label = lambda: random.uniform(0.7,1.2)
fake_label = lambda: random.uniform(0.0,0.3)

optimizerD = optim.SGD(D.parameters(), lr=0.01, momentum=0.9)
optimizerG = optim.Adam(G.parameters(), lr=0.0001, betas=(beta1, 0.999))

img_list = []
G_losses = []
D_losses = []
iters = 0

for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        D.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        # Forward pass real batch through D
        output = D(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1




