__author__ = 'Hang Wu'

import torchvision
import torchvision.transforms as transforms
import torch
from utils import *

import matplotlib.pyplot as plt
import numpy as np
def imshow(img, ax):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    #ax.imshow(np.transpose(npimg, (1,2,0)))
    ax.imshow(npimg, cmap=plt.cm.Greys)
    ax.set_axis_off()

transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True, num_workers=2)

dataiter = iter(trainloader)
images, labels = dataiter.next()


fig, axes = plt.subplots(nrows=10, ncols=7)
#print(axes)

noise_free_idxs = [np.where(labels.numpy() == i)[0][0] for i in range(10)]
for i, cur_img_idx in enumerate(noise_free_idxs):
    cur_img = images[cur_img_idx,0,:,:]
    for j, noise_std in enumerate([0, 8, 32, 64, 128, 256, 512]):
        noise_X = add_gaussian_noise(cur_img, 0, noise_std/255)
        imshow(noise_X, axes[i][j])
plt.axis('off')
fig.savefig('out/exp2_imgs.pdf')
