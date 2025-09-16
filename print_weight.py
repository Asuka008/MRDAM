import torch
checkpoint = torch.load("./results/SRGAN_x4-SRGAN_ImageNet/d_last.pth.tar")
print(checkpoint.keys())