"""
Description: spectrum loss for GAN
"""
import torch
import pdb

def calc_fft(image):
    '''image is tensor, N*C*H*W'''
    # fft = torch.rfft(image, 2, onesided=False)
    fft = torch.fft.fft2(image, dim=(-2, -1))
    fft = torch.stack((fft.real, fft.imag), -1)
    fft_mag = torch.log(1 + torch.sqrt(fft[..., 0] ** 2 + fft[..., 1] ** 2 + 1e-8))
    return fft_mag


def fft_L1_loss(fake_image, real_image):
    criterion_L1 = torch.nn.L1Loss()
     
    fake_image_gray = fake_image[:,0]*0.299 + fake_image[:,1]*0.587 + fake_image[:,2]*0.114
    real_image_gray = real_image[:,0]*0.299 + real_image[:,1]*0.587 + real_image[:,2]*0.114

    fake_fft = calc_fft(fake_image_gray)
    real_fft = calc_fft(real_image_gray)
    loss = criterion_L1(fake_fft, real_fft)
    return loss


def fft_L1_loss_mask(fake_image, real_image, mask):
    criterion_L1 = torch.nn.L1Loss()

    fake_image_gray = fake_image[:, 0] * 0.299 + fake_image[:, 1] * 0.587 + fake_image[:, 2] * 0.114
    real_image_gray = real_image[:, 0] * 0.299 + real_image[:, 1] * 0.587 + real_image[:, 2] * 0.114

    fake_fft = calc_fft(fake_image_gray)
    real_fft = calc_fft(real_image_gray)
    loss = criterion_L1(fake_fft * mask, real_fft * mask)
    return loss


def fft_L1_loss_color(fake_image, real_image):
    criterion_L1 = torch.nn.L1Loss()

    fake_fft = calc_fft(fake_image)
    real_fft = calc_fft(real_image)
    loss = criterion_L1(fake_fft, real_fft)
    return loss



def decide_circle(N=4,  L=256,r=96, size = 256):
    x=torch.ones((N, L, L))
    for i in range(L):
        for j in range(L):
            if (i- L/2 + 0.5)**2 + (j- L/2 + 0.5)**2 < r **2:
                x[:,i,j]=0
    return x, torch.ones((N, L, L)) - x