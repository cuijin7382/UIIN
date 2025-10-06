from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
from DISTS_pytorch import DISTS
import time
from einops import rearrange
# from HVI_transformer import RGB_HVI
import random
from measure import metrics
import matplotlib.pyplot as plt
import cv2
from loss import LossFunction
import os
from bsnDBSNl import DBSNl
from LLCaps import LLCaps,CWA
import util2
from timm.models.layers import trunc_normal_
from blocks import CBlock_ln, SwinTransformerBlock
from global_net import Global_pred
from PIL import Image
import torch
from logger import Logger
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
# from pytorch_msssim import ssim, ms_ssim
from complexPyTorch.complexLayers import ComplexConv2d, complex_relu
from tensorboardX import SummaryWriter
import math
from scipy.stats import wasserstein_distance
import torchvision.transforms as transforms


#
# from skimage.metrics import peak_signal_noise_ratio, structural_similarity
# import lpips
# from DISTS_pytorch import DISTS
# import time
# import random
# import matplotlib.pyplot as plt
# import cv2
# from IQA_pytorch import SSIM
# from .loss import LossFunction
# import os
# from einops import rearrange
#
# from .bsnDBSNl import DBSNl
# # from LLCaps import LLCaps,CWA
# from . import util2
# from timm.models.layers import trunc_normal_
# from .blocks import CBlock_ln, SwinTransformerBlock
# from .global_net import Global_pred
# from PIL import Image
# import torch
# from .logger import Logger
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.autograd import Variable
# import numpy as np
# # from pytorch_msssim import ssim, ms_ssim
# from complexPyTorch.complexLayers import ComplexConv2d, complex_relu
# from tensorboardX import SummaryWriter
# import math
# from scipy.stats import wasserstein_distance
# import torchvision.transforms as transforms
def np2tensor(n:np.array):
    '''
    transform numpy array (image) to torch Tensor
    BGR -> RGB
    (h,w,c) -> (c,h,w)
    '''
    # gray
    if len(n.shape) == 2:
        return torch.from_numpy(np.ascontiguousarray(np.transpose(n, (2,0,1))))
    # RGB -> BGR
    elif len(n.shape) == 3:
        return torch.from_numpy(np.ascontiguousarray(np.transpose(np.flip(n, axis=2), (2,0,1))))
    else:
        raise RuntimeError('wrong numpy dimensions : %s'%(n.shape,))
def lpips2(img1, img2):
    '''
    image value range : [0 - 255]
    clipping for model output
    '''
    # print(img1.is_cuda, img2.is_cuda)

    device = torch.device("cuda:0")

    lpips_model = lpips.LPIPS(net="alex").to(device)#alex
    # numpt to tensor
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])


    # img1 = img1.transpose(1, 2, 0)
    # print(img1.shape) (600, 3, 400)
    img1 = np2tensor(img1).to(device)
    img2 = np2tensor(img2).to(device)
    # print(img1.shape) torch.Size([400, 600, 3])

    # img1 = Image.fromarray(np.uint8(img1))
    # img2 = Image.fromarray(np.uint8(img2))
    # img1 = preprocess(img1).unsqueeze(0).to(device)
    # img2 = preprocess(img2).unsqueeze(0).to(device)

    distance = lpips_model(img1, img2)
    return distance.item()

def dists2(img1, img2):
    '''
    image value range : [0 - 255]
    clipping for model output
    '''
    # print(img1.is_cuda, img2.is_cuda)

    device = torch.device("cuda:0")

    D = DISTS().to(device)
    # numpt to tensor
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    img1 = img1.transpose(1, 2, 0)

    img1 = Image.fromarray(np.uint8(img1))
    img2 = Image.fromarray(np.uint8(img2))

    img1 = preprocess(img1).unsqueeze(0).to(device)
    img2 = preprocess(img2).unsqueeze(0).to(device)


    dists_value = D(img1, img2)

    return dists_value.item()


def ssim2(img1, img2):
    '''
    image value range : [0 - 255]
    clipping for model output
    '''
    # if len(img1.shape) == 4:
    #     img1 = img1[0]
    # if len(img2.shape) == 4:
    #     img2 = img2[0]
    #
    # # tensor to numpy
    # if isinstance(img1, torch.Tensor):
    #     img1 = tensor2np(img1)
    # if isinstance(img2, torch.Tensor):
    #     img2 = tensor2np(img2)

    # numpy value cliping
    img2 = np.clip(img2, 0, 255)
    img1 = np.clip(img1, 0, 255)

    return structural_similarity(img1, img2, multichannel=True, data_range=255)

def psnr2(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
#==pie-enhance======

class enhance_net_nopool(nn.Module):

    def __init__(self):
        super(enhance_net_nopool, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        number_f = 32
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, 24, 3, 1, 1, bias=True)

        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        # p1 = self.maxpool(x1)
        x2 = self.relu(self.e_conv2(x1))
        # p2 = self.maxpool(x2)
        x3 = self.relu(self.e_conv3(x2))
        # p3 = self.maxpool(x3)
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        # x5 = self.upsample(x5)
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))

        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)

        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)
        # r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)
        return enhance_image
#==pie-enhance======

def frequency_loss(im1, im2):
    im1_fft = torch.fft.fftn(im1)
    im1_fft_real = im1_fft.real
    im1_fft_imag = im1_fft.imag
    im2_fft = torch.fft.fftn(im2)
    im2_fft_real = im2_fft.real
    im2_fft_imag = im2_fft.imag
    loss = 0
    for i in range(im1.shape[0]):
        real_loss = wasserstein_distance(im1_fft_real[i].reshape(im1_fft_real[i].shape[0]*im1_fft_real[i].shape[1]*im1_fft_real[i].shape[2]).cpu().detach(),
                                         im2_fft_real[i].reshape(im2_fft_real[i].shape[0]*im2_fft_real[i].shape[1]*im2_fft_real[i].shape[2]).cpu().detach())
        imag_loss = wasserstein_distance(im1_fft_imag[i].reshape(im1_fft_imag[i].shape[0]*im1_fft_imag[i].shape[1]*im1_fft_imag[i].shape[2]).cpu().detach(),
                                         im2_fft_imag[i].reshape(im2_fft_imag[i].shape[0]*im2_fft_imag[i].shape[1]*im2_fft_imag[i].shape[2]).cpu().detach())
        total_loss = real_loss + imag_loss
        loss += total_loss
    return torch.tensor(loss / (im1.shape[2] * im2.shape[3]))

#
# class Vgg16(nn.Module):
#     def __init__(self):
#         super(Vgg16, self).__init__()
#         self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
#         self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#
#         self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
#
#         self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
#         self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#
#         self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
#         self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#         self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#
#         self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#         self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#         self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#
#     def forward(self, X):
#         h0 = F.relu(self.conv1_1(X), inplace=True)
#         h1 = F.relu(self.conv1_2(h0), inplace=True)
#         h2 = F.max_pool2d(h1, kernel_size=2, stride=2)
#
#         h3 = F.relu(self.conv2_1(h2), inplace=True)
#         h4 = F.relu(self.conv2_2(h3), inplace=True)
#         h5 = F.max_pool2d(h4, kernel_size=2, stride=2)
#
#         h6 = F.relu(self.conv3_1(h5), inplace=True)
#         h7 = F.relu(self.conv3_2(h6), inplace=True)
#         h8 = F.relu(self.conv3_3(h7), inplace=True)
#         h9 = F.max_pool2d(h8, kernel_size=2, stride=2)
#         h10 = F.relu(self.conv4_1(h9), inplace=True)
#         h11 = F.relu(self.conv4_2(h10), inplace=True)
#         conv4_3 = self.conv4_3(h11)
#         result = F.relu(conv4_3, inplace=True)
#
#         return result
# #
#
#
# def load_vgg16(model_dir):
#     """ Use the model from https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/utils.py """
#     if not os.path.exists(model_dir):
#         os.mkdir(model_dir)
#     vgg = Vgg16()
#     vgg.cuda()
#     vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.weight')))
#
#     return vgg
#
#
# def compute_vgg_loss(enhanced_result, input_high):
#     instance_norm = nn.InstanceNorm2d(512, affine=False)
#     vgg = load_vgg16("./model")
#     vgg.eval()
#     for param in vgg.parameters():
#         param.requires_grad = False
#     img_fea = vgg(enhanced_result)
#     target_fea = vgg(input_high)
#
#     loss = torch.mean((instance_norm(img_fea) - instance_norm(target_fea)) ** 2)
#
#     return loss
# class CalibrateNetwork(nn.Module):
#     def __init__(self, layers, channels):
#         super(CalibrateNetwork, self).__init__()
#         kernel_size = 3
#         dilation = 1
#         padding = int((kernel_size - 1) / 2) * dilation
#         self.layers = layers
#
#         self.in_conv = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
#             nn.BatchNorm2d(channels),
#             nn.ReLU()
#         )
#
#         self.convs = nn.Sequential(
#             nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
#             nn.BatchNorm2d(channels),
#             nn.ReLU()
#         )
#         self.blocks = nn.ModuleList()
#         for i in range(layers):
#             self.blocks.append(self.convs)
#
#         self.out_conv = nn.Sequential(
#             nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, input):
#         fea = self.in_conv(input)
#         for conv in self.blocks:
#             fea = fea + conv(fea)
#
#         fea = self.out_conv(fea)
#         delta = input - fea
#
#         return delta
# #矫正 改进↓
class CalibrateNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(CalibrateNetwork, self).__init__()
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        self.layers = layers
        self.atten= IGAB(
                    dim=channels, num_blocks=2, dim_head=channels, heads=1)
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        ) #最初处理 残差

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        ) #循环 加残差
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.convs)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels*2, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        # self.cou= nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1, stride=1, padding=1)

    def forward(self, input,illu_fea):
        fea = self.in_conv(input)
        attfea=self.atten(fea,illu_fea)

        for conv in self.blocks:
            fea=conv(fea)
            fea = fea + conv(fea)
        catfea=torch.cat([fea,attfea],dim=1)
        fea = self.out_conv(catfea)
        delta = input - fea
        # d2=input+attfea#这 是干啥 矫正的？
        # delta=fea
        return delta
# class ResidualModule0(nn.Module):
#     def __init__(self, channel=64, kernel_size=3):
#         super(ResidualModule0, self).__init__()
#         self.Relu = nn.LeakyReLU()
#
#         self.conv0 = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
#         self.conv1 = nn.Conv2d(channel, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv2 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv3 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv4 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
#
#         self.conv = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         residual = x
#         out0 = self.Relu(self.conv0(x))
#         out1 = self.Relu(self.conv1(out0))
#         out2 = self.Relu(self.conv2(out1))
#         out3 = self.Relu(self.conv3(out2))
#         out4 = self.Relu(self.conv4(out3))
#         out = self.Relu(self.conv(residual))
#
#         final_out = torch.cat((out, out4), dim=1)
#
#         return final_out
#
#
# class ResidualModule1(nn.Module):
#     def __init__(self, channel=64, kernel_size=3):
#         super(ResidualModule1, self).__init__()
#         self.Relu = nn.LeakyReLU()
#
#         self.conv0 = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
#         self.conv1 = nn.Conv2d(channel, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv2 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv3 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv4 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
#
#         self.conv = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         residual = x
#         out0 = self.Relu(self.conv0(x))
#         out1 = self.Relu(self.conv1(out0))
#         out2 = self.Relu(self.conv2(out1))
#         out3 = self.Relu(self.conv3(out2))
#         out4 = self.Relu(self.conv4(out3))
#         out = self.Relu(self.conv(residual))
#
#         final_out = torch.cat((out, out4), dim=1)
#
#         return final_out
#
#
# class ResidualModule2(nn.Module):
#     def __init__(self, channel=64, kernel_size=3):
#         super(ResidualModule2, self).__init__()
#         self.Relu = nn.LeakyReLU()
#
#         self.conv0 = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
#         self.conv1 = nn.Conv2d(channel, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv2 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv3 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv4 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
#
#         self.conv = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         residual = x
#         out0 = self.Relu(self.conv0(x))
#         out1 = self.Relu(self.conv1(out0))
#         out2 = self.Relu(self.conv2(out1))
#         out3 = self.Relu(self.conv3(out2))
#         out4 = self.Relu(self.conv4(out3))
#         out = self.Relu(self.conv(residual))
#
#         final_out = torch.cat((out, out4), dim=1)
#
#         return final_out
#
#
# class ResidualModule3(nn.Module):
#     def __init__(self, channel=64, kernel_size=3):
#         super(ResidualModule3, self).__init__()
#         self.Relu = nn.LeakyReLU()
#
#         self.conv0 = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
#         self.conv1 = nn.Conv2d(channel, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv2 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv3 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv4 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
#
#         self.conv = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         residual = x
#         out0 = self.Relu(self.conv0(x))
#         out1 = self.Relu(self.conv1(out0))
#         out2 = self.Relu(self.conv2(out1))
#         out3 = self.Relu(self.conv3(out2))
#         out4 = self.Relu(self.conv4(out3))
#         out = self.Relu(self.conv(residual))
#
#         final_out = torch.cat((out, out4), dim=1)
#
#         return final_out
#
#
# class Resblock(nn.Module):
#     def __init__(self, channels, kernel_size=3, stride=1):
#         super(Resblock, self).__init__()
#
#         self.kernel_size = (kernel_size, kernel_size)
#         self.stride = (stride, stride)
#         self.activation = nn.LeakyReLU(True)
#
#         sequence = list()
#
#         sequence += [
#             nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='replicate'),
#             nn.LeakyReLU(),
#             nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='replicate'),
#         ]
#         self.model = nn.Sequential(*sequence)
#
#     def forward(self, x):
#
#         residual = x
#         output = self.activation(self.model(x) + residual)
#
#         return output
#
#
# class complex_net(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(complex_net, self).__init__()
#
#         self.complex_conv0 = ComplexConv2d(in_channels, out_channels*2, kernel_size=3, stride=1, padding=1)
#         self.complex_conv1 = ComplexConv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1)
#         self.complex_conv2 = ComplexConv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1)
#
#     def forward(self, x):
#
#         residual = x
#         out0 = complex_relu(self.complex_conv0(x))
#         out1 = complex_relu(self.complex_conv1(out0))
#         out2 = complex_relu(self.complex_conv2(out1))
#         output = residual + out2
#
#         return output
#
#
# class feature_block(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(feature_block, self).__init__()
#
#         self.resblock0 = Resblock(in_channels)
#         self.complex_block = complex_net(out_channels, out_channels)
#         self.resblock1 = Resblock(out_channels)
#
#     def forward(self, x):
#
#         residual = x
#         out0 = self.resblock0(x)
#         fft_out0 = torch.fft.rfftn(out0)
#         out1 = self.complex_block(fft_out0)
#         ifft_out1 = torch.fft.irfftn(out1)
#         out2 = self.resblock1(ifft_out1)
#
#         output = residual + out2
#
#         return output
#
#
# class fft_processing(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(fft_processing, self).__init__()
#         self.complex_conv1 = ComplexConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
#
#         self.complex_block0 = complex_net(out_channels, out_channels)
#         self.complex_block1 = complex_net(out_channels, out_channels)
#
#         self.complex_conv2 = ComplexConv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#
#         conv1_out = complex_relu(self.complex_conv1(x))
#         complex_block_out0 = self.complex_block0(conv1_out)
#         complex_block_out1 = self.complex_block1(complex_block_out0)
#         conv2_out = complex_relu(self.complex_conv2(complex_block_out1))
#
#         return conv2_out


# class Local_pred(nn.Module):
#     def __init__(self, dim=16, number=4, type='ccc'):
#         super(Local_pred, self).__init__()
#         # initial convolution
#         self.conv1 = nn.Conv2d(3, dim, 3, padding=1, groups=1)
#         self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#         # main blocks
#         block = CBlock_ln(dim)
#         block_t = SwinTransformerBlock(dim)  # head number
#         if type == 'ccc':
#             # blocks1, blocks2 = [block for _ in range(number)], [block for _ in range(number)]
#             blocks1 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
#             blocks2 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
#         elif type == 'ttt':
#             blocks1, blocks2 = [block_t for _ in range(number)], [block_t for _ in range(number)]
#         elif type == 'cct':
#             blocks1, blocks2 = [block, block, block_t], [block, block, block_t]
#         #    block1 = [CBlock_ln(16), nn.Conv2d(16,24,3,1,1)]
#         self.mul_blocks = nn.Sequential(*blocks1, nn.Conv2d(dim, 3, 3, 1, 1), nn.ReLU())
#         self.add_blocks = nn.Sequential(*blocks2, nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())
#
#     def forward(self, img):
#         img1 = self.relu(self.conv1(img))
#         mul = self.mul_blocks(img1)
#         add = self.add_blocks(img1)
#
#         return mul, add


# Short Cut Connection on Final Layer
class Local_pred_S(nn.Module):
    def __init__(self, in_dim=3, dim=16, number=4, type='ccc'):
        super(Local_pred_S, self).__init__()
        # initial convolution
        self.conv1 = nn.Conv2d(in_dim, dim, 3, stride=1,padding=1, groups=1)

        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # main blocks
        block = CBlock_ln(dim)
        block_t = SwinTransformerBlock(dim)  # head number
        if type == 'ccc':
            blocks1 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
            blocks2 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
        elif type == 'ttt':
            blocks1, blocks2 = [block_t for _ in range(number)], [block_t for _ in range(number)]
        elif type == 'cct':
            blocks1, blocks2 = [block, block, block_t], [block, block, block_t]
        #    block1 = [CBlock_ln(16), nn.Conv2d(16,24,3,1,1)]
        self.mul_blocks = nn.Sequential(*blocks1)
        self.add_blocks = nn.Sequential(*blocks2)

        self.mul_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.ReLU())
        self.add_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, img):
        # print(img)#[[0.0863, 0.1176, 0.1216,  ..., 0.1137, 0.1176, 0.1098],
        # img1 = self.relu(self.conv1(img))
        # e-01   ====10 ^ -01
        img11=self.conv1(img)
        # print(img11) #[ 9.9695e-03, -2.3828e-03, -6.4886e-03,  ..., -3.4269e-03,
        img1 = self.relu(img11)
        # print(img1) #[ 9.9695e-03, -4.7655e-04, -1.2977e-03,  ..., -6.8539e-04,
        # short cut connection
        mul = self.mul_blocks(img1) + img1
        # print(mul) # 5.4162e-02,  3.9797e-02,  3.8274e-02,  ...,  3.9620e-02,
        add = self.add_blocks(img1) + img1

        # print(add) #0.0374,  0.0076,  0.0074,  ...,  0.0090,  0.0096,  0.0085],
        mul = self.mul_end(mul)
        # print(mul) #[0., 0., 0.,  ..., 0., 0., 0.],
        add = self.add_end(add)
        # print(add) #[0.9997, 1.0000, 1.0000,  ..., 1.0000, 1.0000, 0.9996]
        return mul, add

#
# class EnhanceNetwork(nn.Module):
#     def __init__(self, layers, channels):
#         super(EnhanceNetwork, self).__init__()
#
#         kernel_size = 3
#         dilation = 1
#         padding = int((kernel_size - 1) / 2) * dilation
#
#         self.in_conv = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
#             nn.ReLU()
#         )
#
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
#             nn.BatchNorm2d(channels),
#             nn.ReLU()
#         )
#
#         self.blocks = nn.ModuleList()
#         for i in range(layers):
#             self.blocks.append(self.conv)
#
#         self.out_conv = nn.Sequential(
#             nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, input):
#         fea = self.in_conv(input)
#         for conv in self.blocks:
#             fea = fea + conv(fea)
#         fea = self.out_conv(fea)
#
#         illu = fea + input
#         illu = torch.clamp(illu, 0.0001, 1)
#
#         return illu


class Illumination_Estimator(nn.Module):  # 视网膜电流发生器采用了所提出的由照明估计器组成的ORF
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):  # __init__部分是内部属性，而forward的输入才是外部输入
        super(Illumination_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)
        self.depth_conv3 = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=3, padding=1, bias=True, groups=n_fea_in)
        self.conv3 = nn.Conv2d(n_fea_middle*2, n_fea_middle, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w

        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        mean_c = img.mean(dim=1).unsqueeze(1)  # 平均通道=照明先验
        # stx()
        input = torch.cat([img, mean_c], dim=1)
        # input=img
        x_1 = self.conv1(input)
        illu_fea1 = self.depth_conv(x_1)
        illu_fea2= self.depth_conv3(x_1)
        illu_fea3=torch.cat([illu_fea1,illu_fea2],dim=1)
        illu_fea = self.conv3(illu_fea3)# 特征
        illu_map = self.conv2(illu_fea)  # 亮度映射直接乘原图能得到亮图


        return illu_fea, illu_map
#
# class Illumination_Estimator(nn.Module):  # 视网膜电流发生器采用了所提出的由照明估计器组成的ORF
#     def __init__(
#             self, n_fea_middle, n_fea_in=4, n_fea_out=3):  # __init__部分是内部属性，而forward的输入才是外部输入
#         super(Illumination_Estimator, self).__init__()
#
#         self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
#
#         self.depth_conv = nn.Conv2d(
#             n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)
#
#         self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)
#
#     def forward(self, img):
#         # img:        b,c=3,h,w
#         # mean_c:     b,c=1,h,w
#
#         # illu_fea:   b,c,h,w
#         # illu_map:   b,c=3,h,w
#
#         mean_c = img.mean(dim=1).unsqueeze(1)  # 平均通道=照明先验
#         # stx()
#         input = torch.cat([img, mean_c], dim=1)
#         # input=img
#         x_1 = self.conv1(input)
#         illu_fea = self.depth_conv(x_1)  # 特征
#         illu_map = self.conv2(illu_fea)  # 亮度映射直接乘原图能得到亮图
#
#
#         return illu_fea, illu_map


class LightNet(nn.Module): #照度
    def __init__(self, nf): #nf=32
        super(LightNet, self).__init__()
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.in2 = nn.InstanceNorm2d(nf, affine=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(nf, nf * 2, 3, 2, 1, bias=True)
        self.in3 = nn.InstanceNorm2d(nf * 2, affine=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(nf * 2, nf * 2, 3, 1, 1, bias=True)
        self.in4 = nn.InstanceNorm2d(nf * 2, affine=True)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(nf * 2, nf * 4, 3, 2, 1, bias=True)
        self.in5 = nn.InstanceNorm2d(nf * 4, affine=True)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(nf * 4, nf * 4, 3, 1, 1, bias=True)
        self.in6 = nn.InstanceNorm2d(nf * 4, affine=True)
        self.relu6 = nn.ReLU(inplace=True)

        self.conv7 = nn.Conv2d(nf * 4, nf * 2, 1, 1, 0, bias=True)
        self.in7 = nn.InstanceNorm2d(nf * 2, affine=True)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv2d(nf * 2, nf * 2, 3, 1, 1, bias=True)
        self.in8 = nn.InstanceNorm2d(nf * 2, affine=True)
        self.relu8 = nn.ReLU(inplace=True)

        self.conv9 = nn.Conv2d(nf * 2, nf, 1, 1, 0, bias=True)
        self.in9 = nn.InstanceNorm2d(nf, affine=True)
        self.relu9 = nn.ReLU(inplace=True)
        self.conv10 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.in10 = nn.InstanceNorm2d(nf, affine=True)
        self.relu10 = nn.ReLU(inplace=True)
    def forward(self, x):
        out2 = self.relu2(self.in2(self.conv2(x)))

        out3 = self.relu3(self.in3(self.conv3(out2)))
        out4 = self.relu4(self.in4(self.conv4(out3)))

        out5 = self.relu5(self.in5(self.conv5(out4)))
        out6 = self.relu6(self.in6(self.conv6(out5)))

        up1 = F.interpolate(out6, size=[out4.size()[2], out4.size()[3]], mode='bilinear')
        out7 = self.relu7(self.in7(self.conv7(up1)))
        out8 = self.relu8(self.in8(self.conv8(out7 + out4)))

        up2 = F.interpolate(out8, size=[out2.size()[2], out2.size()[3]], mode='bilinear')
        out9 = self.relu9(self.in9(self.conv9(up2)))
        out10 = self.relu10(self.in10(self.conv10(out9 + out2)))

        return out10
def transform_invert(img_):
    img_ = img_.squeeze(0).transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    img_ = img_.detach().cpu().numpy() * 255.0

    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_.shape[2]))

    return img_

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class IG_MSA(nn.Module): #MSA使用ORF捕获的照明表示来指导自注意的计算。照度引导注意力
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, illu_fea_trans): #两个接收的
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x) #torch.Size([1, 240000, 36]
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        illu_attn = illu_fea_trans # illu_fea: b,c,h,w -> b,h,w,c
        q, k, v, illu_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2)))
        # print(q.shape) #torch.Size([1, 1, 240000, 36])

        v = v * illu_attn #cheng 16 36
        # print(v.shape)#torch.Size([1, 1, 240000, 36])
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        # print(q.shape) #torch.Size([1, 1, 36, 240000])
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)



class IGAB(nn.Module): #IGT的基本单元是IGAB，它由两层归一化（LN）、一个IG-MSA和一个前馈网络（FFN）组成。(
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IG_MSA(dim=dim, dim_head=dim_head, heads=heads),  # 注意力
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, illu_fea):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1) #将张量x的维度顺序进行调整,并将结果存储在一个新的张量 统一到一个维度  归一化
        for (attn, ff) in self.blocks:
            x = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x #处理注意力
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2) #转回之前维度 二个维度 不懂
        return out
#
# class Illumination_Estimator(nn.Module):
#     def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
#         super(Illumination_Estimator, self).__init__()
#
#         self.initial_conv = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
#
#         # 多尺度分支
#         self.branch3x3 = nn.Conv2d(n_fea_middle, n_fea_middle, kernel_size=3, padding=1, bias=True, groups=n_fea_in)
#         self.branch5x5 = nn.Conv2d(n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)
#         self.branch_dilated = nn.Conv2d(n_fea_middle, n_fea_middle, kernel_size=3, padding=3, dilation=3, bias=True, groups=n_fea_in)
#         self.branch_pool = nn.Sequential(
#             nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
#             nn.Conv2d(n_fea_middle, n_fea_middle, kernel_size=1, bias=True)
#         )
#
#         self.fuse = nn.Conv2d(n_fea_middle * 4, n_fea_middle, kernel_size=1, bias=True)
#         self.out_conv = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)
#
#     def forward(self, img):
#         # img: b,3,h,w
#         mean_c = img.mean(dim=1, keepdim=True)
#         x = torch.cat([img, mean_c], dim=1)  # b,4,h,w
#
#         x = self.initial_conv(x)
#
#         # 多尺度特征提取
#         f1 = self.branch3x3(x)
#         f2 = self.branch5x5(x)
#         f3 = self.branch_dilated(x)
#         f4 = self.branch_pool(x)
#
#         multi_scale_fea = torch.cat([f1, f2, f3, f4], dim=1)
#         illu_fea = self.fuse(multi_scale_fea)
#         illu_map = self.out_conv(illu_fea)
#
#         return illu_fea, illu_map

class IAT(nn.Module):
    def __init__(self, in_dim=3, with_global=True, type='lol'):
        super(IAT, self).__init__()
        # self.local_net = Local_pred()
        # if self.training:
        self.stage=5
        self.n_feat = 36
        # else:
        #     self.stage=1
        self.local_net = Local_pred_S(in_dim=in_dim)
        self.estimator=Illumination_Estimator(self.n_feat)
        self.calibrate=CalibrateNetwork(layers=3, channels=36)
        self.enhance=enhance_net_nopool()
        self.enhance =CWA(n_feat=64)
        # self.enhance=LLCaps()
        self.with_global = with_global
        if self.with_global:
            self.global_net = Global_pred(in_channels=in_dim, type=type)
        self.conv3 = nn.Conv2d(4, 3, kernel_size=1)
        self._criterion=LossFunction()

        nf=32

        self.Light_conv_in = nn.Conv2d(1, nf, kernel_size=3, stride=1, padding=1, bias=False)
        self.Light_relu_in = nn.ReLU()
        self.Light_net = LightNet(nf)
        self.Light_conv_out = nn.Conv2d(nf, 1,  kernel_size=3, stride=1, padding=1, bias=False)
        self.Light_sigmoid = nn.Sigmoid()
        #
        # self.trans =   RGB_HVI().cuda()
        # self.HVE_block0 = nn.Sequential(  # map
        #     nn.ReplicationPad2d(1),
        #     nn.Conv2d(3, nf, 3, stride=1, padding=0, bias=False)
        # )

    def apply_color(self, image, ccm):
        shape = image.shape
        image = image.view(-1, 3)
        image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
        image = image.view(shape)
        return torch.clamp(image, 1e-8, 1.0)

    def forward(self, input):
        # ilist,inlist = [], []

        ilist, rlist, inlist, attlist,rimglist = [], [], [], [],[]
        input_op = input
        # print(input_op.shape) #train[4, 3, 100, 100])  predict[1, 3, 400, 600]
        # print(input_op)
        # mul, add = self.local_net(img_low)
        # img_high = (img_low.mul(mul)).add(add)

        # if not self.with_global:
        #     return mul, add, img_high
        # if self.training:
       # ====================循环
        for i in range(self.stage):
            inlist.append(input_op)


            # img_highx = input * illu_map+input


            # dtypes =input.dtype
            # hvi = self.trans.HVIT(input)  ##rgb转hvi
            # i = hvi[:, 2, :, :].unsqueeze(1).to(dtypes)
            # # low
            # i_enc0 = self.IE_block0(i)  # map#itensity map？
            # i_enc1 = self.IE_block1(i_enc0)  # 下采样
            # hv_0 = self.HVE_block0(hvi)  # hv color map
            #
            #
            l1 = torch.clamp(torch.max(input_op, dim=1)[0].unsqueeze(1), min=0.0, max=1.0) #v空间
#换成rgb空间，直接用平均值?
            #v空间是最大值？
            # l1 = input_op.mean(dim=1).unsqueeze(1)


            l2 = self.Light_relu_in(self.Light_conv_in(l1))
            l3 = self.Light_net(l2)  #fea
            l4 = self.Light_conv_out(l3) #向上补充
            i = self.Light_sigmoid(l4 + l1)
            # i=torch.cat([i,illu_map],dim=1)
            # mul, add = self.local_net(input_op)
            # # print(mul )
            # img_high0 = (input_op.mul(mul)).add(add)
            #
            # img_highx = input_op * illu_map
            # gamma, color = self.global_net(input_op)
            #
            # b = img_high0.shape[0]
            # img_high1 = img_high0.permute(0, 2, 3, 1)  # (B,C,H,W) -- (B,H,W,C)
            # img_high = torch.stack(
            #     [self.apply_color(img_high1[i, :, :, :], color[i, :, :]) ** gamma[i, :] for i in range(b)], dim=0)
            # img_high = img_high.permute(0, 3, 1, 2)  # (B,H,W,C) -- (B,C,H,W)
            # print(img_high.shape)
 #xianshi
            #
            # img_higho = transform_invert(img_highx)
            # # img_higho = transform_invert(torch.mean(input, dim=1, keepdim=True))
            # # im = Image.fromarray(img_high)
            # img_higho.save("img_higho.png")
            # # xianshi

            # illu_fea2, illu_map2 = self.estimator(img_high)
            # illu_fea1, illu_map1 = self.estimator(img_high0)
            # i = illu_map2
            # i1=input/illu_map1
            # i1 = torch.clamp(i1, 0, 1)
            # i1= self.calibrate(i1)
            # i= illu_map
            # i=输出
            # r=i
            r1=input/i
            # r2=torch.cat([r1,i],dim=1)
            # r2=i
            # r2=self.conv3(r2)
            # r2=input*r2
            # r1=r1*illu_map+input
            # gamma, color = self.global_net(input_op)
            # b = r1.shape[0]
            # img_high1 = r1.permute(0, 2, 3, 1)  # (B,C,H,W) -- (B,H,W,C)
            # img_high = torch.stack(
            #     [self.apply_color(img_high1[i, :, :, :], color[i, :, :]) ** gamma[i, :] for i in range(b)], dim=0)
            # img_high = img_high.permute(0, 3, 1, 2)  # (B,H,W,C) -- (B,C,H,W)

            # input_op=
            # r1=r1+i   ######
            # r = r- input#是一个很亮的图
            # r=img_high1
            # r = torch.cat([r, img_high], dim=1)
            # r = self.conv3(r)
            # r =img_high # 得到r y/xt      input/high
            r = torch.clamp(r1, 0, 1)
            # i2=illu_map
            # r2=input/illu_map
            illu_fea, illu_map = self.estimator(input)
            rimg=illu_map*input+input
            # att = self.calibrate(r)
            att = self.calibrate(r,illu_fea)
            # r=input_op# st   delta = input - fea   return delta
            input_op = input+att


         # print(illu_fea.shape)
            # illu_fea torch.Size([1, 36, 400, 600])
           # #保存
           #  img_higho = transform_invert(torch.mean(l1, dim=1, keepdim=True))
           #  img_higho = transform_invert(l1)
           #  img_higho.save("img_higho.png")
           #  # 保存
            # att=illu_map
            # v=y+st
            # img_higho = transform_invert(att)
            # img_higho = transform_invert(torch.mean(input, dim=1, keepdim=True))
            # im = Image.fromarray(img_high)
            # img_higho.save("img_higho.png")
            # input_op=img_high1
            # r=r1 +i
            ilist.append(i)  # st
            rlist.append(r)
            rimglist.append(rimg)
            # attlist.append(att)
            attlist.append(torch.abs(att))

        return ilist, rlist, inlist, attlist,rimglist,l4
            # img_high=self.enhance(input_op)

            # maskill2 = img_high.cpu().detach().numpy()
            # # print(maskill2.shape)
            # plt.subplot(221), plt.title("1. B channel"), plt.axis('off')
            # bImg = cv2.cvtColor(maskill2, cv2.COLOR_BGR2RGB)  # 图片格式转换：BGR(OpenCV) -> RGB(PyQt5)
            # plt.imshow(bImg)
            # plt.show()# matplotlib 显示 channel B
        # ==local==========
        #     mul, add = self.local_net(input_op)
            # print('mul',mul) #0
            # print('add',add)#0.9997, 1.0000, 1.0000,  ..., 1.0000, 1.0000, 0.9996],
            # img_high1 = (input_op.mul(mul)).add(add) #[[[0.9997, 1.0000, 1.0000,  ..., 1.0000, 1.0000, 0.9996],
        # img_high=input/img_high
        # img_high= img_high.mul(255).byte()
        # print(img_high)
        # ====
        # img_high=img_high.cpu().numpy()
        # # plt.subplot(221), plt.title("1. B channel"), plt.axis('off')
        # bImg = cv2.cvtColor(img_high, cv2.COLOR_BGR2RGB)  # 图片格式转换：BGR(OpenCV) -> RGB(PyQt5)
        # plt.imshow(bImg)
        # plt.show()# matplotlib 显示 channel B
        # ====###==global=====
        #     gamma, color = self.global_net(input_op)
        # # #     # print(gamma)
        #     b = img_high1.shape[0]
        #     img_high = img_high1.permute(0, 2, 3, 1)  # (B,C,H,W) -- (B,H,W,C)
        #     img_high = torch.stack(
        #         [self.apply_color(img_high[i, :, :, :], color[i, :, :]) ** gamma[i, :] for i in range(b)], dim=0)
        #     img_high = img_high.permute(0, 3, 1, 2)  # (B,H,W,C) -- (B,C,H,W)
        #     st=input/img_high

        # #     ========XIANSHI
        #     maskill2 = img_high.cpu().detach().numpy()
        #     # print(maskill2.shape)
        #     plt.subplot(221), plt.title("1. B channel"), plt.axis('off')
        #     bImg = cv2.cvtColor(maskill2, cv2.COLOR_BGR2RGB)  # 图片格式转换：BGR(OpenCV) -> RGB(PyQt5)
        #     plt.imshow(bImg)
        #     plt.show()  # matplotlib 显示 channel B
            #     ========XIANSHI
        # print('imghigh',img_high)#[0.1740, 0.1740, 0.1740,  ..., 0.1740, 0.1740, 0.1739],
       # ==========
       #  img_high=img_high.cpu().numpy()
       #  img_high = img_high[0, :, :, :]
       #  img_high = img_high.cpu().numpy().transpose(1, 2, 0)
       #  # img_high = img_high.cpu().detach().numpy().transpose(1, 2, 0)
       #  plt.imshow(img_high)
       #  plt.show()  # matplotlib 显示 channel B
       #  return img_high
        # ===============
        # img_high = torch.clamp(img_high, 0.0001, 1)
        # illu = img_high + input
        # illu = torch.clamp(illu, 0.0001, 1)
        # 增强输出
        # r = input_op
        # img_high=input_op
        # print(r.shape)
        # r= input_op
# ===============多输出
#             inlist.append(input_op)  # tu y+st
              # 增强输出 01
            # r = input / st  # 得到r y/xt      input/high
            # r = torch.clamp(img_high, 0, 1)
        #     r=img_high #改成imghigh
        #     att = self.calibrate(img_high)  # st   delta = input - fea   return delta
        #
        #     input_op=input+att
        #     # input_op = input + att  # v=y+st
        #     # input_op =i
        #     i= input /img_high
        #     # i=input/img_high
        #     ilist.append(i)  # st
        #     rlist.append(r)
        #     # attlist.append(torch.abs(att))
        #
        # # ===============多输出
        # # return ilist, rlist,inlist
        # return ilist, rlist, inlist,attlist

    # ilist是输出 rlist是矫正处理的r 保存r[0] inlist是下一个输入 att是st
    def sciloss(self, input):
        i_list, r_list, in_list,att_list,rimglist,illu_map = self(input)
        # i_list,en_list, in_list = self(input)
        lossirt = 0
        loss2=0
        loss3 = 0
        for i in range(self.stage):
            lossirt += self._criterion(in_list[i], i_list[i])
            loss2+=F.l1_loss(att_list[i],rimglist[i]+i_list[i])# 后面是st\
            loss3 += F.l1_loss(rimglist[i]+i_list[i],r_list[i]+i_list[i])
        return lossirt+2*loss2+2*loss3
        #

            # gamma, color = self.global_net(img_low)
            # b = img_high.shape[0]
            # img_high = img_high.permute(0, 2, 3, 1)  # (B,C,H,W) -- (B,H,W,C)
            # img_high = torch.stack(
            #     [self.apply_color(img_high[i, :, :, :], color[i, :, :]) ** gamma[i, :] for i in range(b)], dim=0)
            # img_high = img_high.permute(0, 3, 1, 2)  # (B,H,W,C) -- (B,C,H,W)
            #
            # # i = self.enhance(img_high)  # 增强输出
            # r = input / img_high  # 得到r y/xt
            # r = torch.clamp(r, 0, 1)
            # att = self.calibrate(r)  # st
            # input_op = input + att  # v=y+st

            # return  img_high,input_op


#
# class DecomNet(nn.Module):
#     def __init__(self, channel=64, kernel_size=3):
#         super(DecomNet, self).__init__()
#
#         self.activation = nn.LeakyReLU()
#
#         self.conv0 = nn.Conv2d(4, channel, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.RM0 = ResidualModule0()
#         self.RM1 = ResidualModule1()
#         self.RM2 = ResidualModule2()
#         self.RM3 = ResidualModule3()
#         self.conv1 = nn.Conv2d(channel * 2, channel, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv2 = nn.Conv2d(channel * 2, channel, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv3 = nn.Conv2d(channel * 2, channel, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv4 = nn.Conv2d(channel * 2, channel, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv5 = nn.Conv2d(channel, 4, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, input_im):
#         input_max = torch.max(input_im, dim=1, keepdim=True)[0]
#         input_img = torch.cat((input_max, input_im), dim=1)
#         # print('input_max',input_max.shape)
#         # print('input_img', input_img.shape)
#
#         out0 = self.activation(self.conv0(input_img))
#         out1 = self.RM0(out0)
#         out2 = self.activation(self.conv1(out1))
#         out3 = self.RM1(out2)
#         out4 = self.activation(self.conv2(out3))
#         out5 = self.RM2(out4)
#         out6 = self.activation(self.conv3(out5))
#         out7 = self.RM3(out6)
#         out8 = self.activation(self.conv4(out7))
#         out9 = self.activation(self.conv5(out8))
#
#         R = torch.sigmoid(out9[:, 0:3, :, :])
#         L = torch.sigmoid(out9[:, 3:4, :, :])
#
#         return R, L
#         #得到两个输出 反射和照明
#
#
# class DenoiseNet(nn.Module):
#     def __init__(self, channel=64, kernel_size=3):
#         super(DenoiseNet, self).__init__()
#         self.Relu = nn.LeakyReLU()
#         self.Denoise_conv0_1 = nn.Conv2d(4, channel, kernel_size, padding=2, padding_mode='replicate', dilation=2)
#         self.Denoise_conv0_2 = nn.Conv2d(channel, channel, kernel_size, padding=2, padding_mode='replicate',
#                                          dilation=2)  # 96*96
#         self.conv0 = nn.Conv2d(channel, channel * 2, kernel_size=1, stride=1, padding=0)
#         self.conv1 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv2 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv3 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv4 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
#         self.Denoise_subsampling0 = nn.Conv2d(channel*2, channel, kernel_size=2, stride=2, padding=0)  # 48*48
#         self.conv5 = nn.Conv2d(channel, channel * 2, kernel_size=1, stride=1, padding=0)
#         self.conv6 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv7 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv8 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv9 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
#         self.Denoise_subsampling1 = nn.Conv2d(channel*2, channel, kernel_size=2, stride=2, padding=0)  # 24*24
#         self.conv10 = nn.Conv2d(channel, channel * 2, kernel_size=1, stride=1, padding=0)
#         self.conv11 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv12 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv13 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv14 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
#         self.Denoise_subsampling2 = nn.Conv2d(channel*2, channel, kernel_size=2, stride=2, padding=0)  # 12*12
#         self.conv15 = nn.Conv2d(channel, channel * 2, kernel_size=1, stride=1, padding=0)
#         self.conv16 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv17 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv18 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv19 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
#         self.Denoise_deconv0 = nn.ConvTranspose2d(channel*2, channel, kernel_size=2, stride=2, padding=0,
#                                                   output_padding=0)  # 24*24
#         self.conv20 = nn.Conv2d(channel * 2, channel * 2, kernel_size=1, stride=1, padding=0)
#         self.conv21 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv22 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv23 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv24 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
#         self.Denoise_deconv1 = nn.ConvTranspose2d(channel*2, channel, kernel_size=2, stride=2, padding=0,
#                                                   output_padding=0)  # 48*48
#         self.conv25 = nn.Conv2d(channel * 2, channel * 2, kernel_size=1, stride=1, padding=0)
#         self.conv26 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv27 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv28 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv29 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
#         self.Denoise_deconv2 = nn.ConvTranspose2d(channel*2, channel, kernel_size=2, stride=2, padding=0,
#                                                   output_padding=0)  # 96*96
#
#         self.Denoiseout0 = nn.Conv2d(channel * 2, channel, kernel_size, padding=1, padding_mode='replicate')
#         self.Denoiseout1 = nn.Conv2d(channel, 3, kernel_size=1, stride=1)
#
#     def forward(self, input_L, input_R):
#         input_img = torch.cat((input_R, input_L), dim=1)
#         out0 = self.Relu(self.Denoise_conv0_1(input_img))
#         out1 = self.Relu(self.Denoise_conv0_2(out0))
#         out2 = self.Relu(self.conv0(out1))
#         out3 = self.Relu(self.conv1(out2))
#         out4 = self.Relu(self.conv2(out3))
#         out5 = self.Relu(self.conv3(out4))
#         out6 = self.Relu(self.conv4(out5))
#         down0 = self.Relu(self.Denoise_subsampling0(torch.cat((out1, out6), dim=1)))
#         out7 = self.Relu(self.conv5(down0))
#         out8 = self.Relu(self.conv6(out7))
#         out9 = self.Relu(self.conv7(out8))
#         out10 = self.Relu(self.conv8(out9))
#         out11 = self.Relu(self.conv9(out10))
#         down1 = self.Relu(self.Denoise_subsampling1(torch.cat((down0, out11), dim=1)))
#         out12 = self.Relu(self.conv10(down1))
#         out13 = self.Relu(self.conv11(out12))
#         out14 = self.Relu(self.conv12(out13))
#         out15 = self.Relu(self.conv13(out14))
#         out16 = self.Relu(self.conv14(out15))
#         down2 = self.Relu(self.Denoise_subsampling2(torch.cat((down1, out16), dim=1)))
#         out17 = self.Relu(self.conv15(down2))
#         out18 = self.Relu(self.conv16(out17))
#         out19 = self.Relu(self.conv17(out18))
#         out20 = self.Relu(self.conv18(out19))
#         out21 = self.Relu(self.conv19(out20))
#         up0 = self.Relu(self.Denoise_deconv0(torch.cat((down2, out21), dim=1)))
#         out22 = self.Relu(self.conv20(torch.cat((up0, out16), dim=1)))
#         out23 = self.Relu(self.conv21(out22))
#         out24 = self.Relu(self.conv22(out23))
#         out25 = self.Relu(self.conv23(out24))
#         out26 = self.Relu(self.conv24(out25))
#         up1 = self.Relu(self.Denoise_deconv1(torch.cat((up0, out26), dim=1)))
#         out27 = self.Relu(self.conv25(torch.cat((up1, out11), dim=1)))
#         out28 = self.Relu(self.conv26(out27))
#         out29 = self.Relu(self.conv27(out28))
#         out30 = self.Relu(self.conv28(out29))
#         out31 = self.Relu(self.conv29(out30))
#         up2 = self.Relu(self.Denoise_deconv2(torch.cat((up1, out31), dim=1)))
#         out32 = self.Relu(self.Denoiseout0(torch.cat((out6, up2), dim=1)))
#         out33 = self.Relu(self.Denoiseout1(out32))
#         denoise_R = out33
#
#         return denoise_R

#
# class EnhanceModule0(nn.Module):
#     def __init__(self, channel=64, kernel_size=3):
#         super(EnhanceModule0, self).__init__()
#         self.activation = nn.LeakyReLU()
#         self.conv0 = nn.Conv2d(4, channel, kernel_size=(3, 3), stride=(1, 1), padding=2,
#                                dilation=(2, 2))
#         self.conv1 = nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=(1, 1), padding=2,
#                                dilation=(2, 2))
#
#         self.feature_block0 = feature_block(channel, channel)
#
#         self.down_sampling = nn.Conv2d(channel, channel, kernel_size=(2, 2), stride=(2, 2), padding=0)
#
#         self.fft_processing = fft_processing(channel, channel)
#
#         self.up_sampling = nn.UpsamplingBilinear2d(scale_factor=2)
#
#         self.feature_block1 = feature_block(channel, channel)
#         self.channel_up = ComplexConv2d(4, channel, kernel_size=3, stride=1, padding=1)
#
#         self.conv2 = nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=(1, 1), padding=1)
#         self.conv3 = nn.Conv2d(channel, channel, kernel_size=(1, 1), stride=(1, 1), padding=0)
#
#     def forward(self, input_img):
#         input_img_down = F.interpolate(input_img, scale_factor=0.5, mode='bilinear')
#         input_img_down_fft = torch.fft.rfftn(input_img_down)
#         input_img_down_fft_up = self.channel_up(input_img_down_fft)
#
#         out0 = self.activation(self.conv0(input_img))
#         out1 = self.activation(self.conv1(out0))
#         feature_block_out0 = self.feature_block0(out1)
#         down_sampling = self.activation(self.down_sampling(feature_block_out0))
#
#         down_sampling_fft = torch.fft.rfftn(down_sampling)
#         fft_processing_out = self.fft_processing(down_sampling_fft + input_img_down_fft_up)
#         fft_processing_out_ifft = torch.fft.irfftn(fft_processing_out + input_img_down_fft_up)
#
#         up_sampling = self.up_sampling(fft_processing_out_ifft)
#         feature_block_out1 = self.feature_block1(up_sampling + feature_block_out0)
#
#         out2 = self.activation(self.conv2(feature_block_out1 + out1))
#         out3 = self.activation(self.conv3(out2))
#
#         return out3
#
#
# class EnhanceModule1(nn.Module):
#     def __init__(self, channel=64, kernel_size=3):
#         super(EnhanceModule1, self).__init__()
#
#         self.Relu = nn.LeakyReLU()
#         self.Enhance_conv0_1 = nn.Conv2d(4, channel, kernel_size, padding=2, padding_mode='replicate', dilation=2)
#         self.Enhance_conv0_2 = nn.Conv2d(channel, channel, kernel_size, padding=2, padding_mode='replicate',
#                                          dilation=2)  # 96*96
#         self.conv0 = nn.Conv2d(channel, channel * 2, kernel_size=1, stride=1, padding=0)
#         self.conv1 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv2 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv3 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv4 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
#         self.Enhance_subsampling0 = nn.Conv2d(channel * 2, channel, kernel_size=2, stride=2, padding=0)  # 48*48
#         self.conv5 = nn.Conv2d(channel, channel * 2, kernel_size=1, stride=1, padding=0)
#         self.conv6 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv7 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv8 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv9 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
#         self.Enhance_subsampling1 = nn.Conv2d(channel * 2, channel, kernel_size=2, stride=2, padding=0)  # 24*24
#         self.conv10 = nn.Conv2d(channel, channel * 2, kernel_size=1, stride=1, padding=0)
#         self.conv11 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv12 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv13 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv14 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
#         self.Enhance_subsampling2 = nn.Conv2d(channel * 2, channel, kernel_size=2, stride=2, padding=0)  # 12*12
#         self.conv15 = nn.Conv2d(channel, channel * 2, kernel_size=1, stride=1, padding=0)
#         self.conv16 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv17 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv18 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv19 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
#         self.Enhance_deconv0 = nn.UpsamplingBilinear2d(scale_factor=2.0)
#         self.down_channel0 = nn.Conv2d(channel*2, channel, kernel_size=1, stride=1, padding=0)
#         self.conv20 = nn.Conv2d(channel * 2, channel * 2, kernel_size=1, stride=1, padding=0)
#         self.conv21 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv22 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv23 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv24 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
#         self.Enhance_deconv1 = nn.UpsamplingBilinear2d(scale_factor=2.0)
#         self.down_channel1 = nn.Conv2d(channel*2, channel, kernel_size=1, stride=1, padding=0)
#         self.conv25 = nn.Conv2d(channel * 2, channel * 2, kernel_size=1, stride=1, padding=0)
#         self.conv26 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv27 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv28 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv29 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
#         self.Enhance_deconv2 = nn.UpsamplingBilinear2d(scale_factor=2.0)
#         self.down_channel2 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
#         self.Enhanceout0 = nn.Conv2d(channel * 2, channel, kernel_size, padding=1, padding_mode='replicate')
#         self.Enhanceout1 = nn.Conv2d(channel * 4, channel, kernel_size, padding=1, padding_mode='replicate')
#         self.Enhanceout2 = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, input_img):
#         out0 = self.Relu(self.Enhance_conv0_1(input_img))
#         out1 = self.Relu(self.Enhance_conv0_2(out0))
#         out2 = self.Relu(self.conv0(out1))
#         out3 = self.Relu(self.conv1(out2))
#         out4 = self.Relu(self.conv2(out3))
#         out5 = self.Relu(self.conv3(out4))
#         out6 = self.Relu(self.conv4(out5))
#         down0 = self.Relu(self.Enhance_subsampling0(torch.cat((out1, out6), dim=1)))
#         out7 = self.Relu(self.conv5(down0))
#         out8 = self.Relu(self.conv6(out7))
#         out9 = self.Relu(self.conv7(out8))
#         out10 = self.Relu(self.conv8(out9))
#         out11 = self.Relu(self.conv9(out10))
#         down1 = self.Relu(self.Enhance_subsampling1(torch.cat((down0, out11), dim=1)))
#         out12 = self.Relu(self.conv10(down1))
#         out13 = self.Relu(self.conv11(out12))
#         out14 = self.Relu(self.conv12(out13))
#         out15 = self.Relu(self.conv13(out14))
#         out16 = self.Relu(self.conv14(out15))
#         down2 = self.Relu(self.Enhance_subsampling2(torch.cat((down1, out16), dim=1)))
#         out17 = self.Relu(self.conv15(down2))
#         out18 = self.Relu(self.conv16(out17))
#         out19 = self.Relu(self.conv17(out18))
#         out20 = self.Relu(self.conv18(out19))
#         out21 = self.Relu(self.conv19(out20))
#         up0 = self.Relu(self.Enhance_deconv0(torch.cat((down2, out21), dim=1)))
#         up0 = self.Relu(self.down_channel0(up0))
#         out22 = self.Relu(self.conv20(torch.cat((up0, out16), dim=1)))
#         out23 = self.Relu(self.conv21(out22))
#         out24 = self.Relu(self.conv22(out23))
#         out25 = self.Relu(self.conv23(out24))
#         out26 = self.Relu(self.conv24(out25))
#         up1 = self.Relu(self.Enhance_deconv1(torch.cat((up0, out26), dim=1)))
#         up1 = self.Relu(self.down_channel1(up1))
#         out27 = self.Relu(self.conv25(torch.cat((up1, out11), dim=1)))
#         out28 = self.Relu(self.conv26(out27))
#         out29 = self.Relu(self.conv27(out28))
#         out30 = self.Relu(self.conv28(out29))
#         out31 = self.Relu(self.conv29(out30))
#         up2 = self.Relu(self.Enhance_deconv2(torch.cat((up1, out31), dim=1)))
#         up2 = self.Relu(self.down_channel2(up2))
#         out32 = self.Relu(self.Enhanceout0(torch.cat((out6, up2), dim=1)))
#         up0_1 = F.interpolate(up0, size=(input_img.size()[2], input_img.size()[3]))
#         up1_1 = F.interpolate(up1, size=(input_img.size()[2], input_img.size()[3]))
#         up2_1 = F.interpolate(up2, size=(input_img.size()[2], input_img.size()[3]))
#         out33 = self.Relu(self.Enhanceout1(torch.cat((up0_1, up1_1, up2_1, out32), dim=1)))
#         out34 = self.Relu(self.Enhanceout2(out33))
#         Enhanced_I = out34
#
#         return Enhanced_I

#
# class RelightNet(nn.Module):
#     def __init__(self, channel=64, kernel_size=3):
#         super(RelightNet, self).__init__()
#         self.Relu = nn.LeakyReLU()
#         self.out_act = nn.ReLU()
#         self.conv0 = nn.Conv2d(channel*2, channel, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv1 = nn.Conv2d(channel, 1, kernel_size=1, stride=1, padding=0)
#         self.EnhanceModule0 = EnhanceModule0()
#         self.EnhanceModule1 = EnhanceModule1()
#
#     def forward(self, input_L, denoise_R):
#         input_img = torch.cat((input_L, denoise_R), dim=1)
#
#         out0 = self.EnhanceModule0(input_img)
#         out1 = self.EnhanceModule1(input_img)
#         out3 = self.Relu(self.conv0(torch.cat((out0, out1), dim=1)))
#
#         output = self.out_act(self.conv1(out3))
#         return output
#         #增强照明


writer = SummaryWriter('./runs')

class APBSN(nn.Module):
    '''
    Asymmetric PD Blind-Spot Network (AP-BSN)
    '''

    def __init__(self, pd_a=5, pd_b=2, pd_pad=0, R3=False, R3_T=12, R3_p=0.16,
                 bsn='DBSNl', in_ch=3, bsn_base_ch=128, bsn_num_module=9
):
        '''
        Args:
            pd_a           : 'PD stride factor' during training
            pd_b           : 'PD stride factor' during inference
            pd_pad         : pad size between sub-images by PD process
            R3             : flag of 'Random Replacing Refinement'

            R3_T           : number of masks for R3
            R3_p           : probability of R3
            bsn            : blind-spot network type
            in_ch          : number of input image channel
            bsn_base_ch    : number of bsn base channel
            bsn_num_module : number of module
        '''
        super().__init__()

        # network hyper-parameters
        self.pd_a = pd_a
        self.pd_b = pd_b
        self.pd_pad = pd_pad
        self.R3 = R3
        self.R3_T = R3_T
        self.R3_p = R3_p
        self.threshold=160
        if bsn == 'DBSNl':
            self.bsn = DBSNl(in_ch, in_ch, bsn_base_ch, bsn_num_module)
        else:
            raise NotImplementedError('bsn %s is not implemented' % bsn)


    def forward(self, img ,illu_map,i,pd):
        '''
        Foward function includes sequence of PD, BSN and inverse PD processes.
        Note that denoise() function is used during inference time (for differenct pd factor and R3).
        '''
        # default pd factor is training factor (a)
        # print(i.shape)
        # print(i)/
        # means = get_mean(self,i)
        # maskill = (i > i.max()).float()

        # mean_c = img.mean(dim=1).unsqueeze(1)  # 平均通道=照明先验
        # maskill = (mean_c > 0.18).float()

        # maskill2 = maskill.cpu().numpy()
        # print(maskill2.shape)
        # plt.subplot(221), plt.title("1. B channel"), plt.axis('off')
        # bImg = cv2.cvtColor(maskill2, cv2.COLOR_BGR2RGB)  # 图片格式转换：BGR(OpenCV) -> RGB(PyQt5)
        # plt.imshow(bImg)
        # plt.show()# matplotlib 显示 channel B

#
#         # ===========tensor to numpy转numpy=============
#         illuimg = img.mul(255).byte()
#         illuimg=illuimg[0,:,:,:]
#         # print(illuimg.shape)
#         bgr_image = illuimg.cpu().numpy().transpose(1, 2, 0)
#         # print(bgr_image.shape) #(400, 600, 3)
#         # ===========tensor to numpy转numpy==================
#
#         # print("b",bgr_image.shape)
#      ##=================avg直接mask=====================
#         gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
#         # print('gary',gray.shape)
#         mean_thr = cv2.mean(gray)[0]
#         mean_thg = cv2.mean(gray)[1]
#         mean_thb = cv2.mean(gray)[2]
#         mean_th = (mean_thr+mean_thg+mean_thb)/3
#         # print('avg',mean_th) #112.95401666666667
#         rImg = torch.from_numpy(np.ascontiguousarray(gray))
#         # print(rImg)#197, 234, 228,  ..., 210, 195, 197],
#         # print("r",rImg.shape)
#         # .transpose(1, 2, 0)
#         maskill = (rImg > mean_th).float()
#         # print(maskill.shape) #ze([400, 600])
#         # maskill2 = maskill.numpy()
#         # maskill = maskill[None, None, :, :].cuda()
#         # print(maskill.shape) #([1, 1, 400, 600])
#         ##=================avg直接mask=====================
#
# #====================最大通道再判断=========================
#         # b, g, r = cv2.split(bgr_image)  # 灰度
#         # rgb_value = np.mean(bgr_image, axis=(0, 1))
#         # # print('mean',rgb_value) #[108.85957917 114.61871667 111.263275  ]
#         # max_data = cv2.max(cv2.max(rgb_value[0], rgb_value[1]), rgb_value[2])
#         #
#         # # blue 获取蓝的
#         # if max_data[0] == rgb_value[0]:
#         #     # mImg = bgr_image.copy()  # 获取 BGR
#         #     # mImg[:, :, 1] = 0  # G=0
#         #     # mImg[:, :, 2] = 0  # R=0
#         #     mImg = b
#         # # g 获取绿的
#         # elif max_data[0] == rgb_value[1]:
#         #     # mImg = bgr_image.copy()  # 获取 BGR
#         #     # mImg[:, :, 0] = 0  # B=0
#         #     # mImg[:, :, 2] = 0  # R=0
#         #     mImg = g
#         # # r 获取红的
#         # else:
#         #     # mImg = bgr_image.copy()  # 获取 BGR
#         #     # mImg[:, :, 0] = 0  # B=0
#         #     # mImg[:, :, 1] = 0  # G=0
#         #     mImg = r
#         # # numpy zhuan tensor
#         # rImg = torch.from_numpy(np.ascontiguousarray(mImg))
#         # # print("r",rImg.shape)
#         # # .transpose(1, 2, 0)
#         # maskill = (rImg > self.threshold).float()
#
#         #
#
#         # # ====================最大通道再判断=====================
#
#         # ==============xianshi==================
#         # maskill2 = maskill.numpy()
#         # # print(maskill2.shape)
#         # plt.subplot(221), plt.title("1. B channel"), plt.axis('off')
#         # bImg = cv2.cvtColor(maskill2, cv2.COLOR_BGR2RGB)  # 图片格式转换：BGR(OpenCV) -> RGB(PyQt5)
#         # plt.imshow(bImg)
#         # plt.show()# matplotlib 显示 channel B
#        #############转换进网络
#         maskill = maskill[None, None, :, :].cuda()
#         # print(maskill.shape)
#         # print(maskill.shape)#torch.Size([1, 1, 404, 604])
#         # ==============xianshi==================
#bsn==========================================================================
        # if pd is None: pd = self.pd_a
        b,c,h,w=img.shape
        # maskill = (i >0.18).float()
        maskill = (i > i.mean()).float()
        # print('maskill',maskill.shape)
        #保存=================================
        # illuimg = maskill.mul(255).byte()
        # illuimg=illuimg[0,:,:,:]
        # bgr_image = illuimg.cpu().numpy().transpose(1, 2, 0)
        # bImg = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        # plt.imshow(bImg)
        # plt.axis('off')
        # plt.savefig('image3.png', bbox_inches='tight', pad_inches=0)
        # plt.show()
        # 保存====================================

        # maskill = (i > i.mean()).float()
        maskill1 = (i > i.max()).float()
        # pad images for PD process
        if h % pd != 0:
            img = F.pad(img, (0, 0, 0, pd - h % pd), mode='constant', value=0)
            maskill = F.pad(maskill, (0, 0, 0, pd - h % pd), mode='constant', value=0)
            maskill1 = F.pad(maskill1, (0, 0, 0, pd - h % pd), mode='constant', value=0)
        if w % pd != 0:
            img = F.pad(img, (0, pd - w % pd, 0, 0), mode='constant', value=0)
            maskill = F.pad(maskill, (0, pd - w % pd, 0, 0), mode='constant', value=0)
            maskill1 = F.pad(maskill1, (0, 0, 0, pd - h % pd), mode='constant', value=0)
        pd_img = util2.pixel_shuffle_down_sampling(img, f=pd, pad=self.pd_pad)
        maskill = util2.pixel_shuffle_down_sampling(maskill, f=pd, pad=self.pd_pad)
        maskill1 = util2.pixel_shuffle_down_sampling(maskill1, f=pd, pad=self.pd_pad)
        pd_img, random2seq = util2.randomArrangement(pd_img, pd)
        # print(pd_img.shape)
        # print(maskill.shape)

        pd_img_denoised = self.bsn(pd_img, maskill)

        pd_img_denoised = util2.inverseRandomArrangement(pd_img_denoised, random2seq, pd)
        img_pd_bsn = util2.pixel_shuffle_up_sampling(pd_img_denoised, f=pd, pad=self.pd_pad)


    # #==============================================================================================================================
    #
        # b,c,h,w=img.shape
        # img_pd2_bsn = img
        # maskill2 = (i >0.18).float()
        # if h % 2 != 0:
        #     img_pd2_bsn = F.pad(img_pd2_bsn, (0, 0, 0, 2 - h % 2), mode='constant', value=0)
        #     maskill2 = F.pad(maskill2, (0, 0, 0, 2 - h % 2), mode='constant', value=0)
        # if w % 2 != 0:
        #     img_pd2_bsn = F.pad(img_pd2_bsn, (0, 2 - w % 2, 0, 0), mode='constant', value=0)
        #     maskill2 = F.pad(maskill2, (0, 2 - w % 2, 0, 0), mode='constant', value=0)
        # pd2_img = util2.pixel_shuffle_down_sampling(img_pd2_bsn, f=2, pad=self.pd_pad)
        # maskill2 = util2.pixel_shuffle_down_sampling(maskill2, f=2, pad=self.pd_pad)
        # pd2_img, random2seq = util2.randomArrangement(pd2_img, 2)
        # pd2_img_denoised = self.bsn(pd2_img, maskill2)
        # pd2_img_denoised = util2.inverseRandomArrangement(pd2_img_denoised, random2seq, 2)
        # img_pd2_bsn = util2.pixel_shuffle_up_sampling(pd2_img_denoised, f=2, pad=self.pd_pad)
        #
        #
        #
        # # img_pd2_bsn = forward_mpd(img_pd2_bsn, pd=2)
        #
        # # # ============== PD = 5 ====================
        # img_pd5_bsn = img
        # maskill5 = (i > i.mean()).float()
        # if h % 5 != 0:
        #     img_pd5_bsn = F.pad(img_pd5_bsn, (0, 0, 0, 5 - h % 5), mode='constant', value=0)
        #     maskill5 = F.pad(maskill5, (0, 0, 0, 5 - h % 5), mode='constant', value=0)
        # if w % 5 != 0:
        #     img_pd5_bsn = F.pad(img_pd5_bsn, (0,5 - w % 5, 0, 0), mode='constant', value=0)
        #     maskill5 = F.pad(maskill5, (0, 5 - w % 5, 0, 0), mode='constant', value=0)
        # pd5_img = util2.pixel_shuffle_down_sampling(img_pd5_bsn, f=5, pad=self.pd_pad)
        # maskill5 = util2.pixel_shuffle_down_sampling(maskill5, f=5, pad=self.pd_pad)
        # pd5_img, random5seq = util2.randomArrangement(pd5_img, 5)
        # pd5_img_denoised = self.bsn(pd5_img, maskill5)
        # pd5_img_denoised = util2.inverseRandomArrangement(pd5_img_denoised, random5seq, 5)
        # img_pd5_bsn = util2.pixel_shuffle_up_sampling(pd5_img_denoised, f=5, pad=self.pd_pad)
        # img_pd5_bsn = img_pd5_bsn[:, :, :h, :w]
        # # ============== FUSE 1 ====================
        # maskill1 = (i > i.max()).float()
        # img_pd1_bsn = self.bsn(img, maskill1)
        # img_pd_bsn=img_pd1_bsn
        # img_pd_bsn = torch.add(torch.mul(img_pd5_bsn, 0.7), torch.mul(img_pd1_bsn, 0.3))  # 鍘? 9锛?1
        #
        # # ============== FUSE 2 ====================
        # img_pd_bsn = torch.add(torch.mul(img_pd_bsn, 0.2), torch.mul(img_pd2_bsn, 0.8))
    #     == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =

        if not self.R3:
            # print('no r3')
            return img_pd_bsn,maskill1
        # return img_pd_bsn
        else:
            denoised = torch.empty(*(img.shape), self.R3_T, device=img.device)
            for t in range(self.R3_T):
                indice = torch.rand_like(img)
                mask = indice < 0.16

                tmp_input = torch.clone(img_pd_bsn).detach()
                tmp_input[mask] = img[mask]
                p = self.pd_pad
                tmp_input = F.pad(tmp_input, (p, p, p, p), mode='reflect')
                # print(tmp_input.shape) torch.Size([1, 3, 404, 604])
                if self.pd_pad == 0:
                    denoised[..., t] = self.bsn(tmp_input,maskill1)
                    # print(denoised.shape) #(400, 600, 3)
                else:
                    denoised[..., t] = self.bsn(tmp_input,maskill1)[:, :, p:-p, p:-p]
            rimg = torch.mean(denoised, dim=-1)
            # print('yes r3apbsn')
            return rimg, maskill1
            # pd_img_denoised = self.bsn(oimg, maskill)
            #
            #=======================迭代======================
            # rimg=self.bsn(img_pd_bsn,maskill)
            # r2img=self.bsn(rimg,maskill)
            # r3img = self.bsn(r2img, maskill)
            # r4img = self.bsn(r3img, maskill)
            # r5img = self.bsn(r4img, maskill)
            # r6img = self.bsn(r5img, maskill)
            # # r7img = self.bsn(r6img, maskill)
            # # r8img = self.bsn(r7img, maskill)
            # # r10img[...]=rimg, r2img, r3img, r4img, r5img, r6img, r7img, r8img
            # # r9img=torch.mean(r10img, dim=-1)
            # return r6img,maskill
    # =======================迭代======================
def tensor2np(t:torch.Tensor):
    '''
    transform torch Tensor to numpy having opencv image form.
    RGB -> BGR
    (c,h,w) -> (h,w,c)
    '''
    # t = t.detach()

    # gray
    if len(t.shape) == 2:
        return t.permute(1,2,0).numpy()
    # RGB -> BGR
    elif len(t.shape) == 3:
        return np.flip(t.permute(1,2,0).numpy(), axis=2)
    # image batch
    elif len(t.shape) == 4:
        return np.flip(t.permute(0,2,3,1).numpy(), axis=3)
    else:
        raise RuntimeError('wrong tensor dimensions : %s'%(t.shape,))
def get_mean(self, tensor_shapes):
    means = []
    for i in tensor_shapes:
        a = i.mean()  # mean方法可以直接用于tensor类型数据的计算
        means.append(a)
    # print("获得平均值了")
    return means[0]

class R2RNet(nn.Module):#总的
    def __init__(self):
        super(R2RNet, self).__init__()
        # self.bsn=DBSNl(in_ch=3, out_ch=3,R3=True)

        # self.DecomNet = DecomNet()#分解 需要vgg感知损失
        # self.DenoiseNet = DenoiseNet()#去噪cnn
        # self.RelightNet = RelightNet() #增强
        self.DecomNet = IAT()  # 分解 需要vgg感知损失
        self.DenoiseNet = APBSN()  # 去噪cnn
        # self.RelightNet = RelightNet()  # 增强
        # self.vgg = load_vgg16("./model")


    def forward(self, input_low):
        # print(input_low.shape)
        # device = torch.device('cuda:0')
        # inputs = inputs
        # input_low1=torch.tensor(input_low)
        # input_low = tensor2np(input_low)
        # input_low = Variable(torch.FloatTensor(input_low1))

        # input_low = torch.FloatTensor(torch.tensor(input_low))
        input_low = Variable(torch.FloatTensor(torch.from_numpy(input_low))).cuda()
        # input_low = Variable(input_low, requires_grad=False)
        # Forward DecomNet
        ilist, rlist, inlist, attlist,rimg1,illu_map= self.DecomNet(input_low)
        # # print(illu_map.shape) 1
        # print(ilist[4].shape) # 1
        # ilist,rlist, inlist = self.DecomNet(input_low)
        # R_high, I_high = self.DecomNet(input_high) #干净图像
        # Forward DenoiseNet
       #gaichengshuru pd=5
        # denoise_R1, maskill = self.DenoiseNet(rlist[2]+ilist[2],illu_map,ilist[2], pd=5)


        # 三通道中像素最小值

        self.out1 = rlist[4].detach().cpu()+ilist[0].detach().cpu()

        denoise_R2, maskill2 = self.DenoiseNet(rlist[4] + ilist[0], illu_map, ilist[0], pd=1)
        self.out3 = denoise_R2.detach().cpu()

        # self.out1 = 0.4*inlist[2].detach().cpu() + 0.6*input_low.detach().cpu()
        # self.out1 = 0.2 * rlist[2].detach().cpu() + 0.8 * input_low.detach().cpu()
        # self.out1 = attlist[2].detach().cpu()+input_low.detach().cpu()+ilist[2].detach().cpu()
        # self.out1= 1.2*rlist[2].detach().cpu()+ilist[2].detach().cpu()
        # self.out1= rlist[0].detach().cpu()+ilist[0].detach().cpu()

        # denoise_R2, maskill2 = self.DenoiseNet(rlist[2] +0.5*attlist[2], illu_map, ilist[2], pd=1)
        # denoise_R2, maskill2 = self.DenoiseNet(attlist[2]+input_low+ilist[2],illu_map, ilist[2],pd=1)
        # denoise_R2, maskill2 = self.DenoiseNet(0.2*rlist[2] + 0.8*input_low, illu_map, ilist[2], pd=1)
        # denoise_R2, maskill2 = self.DenoiseNet(0.4* inlist[2] + 0.6* input_low, illu_map, ilist[2], pd=1)
        # denoise_R1= rlist[2]

        # self.out3 = 0.5*self.out3 +0.5* self.out1
        # self.out1=attlist[1].detach().cpu()
        # self.out1 =inlist[2].detach().cpu()
        # self.tar1=inlist[2].detach().cpu()
        # self.out2=denoise_R1.detach().cpu()
        # if not self.R3:
        #
        #     self.out3 = denoise_R2.detach().cpu()
        #     print('no r3')
        # else:
        #     #==================R3===========================
        #     self.input2=rlist[2]
        #
        #     x = self.input2
        #     # print(x.shape)
        #     denoised = torch.empty(*(x.shape),8, device=torch.device('cuda:0'))
        #
        #     for t in range(8):
        #         indice = torch.rand_like(x)
        #         mask = indice < 0.16
        #         tmp_input = torch.clone(denoise_R2).detach()
        #         tmp_input[mask] = x[mask]
        #         p = 0
        #         tmp_input = F.pad(tmp_input, (p, p, p, p), mode='reflect')
        #         # print(tmp_input.shape)#torch.Size(2[1, 3, 404, 604]) 0[400,600]
        #         # denoised=self.bsn(tmp_input,maskill2)
        #         # print(denoised[:, :, p:-p, p:-p].shape)#torch.Size([1, 3, 400, 600])
        #         outbsn= self.DenoiseNet(tmp_input,pd=1)[0]
        #         denoised[..., t]=outbsn
        #         # denoised[..., t]=denoised[..., t]
        #         # print(denoised.shape)
        #     rimg_pd_bsn = torch.mean(denoised, dim=-1)
        #     # rimg_pd_bsn= denoised[..., 6]
        #     self.out3 = rimg_pd_bsn.detach().cpu()
        #     # ==================R3===========================
        #     # denoised=self.DenoiseNet(denoise_R2,pd=1)[0]
        #     # self.out3 = denoised.detach().cpu()
        #     print('true r3')
        #

        # self.tar2=rlist[0].detach().cpu() #meiyongxianhg
        # Forward RelightNet
        # I_delta = self.RelightNet(I_low, denoise_R) #输入两个

        # Other variables
        # I_low_3 = torch.cat((I_low, I_low, I_low), dim=1)
        # I_high_3 = torch.cat((I_high, I_high, I_high), dim=1)#干净图像
        # I_delta_3 = torch.cat((I_delta, I_delta, I_delta), dim=1) #输出三个？

        # DecomNet_loss分解loss
        # self.vgg_loss = compute_vgg_loss(R_low * I_low_3,  input_low).cuda() + compute_vgg_loss(R_high * I_high_3, input_high).cuda()#干净图像
        # self.recon_loss_low = F.l1_loss(R_low * I_low_3, input_low).cuda()
        # self.recon_loss_high = F.l1_loss(R_high * I_high_3, input_high).cuda()#干净图像
        # self.recon_loss_mutal_low = F.l1_loss(R_high * I_low_3, input_low).cuda()
        # self.recon_loss_mutal_high = F.l1_loss(R_low * I_high_3, input_high).cuda()#干净图像

        # self.loss_Decom = self.recon_loss_low + \
        #                   self.recon_loss_high + \
        #                   0.1 * self.recon_loss_mutal_low + \
        #                   0.1 * self.recon_loss_mutal_high + \
        #                   0.1 * self.vgg_loss
#改掉loss
        self.itrloss1=self.DecomNet.sciloss(input_low)
        self.loss_Decom =self.itrloss1
    # 改掉loss
        # self.denoise_vgg = compute_vgg_loss(denoise_R, R_high).cuda() #干净
        # DenoiseNet_loss 去噪损失
        # self.denoise_loss = F.l1_loss(denoise_R1, rlist[2]+ilist[2])
        # self.loss_Denoise = self.denoise_loss
    # DenoiseNet_loss 去噪损失
        #
        # # RelightNet_loss
        # self.Relight_loss = F.l1_loss(denoise_R * I_delta_3, input_high).cuda() #干净 噪声乘光照
        # self.Relight_vgg = compute_vgg_loss(denoise_R * I_delta_3, input_high).cuda() #干净
        # self.fre_loss = frequency_loss(denoise_R * I_delta_3, input_high).cuda() #干净
        #
        # self.loss_Relight = self.Relight_loss + 0.1 * self.Relight_vgg + 0.01 * self.fre_loss
        #
        # self.output_R_low = R_low.detach().cpu()
        # self.output_I_low = I_low_3.detach().cpu()
        # self.output_I_delta = I_delta_3.detach().cpu()
        # self.output_R_denoise = denoise_R.detach().cpu()
        # self.output_S = denoise_R.detach().cpu() * I_delta_3.detach().cpu() #最终输出是去噪的



    def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda()
        # self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2))

        self.smooth_kernel_y = torch.transpose(self.smooth_kernel_x, 2, 3)

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        grad_out = torch.abs(F.conv2d(input_tensor, kernel,
                                      stride=1, padding=1))
        return grad_out

    def ave_gradient(self, input_tensor, direction):
        return F.avg_pool2d(self.gradient(input_tensor, direction),
                            kernel_size=3, stride=1, padding=1)

    def smooth(self, input_I, input_R):
        input_R = 0.299*input_R[:, 0, :, :] + 0.587*input_R[:, 1, :, :] + 0.114*input_R[:, 2, :, :]
        input_R = torch.unsqueeze(input_R, dim=1)
        return torch.mean(self.gradient(input_I, "x") * torch.exp(-10 * self.ave_gradient(input_R, "x")) +
                          self.gradient(input_I, "y") * torch.exp(-10 * self.ave_gradient(input_R, "y")))

    def evaluate(self, epoch_num, eval_low_data_names, eval_high_data_names, vis_dir, train_phase):
        print("Evaluating for phase %s / epoch %d..." % (train_phase, epoch_num))
        spsnr = 0
        sssim=0
        sdists=0
        count=0
        slpips=0
        self.logger = Logger()
        # self.logger = Logger((33, 70191), log_dir=self.file_manager.get_dir(''),log_file_option='a')
        with torch.no_grad():# Otherwise the intermediate gradient would take up huge amount of CUDA memory
            for idx in range(len(eval_low_data_names)):
                eval_low_img = Image.open(eval_low_data_names[idx])
                eval_low_img = np.array(eval_low_img, dtype="float32") / 255.0
                eval_low_img = np.transpose(eval_low_img, [2, 0, 1])

                input_low_eval = np.expand_dims(eval_low_img, axis=0)
                eval_high_img = Image.open(eval_high_data_names[idx])
                eval_high_img = np.array(eval_high_img, dtype="float32")

                if train_phase == "Decom":
                    self.forward(input_low_eval) #输入都是低质？
                    # result_1 = self.color
                    # result_2 = self.output_I_low
                    # input = np.squeeze(input_low_eval)
                    # result_1 = np.squeeze(result_1)
                    # result_2 = np.squeeze(result_2)
                    dcat_image = self.out1
                    dcat_image = dcat_image.numpy().squeeze(0)
                    cat_image = dcat_image
                    # deval_high_img = self.tar1
                    # deval_high_img = deval_high_img.numpy().squeeze(0)
                    # eval_high_img = deval_high_img
                    # cat_image = np.concatenate([result_1, result_2], axis=2) #输出是两个分解的cat
                if train_phase == 'Denoise':
                    self.forward(input_low_eval)
                    # result_1 = self.outdenoise
                    # input = np.squeeze(input_low_eval)
                    # denoise_R2,maskill = self.DenoiseNet(self.input2,pd=2)
                    # denoise_R2 = self.DenoiseNet.bsn(self.input2)
                    # denoise_R2 = self.DenoiseNet(self.input2, pd=1)
                    # self.out3 = denoise_R2.detach().cpu()
                    # x=self.input2
                    # denoised = torch.empty(*(x.shape), 8, device=x.device)
                    # for t in range(8):
                    #     indice = torch.rand_like(x)
                    #     mask = indice <0.16
                    #
                    #     tmp_input = torch.clone(denoise_R2).detach()
                    #     tmp_input[mask] = x[mask]
                    #     p = 2
                    #     tmp_input = F.pad(tmp_input, (p, p, p, p), mode='reflect')
                    #     # if 2 == 0:
                    #     #     denoised[..., t] = self.bsn(tmp_input, is_masked=True)
                    #     # else:
                    #     denoised[..., t] = self.bsn(tmp_input,maskill)[:, :, p:-p, p:-p]#加入maskill
                    #
                    # rimg_pd_bsn = torch.mean(denoised, dim=-1)



                    # x = self.input2
                    # denoised = torch.empty(*(x.shape), 8, device=torch.device('cuda:0'))
                    # for t in range(8):
                    #     indice = torch.rand_like(x)
                    #     mask = indice < 0.16
                    #     tmp_input = torch.clone(denoise_R2).detach()
                    #     tmp_input[mask] = x[mask]
                    #     p = 2
                    #     tmp_input = F.pad(tmp_input, (p, p, p, p), mode='reflect')
                    #     denoised[..., t] = self.bsn(tmp_input)[:, :, p:-p, p:-p]
                    # rimg_pd_bsn = torch.mean(denoised, dim=-1)
                    # self.out3 = rimg_pd_bsn.detach().cpu()
                    dcat_image = self.out3

                    dcat_image = dcat_image.numpy().squeeze(0)
                    cat_image=dcat_image
                    # deval_high_img = self.tar2
                    # deval_high_img=deval_high_img.numpy().squeeze(0)
                    # eval_high_img = deval_high_img
                # if train_phase == "Relight":
                #     self.forward(input_low_eval, input_low_eval)
                #     result_4 = self.output_S
                #     input = np.squeeze(input_low_eval)
                #     result_4 = result_4.numpy().squeeze(0)
                #     cat_image = result_4

                cat_image = np.transpose(cat_image, (1, 2, 0))
            # print(cat_image.shape)
                im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
                im_test = np.array(im, dtype='float32')
                # im_eval =eval_high_img
                # im_eval= Image.fromarray(np.clip(eval_high_img * 255.0, 0, 255.0).astype('uint8'))
                # im_eval=np.array(im_eval, dtype='float32')
                spsnr += psnr2(im_test, eval_high_img)
                sssim += ssim2(im_test, eval_high_img)
                slpips +=lpips2(im_test, eval_high_img)
                # sdists +=dists2(im_test, eval_high_img)
                count+=count
            print('psnr=', spsnr / len(eval_low_data_names))
            print('ssim=', sssim / len(eval_low_data_names))
            print('lpips=', slpips / len(eval_low_data_names))
            # writer.add_scalar('runs/psnr,runs/ssim,runs/lpips', spsnr / len(eval_low_data_names), sssim / len(eval_low_data_names), slpips / len(eval_low_data_names),epoch_num)
            writer.add_scalars('runs/metrics', {
                'psnr': spsnr / len(eval_low_data_names),
                'ssim': sssim / len(eval_low_data_names),
                'lpips': slpips / len(eval_low_data_names)
            }, epoch_num)

            self.logger.val('[%s] Done! PSNR : %.3f dB, SSIM : %.4f, LPIPS : %.4f' % (epoch_num, spsnr / len(eval_low_data_names), sssim / len(eval_low_data_names),slpips / len(eval_low_data_names)))

    def save(self, iter_num, ckpt_dir):
        save_dir = ckpt_dir + '/' + self.train_phase + '/'
        save_name = save_dir + '/' + str(iter_num) + '.tar'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if self.train_phase == 'Decom':
            torch.save(self.DecomNet.state_dict(), save_name)
        if self.train_phase == 'Denoise':
            torch.save(self.DenoiseNet.state_dict(), save_name)
        # if self.train_phase == 'Relight':
        #     torch.save(self.RelightNet.state_dict(), save_name)

    def load(self, ckpt_dir):
        load_dir = ckpt_dir + '/' + self.train_phase + '/'
        # load_dir = ckpt_dir
        if os.path.exists(load_dir):
            load_ckpts = os.listdir(load_dir)
            load_ckpts.sort()
            load_ckpts = sorted(load_ckpts, key=len)
            if len(load_ckpts) > 0:
                load_ckpt = load_ckpts[-1]
                global_step = int(load_ckpt[:-4])
                ckpt_dict = torch.load(load_dir + load_ckpt)
                if self.train_phase == 'Decom':
                    self.DecomNet.load_state_dict(ckpt_dict)
                if self.train_phase == 'Denoise':
                    self.DenoiseNet.load_state_dict(ckpt_dict)
                # if self.train_phase == 'Relight':
                #     self.RelightNet.load_state_dict(ckpt_dict)
                return True, global_step
            else:
                return False, 0
        else:
            return False, 0


    def train(self,
              train_low_data_names,
              train_high_data_names,
              eval_low_data_names,
              eval_high_data_names,
              batch_size,
              patch_size, epoch,
              lr,
              vis_dir,
              ckpt_dir,
              eval_every_epoch,
              train_phase):
        assert len(train_low_data_names) == len(train_high_data_names)
        numBatch = len(train_low_data_names) // int(batch_size)
        self.patch_size=patch_size
        # Create the optimizers
        self.train_op_Decom = optim.Adam(self.DecomNet.parameters(),
                                         lr=lr[0], betas=(0.9, 0.999), weight_decay=0.0001)
        self.train_op_Denoise = optim.Adam(self.DenoiseNet.parameters(),
                                           lr=lr[0], betas=(0.9, 0.999), weight_decay=0.0001)
        # self.train_op_Relight = optim.Adam(self.RelightNet.parameters(),
        #                                    lr=lr[0], betas=(0.9, 0.999), weight_decay=0.0001)

        # Initialize a network if its checkpoint is available
        self.train_phase = train_phase
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("No pretrained model to restore!")

        print("Start training for phase %s, with start epoch %d start iter %d : " %
             (self.train_phase, start_epoch, iter_num))

        start_time = time.time()
        image_id = 0
        for epoch in range(start_epoch, epoch):
            self.lr = lr[epoch]
            # Adjust learning rate
            for param_group in self.train_op_Decom.param_groups:
                param_group['lr'] = self.lr
            for param_group in self.train_op_Denoise.param_groups:
                param_group['lr'] = self.lr
            # for param_group in self.train_op_Relight.param_groups:
            #     param_group['lr'] = self.lr
            for batch_id in range(start_step, numBatch):
                # Generate training data for a batch
                batch_input_low = np.zeros((batch_size, 3, patch_size, patch_size,), dtype="float32")
                # batch_input_high = np.zeros((batch_size, 3, patch_size, patch_size,), dtype="float32")
                for patch_id in range(batch_size):
                    # Load images
                    train_low_img = Image.open(train_low_data_names[image_id])
                    train_low_img = np.array(train_low_img, dtype='float32')/255.0
                    # train_high_img = Image.open(train_high_data_names[image_id])
                    # train_high_img = np.array(train_high_img, dtype='float32')/255.0
                    # Take random crops
                    h, w, _ = train_low_img.shape
                    x = random.randint(0, h - patch_size)
                    y = random.randint(0, w - patch_size)
                    train_low_img = train_low_img[x: x + patch_size, y: y + patch_size, :]
                    # train_high_img = train_high_img[x: x + patch_size, y: y + patch_size, :]
                    # Data augmentation
                    if random.random() < 0.5:
                        train_low_img = np.flipud(train_low_img)
                        # train_high_img = np.flipud(train_high_img)
                    if random.random() < 0.5:
                        train_low_img = np.fliplr(train_low_img)
                        # train_high_img = np.fliplr(train_high_img)
                    rot_type = random.randint(1, 4)
                    if random.random() < 0.5:
                        train_low_img = np.rot90(train_low_img, rot_type)
                        # train_high_img = np.rot90(train_high_img, rot_type)
                    # Permute the images to tensor format
                    train_low_img = np.transpose(train_low_img, (2, 0, 1))
                    # train_high_img = np.transpose(train_high_img, (2, 0, 1))
                    # Prepare the batch
                    batch_input_low[patch_id, :, :, :] = train_low_img
                    # batch_input_high[patch_id, :, :, :] = train_high_img
                    self.input_low = batch_input_low
                    # self.input_high = batch_input_high

                    image_id = (image_id + 1) % len(train_low_data_names)
                    if image_id == 0:
                        tmp = list(zip(train_low_data_names, train_high_data_names))
                        random.shuffle(list(tmp))
                        train_low_data_names, train_high_data_names = zip(*tmp)

                # Feed-Forward to the network and obtain loss
                self.forward(self.input_low)
                if self.train_phase == "Decom":
                    self.train_op_Decom.zero_grad()
                    self.loss_Decom.backward()
                    self.train_op_Decom.step()
                    global_step += 1
                    loss = self.loss_Decom.item()
                elif self.train_phase == 'Denoise':
                    self.train_op_Denoise.zero_grad()
                    self.loss_Denoise.backward()
                    self.train_op_Denoise.step()
                    global_step += 1
                    loss = self.loss_Denoise.item()
                # elif self.train_phase == "Relight":
                #     self.train_op_Relight.zero_grad()
                #     self.loss_Relight.backward()
                #     self.train_op_Relight.step()
                #     global_step += 1
                #     loss = self.loss_Relight.item()

                print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f"
                      % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                iter_num += 1
                writer.add_scalar('runs/loss', loss, global_step)
                img = torch.rand(3, 3, self.patch_size, self.patch_size).numpy()
                if global_step % 10 == 0:
                    img[:1, :, :, :] = batch_input_low[:1, :, :, :]
                    img[1:2, :, :, :] = self.out1[:1, :, :, :]
                    # img[2:3, :, :, :] = batch_input_high[:1, :, :, :]
                    writer.add_images('results', img)

            # Evaluate the model and save a checkpoint file for it
            if (epoch + 1) % eval_every_epoch == 0:
                self.save(iter_num, ckpt_dir)
                self.evaluate(epoch + 1, eval_low_data_names, eval_high_data_names,
                              vis_dir=vis_dir, train_phase=train_phase)

        print("Finished training for phase %s." % train_phase)

    def predict(self,
                test_low_data_names,
                res_dir1,
                res_dir2,
                ckpt_dir,
                eval_high_data_names):

        # Load the network with a pre-trained checkpoint
        self.train_phase = 'Decom'
        load_model_status, _ = self.load(ckpt_dir)
        print(load_model_status)
        if load_model_status:
            print(self.train_phase, "  : Model restore success!")
        else:
            print("No pretrained model to restore!")
            raise Exception

        self.train_phase = 'Denoise'
        load_model_status, _ = self.load(ckpt_dir)
        if load_model_status:
            print(self.train_phase, ": Model restore success!")
        else:
            print("No pretrained model to restore!")
            raise Exception

        # self.train_phase = 'Relight'
        # load_model_status, _ = self.load(ckpt_dir)
        # if load_model_status:
        #     print(self.train_phase, ": Model restore success!")
        # else:
        #     print("No pretrained model to restore!")
        #     raise Exception

        # Set this switch to True to also save the reflectance and shading maps
        save_R_L = False
        # ssim = SSIM()
        # psnr = PSNR()
        ssim1_list = []
        psnr1_list = []
        lpips1_list = []
        psnr1_value=0
        ssim1_value = 0
        lpips1_value = 0
        psnr2_value = 0
        ssim2_value = 0
        lpips2_value = 0
        count=0
        # psnr1_list = []
        ssim2_list = []
        psnr2_list = []
        lpips2_list = []
        # psnr2_list = []
        # Predict for the test images
        for idx in range(len(test_low_data_names)):
            count += 1
            test_img_path = test_low_data_names[idx]
            test_img_name = test_img_path.split('/')[-1]

            print('Processing ', test_img_name)
            test_low_img = Image.open(test_img_path)
            test_low_img = np.array(test_low_img, dtype="float32") / 255.0
            test_low_img = np.transpose(test_low_img, (2, 0, 1))
            input_low_test = np.expand_dims(test_low_img, axis=0)

            self.forward(input_low_test)
            result_1 = self.out1
            result_2 = self.out3
            # result_3 = self.output_I_delta
            # result_4 = self.output_S
            input = np.squeeze(input_low_test)
            result_1 = np.squeeze(result_1)
            result_2 = np.squeeze(result_2)
            # result_3 = np.squeeze(result_3)
            # result_4 = np.squeeze(result_4)
            if save_R_L:
                cat1_image = np.concatenate([input, result_1], axis=2)
            else:
                cat1_image = result_1.numpy()
                cat2_image = result_2.numpy()
            cat1_image = np.transpose(cat1_image, (1, 2, 0))
            # cat1_image1=np.array(cat1_image, dtype='float32')
            cat2_image = np.transpose(cat2_image, (1, 2, 0))
            # print(cat2_image.shape)
            # cat2_image = cat2_image
            # print(cat_image.shape)
            im1 = Image.fromarray(np.clip(cat1_image * 255.0, 0, 255.0).astype('uint8'))
            # im11 = np.array(im1, dtype='float32')
            filepath = res_dir1 + '/' + test_img_name
            # im1.save(filepath[:-4] + 'illu' + '.jpg')
            im1.save(filepath[:-4] + 'illu' + '.png')
            # im1.save(filepath[:-4] + 'illu' + '.bmp')
            eval_high_img = Image.open(eval_high_data_names[idx])
            eval_high_img = np.array(eval_high_img, dtype="float32")
            # print('im1',cat1_image.device)
            # print('high',eval_                high_img.device)
            im11 = Image.fromarray(np.clip(cat1_image * 255.0, 0, 255.0).astype('uint8'))
            im_test1 = np.array(im11, dtype='float32')
            # print(im_test1.shape)
            # print(eval_high_img.shape)
            score_ssim = ssim2(im_test1, eval_high_img)
            score_psnr = psnr2(im_test1, eval_high_img)
            score_lpips = lpips2(im_test1, eval_high_img)

            ssim1_value += score_ssim
            psnr1_value += score_psnr
            lpips1_value += score_lpips
            # print('单幅图增强的 PSNR1 Value is:', psnr1_value)
            # print('单幅图增强的 SSIM1 Value is:', ssim1_value)
            # print('单幅图增强的 lpips1 Value is:', lpips1_value)
            im2 = Image.fromarray(np.clip(cat2_image * 255.0, 0, 255.0).astype('uint8'))
            filepath = res_dir2 + '/' + test_img_name
            # im2.save(filepath[:-4] + 'deno' + '.jpg')
            im2.save(filepath[:-4] + 'deno' + '.png')
            # im2.save(filepath[:-4] + 'deno' + '.bmp')
            # im22 = Image.fromarray(np.clip(cat2_image * 255.0, 0, 255.0).astype('uint8'))
            im_test2 = np.array(im2, dtype='float32')
            ssim2_value += ssim2(im_test2, eval_high_img)
            psnr2_value += psnr2(im_test2, eval_high_img)
            lpips2_value += lpips2(im_test2, eval_high_img)
            # count+=count
        print('psnr1=', psnr1_value / len(test_low_data_names))
        print('ssim1=', ssim1_value / len(test_low_data_names))
        print('lpips2=', lpips1_value / len(test_low_data_names))
        # print('ssim2=', ssim2_value / len(test_low_data_names))
        # print('psnr2=', psnr2_value / len(test_low_data_names))
        # print('lpips2=', lpips2_value / len(test_low_data_names))
        #
        # ssim2_value = ssim2(im2, eval_high_img)
        # psnr2_value = psnr2(im_test2, eval_high_img)
        # lpips2_value = lpips2(im_test2, eval_high_img)
        # print('单幅图去噪的 PSNR2 Value is:', psnr2_value)
        # print('单幅图增强的 SSIM2 Value is:', ssim2_value)
        # print('单幅图增强的 lpips2 Value is:', lpips2_value)

        #     ssim1_list.append(ssim1_value)
        #     lpips1_list.append(lpips1_value)
        #     psnr1_list.append(psnr1_value)
        #     ssim2_list.append(ssim2_value)
        #     psnr2_list.append(psnr2_value)
        #     lpips2_list.append(lpips2_value)
        # SSIM1_mean = np.mean(ssim1_list)
        # PSNR1_mean = np.mean(psnr1_list)
        # lpips1_mean = np.mean(lpips1_list)
        # print('照度总的 SSIM1 Value is:', SSIM1_mean)
        # print('照度总的 PSNR1 Value is:', PSNR1_mean)
        # print('照度总的 lpips1 Value is:', lpips1_mean)
        # SSIM2_mean = np.mean(ssim2_list)
        # PSNR2_mean = np.mean(psnr2_list)
        # lpips2_mean = np.mean(lpips2_list)
        # print('去噪总的 SSIM2 Value is:', SSIM2_mean)
        # print('去噪总的 PSNR2 Value is:', PSNR2_mean)
        # print('去噪总的 lpips2 Value is:', lpips2_mean)
