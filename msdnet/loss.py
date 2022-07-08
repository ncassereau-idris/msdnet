#-----------------------------------------------------------------------
#Copyright 2020 Centrum Wiskunde & Informatica, Amsterdam
#
#Author: Daniel M. Pelt
#Contact: D.M.Pelt@cwi.nl
#Website: http://dmpelt.github.io/msdnet/
#License: MIT
#
#This file is part of MSDNet, a Python implementation of the
#Mixed-Scale Dense Convolutional Neural Network.
#-----------------------------------------------------------------------

"""Module for training and validation loss functions."""

from . import operations
import abc
import torch
import torch.nn.functional as F
import math
import numpy as np

class Loss(abc.ABC):
    '''Base loss class
    
    Computes loss function and its derivative.
    '''

    @abc.abstractmethod
    def loss(self, im, tar):
        '''Computes loss function for each pixel. To be implemented by each class.

        :param im: network output image
        :param tar: target image
        :returns: image of loss function values
        '''
        pass

    def lossvalue(self, im, tar, msk):
        '''Computes loss function.

        :param im: network output image
        :param tar: target image
        :param msk: mask image (or None)
        :return: loss function value
        '''
        vals = self.loss(im, tar)
        if msk is None:
            return operations.sum(vals)
        else:
            return operations.masksum(vals, msk)

    @abc.abstractmethod
    def deriv(self, im, tar):
        '''Computes derivative of loss function. To be implemented by each class.

        :param im: network output image
        :param tar: target image
        :return: image of loss function derivative values
        '''
        pass


class SSIM(Loss, torch.nn.Module):

    def __init__(self, window_size=11, channel=1, size_average=True, sigma=1.5, C1=0.01**2, C2=0.03**2):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.sigma = sigma
        self.C1 = C1
        self.C2 = C2
        self.padding = self.window_size // 2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.window = self.create_window(window_size, channel).to(self.device)

    def gaussian(self, window_size):
        gauss = torch.Tensor([
            math.exp(-(x - window_size//2) ** 2 / float(2 * self.sigma ** 2))
            for x in range(window_size)
        ])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def conv2d(self, X):
        return F.conv2d(
            X,
            self.window,
            padding=self.padding,
            groups=self.channel
        )

    def ssim(self, img1, img2):
        mu1 = self.conv2d(img1)
        mu2 = self.conv2d(img2)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)

        mu1_mu2 = mu1 * mu2

        sigma1_sq = self.conv2d(img1 * img1) - mu1_sq
        sigma2_sq = self.conv2d(img2 ** 2) - mu2_sq

        sigma12 = self.conv2d(img1 * img2) - mu1_mu2

        U = (2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)
        V = (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        ssim_map = U / V

        loss = torch.clamp(1. - ssim_map, min=0, max=1) / 2.

        if self.size_average:
            return loss.mean()
        else:
            return loss.mean((1, 2, 3))

    def np2torch(self, image):
        return torch.from_numpy(image).to(self.device)[None] # Add a batch dimension

    def loss(self, im, tar):
        with torch.no_grad():
            im = self.np2torch(im)
            tar = self.np2torch(tar)
            output = self.ssim(im, tar).detach().cpu().numpy()
        return output

    def deriv(self, im, tar):
        im = self.np2torch(im).requires_grad_()
        tar = self.np2torch(tar)
        output = self.ssim(im, tar)
        grad, = torch.autograd.grad(output, im)
        return grad[0].cpu().numpy() # remove batch dimension and convert to numpy

    forward = loss

class L2Loss(Loss):
    '''Computes L2-norm loss function.
    '''

    def loss(self, im, tar):
        err = np.zeros_like(tar)
        operations.squarediff(err, im, tar)
        return err
        

    def deriv(self, im, tar):
        err = np.zeros_like(tar)
        operations.diff(err, im, tar)
        return err

class CrossEntropyLoss(Loss):
    '''Computes cross entropy loss function for one-hot data.
    '''

    def loss(self, im, tar):
        err = np.zeros_like(tar)
        operations.crossentropylog(err, im, tar)
        return err
        

    def deriv(self, im, tar):
        err = np.zeros_like(tar)
        operations.crossentropyderiv(err, im , tar)
        return err

