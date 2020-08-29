from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pdb
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self, num_points = 784):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)



        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        permutation = np.array([1,0,0,0,1,0,0,0,1])
        permutation = (torch.from_numpy(permutation.astype(np.float32))).clone()
        iden = Variable(permutation).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, num_points = 784, global_feat = True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points = 784)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat
    def forward(self, x):
        batchsize = x.size()[0]
        trans = self.stn(x)
        x = x.transpose(2,1)
        x = torch.bmm(x, trans)
        x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.mp1(x)
        x = x.transpose(2,1)
        # x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
            return torch.cat([x, pointfeat], 1), trans



class PointNetEncoder(nn.Module):
    def __init__(self, num_points = 784):
        super(PointNetEncoder, self).__init__()
        self.num_points = num_points
        self.feat = PointNetfeat(num_points, global_feat=True)

    def forward(self, x):
        x, trans = self.feat(x)
        # x = F.relu(self.bn1(self.fc1(x)))
        # x = F.relu(self.bn2(self.fc2(x)))
        # x = self.fc3(x)
        # x = F.log_softmax(x)
        return x


class PoiNetDecoder(nn.Module):
    def __init__(self):
        super(PoiNetDecoder,self).__init__()
        # self.deconv1 = torch.nn.ConvTranspose1d(256,512)
        self.fc1 = nn.Linear(12, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256,784)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self,x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x

class PointNetVae(nn.Module):
    def __init__(self,encoder,decoder):
        super(PointNetVae,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encodermu = nn.Linear(256,12)
        self.encodersigama = nn.Linear(256,12)
    def sample_latent(self,latent,cudaj = True):
        mu = self.encodermu(latent)
        log_sigma = self.encodersigama(latent)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0,1,size = sigma.size())).float()

        self.mean = mu
        self.sigma = sigma
        self.log_sigma = log_sigma
        if cudaj:
            return (mu+sigma*Variable(std_z,False).cuda())
        else:
            return mu+sigma*Variable(std_z,False)
    def forward(self,x):
        latent = self.encoder(x)
        latent = torch.squeeze(latent)
        z = self.sample_latent(latent)
        return self.mean,self.log_sigma,self.decoder(z)

class PointNetCls(nn.Module):
    def __init__(self, num_points = 784, k = 2):
        super(PointNetCls, self).__init__()
        self.num_points = num_points
        self.feat = PointNetfeat(num_points, global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
    def forward(self, x):
        x, trans = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        x = F.log_softmax(x)
        return x


# def PointNet():
#     # return PointNetCls(1024,10)
#     encoder = PointNetEncoder()
#     decoder = PoiNetDecoder()
#     pointvae = PointNetVae(encoder, decoder)
#     return pointvae
# # def PVAE()



if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,784))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())

    pointfeat = PointNetfeat(global_feat=True)
    out, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointencoder = PointNetEncoder()
    out = pointencoder(sim_data)
    print("point encoder",out.size())

    encoder = PointNetEncoder()
    decoder = PoiNetDecoder()
    pointvae = PointNetVae(encoder,decoder)
    out =pointvae(sim_data)
    print('point vae',out.size())
