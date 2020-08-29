import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.utils.data as Data
from torchvision import datasets,models,transforms
from torch.utils import data
from PIL import Image
import numpy as np
import torch

# 0,  29,  67,  76, 128, 149, 151, 217, 225
class EcpDataset(data.Dataset):
    def __init__(self,root,type = 'train',transform = True):
        self.root = root
        self._transform = transform
        self.files = None
        self.labels = None
        self.label_dict = {
            0:0,
            29:1,
            67:2,
            76:3,
            128:4,
            149:5,
            151:6,
            217:7,
            225:8
        }
        image_filelist = os.listdir(self.root+'/images')
        label_filelist = os.listdir(self.root+'/labels')
        filterlist = list(map(lambda x:''.join(x.replace('.jpg','_mask.png')),image_filelist))
        image_filelist = list(map(lambda x:x.replace('_mask.png','.jpg'), list(set(filterlist)&set(label_filelist))))
        self.image_filelist = image_filelist[:80] if type == 'train' else image_filelist[80:]
    def __len__(self):
        return len(self.image_filelist)
    def __getitem__(self,index):
        img_path = self.root+'/images/'+self.image_filelist[index]
        lbl_path = self.root+'/labels/'+self.image_filelist[index].replace('.jpg','_mask.png')
        img = np.array(Image.open(img_path))
        img = img.transpose(2,0,1)
        lbl = self.process_label(lbl_path)
        img = torch.from_numpy(img).float()
        # img = img.permute(0,2,3,1)
        lbl = torch.from_numpy(lbl).type(torch.LongTensor)
        return img,lbl
    def process_label(self,lbl_filename):
        label = np.array(Image.open(lbl_filename).convert('L'))
        relabel = np.zeros([label.shape[0],label.shape[1]])
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
#                 print(label[i,j])
                relabel[i,j] = self.label_dict[label[i,j]]
        return relabel

# gg = EcpDataset('data')
# print(gg[:9])

def loaddata(imagepath,labelpath,kind = True):
    if kind:
        x = 60000
    else: x = 10000
    image = np.fromfile(imagepath,'float').reshape(x,784,3).transpose(0,2,1)
    label = np.fromfile(labelpath,'float').reshape(x,784)
  
    images = torch.from_numpy(image).type(torch.FloatTensor)
    labels = torch.from_numpy(label).type(torch.FloatTensor)
    
    dataset = Data.TensorDataset(data_tensor=images, target_tensor=labels)
    loader = Data.DataLoader(dataset=dataset, batch_size=128, shuffle=True)
    return images,labels,loader



def saveimage(grayimage,filename):
    # print(grayimage)
    grayimage = np.squeeze(grayimage)
    nummatrix = np.array([ [255, 255,   0],
    [128 ,255, 255]
    ,[255 ,128 ,  0]
    ,[  0 ,255,   0]
    ,[128 ,128 ,128]
    ,[255  , 0  , 0]
    ,[128  , 0 ,255]
    ,[  0  , 0 ,255]
    ,[0, 0 ,0]])
    numdict = {
        0:nummatrix[8],
        1:nummatrix[7],
        2:nummatrix[6],
        3:nummatrix[5],
        4:nummatrix[4],
        5:nummatrix[3],
        6:nummatrix[2],
        7:nummatrix[1],
        8:nummatrix[0],
    }
    image = np.array(list(map(lambda x:nummatrix[8-x],grayimage)))
    Image.fromarray(np.uint8(image)).save(filename)
def get_mean_and_std(dataset):
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

class TripletLossFunc(nn.Module):
    def __init__(self,anchor,positive,negative,beta):
        super(TripletLossFunc,self).__init__()
        self.anchor = anchor
        self.positive = positive
        self.negative = negative
        self.beta = beta
    def forward(self):
        matched = torch.pow(self.anchor-self.positive,2)
        mimatched = torch.pow(self.anchor-self.negative,2)
        distance = matched-mimatched+self.beta
        loss = torch.max(distance,0)
        return loss

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time,tot_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
