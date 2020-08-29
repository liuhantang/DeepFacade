'''Train train_1311 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
import pandas as pd
import os
import argparse


from mutils import progress_bar
import mutils
import fcn8s

##### maskrcnn_model
import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import json
import pandas as pd
import glob
import PIL
from PIL import Image, ImageOps
import skimage
from skimage import draw
import h5py
import torch.utils.data
from torchvision import datasets
# import coco
from config import Config
import utils
import model as modellib
import visualize
from model import log




torch.cuda.set_device(0)
parser = argparse.ArgumentParser(description='PyTorch train_1311 Training')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')


traindataset = mutils.EcpDataset('data')
testdataset =mutils.EcpDataset('data','test')
trainloader = torch.utils.data.DataLoader(traindataset,batch_size  = 1,shuffle = True,num_workers = 4)
testloader = torch.utils.data.DataLoader(testdataset,batch_size = 1,shuffle = True,num_workers = 4)



class EcpConfig(Config):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    NAME = "fish"
    NUM_CLASSES = 1 + 1
    STEPS_PER_EPOCH = 4
    DETECTION_MIN_CONFIDENCE = 0.90
    COCO_MODEL_PATH = os.getcwd()  + '/checkpoint/mask_rcnn_coco.pth'
    VALIDATION_STEPS = 3
    BATCH_SIZE = 2
    LEARNING_RATE = 0.001
    # Necessary for docker immage to optimize memory usage best
    NUM_WORKERS = 0


# Configurations
config = EcpConfig()
# config.display()

MODEL_DIR = 'checkpoint'
model_detect = modellib.MaskRCNN(model_dir=MODEL_DIR, config=config)
model_detect.load_weights(MODEL_DIR+'/mask_rcnn_windows_0070_segmap.pth')
model_detect = model_detect.cuda()

# MODEL_DIR = 'checkpoint'
model_detect_image = modellib.MaskRCNN(model_dir=MODEL_DIR, config=config)
model_detect_image.load_weights(MODEL_DIR+'/mask_rcnn_windows_0070_image.pth')
model_detect_image = model_detect_image.cuda()

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print('we start from our previous epoch', start_epoch, 'and the acc is', best_acc)

else:
    print('==> Building model..')
    net = fcn8s.FCN8s()

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True


criterion = nn.NLLLoss2d()
# import model as modellib



def loss_function(outputs,targets,inputs):
    lcriterion = nn.NLLLoss2d()
    loss = lcriterion(F.log_softmax(outputs,1),targets)
    # print(loss.type)
    # print(loss.shape)
    inputimage = torch.squeeze(inputs).data.cpu().numpy().transpose(1,2,0)
    # print(inputimage.shape)
    global model_detect_image
    roi_results = model_detect_image.detect([inputimage])[0]['rois']
    anchors = [((roi_result[0]+roi_result[2])/2,(roi_result[1]+roi_result[3]/2)) for roi_result in roi_results if (roi_result[2]-roi_result[0]) >20 and(roi_result[3]-roi_result[1])>20 ]
    # anchors_fwh = [(roi_result[2]-roi_result[0],roi_result[3]-roi_result[1]) for roi_result in roi_results]
    def knn(segmap,anchors):
        prior_width,prior_height = 46,38
        segmap = torch.squeeze(segmap).cpu().numpy()
        anchor_box = []
        for anchor in anchors:
            x_min = int(anchor[0]-prior_width/2)
            x_max = int(anchor[0]+prior_width/2)
            y_min = int(anchor[1]-prior_height/2)
            y_max = int(anchor[1]+prior_height/2)
            # print(x_min,x_max,y_min,y_max)
            anchorbboxes = segmap[x_min:x_max,y_min:y_max] 
            x_cor,y_cor = np.where(anchorbboxes == 3)
            if len(x_cor) == 0 and len(y_cor)==0:
                # print('Liang')
                continue
            else:
                pass
                # print('Niubi')
            w,h = max(x_cor)-min(x_cor),max(y_cor)-min(y_cor)
            anchor_box.append((w,h))

        return anchor_box
    _, predicted = torch.max(outputs.data, 1)
    anchor_wh = knn(predicted,anchors)
    # print(len(anchor_wh))
    flags = np.zeros([len(anchors),])
    count = 1
    while True:
        findflag = 0
        testanchor = []
        for i in range(len(flags)):
            if flags[i]==0:
                core_anchor = anchors[i]
                testanchor.append(i)
                findflag = 1
        if not findflag:
            break
        subvertical = [core_anchor[1]-anchors[i][1] for i in testanchor]
        prop = [testanchor[idx]  for idx,sub in enumerate(subvertical) if -38 < sub < 38]
        # print('pront',prop)  
        flags[prop] = count
        if(not len(prop)):
            continue
        else:
            anchor_wh_tensor = torch.from_numpy(np.array(anchors)[prop])
            # print(anchor_wh_tensor)
            loss = loss + torch.var(anchor_wh_tensor[:,0])+torch.var(anchor_wh_tensor[:,1])
            count = count + 1
    return loss

# def citerion(inputs,targets):
#     inputs = F.log_softmax(inputs,dim=1)
#     targets =
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


# Training
# lr = args.lr
lr_begin = 5*10e-5
lr_end = 5*10e-8
import math
wt = math.pow(1000,-1.0/500)
optimizer = optim.Adam(net.parameters(),lr = args.lr)

# global model_detect_image
# optimizer_detect = optim.Adam(model_detect_image.parameters(),lr = 5*10e-6)
optimizer_detect = optim.Adam(filter(lambda p: p.requires_grad, model_detect_image.parameters()),lr = 5*10e-6)
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    # for
    for optim_para in optimizer.param_groups:
        optim_para['lr'] = optim_para['lr']*wt
        cg = optim_para['lr']
    print('lr',cg)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        
        test_loss = loss_function(outputs,targets,inputs)
        test_loss.backward(retain_graph=True)
        optimizer_detect.step()
        # test_loss.backward()

        # print(outputs.shape)
        # print(targets.shape)
        loss = criterion(F.log_softmax(outputs,1), targets)
        # print(loss.shape)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)*targets.size(1)*targets.size(2)
        correct += predicted.eq(targets.data).cpu().sum()
        mutils.saveimage(predicted.cpu().numpy(),'train.png')
        mutils.saveimage(targets.data.cpu().numpy(),'test.png')
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    train_error_list.append(100.*correct/total)
    train_loss_list.append(train_loss/(batch_idx+1))
    time_list.append(mutils.tot_time)
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    total_modify = 0
    correct_modify = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # print(targets.shape)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        # print(outputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)*targets.size(1)*targets.size(2)
        correct += predicted.eq(targets.data).cpu().sum()
        # print(type(predicted))
        # modifyout = []
        # for idx,input in enumerate(inputs):
        #     in_data = input.cpu().data.numpy().transpose(1,2,0)
        #     # print(in_data)
        #     # input()
        #     result = model_detect.detect([in_data])[0]['rois']
        #     # print(result)
        #     mout = outputs[idx].cpu().data.numpy().copy()
        #     for re_cor in result:
        #         mout[re_cor[0]:re_cor[2], re_cor[1]:re_cor[3]] = 3
        #     modifyout.append(mout)
        #
        # # print(targets.data[0])
        #
        # modify_target = torch.from_numpy(np.array(modifyout)).type(torch.LongTensor).cuda()
        # total_modify += targets.size(0) * targets.size(1) * targets.size(2)
        # # print(type(modify_target))
        # # print(type(targets.data))
        # correct_modify += modify_target.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (test_loss / (batch_idx + 1), 100. * correct_modify / total, correct_modify, total))
        #
        # if correct_modify > correct:
        #     print('modify high')
        # else:
        #     print('org high')
        # print('max_acc %.3f%%'% max(00. * correct_modify / total,100.*correct/total,))
    test_error_list.append(100.*correct/total)
    test_loss_list.append(test_loss/(batch_idx+1))
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc


def val(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    total_modify = 0
    correct_modify = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # print(targets.shape)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        # print(outputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)*targets.size(1)*targets.size(2)
        correct += predicted.eq(targets.data).cpu().sum()
        # print(type(predicted))


        modifyout = []
        for idx,input in enumerate(predicted):
            #in_data = input.cpu().data.numpy().transpose(1,2,0)
            # print(in_data)
            # input()
            in_data = np.squeeze(predicted.cpu().numpy())
            # print(in_data)
            # label = [0,29,67,76,128,149,151,217,225]
            label = np.array([ [255, 255,   0],
                [128 ,255, 255]
                ,[255 ,128 ,  0]
                ,[  0 ,255,   0]
                ,[128 ,128 ,128]
                ,[255  , 0  , 0]
                ,[128  , 0 ,255]
                ,[  0  , 0 ,255]
                ,[0, 0 ,0]])
            lin_data = np.zeros([in_data.shape[0],in_data.shape[1],3],dtype = 'uint8')
            print(in_data.shape)
            for i in range(in_data.shape[0]):
                for j in range(in_data.shape[1]):
                    lin_data[i,j] = label[8-in_data[i,j]]
            lin_data = np.uint8(lin_data)
            print(lin_data) 
            print(lin_data.shape)
            result = model_detect.detect([lin_data])[0]['rois']
            # print(result)

            mout = predicted[idx].cpu().numpy().copy()
            # print(mout.shape)
            for re_cor in result:
                mout[re_cor[0]:re_cor[2], re_cor[1]:re_cor[3]] = 3
            modifyout.append(mout)
      
        #
        # # print(targets.data[0])
        #
        modify_target = torch.from_numpy(np.array(modifyout)).type(torch.LongTensor).cuda()
        total_modify += targets.size(0) * targets.size(1) * targets.size(2)
        # print(type(modify_target))
        # # print(type(targets.data))
        correct_modify += modify_target.eq(targets.data).cpu().sum()
        
        mutils.saveimage(modify_target.cpu().numpy(),'rpnrefine_'+str(batch_idx)+'.png')
        # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            # % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), 100. * correct_modify / total, correct_modify, total))
        #
        # if correct_modify > correct:
        #     print('modify high')
        # else:
        #     print('org high')
        # print('max_acc %.3f%%'% max(00. * correct_modify / total,100.*correct/total,))
    val_error_list.append(100.*correct/total)
    val_loss_list.append(test_loss/(batch_idx+1))
    # Save checkpoint.
    acc = 100.*correct_modify/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc
 






# print(model_detect)
# results = model.detect([original_image])

# r = results[0]
# visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
#                             class_names, r['scores'], ax=get_ax())




# pic_file = 'ECP/images/monge_1.jpg'
# original_image = plt.imread(pic_file)
# # plt.imshow(original_image)
# # plt.show()

# def get_ax(rows=1, cols=1, size=8):
#     """Return a Matplotlib Axes array to be used in
#     all visualizations in the notebook. Provide a
#     central point to control graph sizes.

#     Change the default size attribute to control the size
#     of rendered images
#     """
#     _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
#     return ax

# results = model_detect.detect([original_image])

# r = results[0]
# print(r['rois'])
# visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],'window', r['scores'], ax=get_ax())

epoch_list = []
train_error_list = []
test_error_list = []
time_list = []
train_loss_list = []
test_loss_list = []
val_error_list = []
val_loss_list = []
for epoch in range(start_epoch, start_epoch+500):
    epoch_list.append(epoch)
    train(epoch)
    test(epoch)
    val(epoch)
    df = pd.DataFrame(data={'epoch': epoch_list, 'train_error': train_error_list,
                           'test_error': test_error_list,'tot_time':time_list,'train_loss':train_loss_list,'test_loss':test_loss_list,'val_loss':val_loss_list,'val_error':val_error_list})
    df.to_csv('data_error.csv')
