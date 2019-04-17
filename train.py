import os
import time

import cv2
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from dataset import TrafficDataset


N_CLASS = 2


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.n_class = N_CLASS
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(128, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.linear_layers = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, self.n_class),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


def train(trainloader, net, criterion, optimizer, device, epoch):
    '''
    Function for training.
    '''
    start = time.time()
    running_loss = 0.0
    net = net.train()
    for images, labels in tqdm(trainloader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = net(images)
        loss = calc_loss(output, labels, criterion)
        loss.backward()
        optimizer.step()
        running_loss = loss.item()
    end = time.time()
    print('[epoch %d] loss: %.3f elapsed time %.3f' %
          (epoch, running_loss, end-start))


def test(testloader, net, criterion, device):
    '''
    Function for testing.
    '''
    losses = 0.
    cnt = 0
    with torch.no_grad():
        net = net.eval()
        for images, labels in tqdm(testloader):
            images = images.to(device)
            labels = labels.to(device)
            output = net(images)
            loss = calc_loss(output, labels, criterion)
            losses += loss.item()
            cnt += 1
    print(losses / cnt)
    return (losses/cnt)


def calc_loss(output, labels, criterion):
    output = output.view(output.size(0), N_CLASS)
    labels = labels.view(labels.size(0))
    return criterion(output, labels)


def calc_score(loader, net, device):
    correct = 0
    cnt = 0
    with torch.no_grad():
        net = net.eval()
        for images, labels in tqdm(loader):
            images = images.to(device)
            labels = labels.view(labels.size(0))
            output = net(images).cpu()
            correct += torch.sum(np.argmax(output, axis=1) == labels)
            cnt += images.size(0)

    score = correct.item() / cnt
    print(score)
    return score


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)

    dataset = TrafficDataset(size=2639)
    train_and_test_data, eval_data = dataset.split(ratio=0.9)

    name = 'net_2'
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-5)

    print('\nStart training')
    for epoch in range(10):
        print('-----------------Epoch = %d-----------------' % (epoch+1))
        train_data, test_data = train_and_test_data.split(ratio=0.85)
        train_loader = DataLoader(train_data, batch_size=4)
        test_loader = DataLoader(test_data, batch_size=1)
        train(train_loader, net, criterion, optimizer, device, epoch+1)
        test(test_loader, net, criterion, device)

    train_and_test_loader = DataLoader(train_and_test_data, batch_size=1)
    eval_loader = DataLoader(eval_data, batch_size=1)
    print('\nFinished Training, Testing on test set')
    test(eval_loader, net, criterion, device)
    print('\nCalculating score on train and test set')
    calc_score(train_and_test_loader, net, device)
    print('\nCalculating score on eval set')
    calc_score(eval_loader, net, device)

    torch.save(net.state_dict(), './models/model_{}.pth'.format(name))


if __name__ == "__main__":
    main()
