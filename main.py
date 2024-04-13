'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from enum import Enum

from torchsummary import summary
import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utility.utils import progress_bar
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ModelType(Enum):
    """Docstring for Models."""
    RESNET18 = "ResNet18"
    RESNET34 = "some_other_value"
    

def get_model_instance(model_type : ModelType, input_size=(3,32,32)):
    
    net = nn.Module()

    if(model_type == ModelType.RESNET18):
        net = ResNet18()
    elif(model_type == ModelType.RESNET34):
        net = ResNet34()

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    
    summary(model=net, input_size=input_size)
    return net


def get_sgd_optimizer(model : nn.Module, lr=0.05, momentum=0.9, weight_decay=1e-4) ->  optim.Optimizer:
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    return optimizer


def get_adam_optimizer(model : nn.Module, lr=0.05, weight_decay=1e-4) -> optim.Optimizer:
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer


def get_stepLR_scheduler(optimizer):
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8, verbose=True)
    return scheduler

def get_cross_entropy_loss_criteria():
    return nn.CrossEntropyLoss()


class ModelExecutor:

    MAX_IMAGES_FOR_DISPLAY = 20

    def __init__(self, train_loader, test_loader) -> None:

        self.test_loader = test_loader
        self.train_loader = train_loader

        self.best_acc = 0

        self.train_losses = []
        self.test_losses = []

        self.train_accuracy = []
        self.test_accuracy = []        

        self.correctly_predicted_trained_images = []
        self.wrongly_predicted_trained_images = []
        self.correctly_predicted_test_images = []
        self.wrongly_predicted_test_images = []

    def train(self,epoch, model, optimizer:  optim.Optimizer, criterion):
        print('\nEpoch: %d' % epoch)
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(self.train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


    def test(self, epoch, model,criterion):

        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(self.test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > self.best_acc:
            print('Saving..')
            state = {
                'net': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            self.best_acc = acc

    def execute(self, epochs, model, criterion : nn.CrossEntropyLoss, optimizer :  optim.Optimizer, scheduler: optim.lr_scheduler.LRScheduler):
        for epoch in range(epochs):
            self.train(epoch, model, optimizer, criterion)
            self.test(epoch, model, criterion)
            scheduler.step()