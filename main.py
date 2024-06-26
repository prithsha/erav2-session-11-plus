
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from enum import Enum
from collections import deque
from torchsummary import summary
import matplotlib.pyplot as plt



from models import ResNet18, ResNet34
from utility import utils
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ModelType(Enum):
    """Docstring for Models."""
    RESNET18 = "ResNet18"
    RESNET34 = "Resnet34"


def get_model_instance(model_type: ModelType, input_size=(3, 32, 32), show_summary = True):
    model = nn.Module()
    if (model_type == ModelType.RESNET18):
        model = ResNet18()
    elif (model_type == ModelType.RESNET34):
        model = ResNet34()
    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    if(show_summary):
        summary(model=model, input_size=input_size)
    return model


def get_sgd_optimizer(model: nn.Module, lr=0.05, momentum=0.9, weight_decay=1e-4) -> optim.Optimizer:
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=momentum, weight_decay=weight_decay)
    return optimizer


def get_adam_optimizer(model: nn.Module, lr=0.05, weight_decay=1e-4) -> optim.Optimizer:
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    return optimizer


def get_stepLR_scheduler(optimizer):
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.8, verbose=True)
    return scheduler

def get_one_cycle_lr_scheduler(optimizer, max_lr, pct_start,
                               steps_per_epoch,  epochs ):
    
    return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr,
                                                pct_start = pct_start,
                                                div_factor=10,
                                                final_div_factor=1000,
                                                anneal_strategy = "linear",
                                                steps_per_epoch=steps_per_epoch,
                                                epochs=epochs)


def get_cross_entropy_loss_criteria():
    return nn.CrossEntropyLoss()


class NetworkModelEvaluator:

    MAX_IMAGES_FOR_DISPLAY = 20

    def __init__(self, train_loader, test_loader) -> None:

        self.test_loader = test_loader
        self.train_loader = train_loader

        self.best_acc = 0

        self.train_losses = []
        self.test_losses = []

        self.train_accuracy = []
        self.test_accuracy = []

        # self.correctly_predicted_trained_images = deque(maxlen=ModelExecutor.MAX_IMAGES_FOR_DISPLAY)
        self.wrongly_predicted_trained_images = deque(
            maxlen=NetworkModelEvaluator.MAX_IMAGES_FOR_DISPLAY)
        # self.correctly_predicted_test_images = deque(maxlen=ModelExecutor.MAX_IMAGES_FOR_DISPLAY)
        self.wrongly_predicted_test_images = deque(
            maxlen=NetworkModelEvaluator.MAX_IMAGES_FOR_DISPLAY)

    def train(self, epoch, model, optimizer:  optim.Optimizer, criterion):
        print('\nEpoch: %d' % epoch)
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        last_time = time.time()

        tqdm_batches = tqdm(enumerate(self.train_loader),
                            desc="Train batches", total=len(self.train_loader))
        for batch_idx, (inputs, targets) in tqdm_batches:
            begin_time = time.time()
        # for batch_idx, (inputs, targets) in enumerate(self.train_loader):
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

            argmax_data = outputs.argmax(dim=1)
            _, false_indices = utils.get_true_and_false_indices(
                outputs, targets)
            for image_index in false_indices:
                selected_image = inputs[image_index]
                self.wrongly_predicted_trained_images.appendleft(
                    (selected_image, argmax_data[image_index]))

            
            cur_time = time.time()
            total_time = cur_time - last_time
            step_time = cur_time - begin_time
            progress_description = f"Train: {batch_idx, len(self.train_loader)} Loss: {train_loss/(batch_idx+1): 0.3f} | Acc: {100.*correct/total: 0.3f}%, {correct}, {total}, Total : Step time[{utils.format_time(total_time)} : {utils.format_time(step_time)}]"            
            tqdm_batches.set_description(desc=progress_description)
        
        self.train_losses.append(train_loss)
        self.train_accuracy.append(100*correct/total)


    def test(self, epoch, model, criterion):

        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():

            last_time = time.time()
            tqdm_batches = tqdm(enumerate(self.test_loader),
                                desc="Test batches", total=len(self.test_loader))
            for batch_idx, (inputs, targets) in tqdm_batches:
                # for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                begin_time = time.time()
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                argmax_data = outputs.argmax(dim=1)
                _, false_indices = utils.get_true_and_false_indices(
                    outputs, targets)
                for image_index in false_indices:
                    selected_image = inputs[image_index]
                    self.wrongly_predicted_test_images.appendleft(
                        (selected_image, argmax_data[image_index]))

                
                cur_time = time.time()
                total_time = cur_time - last_time
                step_time = cur_time - begin_time
                progress_description = f"Test:  {batch_idx, len(self.test_loader)} Loss: {test_loss/(batch_idx+1): 0.3f} | Acc: {100.*correct/total: 0.3f}%, {correct}, {total}, Total : Step time[{utils.format_time(total_time)} : {utils.format_time(step_time)}]"
                tqdm_batches.set_description(desc=progress_description)
            
            self.test_losses.append(test_loss)
            self.test_accuracy.append(100*correct/total)

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

    def execute(self, epochs, model, criterion: nn.CrossEntropyLoss, optimizer:  optim.Optimizer, scheduler: optim.lr_scheduler.LRScheduler):
        start_time = time.time()
        for epoch in range(epochs):
            self.train(epoch, model, optimizer, criterion)
            self.test(epoch, model, criterion)
            scheduler.step()
        end_time = time.time()

        print(
            f"Finished! Total execution time: {utils.format_time(end_time-start_time)}")


    def show_train_and_test_accuracy_and_losses(self):
        _, axs = plt.subplots(2,2,figsize=(8,8))
        axs[0, 0].plot(self.train_losses)
        axs[0, 0].set_title("Training Loss")
        axs[1, 0].plot(self.train_accuracy)
        axs[1, 0].set_title("Training Accuracy")
        axs[0, 1].plot(self.test_losses)
        axs[0, 1].set_title("Test Loss")
        axs[1, 1].plot(self.test_accuracy)
        axs[1, 1].set_title("Test Accuracy")




