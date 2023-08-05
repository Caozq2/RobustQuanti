import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
import torchvision
import torch
import torch.nn as nn
import copy
from utils.quantize_model import *
import os
import time
import torchvision
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset_model(data, model):
    if data == 'Imagenet':
        pthfile = r'/model/badnets/square_white_tar0_alpha0.00_mark(6,6).pth'
        model = AlexNet().to(device)

        transform_dataset = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        poisonset = torchvision.datasets.ImageFolder(
            root='/data/',
            transform=transform_dataset)
        poisonloader = torch.utils.data.DataLoader(poisonset, batch_size=64, shuffle=True, num_workers=2)

        testset = torchvision.datasets.ImageFolder(
            root='/data/',
            transform=transform_dataset)
        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)


    model.load_state_dict(torch.load(pthfile))
    model.cuda()


    return testloader, poisonloader, model


def test(model, test_loader):
    correct1 = 0
    total1 = 0
    for data in test_loader:
        model.eval()
        images, labels = data
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total1 += labels.size(0)
        correct1 += (predicted == labels).sum()
    print((100. * correct1 / total1).item())
    print(correct1, total1)
    return (100. * correct1 / total1).item()

def get_layer_output(Model, input, layer):
    n = 0
    for i, m in enumerate(Model.modules()):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or \
                isinstance(m, nn.ReLU) or isinstance(m, nn.MaxPool2d) or isinstance(m, nn.AdaptiveAvgPool2d):
            n = n + 1
            input = m(input)
            if n == layer:
                break;
        elif isinstance(m, nn.Linear):
            n = n + 1
            input = torch.flatten(input, 1)
            input = m(input)
            if n == layer:
                break;
    return input

def get_neural(result, figure):
    c = []
    for i in result:
        if i not in c:
            c.append(i)
    b = []
    for i in c:
        num = 0
        for j in range(len(result)):
            if result[j] == i:
                num += 1
        a = []
        a.append(i)
        a.append(num)
        b.append(a)
    for i in range(len(b)):
        for j in range(i, len(b)):
            if b[i][1] < b[j][1]:
                temp = b[i]
                b[i] = b[j]
                b[j] = temp
    res = []
    for i in range(figure):
        res.append(b[i][0])

    return res

def select_neuron(layer, model, data_loader, num, figure):
    tj = []
    tj1 = []
    for data in data_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        for i in range(images.shape[0]):
            out = get_layer_output(model, images[i:i+1], layer)
            _, index = torch.topk(out[0], num, dim=0, largest=True, sorted=True)
            for i in range(num):
                tj.append(index[i].item())
                tj1.append(out[0][index[i].item()])
    neuron_list = get_neural(tj, figure)
    return neuron_list


def RobustQuanti(model, data, bit, Th_high, Th_low):
    if data == 'Imagenet':

        neuron_list = select_neuron(15, model, testloader0, Th_high, Th_low)
        index = []
        for i in range(4096):
            if i not in neuron_list:
                index.append(i)

        for i in range(len(index)):
            for j in range(320):
                model.state_dict()['fc1.weight'][index[i]][j] = 0


    model = quantize_model(model, bit)

    return model

testloader, poisonloader, model = load_dataset_model('Imagenet', 'resnet18')

test(model, poisonloader)
test(model, testloader)

model = RobustQuanti(model, 'Imagenet', 8, Th_high, Th_low)
