import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
import torchvision
import torch
import torch.nn as nn
import copy
from utils.quantize_model import *
import time
import torchvision


def load_dataset_model(data, model):
    if data == 'Imagenet':
        pthfile = r'/model/badnets/square_white_tar0_alpha0.00_mark(6,6).pth'
        model = AlexNet()

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



def get_neural(result, Th_low):
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
    for i in range(Th_low):
        res.append(b[i][0])

    return res


def select_neuron(model, data_loader, Th_high, Th_low):
    tj = []
    for data in data_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        for i in range(images.shape[0]):

            features = []

            def hook(module, inputdata, output):
                # print(output.data.shape)
                features.append(output.data)

            #            handle = model.features.layer4[1].conv2.register_forward_hook(hook)#输入目标层的输出  gtsrb
            handle = model.features.layer4[2].conv2.register_forward_hook(hook)  # 输入目标层的输出 vggface2
            #            handle = model.features.layer4[1].conv2.register_forward_hook(hook)  # 输入目标层的输出 imagenet
            model(images[i:i + 1])
            handle.remove()
            #            _, index = torch.topk(features[0], num, dim=1, largest=True, sorted=True)   #gtsrb
            _, index = torch.topk(features[0][0].mean(dim=-1).mean(dim=-1), Th_high, dim=0, largest=True,
                                  sorted=True)  # vggface2
            #            _, index = torch.topk(features[0], num, dim=0, largest=True, sorted=True)
            #            index = torch.stack(index)
            print(type(index))
            print(index.shape)
            print(index[0])
            print(index[0].shape)

            for i in range(T1):
                tj.append(index[0][i].item())
    neuron_list = get_neural(tj, Th_low)
    return neuron_list


def RobustQuanti(model, data, bit, Th_high, Th_low):
    if data == 'Imagenet':

        neuron_list = select_neuron(15, model, testloader, Th_high, Th_low)
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

model = RobustQuanti(model, 'Imagenet', 8, Th_high, Th_low)
