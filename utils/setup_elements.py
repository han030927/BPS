import torch
from models.resnet import Reduced_ResNet18, ResNet18,MeanTeacherReducedResNet
from torchvision import transforms
import torch.nn as nn


default_trick = {'labels_trick': False, 'kd_trick': False, 'separated_softmax': False,
                 'review_trick': False, 'ncm_trick': False, 'kd_trick_star': False}


input_size_match = {
    'cifar100': [3, 32, 32],
    'cifar10': [3, 32, 32],
    'core50': [3, 128, 128],
    'mini_imagenet': [3, 84, 84],
    'tiny_imagenet': [3, 64, 64],
    'openloris': [3, 50, 50]
}


n_classes = {
    'cifar100': 100,
    'cifar10': 10,
    'core50': 50,
    'mini_imagenet': 100,
    'tiny_imagenet': 200,
    'openloris': 69
}
from PIL import Image


class ToPILImage(object):
    def __call__(self, pic):
        return Image.fromarray(pic)

transforms_match = {
    'core50': transforms.Compose([
        transforms.ToTensor(),
        ]),
    'cifar100': transforms.Compose([
        transforms.ToTensor(),
        ]),
    'cifar10': transforms.Compose([
        transforms.ToTensor(),
        ]),
    'mini_imagenet': transforms.Compose([
        transforms.ToTensor()]),
    'tiny_imagenet':
        transforms.Compose([
        transforms.ToTensor()]),

    'openloris': transforms.Compose([
            transforms.ToTensor()])
}


def setup_architecture(params):
    nclass = n_classes[params.data]



    if params.agent =='BPS':
        if 'imagenet' in params.data:
            return MeanTeacherReducedResNet(nclass,'imagenet')
        else:
            return MeanTeacherReducedResNet(nclass)

    if params.data == 'cifar100':

        return Reduced_ResNet18(nclass)

    elif params.data == 'cifar10':
        return Reduced_ResNet18(nclass)
    elif params.data == 'core50':
        model = Reduced_ResNet18(nclass)
        model.linear = nn.Linear(2560, nclass, bias=True)
        return model
    elif params.data == 'mini_imagenet' or params.data == 'tiny_imagenet':

        model = Reduced_ResNet18(nclass)
        model.linear = nn.Linear(640, nclass, bias=True)

        return model
    elif params.data == 'openloris':
        return Reduced_ResNet18(nclass)


def setup_opt(optimizer, model, lr, wd):
    if optimizer == 'SGD':
        optim = torch.optim.SGD(model.parameters(),
                                lr=lr,
                                weight_decay=wd)
    elif optimizer == 'Adam':
        optim = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 betas=(0.9, 0.99),
                                 weight_decay=wd)
    else:
        raise Exception('wrong optimizer name')
    return optim
