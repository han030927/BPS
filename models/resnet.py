"""
Code adapted from https://github.com/facebookresearch/GradientEpisodicMemory
                    &
                  https://github.com/kuangliu/pytorch-cifar
"""
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn.functional import relu, avg_pool2d
import copy

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm='BN'):
        super(BasicBlock, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'IN' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'IN' else nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(self.expansion*planes, self.expansion*planes, affine=True) if self.norm == 'IN' else nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm='BN'):
        super(Bottleneck, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'IN' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'IN' else nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(self.expansion*planes, self.expansion*planes, affine=True) if self.norm == 'IN' else nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(self.expansion*planes, self.expansion*planes, affine=True) if self.norm == 'IN' else nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, bias,return_features=False,norm='BN'):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.norm=norm
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1) if norm == 'BN' else nn.GroupNorm(nf*1, nf*1, affine=True)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes, bias=bias)
        self.return_features = return_features


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        '''Features before FC layers'''
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def logits(self, x):
        '''Apply the last FC linear mapping to get logits'''
        x = self.linear(x)
        return x

    def forward(self, x,return_features=False):
        out = self.features(x)
        logits = self.logits(out)
        if return_features:
            return logits,out
        else:
            return logits





def Reduced_ResNet18(nclasses, nf=20, bias=True):
    """
    Reduced ResNet18 as in GEM MIR(note that nf=20).
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, bias)

def Reduced_ResNet18_IN(nclasses, nf=20, bias=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, bias, norm='IN')


def ResNet18(nclasses, nf=64, bias=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, bias)




def ResNet18_IN(nclasses, nf=64, bias=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, bias,norm='IN')

'''
See https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''

def ResNet34(nclasses, nf=64, bias=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], nclasses, nf, bias)



class MeanTeacherReducedResNet(nn.Module):

    def __init__(self, num_classes,data='cifar'):
        super(MeanTeacherReducedResNet, self).__init__()
        self.fast_net=Reduced_ResNet18(100)
        if data != 'cifar':
            self.fast_net.linear=nn.Linear(640, num_classes, bias=True)
        self.slow_net=copy.deepcopy(self.fast_net)
        self.global_step = 0

    def forward(self, x, return_features=False):

        return self.slow_net.forward(x, return_features=return_features)

    # @torch.no_grad()
    def fast_forward(self, x, return_features=False):
        return self.fast_net.forward(x, return_features=return_features)

    @torch.no_grad()
    def update_slow_net(self, m, update_freq=1):
        self.global_step += 1

        alpha = min(1 - 1 / (self.global_step), m)
        if torch.rand(1) < update_freq:
            for (name_q, param_q), (name_k, param_k) in zip(self.fast_net.named_parameters(),
                                                            self.slow_net.named_parameters()):
                param_k.data.copy_(param_k.data * alpha + param_q.data * (1. - alpha))
                param_k.requires_grad = False

    def features(self, x):
        return self.slow_net.features(x)



