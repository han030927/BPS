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
from models.convnet import ConvNet
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

class wideResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, bias,return_features=False,norm='BN'):
        super(wideResNet, self).__init__()
        self.in_planes = nf
        self.norm=norm
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1) if norm == 'BN' else nn.GroupNorm(nf*1, nf*1, affine=True)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        #self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        #self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(nf * 32 * block.expansion, num_classes, bias=bias)
        self.return_features = return_features


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
       # print(len(strides))
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        '''Features before FC layers'''
        out = relu(self.bn1(self.conv1(x)))
       # print(out.size())
        out = self.layer1(out)
       # print(out.size())
        out = self.layer2(out)
      #  print(out.size())
        #out = self.layer3(out)
        # out = self.layer4(out)
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

class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()
        self.n_features = 0
        self._name = "BaseModule"

    def forward(self, x):
        return x

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    def init_weights(self, std=0.01):
        print("Initialize weights of %s with normal dist: mean=0, std=%0.2f" % (type(self), std))
        for m in self.modules():
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, 0, std)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif type(m) == nn.Conv2d:
                nn.init.normal_(m.weight, 0, std)
                if m.bias is not None:
                    m.bias.data.zero_()


class QNet(BaseModule):
    def __init__(self,
                 n_units,
                 n_classes):
        super(QNet, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(2 * n_classes, n_units),
            nn.ReLU(True),
            nn.Linear(n_units, n_classes),
        )

    def forward(self, zcat):
        zzt = self.model(zcat)
        return zzt
class DVCNet(BaseModule):
    def __init__(self,
                 backbone,
                 n_units,
                 n_classes,
                 has_mi_qnet=True):
        super(DVCNet, self).__init__()

        self.backbone = backbone
        self.has_mi_qnet = has_mi_qnet

        if has_mi_qnet:
            self.qnet = QNet(n_units=n_units,
                             n_classes=n_classes)

    def forward(self, x, xt):
        size = x.size(0)
        xx = torch.cat((x, xt))
        zz,fea = self.backbone.forward(xx,return_features=True)
        z = zz[0:size]
        zt = zz[size:]

        fea_z = fea[0:size]
        fea_zt = fea[size:]

        if not self.has_mi_qnet:
            return z, zt, None

        zcat = torch.cat((z, zt), dim=1)
        zzt = self.qnet(zcat)

        return z, zt, zzt,[torch.sum(torch.abs(fea_z), 1).reshape(-1, 1),torch.sum(torch.abs(fea_zt), 1).reshape(-1, 1)]


def Reduced_ResNet18_DVC(nclasses, nf=20, bias=True):
    """
    Reduced ResNet18 as in GEM MIR(note that nf=20).
    """
    backnone = ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, bias,return_features=True)
    return DVCNet(backbone=backnone,n_units=128,n_classes=nclasses,has_mi_qnet=True)

def Reduced_ResNet18(nclasses, nf=20, bias=True):
    """
    Reduced ResNet18 as in GEM MIR(note that nf=20).
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, bias)

def Reduced_ResNet18_IN(nclasses, nf=20, bias=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, bias, norm='IN')


def ResNet18(nclasses, nf=64, bias=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, bias)

def WideResNet18(nclasses, nf=64,bias=True):
    return wideResNet(BasicBlock, [1,1], nclasses, nf,bias,norm='IN')



def ResNet18_IN(nclasses, nf=64, bias=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, bias,norm='IN')

'''
See https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''

def ResNet34(nclasses, nf=64, bias=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], nclasses, nf, bias)

# def ResNet50(nclasses, nf=64, bias=True):
#     return ResNet(Bottleneck, [3, 4, 6, 3], nclasses, nf, bias)
#
#
# def ResNet101(nclasses, nf=64, bias=True):
#     return ResNet(Bottleneck, [3, 4, 23, 3], nclasses, nf, bias)
#
#
# def ResNet152(nclasses, nf=64, bias=True):
#     return ResNet(Bottleneck, [3, 8, 36, 3], nclasses, nf, bias)


class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, dim_in=160, head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        self.encoder = Reduced_ResNet18(100)
        #self.encoder=ResNet18_IN(feat_dim)
        #self.encoder=ResNet18(feat_dim)
        #dim_in=512
        #self.encoder = Reduced_ResNet18_IN(100)
        #self.encoder = ConvNet(feat_dim,im_size=(32,32))
        #dim_in=2048
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        elif head == 'None':
            self.head = None
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x,return_features=False):
        feat = self.encoder.features(x)
        if self.head:
            feats = F.normalize(self.head(feat), dim=1)
        else:
            feats = F.normalize(feat, dim=1)
        if return_features:
            return feats,feat
        return feats

    def features(self, x):
        return self.encoder.features(x)





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


class FastSlowSupConResNet(nn.Module):
    def __init__(self, dim_in=160, head='mlp', feat_dim=128):
        super(FastSlowSupConResNet, self).__init__()
        self.fast_encoder = SupConResNet(dim_in=dim_in, head=head, feat_dim=feat_dim)
        #self.fast_encoder =Reduced_ResNet18(100)
        self.slow_encoder = copy.deepcopy(self.fast_encoder)

        self.global_step=0
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        elif head == 'None':
            self.head = None
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x,return_features=False):
        # feat = self.fast_encoder.features(x)
        # if self.head:
        #     feats = F.normalize(self.head(feat), dim=1)
        # else:
        #     feats = F.normalize(feat, dim=1)
        # if return_features:
        #     return feats,feat
        return self.fast_encoder.forward(x,return_features=return_features)

    #@torch.no_grad()
    def slow_forward(self, x,return_features=False):
        #feat = self.slow_encoder.features(x)
        # if self.head:
        #     feats = F.normalize(self.head(feat), dim=1)
        # else:
        #     feats = F.normalize(feat, dim=1)
        # if return_features:
        #     return feats,feat
        return self.slow_encoder.forward(x,return_features=return_features)

    #@torch.no_grad()
    def update_slow_net(self, m,update_freq=1):
        self.global_step+=1

        alpha = min(1 - 1 / (self.global_step),m)
        if torch.rand(1) < update_freq:
            for (name_q, param_q), (name_k, param_k) in zip(self.fast_encoder.named_parameters(),
                                                            self.slow_encoder.named_parameters()):

                param_k.data.copy_(param_k.data * alpha + param_q.data * (1. - alpha))
                param_k.requires_grad=False


    def features(self, x):
        return self.slow_encoder.features(x)



class Mocov2_SupConResNet(nn.Module):
    def __init__(self, dim_in=160, head='mlp', feat_dim=128, K=5001, m=0.9999):
        super(Mocov2_SupConResNet, self).__init__()
        self.encoder_k = SupConResNet(dim_in=dim_in, head=head, feat_dim=feat_dim)
        # self.fast_encoder =Reduced_ResNet18(100)
        self.encoder_q = copy.deepcopy(self.encoder_k)
        self.K = K
        self.m = m
        #self.encoder = ConvNet(100)


        self.register_buffer("queue", torch.randn(feat_dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_label',torch.zeros(K).long())
        self.queue_label -= 1
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    #
    #


    def forward(self, img_q,img_k,batch_y):
        img_q_feature=self.encoder_q(img_q)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            original_indices = torch.arange(len(img_k))

            # Generate a randomly shuffled indices
            shuffled_indices = torch.randperm(len(img_k))

            # Shuffle the data and labels using the shuffled indices
            img_k_shuffled = img_k[shuffled_indices]

            _, restore_order = torch.sort(shuffled_indices)
            img_k_feature=self.encoder_k(img_k_shuffled)

            img_k_feature = img_k_feature[restore_order]

        # ptr = int(self.queue_ptr)
        # if ptr>0:
        #     queue_l=self.queue_label[:ptr]
        #     queue=self.queue[:,:ptr]
        #     indices=torch.randperm(len(queue_l))
        #     queue_l=queue_l[indices]
        #     queue=queue.t()[indices]
        #     features = torch.cat((img_q_feature, img_k_feature, queue.clone().detach()), dim=0)
        #     targets = torch.cat((batch_y, batch_y,queue_l.clone().detach()), dim=0)
        # else:
        #     features = torch.cat((img_q_feature, img_k_feature), dim=0)
        #     targets = torch.cat((batch_y, batch_y), dim=0)
        #
        #
        # features = torch.cat((img_q_feature,img_k_feature, self.queue.clone().detach().t()), dim=0)
        return img_q_feature,img_k_feature

    def features(self, x):
        return self.encoder_q(x)

    def k_features(self,x):
        return self.encoder_k(x)

    def q_logits(self,img_q):

        return self.encoder_q(img_q)
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """

        for (name_q, param_q), (name_k, param_k) in zip(self.encoder_q.named_parameters(),
                                                        self.encoder_k.named_parameters()):
            param_k.data.copy_(param_k.data * self.m + param_q.data * (1. - self.m))
            param_k.requires_grad = False



