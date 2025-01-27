import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from timm.loss import LabelSmoothingCrossEntropy

from torch.autograd import Variable



def kl_div_with_logit(q_logit, p_logit):

    q = F.softmax(q_logit, dim=1)
    logq = F.log_softmax(q_logit, dim=1)
    logp = F.log_softmax(p_logit, dim=1)

    qlogq = ( q *logq).sum(dim=1).mean(dim=0)
    qlogp = ( q *logp).sum(dim=1).mean(dim=0)

    return qlogq - qlogp


def _l2_normalize(d):

    d = d.detach().numpy()
    d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2, 3))).reshape((-1, 1, 1, 1)) + 1e-16)
    return torch.from_numpy(d)

def cross_entropy_loss(z, zt, ytrue, label_smoothing=0):
    zz = torch.cat((z, zt))
    yy = torch.cat((ytrue, ytrue))
    # if label_smoothing > 0:
    #     ce = LabelSmoothingCrossEntropy(label_smoothing)(zz, yy)
    # else:
    ce = nn.CrossEntropyLoss()(zz, yy)
    return ce


def cross_entropy(z, zt):
    # eps = np.finfo(float).eps
    Pz = F.softmax(z, dim=1)
    Pzt = F.softmax(zt, dim=1)
    # make sure no zero for log
    # Pz  [(Pz   < eps).data] = eps
    # Pzt [(Pzt  < eps).data] = eps
    return -(Pz * torch.log(Pzt)).mean()


def agmax_loss(y, ytrue, dl_weight=1.0):
    z, zt, zzt,_ = y
    Pz = F.softmax(z, dim=1)
    Pzt = F.softmax(zt, dim=1)
    Pzzt = F.softmax(zzt, dim=1)

    dl_loss = nn.L1Loss()
    yy = torch.cat((Pz, Pzt))
    zz = torch.cat((Pzzt, Pzzt))
    dl = dl_loss(zz, yy)
    dl *= dl_weight

    # -1/3*(H(z) + H(zt) + H(z, zt)), H(x) = -E[log(x)]
    entropy = entropy_loss(Pz, Pzt, Pzzt)
    return entropy, dl




def clamp_to_eps(Pz, Pzt, Pzzt):
    eps = np.finfo(float).eps
    # make sure no zero for log
    Pz[(Pz < eps).data] = eps
    Pzt[(Pzt < eps).data] = eps
    Pzzt[(Pzzt < eps).data] = eps

    return Pz, Pzt, Pzzt


def batch_probability(Pz, Pzt, Pzzt):
    Pz = Pz.sum(dim=0)
    Pzt = Pzt.sum(dim=0)
    Pzzt = Pzzt.sum(dim=0)

    Pz = Pz / Pz.sum()
    Pzt = Pzt / Pzt.sum()
    Pzzt = Pzzt / Pzzt.sum()

    # return Pz, Pzt, Pzzt
    return clamp_to_eps(Pz, Pzt, Pzzt)


def entropy_loss(Pz, Pzt, Pzzt):
    # negative entropy loss
    Pz, Pzt, Pzzt = batch_probability(Pz, Pzt, Pzzt)
    entropy = (Pz * torch.log(Pz)).sum()
    entropy += (Pzt * torch.log(Pzt)).sum()
    entropy += (Pzzt * torch.log(Pzzt)).sum()
    entropy /= 3
    return entropy


def vat_loss(model, ul_x, ul_y, xi=1e-6, eps=2.5, num_iters=1):
    # find r_adv
    d = torch.Tensor(ul_x.size()).normal_()
    for i in range(num_iters):
        d = xi *_l2_normalize(d)
        d = Variable(d.cuda(), requires_grad=True)
        y_hat = model(ul_x + d)
        delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)
        delta_kl.backward()
        d = d.clone().cpu()
        model.zero_grad()
    d = _l2_normalize(d)
    d = Variable(d.cuda())
    r_adv = eps * d
    # compute lds
    y_hat = model(ul_x + r_adv.detach())
    delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)
    return delta_kl
