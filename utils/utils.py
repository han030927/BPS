import torch
import time
from torch.utils.data import Dataset
import torch.nn.functional as F
def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
def downscale(image_syn, scale_factor):
    image_syn=image_syn.unsqueeze(0)
    image_syn = F.upsample(image_syn, scale_factor=scale_factor, mode='bilinear')
    return image_syn.squeeze(0)
def maybe_cuda(what, use_cuda=True, **kw):
    """
    Moves `what` to CUDA and returns it, if `use_cuda` and it's available.
        Args:
            what (object): any object to move to eventually gpu
            use_cuda (bool): if we want to use gpu or cpu.
        Returns
            object: the same object but eventually moved to gpu.
    """

    if use_cuda is not False and torch.cuda.is_available():
        what = what.cuda()
    return what
def number_sign_augment(image_syn, label_syn):
    half_length = image_syn.shape[2]//2
    # import pdb; pdb.set_trace()
    a, b, c, d = image_syn[:, :, :half_length, :half_length].clone(), image_syn[:, :, half_length:, :half_length].clone(), image_syn[:, :, :half_length, half_length:].clone(), image_syn[:, :, half_length:, half_length:].clone()
    a, b, c, d = F.upsample(a, scale_factor=2, mode='bilinear'), F.upsample(b, scale_factor=2, mode='bilinear'), \
        F.upsample(c, scale_factor=2, mode='bilinear'), F.upsample(d, scale_factor=2, mode='bilinear')
    # a, b, c, d = image_syn.clone(), image_syn.clone(), image_syn.clone(), image_syn.clone()
    image_syn_augmented = torch.concat([a, b, c, d], dim=0)
    label_syn_augmented = label_syn.repeat(4)
    return image_syn_augmented, label_syn_augmented

class OLDTensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()


    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]

class TensorDataset(Dataset):
    def __init__(self, images, labels,logits=None): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()
        if logits is not None:
            self.logits = logits.detach().float()

    def __getitem__(self, index):
        return self.images[index], self.labels[index], self.logits[index]

    def __len__(self):
        return self.images.shape[0]


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.sum += val * n
        self.count += n

    def avg(self):
        if self.count == 0:
            return 0
        return float(self.sum) / self.count


def mini_batch_deep_features(model, total_x, num):
    """
        Compute deep features with mini-batches.
            Args:
                model (object): neural network.
                total_x (tensor): data tensor.
                num (int): number of data.
            Returns
                deep_features (tensor): deep feature representation of data tensor.
    """
    is_train = False
    if model.training:
        is_train = True
        model.eval()
    if hasattr(model, "features"):
        model_has_feature_extractor = True
    else:
        model_has_feature_extractor = False
        # delete the last fully connected layer
        modules = list(model.children())[:-1]
        # make feature extractor
        model_features = torch.nn.Sequential(*modules)

    with torch.no_grad():
        bs = 64
        num_itr = num // bs + int(num % bs > 0)
        sid = 0
        deep_features_list = []
        for i in range(num_itr):
            eid = sid + bs if i != num_itr - 1 else num
            batch_x = total_x[sid: eid]

            if model_has_feature_extractor:
                batch_deep_features_ = model.features(batch_x)
            else:
                batch_deep_features_ = torch.squeeze(model_features(batch_x))

            deep_features_list.append(batch_deep_features_.reshape((batch_x.size(0), -1)))
            sid = eid
        if num_itr == 1:
            deep_features_ = deep_features_list[0]
        else:
            deep_features_ = torch.cat(deep_features_list, 0)
    if is_train:
        model.train()
    return deep_features_


def euclidean_distance(u, v):
    euclidean_distance_ = (u - v).pow(2).sum(1)
    return euclidean_distance_


def ohe_label(label_tensor, dim, device="cpu"):
    # Returns one-hot-encoding of input label tensor
    n_labels = label_tensor.size(0)
    zero_tensor = torch.zeros((n_labels, dim), device=device, dtype=torch.long)
    return zero_tensor.scatter_(1, label_tensor.reshape((n_labels, 1)), 1)


def nonzero_indices(bool_mask_tensor):
    # Returns tensor which contains indices of nonzero elements in bool_mask_tensor
    return bool_mask_tensor.nonzero(as_tuple=True)[0]


class EarlyStopping():
    def __init__(self, min_delta, patience, cumulative_delta):
        self.min_delta = min_delta
        self.patience = patience
        self.cumulative_delta = cumulative_delta
        self.counter = 0
        self.best_score = None

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.min_delta:
            if not self.cumulative_delta and score > self.best_score:
                self.best_score = score
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.counter = 0
        return False

    def reset(self):
        self.counter = 0
        self.best_score = None
