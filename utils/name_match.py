from continuum.dataset_scripts.cifar100 import CIFAR100
from continuum.dataset_scripts.cifar10 import CIFAR10
from continuum.dataset_scripts.core50 import CORE50
from continuum.dataset_scripts.mini_imagenet import Mini_ImageNet
from continuum.dataset_scripts.tiny_imagenet import TinyImageNet
from continuum.dataset_scripts.openloris import OpenLORIS
from utils.buffer.random_retrieve import Random_retrieve
from utils.buffer.reservoir_update import Reservoir_update

from agents.BPS import BPSReplay


data_objects = {
    'cifar100': CIFAR100,
    'cifar10': CIFAR10,
    'core50': CORE50,
    'mini_imagenet': Mini_ImageNet,
    'tiny_imagenet': TinyImageNet,
    'openloris': OpenLORIS
}

agents = {
    'BPS':BPSReplay,

}

retrieve_methods = {

    'random': Random_retrieve,

}

update_methods = {
    'random': Reservoir_update,
}
