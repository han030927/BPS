from agents.gdumb import Gdumb
from continuum.dataset_scripts.cifar100 import CIFAR100
from continuum.dataset_scripts.cifar10 import CIFAR10
from continuum.dataset_scripts.core50 import CORE50
from continuum.dataset_scripts.mini_imagenet import Mini_ImageNet
from continuum.dataset_scripts.tiny_imagenet import TinyImageNet
from continuum.dataset_scripts.openloris import OpenLORIS
from agents.exp_replay import ExperienceReplay
from agents.agem import AGEM
from agents.ewc_pp import EWC_pp
from agents.cndpm import Cndpm
from agents.lwf import Lwf
from agents.icarl import Icarl
from agents.scr import SupContrastReplay
from agents.summarize import SummarizeContrastReplay
from agents.iid_offline import IID_offline
from utils.buffer.random_retrieve import Random_retrieve, BalancedRetrieve
from utils.buffer.reservoir_update import Reservoir_update
from utils.buffer.summarize_update import SummarizeUpdate
from utils.buffer.mir_retrieve import MIR_retrieve
from utils.buffer.gss_greedy_update import GSSGreedyUpdate
from utils.buffer.aser_retrieve import ASER_retrieve
from utils.buffer.aser_update import ASER_update
from utils.buffer.sc_retrieve import Match_retrieve
from utils.buffer.mem_match import MemMatch_retrieve
from utils.buffer.mgi_retrieve import MGI_retrieve




from agents.exp_replay_dvc import ExperienceReplay_DVC


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
    'ER': ExperienceReplay,
    'EWC': EWC_pp,
    'AGEM': AGEM,
    'CNDPM': Cndpm,
    'LWF': Lwf,
    'ICARL': Icarl,
    'GDUMB': Gdumb,
    'SCR': SupContrastReplay,
    'SSCR': SummarizeContrastReplay,
    'IID_offline':IID_offline,
    'ER_DVC':ExperienceReplay_DVC,
    'MIR':ExperienceReplay,
    'GSS':ExperienceReplay,

    'BPS':DDDReplay,

}

retrieve_methods = {
    'MIR': MIR_retrieve,
    'random': Random_retrieve,
    'ASER': ASER_retrieve,
    'match': Match_retrieve,
    'mem_match': MemMatch_retrieve,
    'MGI':MGI_retrieve,
}

update_methods = {
    'random': Reservoir_update,
    'GSS': GSSGreedyUpdate,
    'ASER': ASER_update,
    'summarize': SummarizeUpdate,
}

