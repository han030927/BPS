import copy
import torch
import numpy as np
from collections import defaultdict

from utils.setup_elements import input_size_match
from utils import name_match
from utils.utils import maybe_cuda
from utils.buffer.buffer_utils import BufferClassTracker
from utils.setup_elements import n_classes


class Buffer(torch.nn.Module):
    def __init__(self, model, params):
        super().__init__()
        self.params = params
        self.model = model
        self.cuda = self.params.cuda
        self.current_index = 0
        self.n_seen_so_far = 0
        self.device = "cuda" if self.params.cuda else "cpu"
        self.num_classes_per_task = self.params.num_classes_per_task
        self.num_classes = 0

        # define buffer
        buffer_size = params.mem_size
        print('buffer has %d slots' % buffer_size)
        input_size = input_size_match[params.data]
        class_number = n_classes[params.data]
        buffer_img = maybe_cuda(torch.FloatTensor(buffer_size, *input_size).fill_(0))
        buffer_label = maybe_cuda(torch.LongTensor(buffer_size).fill_(0))
        buffer_logits = maybe_cuda(torch.FloatTensor(buffer_size, 1, class_number).fill_(0))

        # registering as buffer allows us to save the object using `torch.save`
        self.register_buffer('buffer_img', buffer_img)
        self.register_buffer('buffer_label', buffer_label)
        self.register_buffer('buffer_logits', buffer_logits)

        self.labeldict = defaultdict(list)
        self.labelsize = params.images_per_class
        self.avail_indices = list(np.arange(buffer_size))

        # define update and retrieve method
        self.update_method = name_match.update_methods[params.update](params)
        self.retrieve_method = name_match.retrieve_methods[params.retrieve](params)

        if self.params.buffer_tracker:
            self.buffer_tracker = BufferClassTracker(n_classes[params.data], self.device)

    def update(self, x, y, **kwargs):
        return self.update_method.update(buffer=self, x=x, y=y, **kwargs)

    def retrieve(self, **kwargs):
        return self.retrieve_method.retrieve(buffer=self, **kwargs)

    def new_task(self, **kwargs):
        self.num_classes += self.num_classes_per_task
        self.labelsize = self.params.mem_size // self.num_classes

    def new_condense_task(self, **kwargs):
        self.num_classes += self.num_classes_per_task
        self.update_method.new_task(self.num_classes)


class DynamicBuffer(torch.nn.Module):
    def __init__(self, model, params):
        super().__init__()
        self.params = params
        self.model = model
        self.cuda = self.params.cuda
        self.current_index = 0
        self.n_seen_so_far = 0
        self.device = "cuda" if self.params.cuda else "cpu"
        self.num_classes_per_task = self.params.num_classes_per_task
        self.images_per_class = self.params.images_per_class
        self.num_classes = 0
        self.task_id = 0

        # define buffer
        buffer_size = params.mem_size
        print('buffer has %d slots' % buffer_size)
        input_size = input_size_match[params.data]
        buffer_img = maybe_cuda(torch.FloatTensor(buffer_size, *input_size).fill_(0))
        buffer_label = maybe_cuda(torch.LongTensor(buffer_size).fill_(0))

        # registering as buffer allows us to save the object using `torch.save`
        self.register_buffer('buffer_img', buffer_img)
        self.register_buffer('buffer_img_rep', copy.deepcopy(buffer_img))
        self.register_buffer('buffer_label', buffer_label)
        self.condense_dict = defaultdict(list)
        self.labelsize = params.images_per_class
        self.avail_indices = list(np.arange(buffer_size))

        # define update and retrieve method
        self.update_method = name_match.update_methods[params.update](params)
        self.retrieve_method = name_match.retrieve_methods[params.retrieve](params)

    def update(self, x, y, **kwargs):
        return self.update_method.update(buffer=self, x=x, y=y, **kwargs)

    def retrieve(self, **kwargs):
        return self.retrieve_method.retrieve(buffer=self, **kwargs)

    def new_task(self, **kwargs):
        self.num_classes += self.num_classes_per_task
        self.labelsize = self.params.mem_size // self.num_classes
        self.task_id += 1

    def new_condense_task(self, labels, **kwargs):
        self.num_classes += self.num_classes_per_task
        self.task_id += 1
        self.update_method.new_task(self.num_classes, labels)

    def new_network(self):
        self.update_method.new_network(self.num_classes)




class FSSBuffer(DynamicBuffer):
    def __init__(self, model, params, ):
        super().__init__(model, params)

    def new_condense_task(self, labels, **kwargs):
        self.num_classes += self.num_classes_per_task
        self.task_id += 1
