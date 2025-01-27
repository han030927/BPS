import torch
import numpy as np
from torch.utils import data
from utils.buffer.buffer import Buffer, DynamicBuffer,DreamBuffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform, BalancedSampler
from utils.setup_elements import transforms_match, input_size_match
from utils.utils import maybe_cuda, AverageMeter
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
import torch.nn as nn
from models.convnet import ConvNet
from torchvision.utils import make_grid, save_image
from utils.buffer.new_strategy import NEW_Strategy


class DreamExperienceReplay(ContinualLearner):
    def __init__(self, model, opt, params):
        super(DreamExperienceReplay, self).__init__(model, opt, params)
        self.buffer = DreamBuffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
        self.transform = nn.Sequential(
            RandomResizedCrop(size=(input_size_match[self.params.data][1], input_size_match[self.params.data][2]), scale=(0.2, 1.)),
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            RandomGrayscale(p=0.2)

        )
        self.queue_size = params.queue_size


    def init_syn(self,x_train,y_train):
        transform=transforms_match[self.data]

        self.indices_class={}

        for i in self.new_labels:
            self.indices_class[i]=[]
        for i in range(len(y_train)):
            self.indices_class[y_train[i]].append(i)
        self.class_data={}
        for i in self.new_labels:
            self.class_data[i]=x_train[self.indices_class[i]]
        if self.params.data == 'mini_imagenet':
            im_size = (84, 84)
        elif self.params.data == 'tiny_imagenet':
            im_size = (64, 64)
        else:
            im_size = (32, 32)

        net=maybe_cuda(ConvNet(len(self.new_labels), im_size=im_size),self.params.cuda)


        for i in self.new_labels:
            image=self.class_data[i]
            image=maybe_cuda(torch.stack([transform(ee) for ee in image]),self.params.cuda)

            image_transform=self.transform(image)
            strategy=NEW_Strategy(image_transform,net)
            ipc=self.params.images_per_class
            query_idxs = strategy.query(ipc).cpu().numpy()

            for ii in range(ipc):
                if self.params.mem_size > self.buffer.current_index:
                    self.buffer.buffer_img[self.buffer.current_index].data.copy_(image[query_idxs[ii]])
                    self.buffer.buffer_img_rep[self.buffer.current_index].data.copy_(image[query_idxs[ii]])
                    self.buffer.buffer_label[self.buffer.current_index].data.copy_(i)
                    self.buffer.condense_dict[i].append(self.buffer.current_index)
                    self.buffer.avail_indices.remove(self.buffer.current_index)
                    self.buffer.current_index += 1
                else:
                    replace_index = np.random.choice(self.buffer.avail_indices)
                    # Remove the random sample record
                    self.buffer.buffer_img[replace_index].data.copy_(image[query_idxs[ii]])
                    self.buffer.buffer_img_rep[replace_index].data.copy_(image[query_idxs[ii]])
                    self.buffer.buffer_label[replace_index].data.copy_(i)
                    self.buffer.condense_dict[i].append(replace_index)
                    self.buffer.avail_indices.remove(replace_index)







    def train_learner(self, x_train, y_train, labels):
        self.before_train(x_train, y_train)
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_sampler = BalancedSampler(x_train, y_train, self.batch)
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, num_workers=0,
                                       drop_last=True, sampler=train_sampler)
        # set up model
        self.model = self.model.train()
        self.buffer.new_condense_task(labels)
        self.init_syn(x_train,y_train)
        # setup tracker
        losses = AverageMeter()
        acc_batch = AverageMeter()

        aff_x = []
        aff_y = []
        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)

                for j in range(self.mem_iters):
                    mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)

                    if mem_x.size(0) > 0:
                        mem_x = maybe_cuda(mem_x, self.cuda)
                        mem_y = maybe_cuda(mem_y, self.cuda)
                        combined_batch = torch.cat((mem_x, batch_x))
                        combined_labels = torch.cat((mem_y, batch_y))
                        combined_batch_aug = self.transform(combined_batch)
                        outputs = self.model(combined_batch_aug)
                        #features = torch.cat([self.model.forward(combined_batch).unsqueeze(1), self.model.forward(combined_batch_aug).unsqueeze(1)], dim=1)
                        loss = self.criterion(outputs, combined_labels)
                        losses.update(loss, batch_y.size(0))
                        self.opt.zero_grad()
                        loss.backward()
                        self.opt.step()

                # update memory
                aff_x.append(batch_x)
                aff_y.append(batch_y)
                if len(aff_x) > self.queue_size:
                    aff_x.pop(0)
                    aff_y.pop(0)
                self.buffer.update(batch_x, batch_y, aff_x=aff_x, aff_y=aff_y, update_index=i, transform=self.transform)

                if i % 100 == 1 and self.verbose:
                        print(
                            '==>>> it: {}, avg. loss: {:.6f}, '
                                .format(i, losses.avg(), acc_batch.avg())
                        )
        self.after_train()
