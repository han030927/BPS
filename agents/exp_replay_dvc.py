import torch
from torch.utils import data
from utils.buffer.buffer import Buffer,DynamicBuffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform,BalancedSampler
import torch.nn as nn
from utils.setup_elements import transforms_match, input_size_match
from utils.utils import maybe_cuda, AverageMeter
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
from loss import agmax_loss, cross_entropy_loss
from utils.buffer.new_strategy import NEW_Strategy
import copy
import random
from utils.buffer.augment import DiffAug
from torchvision import transforms
from utils.buffer.summarize_update import dist
from models.convnet import ConvNet
import tqdm
import numpy as np
class ExperienceReplay_DVC(ContinualLearner):
    def __init__(self, model, opt, params):
        super(ExperienceReplay_DVC, self).__init__(model, opt, params)
       # self.buffer = DynamicBuffer(model, params)
        self.buffer =Buffer(model, params)
        self.mem_size = params.mem_size
        self.agent = params.agent
        self.dl_weight = params.dl_weight

        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
        self.transform = nn.Sequential(
            RandomResizedCrop(size=(input_size_match[self.params.data][1], input_size_match[self.params.data][2]), scale=(0.2, 1.)),
            RandomHorizontalFlip(),
           ColorJitter(0.4, 0.4, 0.4, 0.1),
           RandomGrayscale(p=0.2)

        )
        self.queue_size = params.queue_size
        self.summarize_method = params.summarize_method
        self.summarize_iter = params.outer_loop
        self.distill_batch = params.distill_batch
        self.L2loss = torch.nn.MSELoss()
    def summarize_past_task(self):
        if self.params.data == 'mini_imagenet':
            im_size = (84, 84)
        elif self.params.data == 'tiny_imagenet':
            im_size = (64, 64)
        else:
            im_size = (32, 32)
        # self.indices_class={}
        self.class_data = {}
        self.old_buffer_data=copy.deepcopy(self.buffer.buffer_img)
        self.old_buffer_label = copy.deepcopy(self.buffer.buffer_label)
        #self.label_dict = {}
        for idx, label in enumerate(self.old_labels):
           # self.label_dict[label] = idx #+ self.buffer.num_classes - len(self.old_labels)
            self.class_data[label] = self.old_buffer_data[self.buffer.condense_dict[label]]




        if self.summarize_method=='kmeans':
            net = maybe_cuda(ConvNet(len(self.old_labels), im_size=im_size), self.params.cuda)
            for i in self.old_labels:
                image = self.class_data[i]
                #image = maybe_cuda(torch.stack([transform(ee) for ee in image]), self.params.cuda)
                old_condense_dict=copy.deepcopy(self.buffer.condense_dict[i])
                self.buffer.condense_dict[i]=[]

                image_transform = self.transform(image)
                strategy = NEW_Strategy(image_transform, net)
                ipc = self.params.mem_size // self.buffer.task_id//self.params.num_classes_per_task
                query_idxs = strategy.query(ipc).cpu().numpy()
                query_idxs.sort()
                for ii in range(ipc):
                    self.buffer.buffer_img[old_condense_dict[query_idxs[ii]]].data.copy_(image[query_idxs[ii]])
                    self.buffer.buffer_img_rep[old_condense_dict[query_idxs[ii]]].data.copy_(image[query_idxs[ii]])
                    self.buffer.buffer_label[old_condense_dict[query_idxs[ii]]].data.copy_(i)
                    self.buffer.condense_dict[i].append(old_condense_dict[query_idxs[ii]])
                    #old_condense_dict.pop(query_idxs[ii])
                for j in old_condense_dict:
                    if j not in self.buffer.condense_dict[i]:
                        self.buffer.avail_indices.append(j)

        elif self.summarize_method == 'random':
            for i in self.old_labels:
                image = self.class_data[i]
                #image = maybe_cuda(torch.stack([transform(ee) for ee in image]), self.params.cuda)
                old_condense_dict=self.buffer.condense_dict[i].copy()
                self.buffer.condense_dict[i]=[]
                ipc = self.params.mem_size // self.buffer.task_id // self.params.num_classes_per_task
                query_idxs=random.sample(list(range(len(old_condense_dict))), ipc)
                for ii in range(ipc):
                    self.buffer.buffer_img[old_condense_dict[query_idxs[ii]]].data.copy_(image[query_idxs[ii]])
                    self.buffer.buffer_img_rep[old_condense_dict[query_idxs[ii]]].data.copy_(image[query_idxs[ii]])
                    self.buffer.buffer_label[old_condense_dict[query_idxs[ii]]].data.copy_(i)
                    self.buffer.condense_dict[i].append(old_condense_dict[query_idxs[ii]])
                    #old_condense_dict.pop(query_idxs[ii])
                for j in old_condense_dict:
                    if j not in self.buffer.condense_dict[i]:
                        self.buffer.avail_indices.append(j)
        elif self.summarize_method =='mean_feature':
            net = maybe_cuda(ConvNet(len(self.old_labels), im_size=im_size), self.params.cuda)
            for i in self.old_labels:
                image = self.class_data[i]
                # image = maybe_cuda(torch.stack([transform(ee) for ee in image]), self.params.cuda)
                old_condense_dict = copy.deepcopy(self.buffer.condense_dict[i])
                self.buffer.condense_dict[i] = []
                image_transform = self.transform(image)
                strategy = NEW_Strategy(image_transform, net,method='mean_feature')
                ipc = self.params.mem_size // self.buffer.task_id // self.params.num_classes_per_task
                query_idxs = strategy.query(ipc).cpu().numpy()
                query_idxs.sort()
                for ii in range(ipc):
                    self.buffer.buffer_img[old_condense_dict[query_idxs[ii]]].data.copy_(image[query_idxs[ii]])
                    self.buffer.buffer_img_rep[old_condense_dict[query_idxs[ii]]].data.copy_(image[query_idxs[ii]])
                    self.buffer.buffer_label[old_condense_dict[query_idxs[ii]]].data.copy_(i)
                    self.buffer.condense_dict[i].append(old_condense_dict[query_idxs[ii]])
                    # old_condense_dict.pop(query_idxs[ii])
                for j in old_condense_dict:
                    if j not in self.buffer.condense_dict[i]:
                        self.buffer.avail_indices.append(j)
        elif self.summarize_method=='summarize':
            self.summarize_model = maybe_cuda(ConvNet(len(self.old_labels), im_size=im_size), self.params.cuda)
            self.optimizer_model = torch.optim.SGD(self.summarize_model.parameters(), lr=0.01, momentum=0.9)
            #initilaze new condense image
            for i in self.old_labels:
                image = self.class_data[i]
                # image = maybe_cuda(torch.stack([transform(ee) for ee in image]), self.params.cuda)
                old_condense_dict = copy.deepcopy(self.buffer.condense_dict[i])
                self.buffer.condense_dict[i] = []

                image_transform = self.transform(image)
                strategy = NEW_Strategy(image_transform,self.summarize_model,method='mean_feature')
                ipc = self.params.mem_size // self.buffer.task_id // self.params.num_classes_per_task
                query_idxs = strategy.query(ipc).cpu().numpy()
                query_idxs.sort()
                for ii in range(ipc):
                    self.buffer.buffer_img[old_condense_dict[query_idxs[ii]]].data.copy_(image[query_idxs[ii]])
                    self.buffer.buffer_img_rep[old_condense_dict[query_idxs[ii]]].data.copy_(image[query_idxs[ii]])
                    self.buffer.buffer_label[old_condense_dict[query_idxs[ii]]].data.copy_(i)
                    self.buffer.condense_dict[i].append(old_condense_dict[query_idxs[ii]])

                    # old_condense_dict.pop(query_idxs[ii])
                for j in old_condense_dict:
                    if j not in self.buffer.condense_dict[i]:
                        self.buffer.avail_indices.append(j)

            diff_aug = DiffAug(strategy='color_crop', batch=False)
            match_aug = transforms.Compose([diff_aug])
            self.condense_x = [self.buffer.buffer_img[self.buffer.condense_dict[c]] for c in self.old_labels]
            self.condense_x = copy.deepcopy(torch.cat(self.condense_x, dim=0)).requires_grad_()
            self.condense_y = [self.buffer.buffer_label[self.buffer.condense_dict[c]] for c in self.old_labels]
            self.condense_y = torch.cat(self.condense_y)
            self.optimizer_img = torch.optim.SGD([self.condense_x, ], lr=self.params.lr_img, momentum=0.9)
            y = maybe_cuda(torch.tensor([self.lbl_inv_map[lab.item()] for lab in self.old_buffer_label]), self.params.cuda)
            for epoch in tqdm(range(self.summarize_iter), desc="Summarizing epochs"):
                for i in self.old_labels:
                    img_real=self.class_data[i]
                    # if len(img_real)==ipc:
                    #     continue

                    lab_real = maybe_cuda(torch.tensor([self.lbl_inv_map[i] for _ in img_real]),
                                          self.params.cuda)
                    img_syn = self.condense_x[self.condense_y == i]
                    lab_syn = self.condense_y[self.condense_y == i]
                    lab_syn = maybe_cuda(torch.tensor([self.lbl_inv_map[l_real.item()] for l_real in lab_syn]),
                                         self.params.cuda)

                    img_aug = match_aug(torch.cat((img_real, img_syn), dim=0))
                    img_real = img_aug[:len(img_real)]
                    img_syn = img_aug[len(img_real):]

                    img_aug = match_aug(torch.cat((img_real, img_syn), dim=0))
                    img_real = img_aug[:len(img_real)]
                    img_syn = img_aug[len(img_real):]

                    for _ in range(self.params.summarize_model_iter):
                        self.update_model(self.old_buffer_data, y)

                    loss = self.match_loss(img_real, img_syn, lab_real, lab_syn)
                    self.optimizer_img.zero_grad()
                    loss.backward()
                    self.optimizer_img.step()
                    # update the condensed image to the memory
            #for i in self.old_labels:
                    img_new = self.condense_x[self.condense_y == i]
                    self.buffer.buffer_img[self.buffer.condense_dict[i]] = img_new.detach()
    def match_loss(self,img_real, img_syn, lab_real, lab_syn):
        criterion = torch.nn.CrossEntropyLoss()
        loss = 0.

        # check if memory-based matching loss is applicable

        output_real = self.summarize_model(img_real)
        feat_real = self.model.features(img_real)
        output_syn, feat_syn = self.summarize_model(img_syn, return_features=True)
        loss_real = criterion(output_real, lab_real)
        gw_real = torch.autograd.grad(loss_real, self.summarize_model.parameters())
        gw_real = list((_.detach().clone() for _ in gw_real))

        loss_syn = criterion(output_syn, lab_syn)
        gw_syn = torch.autograd.grad(loss_syn, self.summarize_model.parameters(), create_graph=True)
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            if len(gwr.shape) == 1 or len(gwr.shape) == 2:
                continue
            loss += dist(gwr, gws, self.params.metric)
        return loss
    def update_model(self, x, y, transform=None):
        '''Naive model updating
        '''
        criterion = nn.CrossEntropyLoss()
        data_len = len(x)
        if transform is not None:
            x=transform(x)
        for tmp_idx in range(data_len // 10):
            batch_x = x[tmp_idx * 10 : (tmp_idx + 1) * 10]
            batch_y = maybe_cuda(y[tmp_idx * 10 : (tmp_idx + 1) * 10],self.params.cuda)
            output = self.summarize_model(batch_x)
            loss = criterion(output, batch_y)
            self.optimizer_model.zero_grad()
            loss.backward()
            self.optimizer_model.step()

    def init_syn(self,x_train,y_train):

        if self.buffer.task_id>1 and not self.params.with_original:
            self.summarize_past_task()


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
        if not self.params.with_original:
            ipc=len(self.buffer.avail_indices)//len(self.new_labels)
        else:
            ipc=self.params.images_per_class
        for i in self.new_labels:
            image=self.class_data[i]
            image=maybe_cuda(torch.stack([transform(ee) for ee in image]),self.params.cuda)
            if len(self.buffer.avail_indices)<ipc*2 and len(self.buffer.avail_indices)>ipc:
                ipc=len(self.buffer.avail_indices)
            if self.params.initialize_way=='kmeans':
                image_transform=self.transform(image)
                strategy=NEW_Strategy(image_transform,net)
                query_idxs = strategy.query(ipc).cpu().numpy()
            else:
                query_idxs=np.array(range(ipc))

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





    def train_learner(self, x_train, y_train,labels=None):
        self.before_train(x_train, y_train)
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_sampler = BalancedSampler(x_train, y_train, self.batch)
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
                                       drop_last=True)
        distill_train_loader = data.DataLoader(train_dataset, batch_size=self.distill_batch, num_workers=0,
                                               drop_last=True, sampler=train_sampler)
        # set up model
        self.model = self.model.train()
      #  self.buffer.new_task(self.new_labels)
      #   self.init_syn(x_train, y_train)
      #   for key, values in self.buffer.condense_dict.items():
      #       print(f'class:{key},summarize_images:{len(values)}')
      #   self.task_data_distill(distill_train_loader)
        self.transform = self.transform.cuda()
        # setup tracker
        losses_batch = AverageMeter()
        losses_mem = AverageMeter()
        acc_batch = AverageMeter()
        acc_mem = AverageMeter()

        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_x_aug = self.transform(batch_x)
                batch_y = maybe_cuda(batch_y, self.cuda)
                for j in range(self.mem_iters):
                    y = self.model(batch_x, batch_x_aug)
                    z, zt, _,_ = y
                    ce = cross_entropy_loss(z, zt, batch_y, label_smoothing=0)


                    agreement_loss, dl = agmax_loss(y, batch_y, dl_weight=self.dl_weight)
                    loss  = ce + agreement_loss + dl

                    if self.params.trick['kd_trick']:
                        loss = 1 / (self.task_seen + 1) * loss + (1 - 1 / (self.task_seen + 1)) * \
                                   self.kd_manager.get_kd_loss(z, batch_x)
                    if self.params.trick['kd_trick_star']:
                        loss = 1/((self.task_seen + 1) ** 0.5) * loss + \
                               (1 - 1/((self.task_seen + 1) ** 0.5)) * self.kd_manager.get_kd_loss(z, batch_x)
                    _, pred_label = torch.max(z, 1)
                    correct_cnt = (pred_label == batch_y).sum().item() / batch_y.size(0)
                    # update tracker
                    acc_batch.update(correct_cnt, batch_y.size(0))
                    losses_batch.update(loss, batch_y.size(0))
                    # backward
                    self.opt.zero_grad()
                    loss.backward()

                    # mem update
                    if  self.params.retrieve == 'MGI':
                        mem_x, mem_x_aug, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)

                    else:
                        mem_x, mem_y= self.buffer.retrieve(x=batch_x, y=batch_y)
                        if mem_x.size(0) > 0:
                            mem_x_aug = self.transform(mem_x)

                    if mem_x.size(0) > 0:
                        mem_x = maybe_cuda(mem_x, self.cuda)
                        mem_x_aug = maybe_cuda(mem_x_aug, self.cuda)
                        mem_y = maybe_cuda(mem_y, self.cuda)
                        y = self.model(mem_x, mem_x_aug)
                        z, zt, _,_ = y
                        ce = cross_entropy_loss(z, zt, mem_y, label_smoothing=0)
                        agreement_loss, dl = agmax_loss(y, mem_y, dl_weight=self.dl_weight)
                        loss_mem = ce  + agreement_loss + dl

                        if self.params.trick['kd_trick']:
                            loss_mem = 1 / (self.task_seen + 1) * loss_mem + (1 - 1 / (self.task_seen + 1)) * \
                                       self.kd_manager.get_kd_loss(z, mem_x)
                        if self.params.trick['kd_trick_star']:
                            loss_mem = 1 / ((self.task_seen + 1) ** 0.5) * loss_mem + \
                                   (1 - 1 / ((self.task_seen + 1) ** 0.5)) * self.kd_manager.get_kd_loss(z,
                                                                                                         mem_x)
                        # update tracker
                        losses_mem.update(loss_mem, mem_y.size(0))
                        _, pred_label = torch.max(z, 1)
                        correct_cnt = (pred_label == mem_y).sum().item() / mem_y.size(0)
                        acc_mem.update(correct_cnt, mem_y.size(0))

                        loss_mem.backward()
                    self.opt.step()

                # update mem
                self.buffer.update(batch_x, batch_y)

                if i % 100 == 1 and self.verbose:
                    print(
                        '==>>> it: {}, avg. loss: {:.6f}, '
                        'running train acc: {:.3f}'
                            .format(i, losses_batch.avg(), acc_batch.avg())
                    )
                    print(
                        '==>>> it: {}, mem avg. loss: {:.6f}, '
                        'running mem acc: {:.3f}'
                            .format(i, losses_mem.avg(), acc_mem.avg())
                    )
        self.after_train()



    def task_data_distill(self, train_loader):
        # aff_x = []
        # aff_y = []
        for ep in range(self.summarize_iter):
            for i, batch_data in enumerate(train_loader):
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                # aff_x.append(batch_x)
                # aff_y.append(batch_y)
                # if len(aff_x) > self.queue_size:
                #     aff_x.pop(0)
                #     aff_y.pop(0)
                self.buffer.update(batch_x, batch_y, aff_x=[batch_x], aff_y=[batch_y], update_index=i, transform=self.transform)




