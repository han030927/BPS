import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from utils.utils import maybe_cuda
from models.convnet import ConvNet
from .buffer_utils import random_retrieve
from .augment import DiffAug
from .summarize_update import condense_retrieve,dist
from .new_strategy import NEW_Strategy
from utils.loss import SupConLoss

class With_IM_loss_FullSummarizeUpdate(object):
    def __init__(self, params):
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.params = params
        self.label_dict = {}
        self.optim_flag = True


    def new_task(self, num_classes, labels):
        '''Initialize the label dict for model training
        '''
        if self.params.data == 'mini_imagenet':
            im_size = (84, 84)
        elif self.params.data == 'tiny_imagenet':
            im_size = (64, 64)
        else:
            im_size = (32, 32)
        self.model = maybe_cuda(ConvNet(num_classes, im_size=im_size),self.params.cuda)
        self.optimizer_model = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.syn_model=maybe_cuda(ConvNet(num_classes, im_size=im_size),self.params.cuda)
        self.syn_optimizer = torch.optim.SGD(self.syn_model.parameters(),lr=0.01, momentum=0.9)
        # for idx, label in enumerate(labels):
        #     self.label_dict[label] = idx + num_classes - len(labels)
        for idx, label in enumerate(labels):
            self.label_dict[label] = idx
        self.new_labels = labels
        self.optim_flag = True

    def update(self, buffer, x, y, aff_x=None, aff_y=None, update_index=-1, transform=None, **kwargs):
        condense_flag = True

        # if len(aff_x) < self.params.queue_size:
        #     condense_flag = False
        # elif len(aff_x) > 0:
        #     aff_x = torch.cat(aff_x, dim=0)
        #     aff_y = torch.cat(aff_y)
        aff_x = torch.cat(aff_x, dim=0)
        aff_y = torch.cat(aff_y)
        if self.params.with_original:
            for data, label in zip(x, y):
                label_c = label.item()
                # The condense dict is always full after initializing
                # The buffer is not full
                if self.params.mem_size > buffer.current_index:
                    if buffer.current_index in buffer.avail_indices:
                        buffer.buffer_img[buffer.current_index].data.copy_(data)
                        buffer.buffer_label[buffer.current_index].data.copy_(label)
                        buffer.current_index += 1
                        #buffer.avail_indices.remove(buffer.current_index)
                # The buffer is full
                else:
                    random_index = int(np.random.uniform(buffer.n_seen_so_far+self.params.images_per_class*len(buffer.condense_dict)))
                    if random_index < self.params.mem_size and random_index in buffer.avail_indices:
                        buffer.buffer_img[random_index].data.copy_(data)
                        buffer.buffer_label[random_index].data.copy_(label)
        #buffer&condense dict is always full
        buffer.n_seen_so_far += x.shape[0]

        # if update_index % self.params.summarize_interval != 0:
        #     condense_flag = False

        # conduct sample condense
        if condense_flag:
            labelset = set(y.cpu().numpy())
            # initialize the optimization target at the first iteration of new tasks
            # if self.optim_flag:

            self.condense_x = [buffer.buffer_img[buffer.condense_dict[c]] for c in labelset]
            self.condense_x = copy.deepcopy(torch.cat(self.condense_x, dim=0)).requires_grad_()
            self.condense_y = [buffer.buffer_label[buffer.condense_dict[c]] for c in labelset]
            self.condense_y = torch.cat(self.condense_y)
            self.optimizer_img = torch.optim.SGD([self.condense_x,], lr=self.params.lr_img, momentum=0.9)
            self.optim_flag = False

            diff_aug = DiffAug(strategy='color_crop', batch=False)
            match_aug = transforms.Compose([diff_aug])

            avg_loss = 0.
            for cls_idx, c in enumerate(labelset):
                # obtain samples of each class and add augmentation
                img_real = aff_x[aff_y == c]
                lab_real = aff_y[aff_y == c]
                lab_real = maybe_cuda(torch.tensor([self.label_dict[l_real.item()] for l_real in lab_real]),self.params.cuda)
                img_syn = self.condense_x[self.condense_y == c]
                lab_syn = self.condense_y[self.condense_y == c]
                lab_syn = maybe_cuda(torch.tensor([self.label_dict[l_real.item()] for l_real in lab_syn]),self.params.cuda)

                img_aug = match_aug(torch.cat((img_real, img_syn), dim=0))
                img_real = img_aug[:len(img_real)]
                img_syn = img_aug[len(img_real):]

                # calculate matching loss

                loss = self.match_loss(img_real, img_syn, lab_real, lab_syn, buffer,aff_x,aff_y,c)
                avg_loss += loss.item()

                self.optimizer_img.zero_grad()
                loss.backward()
                self.optimizer_img.step()

                # update the condensed image to the memory
                img_new = self.condense_x[self.condense_y == c]
                buffer.buffer_img[buffer.condense_dict[c]] = img_new.detach()


        # update the matching model
        y = maybe_cuda(torch.tensor([self.label_dict[lab.item()] for lab in y]),self.params.cuda)
        if self.params.with_kmeans_sampling:
            self.update_model_kmeans(x,y,transform)
        else:
            self.retrieve_update_model(x, y, buffer, transform)

    def match_loss(self, img_real, img_syn, lab_real, lab_syn, buffer,aff_x,aff_y,c):
        loss = 0.
        im_criterion =SupConLoss()
        # check if memory-based matching loss is applicable
        output_real = self.model(img_real)
        with torch.no_grad():
            feat_real = self.model.features(img_real)
        output_syn, feat_syn = self.model(img_syn, return_features=True)
        img_mem, lab_mem = condense_retrieve(buffer, self.params.mem_sim_num, excl_labels=self.new_labels)
        if self.params.with_task_matching_loss:
            task_real=aff_x[aff_y!=c]
            if not self.params.using_all:
                indices = torch.randperm(task_real.size(0))

                task_real = task_real[indices[:10]]
            with torch.no_grad():
                feat_task=self.model.features(task_real)
            feat_syn = F.normalize(feat_syn, dim=1)
            feat_task=F.normalize(feat_task,dim=1)
            feat_real = F.normalize(feat_real, dim=1)
            task_matching_loss=dist(
            self.euclidean_dist(feat_syn.mean(0, keepdim=True), feat_task),
            self.euclidean_dist(feat_real.mean(0, keepdim=True), feat_task),
            metric=self.params.task_loss_metric
            )
        else:
            task_matching_loss=0






        if len(img_mem) > 0 and self.params.with_relation_loss:
            with torch.no_grad():
                feat_mem = self.model.features(img_mem)
            feat_syn = F.normalize(feat_syn, dim=1)
            feat_mem = F.normalize(feat_mem, dim=1)
            feat_real = F.normalize(feat_real, dim=1)
            mem_loss = dist(
                self.euclidean_dist(feat_syn.mean(0, keepdim=True), feat_mem),
                self.euclidean_dist(feat_real.mean(0, keepdim=True), feat_mem),
                'l1'
            )
        else:
            mem_loss = 0

        # options of feature distribution matching and gradient matching
        if 'feat' in self.params.match:
            loss += dist(feat_real.mean(0), feat_syn.mean(0), self.params.metric)
        if 'grad' in self.params.match:
            loss_real = self.criterion(output_real, lab_real)
            gw_real = torch.autograd.grad(loss_real, self.model.parameters())
            gw_real = list((_.detach().clone() for _ in gw_real))

            loss_syn = self.criterion(output_syn, lab_syn)
            gw_syn = torch.autograd.grad(loss_syn, self.model.parameters(), create_graph=True)

            for ig in range(len(gw_real)):
                gwr = gw_real[ig]
                gws = gw_syn[ig]
                if len(gwr.shape) == 1 or len(gwr.shape) == 2:
                    continue
                loss += dist(gwr, gws, self.params.metric)
        if 'feat' in self.params.match and 'grad' in self.params.match:
            loss = loss / 2.0




        if self.params.entropy_minimization:
            def softmax_entropy(x):
                return -(x.softmax(1) * x.log_softmax(1)).sum(1)
            em_loss=softmax_entropy(output_syn).sum(0)
        else:
            em_loss=0
        # if mem_loss > 0:

        if self.params.IM_loss_weight>0:
            task_real = aff_x.detach()
            task_lab= maybe_cuda(torch.tensor([self.label_dict[l_real.item()] for l_real in aff_y]),self.params.cuda)
            # with torch.no_grad():
            combined_batch = torch.cat((task_real, img_syn))

            feat_original_model=F.normalize(self.model.features(combined_batch),dim=1)
            feat_syn_model=F.normalize(self.syn_model.features(combined_batch),dim=1)
            # output_original_model=self.model.classifier(feat_original_model)
            # output_syn_model=self.syn_model.classifier(feat_syn_model)
                # feat_task = self.model.features(task_real)
                # feat_task=F.normalize(feat_task, dim=1)
                # feat_syn = self.syn_model.features(img_syn)
                # feat_syn = F.normalize(feat_syn, dim=1)

            #features=torch.cat((feat_task,feat_syn),dim=0)
            features=torch.cat((feat_original_model.unsqueeze(1),feat_syn_model.unsqueeze(1)),dim=1)
            #features=torch.cat((output_original_model.unsqueeze(1),output_syn_model.unsqueeze(1)),dim=1)
            combined_labels = torch.cat((task_lab,lab_syn))
            IM_loss=im_criterion(features,combined_labels)
        else:
            IM_loss=0
        if self.params.mem_extra == 1:
            if self.params.with_task_matching_loss:
                loss = loss + mem_loss * self.params.mem_weight + task_matching_loss*self.params.task_weight+em_loss+IM_loss*self.params.IM_loss_weight
            else:
                loss = loss + mem_loss * self.params.mem_weight+em_loss+IM_loss*self.params.IM_loss_weight
        else:
            loss = loss * (1 - self.params.mem_weight-self.params.task_weight) + mem_loss * self.params.mem_weight+task_matching_loss*self.params.task_weight+em_loss

        return loss

    def update_model(self, x, y, transform):
        '''Naive model updating
        '''
        data_len = len(x)
        for tmp_idx in range(data_len // 10):
            batch_x = x[tmp_idx * 10 : (tmp_idx + 1) * 10]
            batch_y = maybe_cuda(y[tmp_idx * 10 : (tmp_idx + 1) * 10],self.params.cuda)
            output = self.model(batch_x)
            loss = self.criterion(output, batch_y)
            self.optimizer_model.zero_grad()
            loss.backward()
            self.optimizer_model.step()

    def update_model_kmeans(self, x, y, transform):
        random_indices = torch.randperm(len(x))
        r_x = x[random_indices]
        r_y = y[random_indices]
        bs = self.params.batch
        class_data_dict = {}
        label_indices = {}
        for i in range(len(r_y)):
            if r_y[i] in label_indices:
                label_indices[r_y[i]].append(i)
            else:
                label_indices[r_y[i]] = [i]
        for label, indices in label_indices.items():
            class_data_dict[label] = r_x[indices]

        for c, data in class_data_dict.items():
            img = data

            strategy = NEW_Strategy(img, self.model)

            query_idxs = strategy.query(bs)

            batch_x = x[query_idxs]
            batch_y = y[query_idxs]
            output = self.model(batch_x)
            loss = self.criterion(output, batch_y)
            self.optimizer_model.zero_grad()
            loss.backward()
            self.optimizer_model.step()
    def retrieve_update_model(self, x, y, buffer, transform):
        '''Model updating with images of previous tasks
        '''
        labels = list(set(y.cpu().numpy()))
        condense_indices = []
        if self.params.with_past_assist:
            for lab in buffer.condense_dict.keys():
                condense_indices += buffer.condense_dict[lab]
        else:
            for lab in buffer.condense_dict.keys():
                condense_indices += buffer.condense_dict[lab]
        r_x, r_y = random_retrieve(buffer, self.params.eps_mem_batch, excl_indices=condense_indices)
        r_y = maybe_cuda(torch.tensor([self.label_dict[lab.item()] for lab in r_y]),self.params.cuda)
        r_x = torch.cat((x, r_x), dim=0)
        r_y = torch.cat((y, r_y)).long()
        random_indices = torch.randperm(len(r_x))
        r_x = r_x[random_indices]
        r_y = r_y[random_indices]
        data_len = len(r_x)
        bs = self.params.batch
        for j in range(self.params.inner_loop):
            for tmp_idx in range(data_len // bs):
                batch_x = r_x[tmp_idx * bs : (tmp_idx + 1) * bs]
                batch_y = r_y[tmp_idx * bs : (tmp_idx + 1) * bs]
                output = self.model(batch_x)
                loss = self.criterion(output, batch_y)
                self.optimizer_model.zero_grad()
                loss.backward()
                self.optimizer_model.step()

        r_x = copy.deepcopy(self.condense_x.detach())
        r_y =  copy.deepcopy(self.condense_y.detach())
        r_y= maybe_cuda(torch.tensor([self.label_dict[lab.item()] for lab in r_y]), self.params.cuda)
        data_len = len(r_x)
        random_indices = torch.randperm(len(r_x))
        r_x = r_x[random_indices]
        r_y = r_y[random_indices]
        bs = self.params.batch
        for j in range(self.params.inner_loop):
            for tmp_idx in range(data_len // bs):
                batch_x = r_x[tmp_idx * bs: (tmp_idx + 1) * bs]
                batch_y = r_y[tmp_idx * bs: (tmp_idx + 1) * bs]
                output = self.syn_model(batch_x)
                loss = self.criterion(output, batch_y)
                self.syn_optimizer.zero_grad()
                loss.backward()
                self.syn_optimizer.step()

    def euclidean_dist(self, x, y):
        m, n = x.shape[0], y.shape[0]
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(x, y.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()
        return dist

    def distance_wb(gwr, gws):
        shape = gwr.shape
        if len(shape) == 4:  # conv, out*in*h*w
            gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
            gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
        elif len(shape) == 3:  # layernorm, C*h*w
            gwr = gwr.reshape(shape[0], shape[1] * shape[2])
            gws = gws.reshape(shape[0], shape[1] * shape[2])
        elif len(shape) == 2:  # linear, out*in
            tmp = 'do nothing'
        elif len(shape) == 1:  # batchnorm/instancenorm, C; groupnorm x, bias
            gwr = gwr.reshape(1, shape[0])
            gws = gws.reshape(1, shape[0])
            return torch.tensor(0, dtype=torch.float, device=gwr.device)

        dis_weight = torch.sum(
            1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
        dis = dis_weight
        return dis