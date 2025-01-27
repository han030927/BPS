import torch
import numpy as np
from torch.utils import data
from utils.buffer.buffer import Buffer, DynamicBuffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform, BalancedSampler
from utils.setup_elements import transforms_match, input_size_match
from utils.utils import maybe_cuda, AverageMeter
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
import torch.nn as nn
from torchvision.utils import make_grid, save_image
from models.convnet import ConvNet
from torchvision.utils import make_grid, save_image
from utils.buffer.new_strategy import NEW_Strategy
import copy
from utils.setup_elements import n_classes,setup_opt,setup_architecture
import random
from utils.buffer.augment import DiffAug
from torchvision import transforms
from utils.buffer.summarize_update import dist
from tqdm import tqdm
from models.resnet import SupConResNet
from PIL import Image


def scale_tensor(tensor):
    # 获取 tensor 的最小和最大值
    min_val = tensor.min()
    max_val = tensor.max()

    # 将 tensor 线性映射到 [0, 255]
    # 先将 tensor 归一化到 [0, 1]
    tensor = (tensor - min_val) / (max_val - min_val)
    # 再缩放到 [0, 255]
    tensor = tensor * 255
    return tensor
class PDDSummarizeContrastReplay(ContinualLearner):
    def __init__(self, model, opt, params):
        super(PDDSummarizeContrastReplay, self).__init__(model, opt, params)
        self.buffer = DynamicBuffer(model, params)
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
        self.summarize_method=params.summarize_method
        self.summarize_iter=params.summarize_iter
        self.distill_batch=params.distill_batch
        self.stage_num=params.stage_num



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
        #TODO keep past synthetic images, get spaces for new synimages





        # self.
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
            flag=1
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
        distill_train_loader = data.DataLoader(train_dataset, batch_size=self.distill_batch, num_workers=0,
                                       drop_last=True, sampler=train_sampler)
        # set up model
        self.model = self.model.train()
        self.buffer.new_condense_task(self.new_labels)
        self.init_syn(x_train, y_train)
        # for c in self.old_labels:
        #     images_c = self.buffer.buffer_img[self.buffer.condense_dict[c]]
        #     for i in range(len(images_c)):
        #         before_distill = scale_tensor(images_c[i].to('cpu'))
        #         before_distill_array = before_distill.numpy().transpose(1, 2, 0).astype(np.uint8)  # Rearrange dimensions to HxWxC
        #         img = Image.fromarray(before_distill_array, 'RGB')
        #         img.save(f'images/with_mean_feature/selected_distilled/after_distilled_image_{c}_{i}.png')
        # for c in self.new_labels:
        #     flg=4
        #     images_c=self.buffer.buffer_img[self.buffer.condense_dict[c]]
        #     for i in range(len(images_c)):
        #         before_distill=images_c[i].to('cpu').mul(255).byte()
        #         before_distill_array = before_distill.numpy().transpose(1, 2, 0)  # Rearrange dimensions to HxWxC
        #         img = Image.fromarray(before_distill_array, 'RGB')
        #         if len(self.old_labels)>0:
        #             img.save(f'images/with_mean_feature/new_original/before_image_{c}_{i}.png')
        #         else:
        #             img.save(f'images/with_mean_feature/distilled/before_image_{c}_{i}.png')
        #     while flg>0:
        #         cur=self.class_data[c][flg]
        #         img = Image.fromarray(cur)
        #         if len(self.old_labels) > 0:
        #             img.save(f'images/with_mean_feature/new_original/original_image_{c}_{flg}.png')
        #         else:
        #             img.save(f'images/with_mean_feature/original/original_image_{c}_{flg}.png')
        #         flg -= 1

        for key, values in self.buffer.condense_dict.items():
            print(f'class:{key},summarize_images:{len(values)}')
        # if self.old_labels != []:
        #     print('hh')
        self.task_data_distill(distill_train_loader)




        # for c in self.new_labels:
        #     # flg=4
        #     images_c=self.buffer.buffer_img[self.buffer.condense_dict[c]]
        #     for i in range(len(images_c)):
        #         before_distill=scale_tensor(images_c[i].to('cpu'))
        #         before_distill_array = before_distill.numpy().transpose(1, 2, 0).astype(np.uint8) # Rearrange dimensions to HxWxC
        #         img = Image.fromarray(before_distill_array,'RGB')
        #         img.save(f'images/with_mean_feature/distilled/after_distilled_image_{c}_{i}.png')
        # setup tracker
        losses = AverageMeter()
        acc_batch = AverageMeter()


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
                        features = torch.cat([self.model.forward(combined_batch).unsqueeze(1), self.model.forward(combined_batch_aug).unsqueeze(1)], dim=1)
                        loss = self.criterion(features, combined_labels)
                        losses.update(loss, batch_y.size(0))
                        self.opt.zero_grad()
                        loss.backward()
                        self.opt.step()
                    else:
                        batch_aug=self.transform(batch_x)
                        features = torch.cat([self.model.forward(batch_x).unsqueeze(1),
                                              self.model.forward(batch_aug).unsqueeze(1)], dim=1)
                        loss = self.criterion(features, batch_y)
                        losses.update(loss, batch_y.size(0))
                        self.opt.zero_grad()
                        loss.backward()
                        self.opt.step()
                # update memory
                # aff_x.append(batch_x)
                # aff_y.append(batch_y)
                # if len(aff_x) > self.queue_size:
                #     aff_x.pop(0)
                #     aff_y.pop(0)
                # self.buffer.update(batch_x, batch_y, aff_x=aff_x, aff_y=aff_y, update_index=i, transform=self.transform)

                if i % 100 == 1 and self.verbose:
                        print(
                            '==>>> it: {}, avg. loss: {:.6f}, '
                                .format(i, losses.avg(), acc_batch.avg())
                        )




        if self.params.get_distillation_accuracy:
            print('training using original images')
            self.get_original_accruacy(x_train,y_train,labels)
            condense_x_train=torch.cat([self.buffer.buffer_img[self.buffer.condense_dict[c]] for c in self.new_labels],dim=0)
            condense_y_train=torch.cat([self.buffer.buffer_label[self.buffer.condense_dict[c]] for c in self.new_labels],dim=0)
            print("training using distillation images")
            self.get_distillation_accuray(condense_x_train,condense_y_train)


        self.after_train()



    def task_data_distill(self, train_loader):
        # aff_x = []
        # aff_y = []
        #Todo change this into PDD form
        #rewritte dd part(maybe)
        #dict save previous stage synthetic image
        #initilaze model
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

    def get_original_accruacy(self, x_train, y_train, labels):
        label_dict={}
        for i in range(len(self.new_labels)):
            label_dict[self.new_labels[i]]=i
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_sampler = BalancedSampler(x_train, y_train, self.batch)
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, num_workers=0,
                                       drop_last=True, sampler=train_sampler)
        if self.params.data == 'mini_imagenet' or self.params.data == 'tiny_imagenet':
            net_original= SupConResNet(640, head=self.params.head,feat_dim=self.params.num_classes_per_task)
        else:
            net_original = SupConResNet(head=self.params.head,feat_dim=self.params.num_classes_per_task)
        # net_condense=SupConResNet(head=self.params.head,feat_dim=nclass)
        # net_condense = maybe_cuda(net_condense, self.params.cuda)
        net_original = maybe_cuda(net_original, self.params.cuda)
        # net_condense=net_condense.train()
        net_original = net_original.train()

        opt = setup_opt(self.params.optimizer, net_original, self.params.learning_rate, self.params.weight_decay)
        for ep in range(self.params.test_epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(torch.tensor([label_dict[lab.item()] for lab in batch_y]), self.params.cuda)
                batch_aug = self.transform(batch_x)
                features = torch.cat([net_original.forward(batch_x).unsqueeze(1),
                                     net_original.forward(batch_aug).unsqueeze(1)], dim=1)
                loss = self.criterion(features, batch_y)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.evaluate_cur_task(net_original)

    def get_distillation_accuray(self,x,y):
        label_dict = {}
        for i in range(len(self.new_labels)):
            label_dict[self.new_labels[i]] = i
        y = maybe_cuda(torch.tensor([label_dict[lab.item()] for lab in y]), self.params.cuda)
        net = SupConResNet(head=self.params.head,feat_dim=self.params.num_classes_per_task)
        # net_condense=SupConResNet(head=self.params.head,feat_dim=nclass)
        # net_condense = maybe_cuda(net_condense, self.params.cuda)
        net = maybe_cuda(net, self.params.cuda)
        # net_condense=net_condense.train()
        net_original = net.train()
        bs=self.params.batch
        opt = setup_opt(self.params.optimizer, net_original, self.params.learning_rate, self.params.weight_decay)
        data_len = len(x)
        for epoch in range(self.params.test_epoch):
            for tmp_idx in range(data_len // bs):
                batch_x = x[tmp_idx * bs: (tmp_idx + 1) * bs]
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(y[tmp_idx * 10: (tmp_idx + 1) * 10], self.params.cuda)
                batch_aug = self.transform(batch_x)
                features = torch.cat([net_original.forward(batch_x).unsqueeze(1),
                                      net_original.forward(batch_aug).unsqueeze(1)], dim=1)
                loss = self.criterion(features, batch_y)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.evaluate_cur_task(net_original)

    def evaluate_cur_task(self, model):
        model.eval()
        # acc_array = np.zeros(len(self.cur_task_test_loadertest_loaders))
        #  acc_array=[]
        label_dict = {}
        for i in range(len(self.new_labels)):
            label_dict[self.new_labels[i]] = i
        exemplar_means = {}
        cls_exemplar = {label_dict[cls]: [] for cls in self.new_labels}
        task_indices= np.concatenate([self.buffer.condense_dict[c] for c in self.new_labels])
        for x, y in zip(self.buffer.buffer_img[task_indices], self.buffer.buffer_label[task_indices]):
            cls_exemplar[label_dict[y.item()]].append(x)
        for cls, exemplar in cls_exemplar.items():
            features = []
            # Extract feature for each exemplar in p_y
            for ex in exemplar:
                feature = model.features(ex.unsqueeze(0)).detach().clone()
                feature = feature.squeeze()
                feature.data = feature.data / feature.data.norm()  # Normalize
                features.append(feature)
            if len(features) == 0:
                mu_y = maybe_cuda(
                    torch.normal(0, 1, size=tuple(model.features(x.unsqueeze(0)).detach().size())), self.cuda)
                mu_y = mu_y.squeeze()
            else:
                features = torch.stack(features)
                mu_y = features.mean(0).squeeze()
            mu_y.data = mu_y.data / mu_y.data.norm()  # Normalize
            exemplar_means[cls] = mu_y
        with torch.no_grad():
            # for task, test_loader in enumerate(test_loaders):
            acc = AverageMeter()
            for i, (batch_x, batch_y) in enumerate(self.cur_task_test_loader):
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(torch.tensor([label_dict[lab.item()] for lab in batch_y]), self.params.cuda)

                if self.params.trick['ncm_trick'] or self.params.agent in ['ICARL', 'SCR', 'SCP', 'SSCR',
                                                                           'DREAM_SSCR', 'FSR']:
                    feature = model.features(batch_x)  # (batch_size, feature_size)
                    for j in range(feature.size(0)):  # Normalize
                        feature.data[j] = feature.data[j] / feature.data[j].norm()
                    feature = feature.unsqueeze(2)  # (batch_size, feature_size, 1)
                    means = torch.stack(
                        [exemplar_means[label_dict[cls]] for cls in self.new_labels])  # (n_classes, feature_size)

                    # old ncm
                    means = torch.stack([means] * batch_x.size(0))  # (batch_size, n_classes, feature_size)
                    means = means.transpose(1, 2)
                    feature = feature.expand_as(means)  # (batch_size, feature_size, n_classes)
                    dists = (feature - means).pow(2).sum(1)
                    if len(dists.size()) > 2:
                        dists = dists.squeeze()  # (batch_size, n_classes)
                    _, pred_label = dists.min(1)
                    # may be faster
                    # feature = feature.squeeze(2).T
                    # _, preds = torch.matmul(means, feature).max(0)
                    labels=[]
                    for i in self.new_labels:
                        labels.append(label_dict[i])
                    correct_cnt = (np.array(labels)[
                                       pred_label.tolist()] == batch_y.cpu().numpy()).sum().item() / batch_y.size(0)
                else:
                    logits = model.forward(batch_x)
                    _, pred_label = torch.max(logits, 1)
                    correct_cnt = (pred_label == batch_y).sum().item() / batch_y.size(0)

                acc.update(correct_cnt, batch_y.size(0))
                # acc_array[task] = acc.avg()
        print('accuracy: {:.6f}'.format(acc.avg()))

