import torch
import numpy as np
from torch.utils import data
from utils.buffer.buffer import Buffer, DynamicBuffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform, BalancedSampler
from utils.setup_elements import transforms_match, input_size_match
from utils.utils import maybe_cuda, AverageMeter,TensorDataset
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
import torch.nn as nn
import copy
import time
from models.convnet import ConvNet
from utils.buffer.augment import DiffAug
from torchvision import transforms


class SummarizeContrastReplay(ContinualLearner):
    def __init__(self, model, opt, params):
        super(SummarizeContrastReplay, self).__init__(model, opt, params)
        self.buffer = DynamicBuffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
        # self.transform = nn.Sequential(
        #     RandomResizedCrop(size=(input_size_match[self.params.data][1], input_size_match[self.params.data][2]), scale=(0.2, 1.)),
        #     RandomHorizontalFlip(),
        #     ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
        #     RandomGrayscale(p=0.2)
        #
        # )
        diff_aug = DiffAug(strategy='color_crop_flip_scale_rotate', batch=False)
        if not self.params.match_aug:
            self.transform = nn.Sequential(
                RandomResizedCrop(size=(input_size_match[self.params.data][1], input_size_match[self.params.data][2]), scale=(0.2, 1.)),
                RandomHorizontalFlip(),
                ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                RandomGrayscale(p=0.2)

            )
        else:
            self.transform = transforms.Compose([diff_aug])
        self.queue_size = params.queue_size
        if self.params.data == 'mini_imagenet':
            self.im_size = (84, 84)
            self.feature_dim = 12800
        elif self.params.data == 'tiny_imagenet':
            self.feature_dim = 8192
            self.im_size = (64, 64)
        else:
            self.feature_dim = 2048
            self.im_size = (32, 32)

    def calculate_ipc(self):

        current_classes_num=self.params.num_classes_per_task*self.buffer.task_id
        buffer_size=self.params.mem_size
        self.new_ipc = buffer_size // current_classes_num
        self.old_ipc=self.new_ipc+1
        self.new_ipc_index= current_classes_num*(self.old_ipc)-buffer_size
        self.old_ipc_index= current_classes_num-self.new_ipc_index
        self.class_index=0
        if self.new_ipc_index>0:
            self.buffer.images_per_class=self.new_ipc
            self.ipc=self.new_ipc
        else:
            self.buffer.images_per_class=self.old_ipc
            self.ipc=self.old_ipc
        self.ipc=self.params.images_per_class
        #self.outer_loop =1#45+self.ipc*5
        if self.params.match=='feat':
            if self.ipc<5:
                self.buffer.update_method.params.lr_img=1
            elif self.ipc<10:
                self.buffer.update_method.params.lr_img=5
            elif self.ipc<50:
                self.buffer.update_method.params.lr_img=10
            else:
                self.buffer.update_method.params.lr_img= 50
        elif self.params.match=='grad':
            if self.ipc<5:
                self.buffer.update_method.params.lr_img=2e-4
            elif self.ipc<10:
                self.buffer.update_method.params.lr_img=4e-4
            elif self.ipc<50:
                self.buffer.update_method.params.lr_img=1e-3
            else:
                self.buffer.update_method.params.lr_img= 2e-3
       # self.lr_img=min(10*(self.im_size[0]+31)//32,self.ipc)


    def summarize_past_task(self):

        # self.indices_class={}
        self.class_data = {}
        self.class_logit={}
        self.old_buffer_data=copy.deepcopy(self.buffer.buffer_img)
        #self.old_buffer_logit=copy.deepcopy(self.buffer.buffer_logits)
        #self.old_buffer_label = copy.deepcopy(self.buffer.buffer_label)
        self.label_dict = {}
        for idx, label in enumerate(self.old_labels):
            self.label_dict[label] = idx #+ self.buffer.num_classes - len(self.old_labels)
            self.class_data[label] = self.old_buffer_data[self.buffer.condense_dict[label]]
            #self.class_logit[label]=self.old_buffer_logit[self.buffer.condense_dict[label]]


        net = maybe_cuda(ConvNet(len(self.old_labels), im_size=self.im_size), self.params.cuda)
        for i in self.old_labels:
            self.class_index+=1
            if self.class_index<=self.old_ipc_index:
                ipc=self.old_ipc
            else:
                ipc=self.new_ipc
            image = self.class_data[i]
            #logits= self.class_logit[i]
            # image = maybe_cuda(torch.stack([transform(ee) for ee in image]), self.params.cuda)
            old_condense_dict = copy.deepcopy(self.buffer.condense_dict[i])
            self.buffer.condense_dict[i] = []

            #image_transform = self.transform(image)
            #strategy = NEW_Strategy(image_transform, self.model,method='mean_feature')
            #ipc = self.params.mem_size // self.buffer.task_id // self.params.num_classes_per_task
            # query_idxs = strategy.query(ipc).cpu().numpy()
            # query_idxs.sort()
            query_idxs=list(range(ipc))
            for ii in range(ipc):
                self.buffer.buffer_img[old_condense_dict[query_idxs[ii]]].data.copy_(image[query_idxs[ii]])
                self.buffer.buffer_img_rep[old_condense_dict[query_idxs[ii]]].data.copy_(image[query_idxs[ii]])
                #self.buffer.buffer_logits[old_condense_dict[query_idxs[ii]]].data.copy_(logits[query_idxs[ii]])
                self.buffer.buffer_label[old_condense_dict[query_idxs[ii]]].data.copy_(torch.tensor(i))
                self.buffer.condense_dict[i].append(old_condense_dict[query_idxs[ii]])
                # old_condense_dict.pop(query_idxs[ii])
            for j in old_condense_dict:
                if j not in self.buffer.condense_dict[i]:
                    self.buffer.avail_indices.append(j)






    def train_learner(self, x_train, y_train, labels):
        self.before_train(x_train, y_train)
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_sampler = BalancedSampler(x_train, y_train, self.batch)
        if self.params.num_tasks == 1:
            train_loader = data.DataLoader(train_dataset, batch_size=self.batch, num_workers=0,
                                           drop_last=True, shuffle=True)

        else:
            train_loader = data.DataLoader(train_dataset, batch_size=self.batch, num_workers=0,
                                           drop_last=True, sampler=train_sampler)

        # set up model
        self.model = self.model.train()
        self.buffer.new_condense_task(labels)

        # setup tracker
        losses = AverageMeter()
        acc_batch = AverageMeter()
        # self.calculate_ipc()
        # if self.buffer.task_id>1:
        #     self.summarize_past_task()
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
                        features = torch.cat([self.model.forward(combined_batch).unsqueeze(1), self.model.forward(combined_batch_aug).unsqueeze(1)], dim=1)
                        loss = self.criterion(features, combined_labels)
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

        for key,values in self.buffer.condense_dict.items():
            print(f'class:{key},summarize_images:{len(values)}')
        if self.params.get_distillation_accuracy:
            print('training using original images')
            #self.get_original_accruacy(x_train, y_train, labels)
            condense_x_train = torch.cat(
                [self.buffer.buffer_img[self.buffer.condense_dict[c]] for c in self.new_labels], dim=0)
            condense_y_train = torch.cat(
                [self.buffer.buffer_label[self.buffer.condense_dict[c]] for c in self.new_labels], dim=0)
            print("training using distillation images")
            self.get_distillation_accuray(condense_x_train, condense_y_train)
        self.after_train()





    def get_distillation_accuray(self, x, y):
        # x=self.params.num_classes_per_task
        self.label_dict = {}
        for i, label in enumerate(self.new_labels):
            self.label_dict[label] = i
        if self.params.data == 'mini_imagenet':
            self.im_size = (84, 84)
        elif self.params.data == 'tiny_imagenet':
            self.im_size = (64, 64)
        else:
            self.im_size = (32, 32)
        # self.label_dict=self.buffer.update_method.label_dict
        net_eval = maybe_cuda(ConvNet(len(self.new_labels), im_size=self.im_size), self.params.cuda)
        # net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)  # get a random model
        image_syn_eval, label_syn_eval = copy.deepcopy(x.detach()), copy.deepcopy(
            y.detach())  # avoid any unaware modification
        # label_syn_eval=maybe_cuda(torch.tensor([self.label_dict[label_syn.item()] for label_syn in label_syn_eval]),self.params.cuda)
        label_syn_eval = maybe_cuda(label_syn_eval, self.params.cuda)
        self.evaluate_synset(100, net_eval, image_syn_eval, label_syn_eval)
        # accs.append(acc_test)
        # label_dict = {}
        # for i in range(len(self.new_labels)):
        #     label_dict[self.new_labels[i]] = i
        # y = maybe_cuda(torch.tensor([label_dict[lab.item()] for lab in y]), self.params.cuda)
        # net = SupConResNet(head=self.params.head,feat_dim=self.params.num_classes_per_task)
        # # net_condense=SupConResNet(head=self.params.head,feat_dim=nclass)
        # # net_condense = maybe_cuda(net_condense, self.params.cuda)
        # net = maybe_cuda(net, self.params.cuda)
        # # net_condense=net_condense.train()
        # net_original = net.train()
        # bs=self.params.batch
        # opt = setup_opt(self.params.optimizer, net_original, self.params.learning_rate, self.params.weight_decay)
        # data_len = len(x)
        # for epoch in range(self.params.test_epoch):
        #     for tmp_idx in range(data_len // bs):
        #         batch_x = x[tmp_idx * bs: (tmp_idx + 1) * bs]
        #         batch_x = maybe_cuda(batch_x, self.cuda)
        #         batch_y = maybe_cuda(y[tmp_idx * 10: (tmp_idx + 1) * 10], self.params.cuda)
        #         batch_aug = self.transform(batch_x)
        #         features = torch.cat([net_original.forward(batch_x).unsqueeze(1),
        #                               net_original.forward(batch_aug).unsqueeze(1)], dim=1)
        #         loss = self.criterion(features, batch_y)
        #         opt.zero_grad()
        #         loss.backward()
        #         opt.step()
        # self.evaluate_cur_task(net_original)

    def evaluate_synset(self, it_of_eval, net, images_train, labels_train):
        lr = 0.01
        Epoch = self.params.test_epoch
        lr_schedule = [Epoch // 2 + 1]
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        criterion = maybe_cuda(nn.CrossEntropyLoss())

        dst_train = TensorDataset(images_train, labels_train)
        # dst_train= dataset_transform(images_train,labels_train,transform=None)
        # trainloader = data.DataLoader(dst_train, batch_size=self.params.distill_batch)
        trainloader = data.DataLoader(dst_train, batch_size=self.params.distill_batch, shuffle=True, num_workers=0)

        start = time.time()
        for ep in range(Epoch + 1):
            loss_train, acc_train = self.test_epoch('train', trainloader, net, optimizer, criterion, aug=True)
            if ep in lr_schedule:
                lr *= 0.1
                optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

        time_train = time.time() - start
        loss_test, acc_test = self.test_epoch('test', self.cur_task_test_loader, net, optimizer, criterion,
                                              aug=False)
        print(
            'Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (
                Epoch, Epoch, int(time_train), loss_train, acc_train, acc_test))

        return net, acc_train, acc_test

    def test_epoch(self, mode, dataloader, net, optimizer, criterion, aug):
        loss_avg, acc_avg, num_exp = 0, 0, 0
        # net = net.to(args.device)
        # criterion = criterion.to(args.device)

        if mode == 'train':
            net.train()
        else:
            net.eval()

        for i_batch, datum in enumerate(dataloader):
            img = maybe_cuda(datum[0].float(), self.params.cuda)

            # diff_aug = DiffAug(strategy='color_crop', batch=False)
            # match_aug = transforms.Compose([diff_aug])
            # img = match_aug(img)
            lab = datum[1].long()
            # if  mode=='train':
            #     lab = maybe_cuda(lab,self.params.cuda)
            # else:
            lab = maybe_cuda(torch.tensor([self.label_dict[label_syn.item()] for label_syn in lab]),
                             self.params.cuda)
            n_b = lab.shape[0]

            output = net(img)
            loss = criterion(output, lab)
            acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

            loss_avg += loss.item() * n_b
            acc_avg += acc
            num_exp += n_b

            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        loss_avg /= num_exp
        acc_avg /= num_exp

        return loss_avg, acc_avg
