import sys
import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils import data
from utils.buffer.buffer import Buffer, FSSBuffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform, BalancedSampler
from utils.setup_elements import transforms_match, input_size_match
from utils.utils import maybe_cuda, AverageMeter


import datetime



class BPSReplay(ContinualLearner):
    def __init__(self, model, opt, params):
        super(BPSReplay, self).__init__(model, opt, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
        self.buffer = FSSBuffer(self.model,self.params)

        self.m=params.momentum_m
        self.update_global_step=0




    def save_model(self):

        log_path = os.path.join(
            self.params.logs_dir,
            f"{self.params.data}{self.params.tag}",
            self.params.agent,
            f"task_number_{self.params.num_tasks}_{self.params.seed}",
            f"{self.params.update}batch_size{self.params.batch}_epoch{self.params.epoch}_{self.params.mem_size}",
            f"{self.params.data}_buffer_{self.mem_size}.pth"
        )
        torch.save(self.model.state_dict(),log_path)
        print(f"Save model successfully to {log_path}")


    @torch.no_grad()
    def update_distilled_model(self, steps):
        self.model.update_slow_net(self.m)

    def train_learner(self, x_train, y_train, labels):

        self.before_train(x_train, y_train)

        log_dir = "logs/DSA/task" + str(
           self.buffer.task_id) + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #self.loss_file = open(log_dir + ".txt", "w")
        if self.params.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
        self.global_step = 0
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_sampler = BalancedSampler(x_train, y_train, self.batch)
        if self.params.num_tasks == 1 or self.batch<10:
            train_loader = data.DataLoader(train_dataset, batch_size=self.batch, num_workers=0,
                                           drop_last=True, shuffle=True)

        else:
            train_loader = data.DataLoader(train_dataset, batch_size=self.batch, num_workers=0,
                                           drop_last=True, sampler=train_sampler)

        # set up model
        self.model.train()

        self.buffer.new_condense_task(labels)


        losses = AverageMeter()
        acc_batch = AverageMeter()
        con_losses=AverageMeter()
        con_buffer_losses=AverageMeter()


        steps=0
        opt = torch.optim.SGD(self.model.fast_net.parameters(), lr=self.params.learning_rate)

        for ep in range(self.epoch):

            for i, batch_data in enumerate(train_loader):

                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)

                logits = self.model.fast_forward(batch_x)

                loss = self.criterion(logits, batch_y)
                if self.params.batch_cons_weight > 0:
                    ema_logits = self.model.forward(batch_x)

                    l_cons = self.compute_loss(logits, ema_logits.detach(), loss_type=self.params.consistency_type)

                    l_reg = self.params.batch_cons_weight * l_cons
                    loss += l_reg
                    con_losses.update(l_reg, batch_y.size(0))
                _, pred_label = torch.max(logits, 1)
                correct_cnt = (pred_label == batch_y).sum().item() / batch_y.size(0)

                acc_batch.update(correct_cnt, batch_y.size(0))
                losses.update(loss, batch_y.size(0))

                for j in range(self.params.mem_iters):
                    mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y, return_indices=True)

                    if mem_x.size(0) > 0:
                        mem_x = maybe_cuda(mem_x, self.cuda)
                        mem_y = maybe_cuda(mem_y, self.cuda)

                        mem_logits = self.model.fast_forward(mem_x)
                        loss += self.criterion(mem_logits, mem_y)
                        if self.params.buffer_cons_weight > 0:
                            indices = [i for i, y in enumerate(mem_y) if y not in self.new_labels]
                            if len(indices) > 0:
                                batch_logits = mem_logits[indices]

                                ema_logits = self.model.forward(mem_x[indices])

                                l_cons = self.compute_loss(batch_logits, ema_logits.detach(),
                                                           loss_type=self.params.consistency_type)
                                l_reg = l_cons * self.params.buffer_cons_weight

                                loss += l_reg
                                con_buffer_losses.update(l_reg, ema_logits.size(0))

                opt.zero_grad()
                loss.backward()
                opt.step()

                with torch.no_grad():
                    self.update_distilled_model(steps)
                self.buffer.update(batch_x, batch_y)
                if i % 100 == 1 and self.verbose:
                    print(
                        '==>>> it: {}, avg. loss: {:.6f}, con loss:{:.6f}, con buffer loss:{:.6f}'
                        .format(i, losses.avg(), con_losses.avg(), con_buffer_losses.avg())
                    )



        self.get_fast_model_accuracy()
        for key, values in self.buffer.condense_dict.items():
            print(f'class:{key},summarize_images:{len(values)}')



        self.after_train()
        if self.params.save_pth and self.buffer.task_id==self.params.num_tasks:
            self.save_model()


    def get_fast_model_accuracy(self):
        # self.fast_model.eval()
        self.model.eval()
        with torch.no_grad():
            acc_array = np.zeros(len(self.test_loaders))

            for task, test_loader in enumerate(self.test_loaders):
                acc = AverageMeter()

                for i, (batch_x, batch_y) in enumerate(test_loader):
                    batch_x = maybe_cuda(batch_x, self.cuda)
                    batch_y = maybe_cuda(batch_y, self.cuda)
                    logits, features = self.model.fast_forward(batch_x, return_features=True)

                    _, pred_label = torch.max(logits, 1)

                    correct_cnt = (pred_label == batch_y).sum().item() / batch_y.size(0)

                    acc.update(correct_cnt, batch_y.size(0))

                acc_array[task] = acc.avg()

        self.model.train()
        print(f'fast model acc:{acc_array}')

    def compute_loss(self, logits, ema_logits, loss_type='mse'):
        if loss_type == 'mse':
            loss = torch.mean(F.mse_loss(logits, ema_logits.detach(), reduction='none'))
        elif loss_type == 'l1':
            loss = torch.mean(torch.abs(logits - ema_logits.detach()))
        elif loss_type == 'kl':
            p = F.softmax(logits, dim=-1)
            q = F.softmax(ema_logits.detach(), dim=-1)
            loss = torch.mean(F.kl_div(p.log(), q, reduction='none').sum(dim=-1))
        elif loss_type == 'cos':
            cosine_sim = F.cosine_similarity(logits, ema_logits.detach(), dim=-1)
            loss = torch.mean(1 - cosine_sim)  # 1 - 相似度作为损失
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        return loss
