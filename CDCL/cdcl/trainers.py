from __future__ import print_function, absolute_import
import time
import numpy as np
import collections

import torch
import torch.nn as nn
from torch.nn import functional as F
from .evaluation_metrics import accuracy
from .loss import TripletLoss, SoftTripletLoss, CrossEntropyLabelSmooth_s, \
    CrossEntropyLabelSmooth_c, SoftEntropy
from .utils.meters import AverageMeter


class PreTrainer(object):
    def __init__(self, model, num_classes, margin=0.0):
        super(PreTrainer, self).__init__()
        self.model = model
        self.source_classes = num_classes
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()
        self.criterion_ce_s = CrossEntropyLabelSmooth_s(num_classes).cuda()

    def train(self, epoch, train_loader_source, train_loader_target,
              optimizer, print_freq=10, train_iters=400):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            source_inputs = train_loader_source.next()
            target_inputs = train_loader_target.next()

            data_time.update(time.time() - end)

            # process inputs
            source_inputs, _, source_pid, _ = self._parse_data(source_inputs)
            target_inputs, _, _, _ = self._parse_data(target_inputs)

            # forward
            s_features, s_cls_out, _, _ = self.model(source_inputs, training=True)
            # target samples: only forward
            _, _, _, _ = self.model(target_inputs, training=True)

            loss_ce, loss_tr, prec1 = self._forward(s_features, s_cls_out[0], source_pid)
            loss = loss_ce + loss_tr

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            precisions.update(prec1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % print_freq == 0):
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tr {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tr.val, losses_tr.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs1, imgs2, _, pids, _, indexes = inputs
        return imgs1.cuda(), imgs2.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, s_features, s_outputs, targets):
        loss_ce = self.criterion_ce_s(s_outputs, targets)
        if isinstance(self.criterion_triple, SoftTripletLoss):
            loss_tr = self.criterion_triple(s_features, s_features, targets)
        elif isinstance(self.criterion_triple, TripletLoss):
            loss_tr, _ = self.criterion_triple(s_features, targets)
        prec, = accuracy(s_outputs.data, targets.data)
        prec = prec[0]
        return loss_ce, loss_tr, prec


class CDCLTrainer_UDA(object):
    def __init__(self, encoder1, encoder2, memory1, memory2):
        super(CDCLTrainer_UDA, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.memory1 = memory1
        self.memory2 = memory2

    def train(self, epoch, data_loader1_target, data_loader2_target,
              optimizer, print_freq=10, train_iters=400):
        self.encoder1.train()
        self.encoder2.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        losses_t_1 = AverageMeter()
        losses_t_2 = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            target_inputs_1 = data_loader1_target.next()

            data_time.update(time.time() - end)

            # process inputs
            target_inputs1, target_inputs2, _, target_indexes = self._parse_data(target_inputs_1)

            bn_x1, full_conect1, bn_x2, full_conect2 = self._forward(target_inputs1, target_inputs2)
            # compute loss with the hybrid memory
            loss_t_1 = self.memory1(bn_x1, full_conect2.clone(), target_indexes,back = False)
            loss_t_2 = self.memory2(bn_x2, full_conect1.clone(), target_indexes, back=True)
            criterion = nn.CrossEntropyLoss()
            lossSce = criterion(full_conect1, full_conect2.argmax(dim=1))

            loss = loss_t_1 + loss_t_2 +lossSce
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_t_1.update(loss_t_1.item())
            losses_t_2.update(loss_t_2.item())
            losses.update(loss.item())
            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_t_1 {:.3f} ({:.3f})\t'
                      'Loss_t_2 {:.3f} ({:.3f})\t'
                      'AllLoss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader1_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_t_1.val, losses_t_1.avg,
                              losses_t_2.val, losses_t_2.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs1, imgs2, _, pids, _, indexes = inputs
        return imgs1.cuda(), imgs2.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs1, inputs2):
        bn_x1, full_conect1 = self.encoder1(inputs1)
        bn_x2, full_conect2 = self.encoder2(inputs2)
        return bn_x1, full_conect1, bn_x2, full_conect2


class CDCLTrainer_USL(object):
    def __init__(self, encoder, memory):
        super(CDCLTrainer_USL, self).__init__()
        self.encoder = encoder
        self.memory = memory

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            inputs = data_loader.next()
            data_time.update(time.time() - end)
            inputs, _, indexes = self._parse_data(inputs)
            f_out = self._forward(inputs)
            loss = self.memory(f_out, indexes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item())
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)


class CDCLSIC_USL(object):
    def __init__(self, encoder1, encoder2, memory1, memory2):
        super(CDCLSIC_USL, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.memory1 = memory1
        self.memory2 = memory2

    def train(self, epoch, data_loader1, data_loader2, optimizer, print_freq=10, train_iters=400):
        self.encoder1.train()
        self.encoder2.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        losses1 = AverageMeter()
        losses2 = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs1 = data_loader1.next()
            inputs3 = data_loader2.next()

            data_time.update(time.time() - end)

            inputs1, inputs2, _, indexes1 = self._parse_data(inputs1)

            bn_x1, full_conect1, bn_x2, full_conect2 = self._forward(inputs1, inputs2)

            flag = 0
            loss1 = self.memory1(bn_x1, full_conect2.clone(), indexes1, back=flag)
            flag = 1
            loss2 = self.memory2(bn_x2, full_conect1.clone(), indexes1, back=flag)

            loss = (loss1 + loss2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader1),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs1, imgs2, _, pids, _, indexes = inputs
        return imgs1.cuda(), imgs2.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs1, inputs2):
        bn_x1, full_conect1 = self.encoder1(inputs1)
        bn_x2, full_conect2 = self.encoder2(inputs2)
        return bn_x1, full_conect1, bn_x2, full_conect2
