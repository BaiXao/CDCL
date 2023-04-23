from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import time
from datetime import timedelta
from sklearn.cluster import DBSCAN
import torch
from torch import nn
from torch.nn import Parameter
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image, ImageOps
import os

sys.path.append(os.path.dirname(sys.path[0]))
from cdcl import datasets
from cdcl import models
from cdcl.models.hm import HybridMemory
from cdcl.trainers import CDCLTrainer_USL, CDCLTrainer_UDA
from cdcl.evaluators import Evaluator, extract_features
from cdcl.utils.data import IterLoader
from cdcl.utils.data import transforms as T
from cdcl.utils.data.sampler import RandomMultipleGallerySampler
from cdcl.utils.data.preprocessor import Preprocessor
from cdcl.utils.logging import Logger
from cdcl.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from cdcl.utils.faiss_rerank import compute_jaccard_distance


def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


# 图片百分比灰度化
def gray_top_half(image):
    # 图片灰度化的比例
    ratio = 0.6
    # 将图片裁剪为上下两部分
    width, height = image.size
    top_half = image.crop((0, 0, width, int(height * ratio)))
    bottom_half = image.crop((0, int(height * ratio), width, height))

    # 将上半部分转换为灰度图像
    top_half_gray = ImageOps.grayscale(top_half)

    # 将处理后的上半部分和下半部分合并
    new_image = Image.new('RGB', (width, height))
    new_image.paste(top_half_gray, (0, 0))
    new_image.paste(bottom_half, (0, int(height * ratio)))

    return new_image


# Gray 参数控制图像要不要进行灰度化处理
def get_train_loader(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]),
    ])
    train_transformer2 = T.Compose([
        T.Resize((height, width), interpolation=3),
        # 调用图片部分灰度化函数
        T.Lambda(gray_top_half),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform1=train_transformer,
                                transform2=train_transformer2),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform1=test_transformer, transform2=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def copy_state_dict(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        name = name.replace('module.', '')
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)

    return model


def create_model(args, loadstate_dict):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout, num_classes=0)
    if loadstate_dict:
        initial_weights = load_checkpoint(args.init_1)
        copy_state_dict(initial_weights['state_dict'], model)
    model.cuda()
    model = nn.DataParallel(model)
    return model


def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP

    best_mAP = 0
    start_time = time.monotonic()
    cudnn.benchmark = True
    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    iters = args.iters if (args.iters > 0) else None
    # 泰坦
    args.data_dir = '/home/tq_sunjx/data'
    # 3090
    # args.data_dir = '/home/tq_sjx/data'
    # 租用
    # args.data_dir = '/root/autodl-tmp/data'

    print("==> Load target-domain dataset")
    dataset_target = get_data(args.dataset_target, args.data_dir)
    test_loader_target = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers)

    # Create model
    model1 = create_model(args, loadstate_dict=True)
    model2 = create_model(args, loadstate_dict=False)
    # model1.module是nn.DataParallel包装后的模型，num_features是模型中的一个属性，表示输入特征的维度数。
    memory1 = HybridMemory(model1.module.num_features, len(dataset_target.train),
                           temp=args.temp, momentum=args.momentum).cuda()
    memory2 = HybridMemory(model2.module.num_features, len(dataset_target.train),
                           temp=args.temp, momentum=args.momentum).cuda()

    cluster_loader = get_test_loader(dataset_target, args.height, args.width,
                                     args.batch_size, args.workers, testset=sorted(dataset_target.train))

    # Initialize target-domain instance features
    def getTarget_Domain_Data(dataset_target, model, cluster_loader, modelnum):
        print("==> Initialize target-domain instance features use model{} in the hybrid memory".format(modelnum))
        target_features, _, _ = extract_features(model, cluster_loader, print_freq=50)
        target_features = torch.cat([target_features[f].unsqueeze(0) for f, _, _ in sorted(dataset_target.train)], 0)
        target_features = F.normalize(target_features, dim=1)
        return target_features

    target_features = getTarget_Domain_Data(dataset_target, model1, cluster_loader, modelnum=1)
    memory1.features = target_features.cuda()
    target_features = getTarget_Domain_Data(dataset_target, model2, cluster_loader, modelnum=2)
    memory2.features = target_features.cuda()

    # Evaluator
    evaluator1 = Evaluator(model1)

    params = []
    print('prepare parameter')
    for key, value in model1.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    for key, value in model2.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    # Optimizer
    optimizer = torch.optim.Adam(params)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # Trainer
    trainer = CDCLTrainer_UDA(model1, model2, memory1, memory2)

    for epoch in range(args.epochs):
        # Calculate distance
        if (epoch == 0):
            eps = args.eps
            eps_tight = eps - args.eps_gap
            eps_loose = eps + args.eps_gap
            print('Clustering criterion: eps: {:.3f}, eps_tight: {:.3f}, eps_loose: {:.3f}'.format(eps, eps_tight,
                                                                                                   eps_loose))
            cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
            cluster_tight = DBSCAN(eps=eps_tight, min_samples=4, metric='precomputed', n_jobs=-1)

        def generate_pseudo_labels(cluster_id, num):
            labels = []
            outliers = 0
            for i, ((fname, _, cid), id) in enumerate(zip(sorted(dataset_target.train), cluster_id)):
                if id != -1:
                    labels.append(id)
                else:
                    labels.append(num + outliers)
                    outliers += 1
            return torch.Tensor(labels).long()

        print('==> Create pseudo labels for unlabeled data with self-paced policy')
        features = memory1.features.clone()
        now_time_before_cluster = time.monotonic()
        # Jaccard距离是一种用于衡量两个集合之间差异性的指标
        # 杰卡德相似系数计算距离能够更好地体现集合的相似度，从而得到更好的聚类效果。
        rerank_dist = compute_jaccard_distance(features, k1=args.k1, k2=args.k2)
        # 计算余弦相似度
        cosine_sim = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)

        pseudo_labels = cluster.fit_predict(rerank_dist)
        pseudo_labels_tight = cluster_tight.fit_predict(rerank_dist)
        num_ids = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
        num_ids_tight = len(set(pseudo_labels_tight)) - (1 if -1 in pseudo_labels_tight else 0)

        pseudo_labels = generate_pseudo_labels(pseudo_labels, num_ids)
        pseudo_labels_tight = generate_pseudo_labels(pseudo_labels_tight, num_ids_tight)

        index2label = collections.defaultdict(int)
        for label in pseudo_labels:
            index2label[label.item()] += 1
        index2label = np.fromiter(index2label.values(), dtype=float)
        print('==> Statistics for epoch {}: {} clusters, {} un-clustered instances'
              .format(epoch, (index2label > 1).sum(), (index2label == 1).sum()))
        print('=======> Start pseudo labels refinement for unlabeled data')
        # ==========================重新聚类开始================================================
        rerank_dist_tensor = torch.tensor(rerank_dist)
        rerank_cos_tensor = torch.tensor(cosine_sim)
        N = pseudo_labels.size(0)

        # 计算聚类标签相同的样本对之间的相似度矩阵
        label_sim = pseudo_labels.expand(N, N).eq(pseudo_labels.expand(N, N).t()).float()
        label_sim_tight = pseudo_labels_tight.expand(N, N).eq(pseudo_labels_tight.expand(N, N).t()).float()
        # 计算每个样本到其聚类中其他样本的加权平均 Jaccard 距离
        sim_distance = rerank_dist_tensor.clone() * label_sim
        dists_label_add = (label_sim.sum(-1))
        for i in range(len(dists_label_add)):
            if dists_label_add[i] > 1:
                dists_label_add[i] = dists_label_add[i] - 1
        dists_labels = (label_sim.sum(-1))
        sim_add_averge = sim_distance.sum(-1) / torch.pow(dists_labels, 2)
        # 计算每个簇的平均相似度
        cluster_I_average = torch.zeros((torch.max(pseudo_labels).item() + 1))
        for sim_dists, label in (zip(sim_add_averge, pseudo_labels)):
            cluster_I_average[label.item()] = cluster_I_average[label.item()] + sim_dists
        # 处理紧密度信息
        sim_tight = label_sim.eq(1 - label_sim_tight.clone()).float()
        dists_tight = sim_tight * rerank_dist_tensor.clone()
        cos_dists_tight = sim_tight * rerank_cos_tensor.clone()
        dists_label_tight_add = (1 + sim_tight.sum(-1))
        for i in range(len(dists_label_tight_add)):
            if dists_label_tight_add[i] > 1:
                dists_label_tight_add[i] = dists_label_tight_add[i] - 1
        sim_add_averge = dists_tight.sum(-1) / torch.pow(dists_label_tight_add, 2)
        cos_sim_add_averge = cos_dists_tight.sum(-1) / torch.pow(dists_label_tight_add, 2)
        # 根据阈值判断样本是否为噪声点，将其归为新的聚类或原有的聚类
        cluster_tight_average = torch.zeros((torch.max(pseudo_labels_tight).item() + 1))
        for sim_dists, label in (zip(sim_add_averge, pseudo_labels_tight)):
            cluster_tight_average[label.item()] = cluster_tight_average[label.item()] + sim_dists
        cluster_final_average = torch.zeros(len(sim_add_averge))
        for i, label_tight in enumerate(pseudo_labels_tight):
            cluster_final_average[i] = cluster_tight_average[label_tight.item()]

        cos_cluster_tight_average = torch.zeros((torch.max(pseudo_labels_tight).item() + 1))
        for sim_dists, label in (zip(cos_sim_add_averge, pseudo_labels_tight)):
            cos_cluster_tight_average[label.item()] = cos_cluster_tight_average[label.item()] + sim_dists
        cos_cluster_final_average = torch.zeros(len(cos_sim_add_averge))
        for i, label_tight in enumerate(pseudo_labels_tight):
            cos_cluster_final_average[i] = cos_cluster_tight_average[label_tight.item()]

        pseudo_labeled_dataset = []
        outliers = 0
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_target.train), pseudo_labels)):
            D_score = cluster_final_average[i]
            cos_Score = cos_cluster_final_average[i]
            if args.ratio_cluster * D_score.item() + (1-args.ratio_cluster) * cos_Score.item() <= cluster_I_average[label.item()]:
                pseudo_labeled_dataset.append((fname, label.item(), cid))
            else:
                pseudo_labeled_dataset.append((fname, len(cluster_I_average) + outliers, cid))
                pseudo_labels[i] = len(cluster_I_average) + outliers
                outliers += 1
        # ==========================重新聚类结束================================================

        index2label = collections.defaultdict(int)
        for label in pseudo_labels:
            index2label[label.item()] += 1
        index2label = np.fromiter(index2label.values(), dtype=float)
        print('==> Statistics for epoch {}: {} clusters, {} un-clustered instances\n'
              .format(epoch, (index2label > 1).sum(), (index2label == 1).sum()))
        print('=======> Finish pseudo labels refinement for unlabeled data on model')
        now_time_after_epoch = time.monotonic()

        print(
            'the time of cluster refinement is {}'.format(now_time_after_epoch - now_time_before_cluster)
        )
        memory1.labels = pseudo_labels.cuda()
        train_loader1_target = get_train_loader(args, dataset_target, args.height, args.width,
                                                args.batch_size, args.workers, args.num_instances, iters,
                                                trainset=pseudo_labeled_dataset)
        memory2.labels = pseudo_labels.cuda()
        train_loader2_target = get_train_loader(args, dataset_target, args.height, args.width,
                                                args.batch_size, args.workers, args.num_instances, iters,
                                                trainset=pseudo_labeled_dataset)
        train_loader1_target.new_epoch()
        train_loader2_target.new_epoch()

        trainer.train(epoch, train_loader1_target, train_loader2_target, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader1_target))

        if ((epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1)):
            cmc_socore1, mAP1 = evaluator1.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery,
                                                    cmc_flag=False)
            mAP = mAP1
            print('model1 is better')
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model1.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))
            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.format(epoch, mAP, best_mAP,
                                                                                            ' *' if is_best else ''))
        lr_scheduler.step()

    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model1.load_state_dict(checkpoint['state_dict'])
    evaluator1.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Cluster-guided Asymmetric Contrastive Learning for Unsupervised Person Re-Identification")
    # data
    parser.add_argument('-ds', '--dataset-source', type=str, default='dukemtmcreid',
                        choices=datasets.names())
    parser.add_argument('-dt', '--dataset-target', type=str, default='market1501',
                        choices=datasets.names())
    # parser.add_argument('-ds', '--dataset-source', type=str, default='market1501',
    #                     choices=datasets.names())
    # parser.add_argument('-dt', '--dataset-target', type=str, default='dukemtmcreid',
    #                     choices=datasets.names())
    # parser.add_argument('-ds', '--dataset-source', type=str, default='market1501',
    #                     choices=datasets.names())
    # parser.add_argument('-dt', '--dataset-target', type=str, default='msmt17',
    #                     choices=datasets.names())

    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.60,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,

                        help="hyperparameter for jaccard distance")
    parser.add_argument('--output_weight', type=float, default=1.0,
                        help="loss outputs for weight ")
    parser.add_argument('--ratio_cluster', type=float, default=1.0,
                        help="cluster hypter ratio ")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    parser.add_argument('--loss-size', type=int, default=2)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)
    parser.add_argument('--sic_weight', type=float, default=1,
                        help="loss outputs for sic ")
    # training configs
    parser.add_argument('--seed', type=int, default=111)  #
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=5)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    # 需要更改参数
    # preTrainMode
    # Taitan
    parser.add_argument('--init-1', type=str,
                        default='/home/tq_sunjx/preTrain/dukemtmcreid2market1501/resnet50-pretrain/model_best.pth.tar',
                        metavar='PATH')
    # parser.add_argument('--init-1', type=str,
    #                     default='/home/tq_sunjx/preTrain/market15012dukemtmcreid/resnet50-pretrain/model_best.pth.tar',
    #                     metavar='PATH')
    # parser.add_argument('--init-1', type=str,
    #                     default='/home/tq_sunjx/preTrain/market15012msmt17/resnet50-pretrain/model_best.pth.tar',
    #                     metavar='PATH')
    # parser.add_argument('--init-1', type=str,
    #                     default='/home/tq_sunjx/preTrain/dukemtmcreid2msmt17/resnet50-pretrain/model_best.pth.tar',
    #                     metavar='PATH')
    parser.add_argument('--imgGrayRatio', type=int, default=0.6)
    main()
