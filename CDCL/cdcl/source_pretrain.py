from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import time
from datetime import timedelta
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from PIL import Image, ImageOps
import os

sys.path.append(os.path.dirname(sys.path[0]))
from cdcl import datasets
from cdcl import models
from cdcl.trainers import CDCLTrainer_USL, CDCLTrainer_UDA, PreTrainer
from cdcl.evaluators import Evaluator, extract_features
from cdcl.utils.data import IterLoader
from cdcl.utils.data import transforms as T
from cdcl.utils.data.sampler import RandomMultipleGallerySampler
from cdcl.utils.data.preprocessor import Preprocessor
from cdcl.utils.logging import Logger
from cdcl.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict


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


def create_model(args,num_classes):
    model = models.create(args.arch, num_features=args.features, dropout=args.dropout,
                          num_classes=[num_classes],mb_h=2048, sour_class=751)
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

    print("==> Load source-domain dataset")
    dataset_source = get_data(args.dataset_source, args.data_dir)
    print("==> Load target-domain dataset")
    dataset_target = get_data(args.dataset_target, args.data_dir)

    test_loader_target = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers)
    train_loader_source = get_train_loader(args, dataset_source, args.height, args.width,
                                            args.batch_size, args.workers, args.num_instances, iters)
    source_classes = dataset_source.num_train_pids
    train_loader_target = get_train_loader(args, dataset_target, args.height, args.width,
                                            args.batch_size, args.workers, args.num_instances, iters)
    # Create model
    model = create_model(args,source_classes)

    # Evaluator
    evaluator = Evaluator(model)

    params = []
    print('prepare parameter')
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    optimizer = torch.optim.Adam(params)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # Trainer
    trainer = PreTrainer(model,source_classes)

    for epoch in range(args.epochs):
        train_loader_source.new_epoch()
        train_loader_target.new_epoch()

        trainer.train(epoch, train_loader_source, train_loader_target,
                      optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader_source))

        if ((epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1)):
            cmc_socore1, mAP1 = evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery,
                                                    cmc_flag=False)
            mAP = mAP1
            print('model1 is better')
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))
            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.format(epoch, mAP, best_mAP,
                                                                                            ' *' if is_best else ''))
        lr_scheduler.step()

    print('==> Test with the best model on the target:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Cluster-guided Asymmetric Contrastive Learning for Unsupervised Person Re-Identification")
    # data
    parser.add_argument('-ds', '--dataset-source', type=str, default='dukemtmcreid',
                        choices=datasets.names())
    parser.add_argument('-dt', '--dataset-target', type=str, default='msmt17',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=32)
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
    # 200和300相差1个点，所以就用200
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)
    parser.add_argument('--sic_weight', type=float, default=1,
                        help="loss outputs for sic ")
    # training configs
    parser.add_argument('--seed', type=int, default=1)  #
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

    main()
