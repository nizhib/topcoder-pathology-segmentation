import argparse
import os
import shutil
import time
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycrayon import CrayonClient
from scipy.special import expit as sigmoid
from torch import nn
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm

from loader import get_loaders
from loss import dice_th, fmicro_th, score_th
from models import linknet, unet, fusion, pspnet
from utils import make_experiment

models = {
    'linknet': linknet.LinkNet,
    'unet': unet.UNet,
    'fusion': fusion.FusionNet,
    'pspnet': pspnet.PSPNet
}

parser = argparse.ArgumentParser(description='Konika Training')
parser.add_argument('--name', default='invalid_9000', type=str, metavar='M',
                    help='model name to train')

parser.add_argument('-l', '--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-e', '--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-d', '--num-epochs-per-decay', default=100, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('-w', '--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
args = None

criterion = nn.modules.loss.BCEWithLogitsLoss().cuda()

colormap = np.load('cmap.npy')


def apply_colormap(probs):
    colored = np.stack([np.take(colormap[:, :, 0], probs),
                        np.take(colormap[:, :, 1], probs),
                        np.take(colormap[:, :, 2], probs)], -1)
    return colored


def adjust_lr(optimizer, epoch, init_lr=0.1, num_epochs_per_decay=10, lr_decay_factor=0.1):
    lr = init_lr * (lr_decay_factor ** (epoch // num_epochs_per_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def run(net, name, fold, optimizer, epoch, metrics, loader, exp):
    training = optimizer is not None

    accum = OrderedDict({'loss': 0})
    accum.update({m[0]: 0.0 for m in metrics})
    dtime = 0.0
    ttime = 0.0

    images = []
    ytrues = []
    ypreds = []

    if training:
        exp.add_scalar_value(f'optimizer/lr', optimizer.param_groups[0]['lr'], step=epoch + 1)

    for i, data in tqdm(enumerate(loader), total=len(loader)):
        # acquire data
        t0 = time.time()

        images, ytrues, _, _ = data
        if training:
            images = Variable(images.cuda())
            ytrues = Variable(ytrues.cuda(async=True))
        else:
            images = Variable(images.cuda(), volatile=True)
            ytrues = Variable(ytrues.cuda(async=True), volatile=True)

        dtime += time.time() - t0

        # optimize | predict & evaluate
        t0 = time.time()

        if training:
            optimizer.zero_grad()

        ypreds = net(images)
        if isinstance(ypreds, tuple):
            loss = sum(criterion(o, ytrues) for o in ypreds)
        else:
            loss = criterion(ypreds, ytrues)

        if training:
            loss.backward()
            optimizer.step()

        accum['loss'] += loss.data.cpu()[0] * ypreds.size()[0]
        for acc, func in metrics:
            value = func(F.sigmoid(ypreds)[:, :, 6:-6, 6:-6].contiguous(),
                         ytrues[:, :, 6:-6, 6:-6].contiguous())
            accum[acc] += value * ypreds.size()[0]

        ttime += time.time() - t0

    for acc in accum:
        accum[acc] = accum[acc] / len(loader.dataset)

    for i in range(4):
        ims = []
        for image, ytrue, ypred in zip(images.cpu().data.numpy()[i::4],
                                       ytrues.cpu().data.numpy()[i::4],
                                       ypreds.cpu().data.numpy()[i::4]):
            image = np.uint8(np.swapaxes((np.swapaxes(image, 1, 2) + 3) / 6, 0, 2) * 255)
            ytrue = np.uint8(ytrue[0] * 255)
            yprob = np.uint8(sigmoid(ypred[0]) * 255)
            im = np.hstack([image[:, :, ::-1],
                            apply_colormap(ytrue),
                            apply_colormap(yprob)])
            ims.append(im)
        im = np.vstack(ims)
        if training:
            cv2.imwrite(f'output/{name}/fold{fold}/train/{epoch + 1:03d}_{i}.png', im)
        else:
            cv2.imwrite(f'output/{name}/fold{fold}/valid/{i}_{epoch + 1:03d}.png', im)

    if training:
        print(f'| Train: [{epoch + 1}]  ', end='')
    else:
        print(f'| Valid: [{epoch + 1}]  ', end='')
    print(f'  Time: {ttime:0.3f}  Data: {dtime:0.3f}', end='')
    for acc in accum:
        if accum[acc] < 1:
            exp.add_scalar_value(f'loss/{acc}', accum[acc], step=epoch + 1)
        print(f' {acc} {accum[acc]:.4f} ', end='')
    print()

    return accum['score']


def train_eval(net, name, trainloader, validloader, fold, exp,
               init_lr=0.1, epochs=30, num_epochs_per_decay=10):
    metrics = (
        ('f1', fmicro_th),
        ('dice', dice_th),
        ('score', score_th)
    )

    # optimizer = optim.SGD(net.parameters(), lr=init_lr,
    #                       momentum=args.momentum,
    #                       weight_decay=args.weight_decay)

    optimizer = optim.Adam(net.parameters(), lr=init_lr, weight_decay=args.wd)

    train_exp, valid_exp = exp

    best_epoch = -1
    best_score = -1.0
    for epoch in range(epochs):
        adjust_lr(optimizer, epoch, init_lr=init_lr, num_epochs_per_decay=num_epochs_per_decay)

        net.train()
        run(net, name, fold, optimizer, epoch, metrics, trainloader, train_exp)

        net.eval()
        score = run(net, name, fold, None, epoch, metrics, validloader, valid_exp)

        if not (epoch + 1) % 10 and score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(net, f'weights/{name}/{name}_fold{fold}_best.pth')
    torch.save(net, f'weights/{name}/{name}_fold{fold}_last.pth')
    print(f'Finished {name}_f{fold} with best score {best_score} on epoch {best_epoch}')

NUM_SPLITS = 5


def main():
    global args
    args = parser.parse_args()

    cc = CrayonClient(port=8089)

    for name in args.name.split(','):
        shutil.rmtree(f'weights/{name}/', ignore_errors=True)
        shutil.rmtree(f'output/{name}/', ignore_errors=True)
        os.makedirs(f'weights/{name}')

        for fold in range(NUM_SPLITS):
            print(f'=> Targeting {name} fold {fold+1}/{NUM_SPLITS}')
            os.makedirs(f'output/{name}/fold{fold}/train')
            os.makedirs(f'output/{name}/fold{fold}/valid')

            arch = name.split('_')[0]
            model = models[arch](1)
            model = nn.DataParallel(model)
            model.cuda()

            train_loader, valid_loader, _ = get_loaders(args.batch_size, NUM_SPLITS, fold)

            train_eval(model, name, train_loader, valid_loader,
                       fold, make_experiment(cc, name, fold),
                       init_lr=args.lr, epochs=args.epochs,
                       num_epochs_per_decay=args.num_epochs_per_decay)

            del model

if __name__ == '__main__':
    main()
