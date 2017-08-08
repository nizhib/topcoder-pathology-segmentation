import numpy as np
from scipy.spatial.distance import dice as sdice
from sklearn.metrics import f1_score

import torch
from torch.autograd import Variable


def fmicro_np(ytrue, ypred):
    ytrue = ytrue.reshape(ytrue.shape[0], -1).astype(np.uint8)
    ypred = ypred.reshape(ytrue.shape[0], -1).astype(np.uint8)
    return f1_score(ytrue, ypred, average='micro')


def dice_np(ytrue, ypred):
    ytrue = ytrue.reshape(ytrue.shape[0], -1).astype(np.uint8)
    ypred = ypred.reshape(ytrue.shape[0], -1).astype(np.uint8)
    ds = []
    for true, pred in zip(ytrue, ypred):
        if np.any(true) or np.any(pred):
            d = 1 - sdice(true.astype(int), pred.astype(int))
        else:
            d = 1
        ds.append(d)
    dc: float = np.mean(ds)
    return dc


def score_np(ytrue, ypred):
    return int(round(1e8 * (fmicro_np(ytrue, ypred) + dice_np(ytrue, ypred)) / 2)) / 100


# noinspection PyUnresolvedReferences
def fmicro_th(preds, trues, thr=0.5, eps=1e-8):
    preds = (preds.view(-1).float() > thr).float()
    trues = (trues.view(-1).float() > thr).float()

    tp = torch.sum(preds * trues)
    fp = torch.sum(preds * (1 - trues))
    fn = torch.sum((1 - preds) * trues)

    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)

    return (2 * p * r / (p + r + eps)).data.cpu()[0]


# noinspection PyUnresolvedReferences,PyArgumentList
def dice_th(preds, trues, thr=0.5, eps=1e-8):
    preds = (preds.view(preds.size(0), -1).float() > thr).float()
    trues = (trues.view(trues.size(0), -1).float() > thr).float()

    tp = torch.sum(preds * trues, 1)
    fp = torch.sum(preds * (1 - trues), 1)
    fn = torch.sum((1 - preds) * trues, 1)

    return torch.mean((2 * tp + eps) / (2 * tp + fp + fn + eps)).data.cpu()[0]


# noinspection PyUnresolvedReferences
def score_th(preds, trues):
    return (fmicro_th(preds, trues) + dice_th(preds, trues)) / 2


if __name__ == '__main__':
    an = np.greater(np.random.rand(4, 10, 10, 10), 0.7).astype(float)
    bn = np.greater(np.random.rand(4, 10, 10, 10), 0.7).astype(float)
    at = Variable(torch.from_numpy(an), volatile=True).cuda()
    bt = Variable(torch.from_numpy(bn), volatile=True).cuda()
    assert np.allclose([fmicro_th(at, bt)], [fmicro_np(an, bn)])
    assert np.allclose([dice_th(at, bt)], [dice_np(an, bn)])
    print([score_th(at, bt)], [score_np(an, bn)])
