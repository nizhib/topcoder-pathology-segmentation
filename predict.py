import os
import tempfile
import zipfile
from io import BytesIO

import cv2
import click
import numpy as np
from skimage import filters
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from dataset import rtranspose
from loader import get_loaders
from loss import fmicro_th, dice_th, fmicro_np, dice_np

imsize = 512


def predict(net, loader, verbose=0):
    ypred = torch.zeros([len(loader.dataset), imsize, imsize])
    ytrue = torch.zeros([len(loader.dataset), imsize, imsize])
    ypath = [''] * len(loader.dataset)
    ytidx = torch.zeros(len(loader.dataset))

    gen = enumerate(loader, 0)
    if verbose == 1:
        gen = tqdm(list(gen))
    for i, data in gen:
        images, ytrues, paths, ts = data
        images = Variable(images.cuda(), volatile=True)

        ypreds = net(images).select(1, 0)
        ypred[i * loader.batch_size:(i + 1) * loader.batch_size] = ypreds.data.cpu()
        if ytrues is not None:
            ytrue[i * loader.batch_size:(i + 1) * loader.batch_size] = ytrues.select(1, 0)

        ypath[i * loader.batch_size:(i + 1) * loader.batch_size] = paths
        ytidx[i * loader.batch_size:(i + 1) * loader.batch_size] = ts

    return ypred, ytrue, ypath, ytidx


@click.command()
@click.option('-n', '--name', default='invalid9000', help='Model name')
@click.option('-m', '--mode', default='best', help='Checkpoint to use')
@click.option('-f', '--nfolds', type=int, prompt=True, help='Number of folds')
@click.option('-b', '--batch-size', default=16, help='Batch size')
def main(name, mode, nfolds, batch_size):
    out_root = f'output/{name}/'
    os.makedirs(out_root, exist_ok=True)

    paths = []
    tomix = []
    trues = []
    probs = []
    tidxs = []

    EXCLUDED = ['quading/i628806.tif_i282989.tif_i417677.tif_i659777.tif.tif',
                'quading/i933123.tif_i154348.tif_i435969.tif_i385761.tif.tif']
    enames = sum([os.path.splitext(os.path.basename(p))[0].split('_') for p in EXCLUDED], [])

    tmpzip = BytesIO()
    with zipfile.ZipFile(tmpzip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # for fold in range(nfolds):
        #     print(f'fold{fold}:')
        #
        #     mpath = f'weights/{name}/{name}_fold{fold}_{mode}.pth'
        #     model = torch.load(mpath)
        #     model.cuda()
        #     model.eval()
        #
        #     splits = ['train', 'valid', 'test']
        #
        #     for split, loader in zip(splits, get_loaders(batch_size, nfolds, fold, training=False)):
        #         ypred, ytrue, ypath, ts = predict(model, loader, verbose=1)
        #         ypred = ypred[:, 6:-6, 6:-6].contiguous()
        #         ytrue = ytrue[:, 6:-6, 6:-6].contiguous()
        #         yprob = torch.sigmoid(ypred)
        #
        #         if split != 'test':
        #             vprob = Variable(yprob, volatile=True).cuda()
        #             vtrue = Variable(ytrue, volatile=True).cuda()
        #             ll = F.binary_cross_entropy(vprob, vtrue).data[0]
        #
        #             f1 = fmicro_th(vprob > 0.5, vtrue)
        #             dc = dice_th(vprob > 0.5, vtrue)
        #             sc = int(round(1e8 * (f1 + dc) / 2)) / 100
        #             print(f'[{0.5:0.1f}] '
        #                   f'loss {ll:0.3f}  f1 {f1:0.4f}  '
        #                   f'dice {dc:0.4f}  score {sc:0.2f}')
        #
        #         if split != 'train':
        #             store = [True for _ in ypath]
        #         else:
        #             store = [split == 'valid' or
        #                      fold == 1 and np.any([p.find(str(name)) != -1 for name in enames])
        #                      for p in ypath]
        #
        #         tomix.extend(np.array([split == 'valid' for _ in store])[store])
        #         paths.extend(np.array(ypath)[store])
        #         trues.extend(ytrue.numpy()[store])
        #         probs.extend(yprob.numpy()[store])
        #         tidxs.extend(ts.numpy()[store])
        #
        # # untranspose
        # for i, (true, prob, t) in enumerate(zip(trues, probs, tidxs)):
        #     trues[i] = rtranspose(true, t)
        #     probs[i] = rtranspose(prob, t)
        #
        # tomix = np.stack(tomix)
        # paths = np.stack(paths)
        # trues = np.stack(trues)
        # probs = np.stack(probs)
        #
        # np.save(out_root + f'{name}_{mode}_tomix.npy', tomix)
        # np.save(out_root + f'{name}_{mode}_paths.npy', paths)
        # np.save(out_root + f'{name}_{mode}_trues.npy', trues)
        # np.save(out_root + f'{name}_{mode}_probs.npy', probs)

        tomix = np.load(out_root + f'{name}_{mode}_tomix.npy')
        paths = np.load(out_root + f'{name}_{mode}_paths.npy')
        trues = np.load(out_root + f'{name}_{mode}_trues.npy')
        probs = np.load(out_root + f'{name}_{mode}_probs.npy')

        print('CVOOF:')
        cvpaths = paths[tomix]
        cvtrues = trues[tomix]
        cvprobs = probs[tomix]
        meantrues = []
        meanprobs = []
        for cvpath in np.unique(cvpaths):
            thiscvtrues = cvtrues[cvpaths == cvpath]
            assert np.alltrue(np.std(thiscvtrues, 0) == 0)  # all rotated
            thiscvprobs = cvprobs[cvpaths == cvpath]
            meantrues.append(np.mean(thiscvtrues, 0))
            meanprobs.append(np.mean(thiscvprobs, 0))
        meantrue = np.stack(meantrues)
        meanprob = np.stack(meanprobs)
        for thr in [0.4, 0.5]:
            for i, (prob, true) in enumerate(zip(meanprob, meantrue)):
                cv2.imwrite(f'rounds/pred/{i}_{thr}.png', (np.uint8(prob > thr) * 255))
            f1 = fmicro_np(meanprob > thr, meantrue)
            dc = dice_np(meanprob > thr, meantrue)
            sc = int(round(1e8 * (f1 + dc) / 2)) / 100
            print(f'[{thr:0.1f}] '
                  f'loss        f1 {f1:0.4f}  '
                  f'dice {dc:0.4f}  score {sc:0.2f}')

        meanpred = np.zeros_like(meanprob)
        for i, prob in enumerate(meanprob):
            meanpred[i] = cv2.adaptiveThreshold(np.uint8(prob * 255), 255,
                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 11, 2)
        f1 = fmicro_np(meanpred, meantrue)
        dc = dice_np(meanpred, meantrue)
        sc = int(round(1e8 * (f1 + dc) / 2)) / 100
        print(f'[cv2] '
              f'loss        f1 {f1:0.4f}  '
              f'dice {dc:0.4f}  score {sc:0.2f}')

        for method in [filters.threshold_isodata]:
            print(method)
            meanpred = np.zeros_like(meanprob)
            for i, (prob, true) in enumerate(zip(meanprob, meantrue)):
                thr = method(prob)
                meanpred[i] = prob > thr
                cv2.imwrite(f'rounds/pred/{i}_prob.png', np.uint8(prob * 255))
                cv2.imwrite(f'rounds/pred/{i}_true.png', np.uint8(true) * 255)
                cv2.imwrite(f'rounds/pred/{i}_iso.png', np.uint8(meanpred[i]) * 255)
            f1 = fmicro_np(meanpred, meantrue)
            dc = dice_np(meanpred, meantrue)
            sc = int(round(1e8 * (f1 + dc) / 2)) / 100
            print(f'[thr] '
                  f'loss        f1 {f1:0.4f}  '
                  f'dice {dc:0.4f}  score {sc:0.2f}')

        for path in np.unique(paths):
            thistrues = trues[paths == path]
            assert np.alltrue(np.std(thistrues, 0) == 0)  # all rotated
            thisprobs = probs[paths == path]
            prob = np.mean(thisprobs, 0)
            pred = np.greater(prob, 0.5)
            with tempfile.NamedTemporaryFile() as tmpmask:
                np.savetxt(tmpmask.name, pred.T, fmt='%d', delimiter='')
                zipf.write(tmpmask.name, os.path.basename(path).replace('.tif', '_mask.txt'))

    tmpzip.seek(0)
    with open(out_root + f'{name}_{mode}.zip', 'wb') as out:
        out.write(tmpzip.read())

if __name__ == '__main__':
    main()
