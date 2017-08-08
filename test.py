import os
import zipfile

import click
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from loss import dice_np, fmicro_np, score_np

TRUEDIR = 'data/training/truth/'


@click.command()
@click.option('-n', '--name', default='invalid9000', help='Model name')
@click.option('-m', '--mode', default='best', help='Checkpoint to use')
def main(name, mode):
    trues = []
    preds = []

    with zipfile.ZipFile(f'output/{name}/{name}_{mode}.zip') as myzip:
        for i, name in enumerate(tqdm([m for m in os.listdir(TRUEDIR) if m.endswith('.txt')])):
            if not name.endswith('txt'):
                continue

            true = np.array([[c == '1' for c in s] for s in
                             [_.strip() for _ in open(TRUEDIR + name).readlines()]]) + 0
            pred = np.array([[c == '1' for c in s] for s in
                             [_.decode().strip() for _ in myzip.open(name).readlines()]]) + 0
            trues.append(true)
            preds.append(pred)

            # if i < 5:
            #     plt.imsave(os.path.expanduser('~') + f'/{name}.true.jpg', true)
            #     plt.imsave(os.path.expanduser('~') + f'/{name}.pred.jpg', pred)

    ypred = np.hstack(np.array(preds)[np.newaxis, :])
    ytrue = np.hstack(np.array(trues)[np.newaxis, :])

    print(dice_np(ytrue.reshape(ytrue.shape[0], -1), ypred.reshape(ypred.shape[0], -1)))
    print(fmicro_np(ytrue.reshape(ytrue.shape[0], -1), ypred.reshape(ypred.shape[0], -1)))
    print(score_np(ytrue.reshape(ytrue.shape[0], -1), ypred.reshape(ypred.shape[0], -1)))

    # for _ in range(5):
    #     ycheat = ypred.copy()
    #     idx = np.random.choice(ycheat.shape[0], 41, replace=False)
    #     ycheat[idx] = ytrue[idx]
    #     print()
    #     print(dice_np(ytrue.reshape(ytrue.shape[0], -1), ycheat.reshape(ycheat.shape[0], -1)))
    #     print(fmicro_np(ytrue.reshape(ytrue.shape[0], -1), ycheat.reshape(ycheat.shape[0], -1)))
    #     print(score_np(ytrue.reshape(ytrue.shape[0], -1), ycheat.reshape(ycheat.shape[0], -1)))

if __name__ == '__main__':
    main()
