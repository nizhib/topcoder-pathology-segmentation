import os
import shutil

import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import KFold
from tqdm import tqdm

import torch
import torch.utils.data as data
from torchvision import transforms

import abstract
import augcolor
import augcrops


def transpose(img, index):
    if index % 2:
        img = np.fliplr(img)
    if index // 2 % 2:
        img = np.flipud(img)
    if index // 4 % 2:
        img = np.rot90(img)
    return img


def rtranspose(img, index):
    if index // 4 % 2:
        img = np.rot90(img, k=-1)
    if index // 2 % 2:
        img = np.flipud(img)
    if index % 2:
        img = np.fliplr(img)
    return img


def _load_pil(path, ismask=False):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            if not ismask:
                return img.convert('RGB')
            else:
                return img.copy()


def _load_cv2(path, ismask=False):
    if not ismask:
        img = cv2.imread(path)
    else:
        img = cv2.imread(path, 0)
    return img

load_image = _load_cv2


EXCLUDED = ['quading/i628806.tif_i282989.tif_i417677.tif_i659777.tif.tif',
            'quading/i933123.tif_i154348.tif_i435969.tif_i385761.tif.tif']


class QuadedDataset(data.Dataset):
    def __init__(self, data_root, split='train', training=False, nfolds=6, fold=0, transform=None):
        assert split in ['train', 'valid', 'test']

        self.split = split
        self.training = training
        self.transform = transform

        self.images_root = None
        self.truth_root = None
        self.ipaths = None
        self.mpaths = None

        if self.split == 'test':
            self.images_root = os.path.join(data_root, 'testing/images')
        else:
            self.images_root = os.path.join(data_root, 'quading/images')
            self.truth_root = os.path.join(data_root, 'quading/truth')

        inames = os.listdir(self.images_root)
        ipaths = sorted([os.path.join(self.images_root, n) for n in inames])

        if self.split == 'test':
            self.ipaths = ipaths
        else:
            kf = KFold(n_splits=nfolds, random_state=42, shuffle=True)
            idx_train, idx_val = list(kf.split(ipaths))[fold]
            if self.split == 'train':
                self.ipaths = list(np.array(ipaths)[idx_train])
                if not self.training:
                    self.ipaths.extend([os.path.join(data_root, p) for p in EXCLUDED])
                if fold == 0:
                    assert self.ipaths[0].endswith('i557461.tif.tif')
                print('1st train:', self.ipaths[0])
            else:
                self.ipaths = list(np.array(ipaths)[idx_val])
                if fold == 0:
                    assert self.ipaths[0].endswith('i341701.tif.tif')
                print('1st valid:', self.ipaths[0])
        self.num_images = len(self.ipaths)

        if self.split != 'test':
            self.mpaths = [os.path.join(self.truth_root,
                                        f'{os.path.splitext(os.path.basename(path))[0]}_mask.png')
                           for path in self.ipaths]

        self.images = [load_image(ipath) for ipath in self.ipaths]

        if self.split != 'test':
            self.masks = [load_image(str(mpath), ismask=True) for mpath in self.mpaths]
        else:
            self.masks = [None for _ in self.ipaths]

    def __getitem__(self, index):
        image = self.images[index % self.num_images]
        mask = self.masks[index % self.num_images]
        path = self.ipaths[index % self.num_images]
        if self.split != 'test':
            path = os.path.splitext(os.path.basename(path))[0]

        index //= self.num_images

        if self.split == 'valid' or self.split == 'train' and not self.training:
            path = np.array(path.split('_')).reshape([2, 2])

            path = path[index % 2]
            if index % 2 == 0:
                image = cv2.copyMakeBorder(image[:506], 6, 0, 0, 0, cv2.BORDER_REFLECT_101)
                mask = cv2.copyMakeBorder(mask[:506], 6, 0, 0, 0, cv2.BORDER_REFLECT_101)
            else:
                image = cv2.copyMakeBorder(image[-506:], 0, 6, 0, 0, cv2.BORDER_REFLECT_101)
                mask = cv2.copyMakeBorder(mask[-506:], 0, 6, 0, 0, cv2.BORDER_REFLECT_101)
            index //= 2

            path = path[index % 2]
            if index % 2 == 0:
                image = cv2.copyMakeBorder(image[:, :506], 0, 0, 6, 0, cv2.BORDER_REFLECT_101)
                mask = cv2.copyMakeBorder(mask[:, :506], 0, 0, 6, 0, cv2.BORDER_REFLECT_101)
            else:
                image = cv2.copyMakeBorder(image[:, -506:], 0, 0, 0, 6, cv2.BORDER_REFLECT_101)
                mask = cv2.copyMakeBorder(mask[:, -506:], 0, 0, 0, 6, cv2.BORDER_REFLECT_101)
            index //= 2

            path = str(path)

        trid = index % 8
        image = transpose(image, trid)
        if mask is not None:
            mask = transpose(mask, trid)

        if self.transform:
            image, mask = self.transform(image, mask)

        if mask is None:
            mask = torch.zeros(1, image.size()[1], image.size()[2])

        return image, mask, path, trid

    def __len__(self):
        if self.split == 'test':
            return 8 * self.num_images  # D4
        else:
            return 32 * self.num_images  # D4 + 4crops


def test():
    shutil.rmtree('rounds', ignore_errors=True)
    os.makedirs('rounds')

    train_transform = abstract.DualCompose([
        augcrops.DualRotateCropPadded(max_angle=45.0, max_shift=384, size=512),
        augcrops.DualToPIL(),
        abstract.Dualized(transforms.ToTensor()),
        abstract.ImageOnly(augcolor.RandomGamma(0.8, 1.2)),
        abstract.ImageOnly(augcolor.RandomBrightness(-0.15, 0.15)),
        abstract.ImageOnly(augcolor.RandomSaturation(-0.15, 0.15))
    ])
    valid_transform = abstract.DualCompose([
        augcrops.DualToPIL(),
        abstract.Dualized(transforms.ToTensor())
    ])
    test_transform = abstract.DualCompose([
        augcrops.DualPad(padding=6),
        augcrops.DualToPIL(),
        abstract.Dualized(transforms.ToTensor())
    ])

    train_1_dataset = QuadedDataset('data', split='train', training=True,
                                    nfolds=5, fold=0, transform=train_transform)
    train_1_loader = data.DataLoader(train_1_dataset, batch_size=16, shuffle=False, num_workers=8)

    train_0_dataset = QuadedDataset('data', split='train', training=False,
                                    nfolds=5, fold=0, transform=valid_transform)
    train_0_loader = data.DataLoader(train_0_dataset, batch_size=16, shuffle=False, num_workers=8)

    valid_dataset = QuadedDataset('data', split='valid', training=True,
                                  nfolds=5, fold=0, transform=valid_transform)
    valid_loader = data.DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=8)

    test_dataset = QuadedDataset('data', split='test', training=True,
                                 nfolds=5, fold=0, transform=test_transform)
    test_loader = data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=8)

    subsets = ['train_1', 'train_0', 'valid', 'test']
    loaders = [train_1_loader, train_0_loader, valid_loader, test_loader]

    num_saved = 0

    for subset, loader in zip(subsets[:1], loaders[:1]):
        os.makedirs(f'rounds/{subset}')
        for i, (ims, mss, ps, ts) in enumerate(tqdm(loader)):
            for j, (im, ms, p, t) in enumerate(zip(ims, mss, ps, ts)):
                im = np.uint8(np.swapaxes(np.swapaxes(im.numpy(), 1, 2), 0, 2) * 255)
                ms = np.uint8(np.swapaxes(np.swapaxes(ms.numpy(), 1, 2), 0, 2) * 255)
                name = os.path.splitext(os.path.basename(p))[0]
                if subset == 'test':
                    cv2.imwrite(f'rounds/{subset}/{name}_{i * 16 + j}.jpg',
                                rtranspose(im[:, :, ::-1], t))
                elif subset == 'valid':
                    cv2.imwrite(f'rounds/{subset}/{name}_{i * 16 + j}.jpg',
                                rtranspose(im[:, :, ::-1], t))
                    cv2.imwrite(f'rounds/{subset}/{name}_{i * 16 + j}_m.jpg',
                                rtranspose(ms, t))
                else:
                    cv2.imwrite(f'rounds/{subset}/{name}_{i * 16 + j}.jpg',
                                im[:, :, ::-1])
                    cv2.imwrite(f'rounds/{subset}/{name}_{i * 16 + j}_m.jpg',
                                ms)
                num_saved += 1
                if num_saved == 32:
                    break
            if num_saved == 32:
                break

if __name__ == '__main__':
    test()
