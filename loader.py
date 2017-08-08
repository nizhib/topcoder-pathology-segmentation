from torch.utils import data
from torchvision import transforms

import abstract
import augcolor
import augcrops
import dataset


def get_loaders(batch_size, nfolds, fold, training=True):
    train_transform = abstract.DualCompose([
        augcrops.DualRotateCropPadded(max_angle=45.0, max_shift=384, size=512),
        augcrops.DualToPIL(),
        abstract.Dualized(transforms.ToTensor()),
        abstract.ImageOnly(augcolor.RandomGamma(0.8, 1.2)),
        abstract.ImageOnly(augcolor.RandomBrightness(-0.15, 0.15)),
        abstract.ImageOnly(augcolor.RandomSaturation(-0.15, 0.15)),
        abstract.ImageOnly(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    ])
    valid_transform = abstract.DualCompose([
        augcrops.DualToPIL(),
        abstract.Dualized(transforms.ToTensor()),
        abstract.ImageOnly(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    ])
    test_transform = abstract.DualCompose([
        augcrops.DualPad(padding=6),
        augcrops.DualToPIL(),
        abstract.Dualized(transforms.ToTensor()),
        abstract.ImageOnly(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    ])

    if training:
        train_dataset = dataset.QuadedDataset('data', split='train', training=training,
                                              nfolds=nfolds, fold=fold, transform=train_transform)
    else:
        train_dataset = dataset.QuadedDataset('data', split='train', training=training,
                                              nfolds=nfolds, fold=fold, transform=valid_transform)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size,
                                   shuffle=True, num_workers=8)

    valid_dataset = dataset.QuadedDataset('data', split='valid', training=True,
                                          nfolds=nfolds, fold=fold, transform=valid_transform)
    valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size,
                                   shuffle=False, num_workers=8)

    test_dataset = dataset.QuadedDataset('data', split='test', training=True,
                                         nfolds=nfolds, fold=fold, transform=test_transform)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=8)

    return train_loader, valid_loader, test_loader
