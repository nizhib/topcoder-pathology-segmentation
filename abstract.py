class DualTransformer(object):
    def transform(self, img, how):
        raise NotImplementedError

    def __call__(self, img, msk):
        img = self.transform(img, 'image')

        if msk is not None:
            msk = self.transform(msk, 'mask')

        return img, msk


class Dualized(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, msk):
        img = self.transform(img)

        if msk is not None:
            msk = self.transform(msk)

        return img, msk


class ImageOnly(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, msk):
        img = self.transform(img)

        return img, msk


class MaskOnly(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, msk):
        if msk is not None:
            msk = self.transform(msk)

        return img, msk


class DualCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, msk):
        for t in self.transforms:
            img, msk = t(img, msk)
        return img, msk
