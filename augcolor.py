import random

import torch


def _blend(img1, img2, alpha):
    """
    Weighted sum of two images
    Arguments
    ---------
    img1 : torch tensor
    img2 : torch tensor
    alpha : float between 0 and 1
        how much weight to put on img1 and 1-alpha weight
        to put on img2
    """
    return img1.mul(alpha).add(1 - alpha, img2)


class Grayscale(object):
    def __init__(self, keep_channels=False):
        """
        Convert RGB image to grayscale
        Arguments
        ---------
        keep_channels : boolean
            If true, will keep all 3 channels and they will be the same
            If false, will just return 1 grayscale channel
        """
        self.keep_channels = keep_channels
        if keep_channels:
            self.channels = 3
        else:
            self.channels = 1

    def __call__(self, *inputs):
        outputs = []
        idx = 0
        for idx, _input in enumerate(inputs):
            _input_dst = _input[0] * 0.299 + _input[1] * 0.587 + _input[2] * 0.114
            # noinspection PyUnresolvedReferences
            _input_gs = _input_dst.repeat(self.channels, 1, 1)
            outputs.append(_input_gs)
        return outputs if idx > 1 else outputs[0]


class Brightness(object):
    def __init__(self, value):
        """
        Alter the Brightness of an image
        Arguments
        ---------
        value : brightness factor
            =-1 = completely black
            <0 = darker
            0 = no change
            >0 = brighter
            =1 = completely white
        """
        self.value = max(min(value, 1.0), -1.0)

    def __call__(self, *inputs):
        outputs = []
        idx = 0
        for idx, _input in enumerate(inputs):
            _input = torch.clamp(_input.float().add(self.value).type(_input.type()), 0, 1)
            outputs.append(_input)
        return outputs if idx > 1 else outputs[0]


class Gamma(object):
    def __init__(self, value):
        """
        Performs Gamma Correction on the input image. Also known as
        Power Law Transform. This function transforms the input image
        pixelwise according
        to the equation Out = In**gamma after scaling each
        pixel to the range 0 to 1.
        Arguments
        ---------
        value : float
            <1 : image will tend to be lighter
            =1 : image will stay the same
            >1 : image will tend to be darker
        """
        self.value = value

    def __call__(self, *inputs):
        outputs = []
        idx = 0
        for idx, _input in enumerate(inputs):
            _input = torch.clamp(torch.pow(_input, self.value).type(_input.type()), 0, 1)
            outputs.append(_input)
        return outputs if idx > 1 else outputs[0]


class Contrast(object):
    def __init__(self, value):
        """
        Adjust Contrast of image.
        Contrast is adjusted independently for each channel of each image.
        For each channel, this Op computes the mean of the image pixels
        in the channel and then adjusts each component x of each pixel to
        (x - mean) * contrast_factor + mean.
        Arguments
        ---------
        value : float
            smaller value: less contrast
            ZERO: channel means
            larger positive value: greater contrast
            larger negative value: greater inverse contrast
        """
        self.value = value

    def __call__(self, *inputs):
        outputs = []
        idx = 0
        for idx, _input in enumerate(inputs):
            channel_means = _input.mean(2, keepdim=True).mean(1, keepdim=True)
            _input = torch.clamp(_input + (channel_means - _input) * self.value, 0, 1)
            outputs.append(_input)
        return outputs if idx > 1 else outputs[0]


class Saturation(object):
    def __init__(self, value):
        """
        Alter the Saturation of image
        Arguments
        ---------
        value : float
            =-1 : gray
            <0 : colors are more muted
            =0 : image stays the same
            >0 : colors are more pure
            =1 : most saturated
        """
        self.value = max(min(value, 1.0), -1.0)

    def __call__(self, *inputs):
        outputs = []
        idx = 0
        for idx, _input in enumerate(inputs):
            _in_gs = Grayscale(keep_channels=True)(_input)
            alpha = 1.0 + self.value
            _in = torch.clamp(_blend(_input, _in_gs, alpha), 0, 1)
            outputs.append(_in)
        return outputs if idx > 1 else outputs[0]


class RandomBrightness(object):
    def __init__(self, min_val, max_val):
        """
        Alter the Brightness of an image with a value randomly selected
        between `min_val` and `max_val`
        Arguments
        ---------
        min_val : float
            min range
        max_val : float
            max range
        """
        self.values = (min_val, max_val)

    def __call__(self, *inputs):
        value = random.uniform(self.values[0], self.values[1])
        outputs = Brightness(value)(*inputs)
        return outputs


class RandomGamma(object):
    def __init__(self, min_val, max_val):
        """
        Performs Gamma Correction on the input image with some
        randomly selected gamma value between min_val and max_val.
        Also known as Power Law Transform. This function transforms
        the input image pixelwise according to the equation
        Out = In**gamma after scaling each pixel to the range 0 to 1.
        Arguments
        ---------
        min_val : float
            min range
        max_val : float
            max range
        """
        self.values = (min_val, max_val)

    def __call__(self, *inputs):
        value = random.uniform(self.values[0], self.values[1])
        outputs = Gamma(value)(*inputs)
        return outputs


class RandomContrast(object):
    def __init__(self, min_val, max_val):
        """
        Alter the Contrast of an image with a value randomly selected
        between `min_val` and `max_val`
        Arguments
        ---------
        min_val : float
            min range
        max_val : float
            max range
        """
        self.values = (min_val, max_val)

    def __call__(self, *inputs):
        value = random.uniform(self.values[0], self.values[1])
        outputs = Contrast(value)(*inputs)
        return outputs


class RandomSaturation(object):
    def __init__(self, min_val, max_val):
        """
        Alter the Saturation of an image with a value randomly selected
        between `min_val` and `max_val`
        Arguments
        ---------
        min_val : float
            min range
        max_val : float
            max range
        """
        self.values = (min_val, max_val)

    def __call__(self, *inputs):
        value = random.uniform(self.values[0], self.values[1])
        outputs = Saturation(value)(*inputs)
        return outputs
