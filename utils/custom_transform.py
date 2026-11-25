import torch


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if torch.rand(1) < self.p:
            image = torch.flip(image, dims=[-1])   # width
            target = torch.flip(target, dims=[-1])
        return image, target


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if torch.rand(1) < self.p:
            image = torch.flip(image, dims=[-2])   # height
            target = torch.flip(target, dims=[-2])
        return image, target