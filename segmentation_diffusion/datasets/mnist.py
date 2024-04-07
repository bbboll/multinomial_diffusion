"""
MNIST data distribution.
"""
from torchvision.datasets import MNIST
import torchvision.transforms as tv_transforms
from torch.utils.data import DataLoader

class Threshold(object):
    """Threshold grayscale image at 0.5 to yield a binary labeling."""
    def __init__(self, threshold=0.5):
        super(Threshold, self).__init__()
        self.threshold = threshold
    
    def __call__(self, x):
        return (x > self.threshold).long()

class ImageOnlyDataLoader(DataLoader):
    """Wrapper for torchvision DataLoader which removes the labels from each batch."""
    def __init__(self, dataset, **kwargs):
        super(ImageOnlyDataLoader, self).__init__(dataset, **kwargs)
    
    def __iter__(self):
        base_iterator = super(ImageOnlyDataLoader, self).__iter__()
        return map(lambda x: x[0], base_iterator)

class BinarizedMNIST(object):
    """
    Binary segmentations of MNIST images produced through thresholding
    """
    def __init__(self):
        super(BinarizedMNIST, self).__init__()
        self.spatial_dims = (32, 32)
        self.dataset = None
        self.num_classes = 2

    def load_data(self, split="train"):
        self.transforms = tv_transforms.Compose([
            tv_transforms.ToTensor(),
            tv_transforms.Pad(2),
            Threshold()
        ])
        assert split in ["train", "val", "test"]
        if split == "val":
            print("Warning: selected \"val\" split for MNIST. This is an alias for the test set.")
        self.dataset = MNIST(root="./mnist", train=(split == "train"), transform=self.transforms, download=True)

    def dataloader(self, split="train", **kwargs):
        if self.dataset is None:
            self.load_data(split)
        return ImageOnlyDataLoader(self.dataset, **kwargs)

