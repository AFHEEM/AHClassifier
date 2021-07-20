from torch.utils.data import IterableDataset
import torch
from PIL import Image


class SingleImage(IterableDataset):
    def __init__(self, path, transformers):
        self.path = path
        self.transformers = transformers

    def read_image(self):
        with open(self.path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            yield self.transformers(img)

        return None

    def __iter__(self):
        return self.read_image()
