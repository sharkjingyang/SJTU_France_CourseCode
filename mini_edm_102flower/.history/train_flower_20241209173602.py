import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from pathlib import Path
from functools import partial
import torch.nn as nn
from torchvision import transforms as T, utils
from PIL import Image
from torch.utils.data import Dataset, DataLoader

def exists(x):
    return x is not None

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

class Dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

ds = Dataset(folder, self.image_size, augment_horizontal_flip = True, convert_image_to = convert_image_to)

assert len(ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'

dl = DataLoader(ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

dl = self.accelerator.prepare(dl)
dl = cycle(dl)