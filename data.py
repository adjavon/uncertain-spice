import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import torch


pepper_dir = Path(__name__).parent / "peppers"
peppers = ["juicy.png", "mild.png", "spicy.png", "hot.png"]


sample_types = [
    [0, 0, 1],  # spicy
    [0, 1, 0],  # mild
    [1, 0, 0],  # juicy
    [0, 0, 2],  # spicy
    [0, 2, 0],  # mild
    [2, 0, 0],  # juicy
    [0, 1, 2],  # spicy 
    [1, 0, 2],  # spicy 
    [0, 2, 1],  # mild  
    [1, 2, 0],  # mild
    [2, 1, 0],  # juicy
    [2, 0, 1],  # juicy
    [1, 1, 2],  # spicy 
    [1, 1, 2],  # spicy 
    [1, 2, 1],  # mild  
    [1, 2, 1],  # mild
    [2, 1, 1],  # juicy
    [2, 1, 1],  # juicy
]
n_types = len(sample_types)


conf_types = [
   [0, 1, 1], 
   [1, 0, 1], 
   [1, 1, 0],
   [0, 2, 2], 
   [2, 0, 2], 
   [2, 2, 0]
]
n_conf = len(conf_types)



class SpicyDataset(Dataset):
    """A dataset that separates spicy from un-spicy images based on pepper emojis."""

    def __init__(
        self, size=160, max_samples=3, label_type="max", length=8000, transform=None
    ):
        super().__init__()
        self.peppers = [Image.open(pepper_dir / im) for im in peppers]
        self.size = size
        self.shape = (size, size, 4)
        self.max_samples = max_samples
        self.length = length
        if transform is not None:
            self.transform = transform
        else:
            self.transform = lambda x: x
        if label_type == "max":
            self.get_label = self._max_label
        else:
            self.get_label = self._ratio_label

    def _max_label(self, numbers):
        max_value = np.max(numbers)
        return np.random.choice(np.where(numbers == max_value)[0])

    def _ratio_label(self, numbers):
        return numbers[1] / sum(numbers)

    def get_background(self):
        return Image.fromarray(np.zeros(self.shape, dtype=np.uint8))

    def add_pepper(self, label, background):
        img = self.peppers[label]
        rotation = np.random.randint(360)
        scale = np.random.randint(32, 96)
        location = tuple(np.random.randint(0, self.size - scale, size=2))
        foreground = img.resize((scale, scale)).rotate(rotation)
        background.paste(foreground, box=location, mask=foreground)

    def generate_sample(self, background, numbers):
        for label, n in enumerate(numbers):
            for _ in range(n):
                self.add_pepper(label, background)
        return self.transform(background)

    def make_prototypes(self):
        tensors = []
        for p in self.peppers: 
            bg = self.get_background()
            center = (bg.width - 96) // 2
            bg.paste(p.resize((96, 96)), box=(center, center)) 
            tensors.append(self.transform(bg))
        return torch.stack(tensors)

    def make_uninformative(self):
        bg = self.get_background()
        ix  = np.random.randint(n_conf)
        numbers = conf_types[ix]
        image = self.generate_sample(bg, numbers)
        return image

    def make_ood(self):
        # Add the sun
        background = self.get_background()
        ix = np.random.randint(n_types)
        numbers = sample_types[ix] 
        image = self.generate_sample(background, numbers+ [np.random.randint(3)])
        return image, self.get_label(numbers)

    def __getitem__(self, item):
        background = self.get_background()
        ix = np.random.randint(n_types)
        numbers = sample_types[ix]
        image = self.generate_sample(background, numbers)
        return image, self.get_label(numbers)

    def __len__(self):
        return self.length
