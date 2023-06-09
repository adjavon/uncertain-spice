# %%
from data import SpicyDataset
from funlib.learn.torch.models import Vgg2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision.utils import make_grid
from torch.nn.functional import softmax
import torch

from utils import plot_pair

torch.backends.cudnn.benchmark = True

# %% Create the dataloader
ds = SpicyDataset(transform=ToTensor(), length=16000)
dataloader = DataLoader(ds, batch_size=16, pin_memory=True)

# %%
x, y = ds[0]
print(y)
to_pil_image(x)
# %% Create the model
model = torch.load("models/baseline.chkpt").cuda()
model.eval()

# %%
@torch.no_grad()
def test_time_aug(sample, model):
    augmentations = [
        lambda x: transforms.functional.rotate(x, 90),
        lambda x: transforms.functional.rotate(x, 180),
        lambda x: transforms.functional.rotate(x, 270),
        transforms.functional.hflip,
        transforms.functional.vflip
    ]
    predictions = []
    for aug in augmentations:
        pred = softmax(model(aug(sample).cuda()).cpu())
        predictions.append(pred)
    return torch.stack(predictions, dim=-1)
        
# %%
prototypes = torch.from_numpy(np.load("test_samples/prototypes.npy"))
uninformative = torch.from_numpy(np.load("test_samples/uninformative.npy"))
ood_x = torch.from_numpy(np.load("test_samples/ood_x.npy"))
ood_y = torch.from_numpy(np.load("test_samples/ood_y.npy"))
ind_x = torch.from_numpy(np.load("test_samples/ind_x.npy"))
ind_y = torch.from_numpy(np.load("test_samples/ind_y.npy"))

# %% Run predictions
with torch.no_grad():
    pred_prototypes = test_time_aug(prototypes, model) 
    pred_uninformative = test_time_aug(uninformative, model)
    pred_ood = test_time_aug(ood_x, model) 
    pred_ind = test_time_aug(ind_x, model) 

# %%
for im, p in zip(prototypes, pred_prototypes):
    p_mean = p.mean(dim=-1) 
    p_std = p.std(dim=-1) 
    plot_pair(im.moveaxis((0, 1, 2), (2, 0, 1)), p_mean, p_std)

# %% Check robustness
from sklearn.metrics import accuracy_score

print("In distribution: ", accuracy_score(pred_ind.mean(dim=-1).argmax(dim=-1), ind_y))
print("OOD: ", accuracy_score(pred_ood.mean(dim=-1).argmax(dim=-1), ood_y))

# %%
plt.hist(
    [
        pred_ind.mean(dim=-1).max(dim=-1).values,
        pred_uninformative.mean(dim=-1).max(dim=-1).values,
        pred_ood.mean(dim=-1).max(dim=-1).values,
    ],
    label=["informative", "uninformative", "ood"],
)
plt.legend()
plt.xlabel("Prediction score")
plt.ylabel("Percentage")
# %%
plt.hist(
    [
        pred_ind.max(dim=1).values.std(dim=-1),
        pred_uninformative.max(dim=1).values.std(dim=-1),
        pred_ood.max(dim=1).values.std(dim=-1),
    ],
    label=["informative", "uninformative", "ood"],
)
plt.legend()
plt.xlabel("std")
plt.ylabel("Percentage")

# %%
print("In-d", pred_ind.max(dim=1).values.mean())
print("Uninformative", pred_uninformative.max(dim=1).values.mean())
print("OOD", pred_ood.max(dim=1).values.mean())
# %%
print("In-d", pred_ind.max(dim=1).values.std())
print("Uninformative", pred_uninformative.max(dim=1).values.std())
print("OOD", pred_ood.max(dim=1).values.std())
# %%
