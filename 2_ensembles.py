# %%
from data import SpicyDataset
from funlib.learn.torch.models import Vgg2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from tqdm import tqdm
from torchvision.transforms import ToTensor
from torch import nn, optim
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
# %% Create n models
n_models = 10

criterion = nn.CrossEntropyLoss()

# %% Train the model
if not all([Path(f"models/ensemble_{i}.chkpt").exists() for i in range(n_models)]):
    models = [
        Vgg2D(input_size=(160, 160), fmaps=8, input_fmaps=4, output_classes=3).cuda()
        for _ in range(n_models)
    ]
    optimizers = [
        optim.Adam(model.parameters(), lr=1e-4)
        for model in models
    ]
    history = {i: [] for i in range(n_models)}
    for epoch in range(1):
        for x, y in tqdm(dataloader, total=len(dataloader)):
            for i, model in enumerate(models):
                optimizers[i].zero_grad()
                pred = model(x.cuda())
                loss = criterion(pred, y.cuda())
                loss.backward()
                optimizers[i].step()
                history[i].append(loss.item())
        for lbl, hist in history.items():
            plt.plot(hist, label='lbl')
        plt.show()
    for i, model in enumerate(models):
        with open(f"models/ensemble_{i}.chkpt", 'wb') as fd:
            torch.save(model, fd)
else:
    models = [
        torch.load(f"models/ensemble_{i}.chkpt") for i in range(n_models)
    ]


# %%
@torch.no_grad()
def ensemble(sample, models):
    predictions = []
    for model in models:
        pred = softmax(model(sample.cuda()).cpu())
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
for model in models:
    model.eval()
with torch.no_grad():
    pred_prototypes = ensemble(prototypes, models) 
    pred_uninformative = ensemble(uninformative, models)
    pred_ood = ensemble(ood_x, models) 
    pred_ind = ensemble(ind_x, models) 

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