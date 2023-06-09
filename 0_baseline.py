# %%
from data import SpicyDataset
from funlib.learn.torch.models import Vgg2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from torchvision.transforms import ToTensor
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision.utils import make_grid
from torch.nn.functional import softmax
import torch

torch.backends.cudnn.benchmark = True

# %% Create the dataloader
ds = SpicyDataset(transform=ToTensor(), length=16000)
dataloader = DataLoader(ds, batch_size=16, pin_memory=True)

# %%
x, y = ds[0]
print(y)
to_pil_image(x)
# %% Create the model
model = Vgg2D(input_size=(160, 160), input_fmaps=4, output_classes=3).cuda()

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# %% Train the model
history = []
for epoch in range(1):
    for x, y in tqdm(dataloader, total=len(dataloader)):
        optimizer.zero_grad()
        pred = model(x.cuda())
        loss = criterion(pred, y.cuda())
        loss.backward()
        optimizer.step()
        history.append(loss.item())
    plt.plot(history)
    plt.show()


# %%
def plot_pair(img, prediction):
    fig, axes = plt.subplot_mosaic(
        """
                                   aaab
                                   aaab
                                   aaab
                                   """
    )
    axes["a"].imshow(img)
    axes["b"].bar(range(3), prediction, color=["yellow", "green", "red"])
    return fig


# %%
prototypes = ds.make_prototypes()
uninformative = torch.stack([ds.make_uninformative() for _ in range(100)])
ood_x = []
ood_y = []
for _ in range(100):
    x, y = ds.make_ood()
    ood_x.append(x)
    ood_y.append(y)
ood_x = torch.stack(ood_x)
ood_y = torch.tensor(ood_y)

ind_x = []
ind_y = []
for _ in range(100):
    x, y = ds[0]
    ind_x.append(x)
    ind_y.append(y)
ind_x = torch.stack(ind_x)
ind_y = torch.tensor(ind_y)
# %% Run predictions
model.eval()
with torch.no_grad():
    pred_prototypes = model(prototypes.cuda()).cpu()
    pred_uninformative = model(uninformative.cuda()).cpu()
    pred_ood = model(ood_x.cuda()).cpu()
    pred_ind = model(ind_x.cuda()).cpu()

# %%
for im, p in zip(prototypes, softmax(pred_prototypes)):
    plot_pair(im.moveaxis((0, 1, 2), (2, 0, 1)), p)

# %% Check robustness
from sklearn.metrics import accuracy_score

print("In distribution: ", accuracy_score(pred_ind.argmax(dim=-1), ind_y))
print("OOD: ", accuracy_score(pred_ood.argmax(dim=-1), ood_y))

# %%
plt.hist(
    [
        softmax(pred_ind).max(dim=-1).values,
        softmax(pred_uninformative).max(dim=-1).values,
        softmax(pred_ood).max(dim=-1).values,
    ],
    label=["informative", "uninformative", "ood"],
)
plt.legend()
plt.xlabel("Prediction score")
plt.ylabel("Percentage")

# %%
plt.hist(
    [
        pred_ind.max(dim=-1).values,
        pred_uninformative.max(dim=-1).values,
        pred_ood.max(dim=-1).values,
    ],
    label=["informative", "uninformative", "ood"],
)
plt.legend()
plt.xlabel("Logit")
plt.ylabel("Percentage")

# %%
