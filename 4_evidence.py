# %%
from data import SpicyDataset
from funlib.learn.torch.models import Vgg2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torchvision.transforms import ToTensor
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision.utils import make_grid
from torch.nn.functional import softmax
import torch

from utils import plot_pair, one_hot_embedding
from edl_losses import edl_mse_loss, relu_evidence

torch.backends.cudnn.benchmark = True

# %% Create the dataloader
ds = SpicyDataset(transform=ToTensor(), length=160)
dataloader = DataLoader(ds, batch_size=16, pin_memory=True)

# %%
x, y = ds[0]
print(y)
to_pil_image(x)
# %% Create the model
model = Vgg2D(input_size=(160, 160), input_fmaps=4, output_classes=3).cuda()

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = edl_mse_loss

# %% Train the model
if not Path("models/edl_model.chkpt").exists():
    history = []
    num_classes = 3
    for epoch in range(100):
        for x, y in tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            pred = model(x.cuda())
            loss = criterion(pred, one_hot_embedding(y).cuda().float(), epoch, num_classes, 20, device="cuda")
            loss.backward()
            optimizer.step()
            history.append(loss.item())
    plt.plot(history)
    plt.show()
    with open("models/edl_model.chkpt", 'wb') as fd:
        torch.save(model, fd)
else:
    model = torch.load("models/baseline.chkpt").cuda()
model.eval()

# %%
prototypes = torch.from_numpy(np.load("test_samples/prototypes.npy"))
uninformative = torch.from_numpy(np.load("test_samples/uninformative.npy"))
ood_x = torch.from_numpy(np.load("test_samples/ood_x.npy"))
ood_y = torch.from_numpy(np.load("test_samples/ood_y.npy"))
ind_x = torch.from_numpy(np.load("test_samples/ind_x.npy"))
ind_y = torch.from_numpy(np.load("test_samples/ind_y.npy"))
# %% Run predictions
with torch.no_grad():
    pred_prototypes = model(prototypes.cuda()).cpu()
    pred_uninformative = model(uninformative.cuda()).cpu()
    pred_ood = model(ood_x.cuda()).cpu()
    pred_ind = model(ind_x.cuda()).cpu()


# %%
def get_prob_uncertainty(output, num_classes=3):
    evidence = relu_evidence(output)
    alpha = evidence + 1
    prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
    uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)
    return prob, uncertainty

# %%
prob_ind, unc_ind = get_prob_uncertainty(pred_ind)
prob_prototypes, unc_prototypes = get_prob_uncertainty(pred_prototypes)
prob_ood, unc_ood = get_prob_uncertainty(pred_ood)
prob_uninformative, unc_uninformative = get_prob_uncertainty(pred_uninformative)

# %% Check robustness

print("In distribution: ", accuracy_score(prob_ind.argmax(dim=-1), ind_y))
print("OOD: ", accuracy_score(prob_ood.argmax(dim=-1), ood_y))

# %%
for im, p, unc in zip(prototypes, prob_prototypes, unc_prototypes):
    plot_pair(im.moveaxis((0, 1, 2), (2, 0, 1)), p, unc)

# %%
plt.hist(
    [
        prob_ind.max(dim=-1).values,
        prob_uninformative.max(dim=-1).values,
        prob_ood.max(dim=-1).values,
    ],
    label=["informative", "uninformative", "ood"],
)
plt.legend()
plt.xlabel("Prediction score")
plt.ylabel("Percentage")

# %%
plt.hist(
    [
        unc_ind.max(dim=-1).values,
        unc_uninformative.max(dim=-1).values,
        unc_ood.max(dim=-1).values,
    ],
    label=["informative", "uninformative", "ood"],
)
plt.legend()
plt.xlabel("Uncertainty")
plt.ylabel("Percentage")

# %% Dirichlet distribution
plt.hist(
    [
        pred_ind.flatten(),
        pred_uninformative.flatten(),
        pred_ood.flatten() 
    ],
    label=["informative", "uninformative", "ood"],
)
plt.legend()
plt.xlabel("Logit")
plt.ylabel("Percentage")

# %%
