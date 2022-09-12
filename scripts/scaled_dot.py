import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.datasets as fetch_openml
import numpy as np
import sys
sys.path.insert(0, "/home/maria/AMC/includes")
from mlp import *
from torch.utils.data import DataLoader
import math


class Dataset(torch.utils.data.Dataset):

  def __init__(self, X, y):
    self.X = X
    self.y = y

  def __len__(self):
    return len(self.X)

  def __getitem__(self, ix):
    return torch.tensor(self.X[ix]).float(), torch.tensor(self.y[ix]).long()

class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, batch_size: int = 64, Dataset = Dataset):
        super().__init__()
        self.batch_size = batch_size
        self.Dataset = Dataset

    def setup(self, stage=None):
        mnist = fetch_openml('mnist_784', version=1)
        X, y = mnist["data"], mnist["target"]
        X_train, X_test, y_train, y_test = X[:60000] / 255., X[60000:] / 255., y[:60000].astype(np.int), y[60000:].astype(np.int)
        self.train_ds = self.Dataset(X_train, y_train)
        self.val_ds = self.Dataset(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)

class AttnDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, patch_size=(7, 7)):
        self.X = X
        self.y = y
        self.patch_size = patch_size

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ix):
        image = torch.tensor(self.X[ix]).float().view(28, 28) # 28 x 28
        h, w = self.patch_size
        patches = image.unfold(0, h, h).unfold(1, w, w) # 4 x 4 x 7 x 7
        patches = patches.contiguous().view(-1, h*w) # 16 x 49
        return patches, torch.tensor(self.y[ix]).long()

attn_dm = MNISTDataModule(Dataset = AttnDataset)
attn_dm.setup()
imgs, labels = next(iter(attn_dm.train_dataloader()))
print(imgs.shape, labels.shape)
fig = plt.figure(figsize=(5,5))

for i in range(4):
    for j in range(4):
        ax = plt.subplot(4, 4, i*4 + j + 1)
        ax.imshow(imgs[6,i*4 + j].view(7, 7), cmap="gray")
        ax.axis('off')
plt.tight_layout()
plt.show()



class ScaledDotSelfAttention(torch.nn.Module):

    def __init__(self, n_embeddings):
        super().__init__()
        self.key = torch.nn.Linear(n_embeddings, n_embeddings)
        self.query = torch.nn.Linear(n_embeddings, n_embeddings)
        self.value = torch.nn.Linear(n_embeddings, n_embeddings)

    def forward(self, x):
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        att = (q @ k.transpose(1,2)) * (1.0/math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        y = att @ v
        return y
class Model(MLP):

    def __init__(self, n_embed=7*7, seq_len=4*4):
        super().__init__()
        self.mlp = None
        self.attn = ScaledDotSelfAttention(n_embed)
        self.actn = torch.nn.ReLU(inplace=True)
        self.fc = torch.nn.Linear(n_embed*seq_len, 10)

    def forward(self, x):
        x = self.attn(x)
        y = self.fc(self.actn(x.view(x.size(0), -1)))
        return y

model = Model()
trainer = pl.Trainer(max_epochs=5)
trainer.fit(model, )