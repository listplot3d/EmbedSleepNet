import numpy as np
import torch.nn.functional as F
import torch.optim
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset import load_split_sleep_dataset


class LightningWrapper(pl.LightningModule):
    def __init__(self, net, learning_rate=1e-3):
        super().__init__()
        self.net = net
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([1., 1.5, 1., 1., 1.]))
        self.learning_rate = learning_rate
        self.max_acc = 0.0

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, train_batch, batch_idx):
        data, labels = train_batch
        outputs = self(data)
        outputs = outputs.reshape(-1, outputs.shape[2])
        labels = labels.reshape(-1)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        self.log('train_epoch', loss, on_epoch=True)

        pred = F.softmax(outputs, dim=1)
        pred = torch.argmax(pred, dim=1)
        acc = (pred == labels).sum() / len(pred)
        self.log('train_acc', acc, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, val_batch, batch_idx):
        data, labels = val_batch
        outputs = self(data)
        outputs = outputs.reshape(-1, outputs.shape[len(outputs.shape) - 1])
        labels = labels.reshape(-1)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss.item(), prog_bar=True)

        pred = F.softmax(outputs, dim=1)
        pred = torch.argmax(pred, dim=1)
        acc = (pred == labels).sum() / len(pred)
        self.val_accs.append(acc.item())
        self.log('val_acc', acc.item(), prog_bar=True)
        return {'val_loss': loss.item()}

    def prepare_data(self):
        self.train_ds, self.val_ds = load_split_sleep_dataset()

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, batch_size=64)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1)

    def on_train_epoch_start(self):
        self.train_ds.reshuffle()

    def on_validation_start(self):
        self.val_accs = []

    def on_validation_end(self):
        acc = np.mean(np.asarray(self.val_accs)).item()
        self.max_acc = max(self.max_acc, acc)
