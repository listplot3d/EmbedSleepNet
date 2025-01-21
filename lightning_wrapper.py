import numpy as np
import torch.nn.functional as F
import torch.optim
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn import metrics

from dataset import load_split_sleep_dataset


class LightningWrapper(pl.LightningModule):
    """PyTorch Lightning Training Pipeline Wrapper
    
    Main Features:
    1. Unified training/validation process management
    2. Automatic class imbalance handling (via weighted cross-entropy)
    3. Training metrics logging (loss, accuracy)
    4. Validation phase Cohen's Kappa and F1-score calculation
    
    Design Characteristics:
    - Class weights: [W, N1, N2, N3, REM] = [1.0, 1.5, 1.0, 1.0, 1.0]
      Compensates for N1 stage sample scarcity
    - Dynamic evaluation: Multi-dimensional metrics after each validation epoch
    - Data augmentation: Reset data order (reshuffle) at each training epoch start
    """
    def __init__(self, net, learning_rate=1e-3):
        super().__init__()
        self.net = net
        # Class weight configuration (handling sleep stage imbalance)
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1., 1.5, 1., 1., 1.])  # Corresponding to W, N1, N2, N3, REM
        )
        self.learning_rate = learning_rate
        self.max_acc = 0.0  # Track best validation accuracy
        self.best_k = None  # Best Kappa coefficient
        self.best_f1 = None  # Best F1 scores (per-class)

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

        self.val_labels.append(labels.cpu().numpy())
        self.val_pred.append(pred.cpu().numpy())

        acc = (pred == labels).sum() / len(pred)
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
        self.val_labels = []
        self.val_pred = []

    def on_validation_end(self):
        """Calculate comprehensive evaluation metrics after validation phase
        1. Combine predictions and true labels from all batches
        2. Calculate overall accuracy, Kappa coefficient, and per-stage F1 scores
        3. Update the best metric records
        4. Print detailed evaluation report for the current epoch
        """
        # Combine data from all batches
        labels = np.concatenate(self.val_labels)  # True labels [n_samples]
        pred = np.concatenate(self.val_pred)      # Predictions [n_samples]
        
        # Calculate basic metrics
        acc = (pred == labels).sum() / len(pred)  # Overall accuracy
        
        # Update the best metrics (only when accuracy improves)
        if acc > self.max_acc:
            self.max_acc = acc
            # Calculate Kappa coefficient (measures annotation agreement, range -1 to 1)
            self.best_k = metrics.cohen_kappa_score(labels, pred)
            # Calculate per-stage F1 scores (no averaging)
            self.best_f1 = metrics.f1_score(labels, pred, average=None)
            
        # Print detailed evaluation report
        stage_names = ['Wake', '  N1  ', '  N2  ', '  N3  ', '  REM  ']
        print(f"\n=== Validation Results [Epoch {self.current_epoch+1}] ===")
        print(f"| Accuracy | Kappa | {' | '.join(stage_names)} |")
        print(f"| {acc*100:.2f}% | {self.best_k:.4f} | " +
              " | ".join(f"{score*100:.2f}%" for score in self.best_f1) + " |")
