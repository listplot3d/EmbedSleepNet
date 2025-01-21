import numpy as np
import torch.nn.functional as F
import torch.optim
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn import metrics

from dataset import load_split_sleep_dataset


class LightningWrapper(pl.LightningModule):
    """PyTorch Lightning训练流程封装器
    
    主要功能：
    1. 统一管理训练/验证流程
    2. 自动处理类别不平衡（通过加权交叉熵）
    3. 记录训练指标（loss, accuracy）
    4. 计算验证阶段的Cohen's Kappa和F1分数
    
    设计特点：
    - 权重分配：[W, N1, N2, N3, REM] = [1.0, 1.5, 1.0, 1.0, 1.0]
      针对N1阶段样本量少的问题进行补偿
    - 动态评估：每个验证epoch结束后计算多维度指标
    - 数据增强：每个训练epoch开始时重置数据顺序（reshuffle）
    """
    def __init__(self, net, learning_rate=1e-3):
        super().__init__()
        self.net = net
        # 类别权重配置（应对睡眠阶段不均衡问题）
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1., 1.5, 1., 1., 1.])  # 对应W, N1, N2, N3, REM
        )
        self.learning_rate = learning_rate
        self.max_acc = 0.0  # 记录最佳验证准确率
        self.best_k = None  # 最佳Kappa系数
        self.best_f1 = None  # 最佳F1分数（各类别独立计算）

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
        """验证阶段结束后计算综合评估指标
        1. 合并所有批次的预测结果和真实标签
        2. 计算整体准确率、Kappa系数和各阶段F1分数
        3. 更新最佳指标记录
        4. 打印当前epoch的详细评估报告
        """
        # 合并所有batch的数据
        labels = np.concatenate(self.val_labels)  # 真实标签 [n_samples]
        pred = np.concatenate(self.val_pred)      # 预测结果 [n_samples]
        
        # 计算基础指标
        acc = (pred == labels).sum() / len(pred)  # 整体准确率
        
        # 更新最佳指标（仅在准确率提升时）
        if acc > self.max_acc:
            self.max_acc = acc
            # 计算Kappa系数（衡量标注一致性，范围-1到1）
            self.best_k = metrics.cohen_kappa_score(labels, pred)
            # 计算各阶段F1分数（不进行平均）
            self.best_f1 = metrics.f1_score(labels, pred, average=None)
            
        # 打印详细评估报告
        stage_names = ['Wake', '  N1  ', '  N2  ', '  N3  ', '  REM  ']
        print(f"\n=== 验证结果 [Epoch {self.current_epoch+1}] ===")
        print(f"| 准确率 | Kappa系数 | {' | '.join(stage_names)} |")
        print(f"| {acc*100:.2f}% | {self.best_k:.4f} | " +
              " | ".join(f"{score*100:.2f}%" for score in self.best_f1) + " |")
