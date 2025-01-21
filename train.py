import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torch
import random
import argparse

from pytorch_lightning.callbacks import ModelCheckpoint

from model import TinySleepNet, EmbedSleepNet
from lightning_wrapper import LightningWrapper

if __name__ == '__main__':
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='Training the model')
    parser.add_argument('--flavor', choices=['embed', 'tiny'], required=True, help='Choose model type: embed or tiny')
    parser.add_argument('--epochs', type=int, default=450, help='Number of epochs to train the model')
    parser.add_argument('--model_name', type=str, default='model', help='Name of the saved model file')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to the checkpoint file to resume from')
    args = parser.parse_args()

    # 设置随机种子
    random.seed(42)

    # 初始化模型
    if args.flavor == 'tiny':
        net = TinySleepNet()
    else:
        net = EmbedSleepNet()

    model = LightningWrapper(net)

    # 检查点回调配置优化
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='checkpoints/',  # 创建专用检查点目录
        filename=f'{args.model_name}-{{epoch:03d}}-{{val_acc:.2f}}',  # 包含epoch和准确率
        save_top_k=3,  # 保留最佳3个检查点
        mode='max',
        save_last=True,  # 额外保存最后epoch的检查点
        auto_insert_metric_name=False  # 美化文件名
    )

    # 初始化日志记录器
    tb_logger = pl_loggers.TensorBoardLogger('d:/logs/')

    # 初始化 Trainer
    trainer = pl.Trainer(
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        reload_dataloaders_every_n_epochs=10,
        accelerator="auto",
        max_epochs=args.epochs
    )

    # 从 checkpoint 文件继续训练
    if args.resume_from:
        print(f"Resuming training from checkpoint: {args.resume_from}")
        trainer.fit(model, ckpt_path=args.resume_from)
    else:
        print("Starting training from scratch.")
        trainer.fit(model)

    # 打印最终模型的准确率
    print(f'Saved model accuracy {model.max_acc * 100}%')
