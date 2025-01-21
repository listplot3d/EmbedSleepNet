import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torch
import random
import argparse

from pytorch_lightning.callbacks import ModelCheckpoint

from model import TinySleepNet, EmbedSleepNet
from lightning_wrapper import LightningWrapper

if __name__ == '__main__':
    # Command line argument parsing
    parser = argparse.ArgumentParser(description='Training the model')
    parser.add_argument('--flavor', choices=['embed', 'tiny'], required=True, help='Choose model type: embed or tiny')
    parser.add_argument('--epochs', type=int, default=450, help='Number of epochs to train the model')
    parser.add_argument('--model_name', type=str, default='model', help='Name of the saved model file')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to the checkpoint file to resume from')
    args = parser.parse_args()

    # Set random seed
    random.seed(42)

    # Initialize the model
    if args.flavor == 'tiny':
        net = TinySleepNet()
    else:
        net = EmbedSleepNet()

    model = LightningWrapper(net)

    # Configure checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='checkpoints/',  # Dedicated checkpoint directory
        filename=f'{args.model_name}-{{epoch:03d}}-{{val_acc:.2f}}',  # Include epoch and accuracy
        save_top_k=3,  # Keep top 3 checkpoints
        mode='max',
        save_last=True,  # Additionally save the checkpoint of the last epoch
        auto_insert_metric_name=False  # Beautify file name
    )

    # Initialize the logger
    logger_dir = 'd:/logs/' #specify absolute path here
    tb_logger = pl_loggers.TensorBoardLogger(logger_dir) 
    print("* Logging files are saved here:",logger_dir)

    # Initialize the Trainer
    trainer = pl.Trainer(
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        reload_dataloaders_every_n_epochs=10,
        accelerator="auto",
        max_epochs=args.epochs
    )

    # Resume training from checkpoint file
    if args.resume_from:
        print(f"Resuming training from checkpoint: {args.resume_from}")
        trainer.fit(model, ckpt_path=args.resume_from)
    else:
        print("Starting training from scratch.")
        trainer.fit(model)

    # Print the final model accuracy
    print(f'Saved model accuracy {model.max_acc * 100}%')
