"""
Convolutional channel configuration comparison experiment
Objective: Evaluate impact of different conv layer channel configurations on model performance and inference speed
Experimental Design:
1. Test channel counts: 128, 96, 64, 32, 16, 8 (exponential decrease)
2. Train each configuration for 150 epochs to ensure convergence
3. Obtain average speed from 1000 inference tests
4. Record accuracy, Kappa coefficient, and per-stage F1 scores
"""

import pytorch_lightning as pl
import random
import numpy as np

from model import TinySleepNet
from lightning_wrapper import LightningWrapper
from benchmark import benchmark

# Experimental parameters configuration
CONV_CHANNELS = [128, 96, 64, 32, 16, 8]  # Convolution channels to test
NUM_EPOCHS = 150                          # Training epochs per configuration
NUM_BENCH_TESTS = 1000                    # Number of benchmark tests

def run_experiment():
    """Execute systematic comparative experiment"""
    random.seed(42)  # Fix random seed for reproducibility
    results = []
    
    for conv_ch in CONV_CHANNELS:
        # Initialize model and trainer
        model = LightningWrapper(TinySleepNet(conv1_ch=conv_ch))
        trainer = pl.Trainer(
            reload_dataloaders_every_epoch=True,
            gpus=1,
            max_epochs=NUM_EPOCHS
        )
        
        # Train the model
        trainer.fit(model)
        
        # Performance benchmark test
        bench_result = benchmark(model.net, num_tests=NUM_BENCH_TESTS)
        
        # Record results (add average F1 score)
        results.append((
            conv_ch,
            model.max_acc,
            model.best_k,
            np.mean(model.best_f1),  # Calculate average F1 score
            bench_result['avg_time']
        ))
        
        # Print real-time progress
        print(f"[{conv_ch} channels] Accuracy: {model.max_acc:.2%} | "
              f"Inference time: {bench_result['avg_time']*1000:.2f}ms")

    # Generate final report
    print("\n=== Summary of Experimental Results ===")
    print("Channels\tAccuracy\tKappa\tAverage F1\tInference Time (ms)")
    for entry in results:
        print(f"{entry[0]}\t{entry[1]:.2%}\t{entry[2]:.4f}\t"
              f"{entry[3]:.4f}\t{entry[4]*1000:.2f}")

if __name__ == '__main__':
    run_experiment()
