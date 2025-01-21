"""
卷积通道数对比实验
目标：评估不同卷积层通道配置对模型性能和推理速度的影响
实验设计：
1. 测试通道数：128, 96, 64, 32, 16, 8（指数级递减）
2. 每个配置训练150个epoch保证收敛
3. 使用1000次推理测试获取平均速度
4. 记录准确率、Kappa系数、各阶段F1分数
"""

import pytorch_lightning as pl
import random
import numpy as np

from model import TinySleepNet
from lightning_wrapper import LightningWrapper
from benchmark import benchmark

# 实验参数配置
CONV_CHANNELS = [128, 96, 64, 32, 16, 8]  # 测试的卷积通道数
NUM_EPOCHS = 150                          # 每个配置的训练epoch数
NUM_BENCH_TESTS = 1000                    # 基准测试次数

def run_experiment():
    """执行系统化对比实验"""
    random.seed(42)  # 固定随机种子保证可重复性
    results = []
    
    for conv_ch in CONV_CHANNELS:
        # 初始化模型和训练器
        model = LightningWrapper(TinySleepNet(conv1_ch=conv_ch))
        trainer = pl.Trainer(
            reload_dataloaders_every_epoch=True,
            gpus=1,
            max_epochs=NUM_EPOCHS
        )
        
        # 训练模型
        trainer.fit(model)
        
        # 性能基准测试
        bench_result = benchmark(model.net, num_tests=NUM_BENCH_TESTS)
        
        # 记录结果（添加平均F1分数）
        results.append((
            conv_ch,
            model.max_acc,
            model.best_k,
            np.mean(model.best_f1),  # 计算平均F1分数
            bench_result['avg_time']
        ))
        
        # 打印实时进度
        print(f"[{conv_ch}通道] 准确率:{model.max_acc:.2%} | "
              f"推理时间:{bench_result['avg_time']*1000:.2f}ms")

    # 生成最终报告
    print("\n=== 实验结果汇总 ===")
    print("通道数\t准确率\tKappa\t平均F1\t推理时间(ms)")
    for entry in results:
        print(f"{entry[0]}\t{entry[1]:.2%}\t{entry[2]:.4f}\t"
              f"{entry[3]:.4f}\t{entry[4]*1000:.2f}")

if __name__ == '__main__':
    run_experiment()
