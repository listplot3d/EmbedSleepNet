import time
import torch
import psutil

def benchmark(net, num_tests=100, input_shape=(1, 1, 3000)):
    """模型性能基准测试工具
    
    参数：
    net: 要测试的PyTorch模型
    num_tests: 测试次数（默认100次）
    input_shape: 输入数据形状（默认[批次=1, 通道=1, 特征=3000]）
    
    返回：
    {
        'min_time': 最小推理时间(ms),
        'avg_time': 平均推理时间(ms),
        'mem_usage': 内存占用(MB),
        'flops': 理论计算量(百万)
    }
    """
    device = next(net.parameters()).device
    net.eval()
    
    # 生成测试数据
    data = torch.randn(input_shape).to(device)
    
    # 预热运行（消除初始化影响）
    with torch.no_grad():
        for _ in range(10):
            _ = net(data)
    
    # 内存基准
    process = psutil.Process()
    mem_before = process.memory_info().rss  # 字节
    
    # 精确计时
    total_time = 0.0
    min_time = float('inf')
    for _ in range(num_tests):
        start_time = time.perf_counter()
        with torch.no_grad():
            _ = net(data)
        elapsed = time.perf_counter() - start_time
        total_time += elapsed
        min_time = min(min_time, elapsed)
    
    # 内存计算
    mem_after = process.memory_info().rss
    mem_usage = (mem_after - mem_before) / (1024 ** 2)  # 转MB
    
    # 理论计算量估算（基于参数量）
    flops = sum(p.numel() for p in net.parameters()) * 2 * input_shape[-1]  # 2*乘加操作
    
    return {
        'min_time': min_time * 1000,          # 毫秒
        'avg_time': (total_time / num_tests) * 1000,
        'mem_usage': mem_usage,
        'flops': flops / 1e6                  # 百万FLOPs
    }
