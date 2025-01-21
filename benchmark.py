import time
import torch
import psutil

def benchmark(net, num_tests=100, input_shape=(1, 1, 3000)):
    """Model performance benchmarking tool
    
    Parameters:
    net: PyTorch model to test
    num_tests: Number of test iterations (default 100)
    input_shape: Input data shape (default [batch=1, channels=1, features=3000])
    
    Returns:
    {
        'min_time': Minimum inference time (ms),
        'avg_time': Average inference time (ms),
        'mem_usage': Memory usage (MB),
        'flops': Theoretical computations (million FLOPs)
    }
    """
    device = next(net.parameters()).device
    net.eval()
    
    # Generate test data
    data = torch.randn(input_shape).to(device)
    
    # Warm-up runs to eliminate initialization overhead
    with torch.no_grad():
        for _ in range(10):
            _ = net(data)
    
    # Memory baseline measurement
    process = psutil.Process()
    mem_before = process.memory_info().rss  # Bytes
    
    # Precision timing tests
    total_time = 0.0
    min_time = float('inf')
    for _ in range(num_tests):
        start_time = time.perf_counter()
        with torch.no_grad():
            _ = net(data)
        elapsed = time.perf_counter() - start_time
        total_time += elapsed
        min_time = min(min_time, elapsed)
    
    # Calculate memory usage
    mem_after = process.memory_info().rss
    mem_usage = (mem_after - mem_before) / (1024 ** 2)  # Convert to MB
    
    # Estimate theoretical computations (based on parameter count)
    flops = sum(p.numel() for p in net.parameters()) * 2 * input_shape[-1]  # Multiply-accumulate operations
    
    return {
        'min_time': min_time * 1000,          # Milliseconds
        'avg_time': (total_time / num_tests) * 1000,
        'mem_usage': mem_usage,
        'flops': flops / 1e6                  # Million FLOPs
    }
