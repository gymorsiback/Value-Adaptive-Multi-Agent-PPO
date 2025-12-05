"""
GPU Configuration and Optimization Tools
Provides best practice configuration for GPU usage
"""

import torch
import gc

def configure_gpu_for_training():
    """
    Configure GPU for optimal training settings
    
    Returns:
        device: Configured device
        gpu_info: GPU information dictionary
    """
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, will use CPU for training")
        return torch.device('cpu'), {}
    
    # Get GPU information
    device = torch.device('cuda')
    device_props = torch.cuda.get_device_properties(device)
    device_name = device_props.name
    
    gpu_info = {
        'device_name': device_name,
        'total_memory_gb': device_props.total_memory / 1024**3,
        'multi_processor_count': device_props.multi_processor_count,
        'memory_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
        'memory_cached_gb': torch.cuda.memory_reserved() / 1024**3
    }
    
    # Clear GPU cache before starting
    torch.cuda.empty_cache()
    
    # Display GPU information
    print(f"Detected GPU: {device_name}")
    print(f"Total Memory: {device_props.total_memory / 1024**3:.2f}GB")
    print(f"Multiprocessor Count: {device_props.multi_processor_count}")
    
    # Set GPU optimization options
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
    
    # Enable cudnn benchmark mode (for networks with fixed input size)
    torch.backends.cudnn.benchmark = True
    print("Enabled cuDNN benchmark mode")
    
    # Set memory allocation strategy
    torch.cuda.set_per_process_memory_fraction(0.9)
    
    return device, gpu_info

def monitor_gpu_memory():
    """Monitor current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, "
              f"Cached: {cached:.2f}GB, Total: {total:.2f}GB")
        
        # Show memory usage percentage
        usage_percentage = (allocated / total) * 100
        print(f"Memory Usage: {usage_percentage:.1f}%")
        
        return {
            'allocated_gb': allocated,
            'cached_gb': cached,
            'total_gb': total,
            'usage_percentage': usage_percentage
        }
    else:
        print("No GPU available for monitoring")
        return {}

def optimize_for_batch_size(batch_size: int):
    """
    Optimize GPU settings for specific batch size
    
    Args:
        batch_size: Training batch size
    """
    if not torch.cuda.is_available():
        return
    
    # Estimate memory needs based on batch size
    estimated_memory_gb = batch_size * 0.01  # Rough estimate: 10MB per sample
    total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    if estimated_memory_gb > total_memory_gb * 0.8:
        print(f"WARNING: Batch size {batch_size} may exceed GPU memory")
        print(f"Estimated usage: {estimated_memory_gb:.2f}GB, Available: {total_memory_gb * 0.8:.2f}GB")
        
        # Suggest smaller batch size
        suggested_batch_size = int(batch_size * 0.8 * total_memory_gb / estimated_memory_gb)
        print(f"Suggested batch size: {suggested_batch_size}")
    else:
        print(f"Batch size {batch_size} should fit comfortably in GPU memory")

def cleanup_gpu():
    """Clean up GPU memory and cache"""
    if torch.cuda.is_available():
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        print("GPU memory cleaned up")
        monitor_gpu_memory()
    else:
        print("No GPU available for cleanup")

def get_optimal_device():
    """Get the optimal device for current system"""
    if torch.cuda.is_available():
        # Check if multiple GPUs are available
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            print(f"Multiple GPUs detected: {gpu_count}")
            # Use the first GPU by default
            device = torch.device('cuda:0')
        else:
            device = torch.device('cuda')
        
        # Verify GPU accessibility
        try:
            test_tensor = torch.randn(10, 10).to(device)
            del test_tensor
            torch.cuda.empty_cache()
            print(f"GPU device verified: {device}")
            return device
        except RuntimeError as e:
            print(f"GPU test failed: {e}")
            print("Falling back to CPU")
            return torch.device('cpu')
    else:
        print("CUDA not available, using CPU")
        return torch.device('cpu')

def set_deterministic_training(seed: int = 42):
    """
    Set deterministic training for reproducible results
    
    Args:
        seed: Random seed for reproducibility
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Make cudnn deterministic (may slow down training)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        print(f"Set deterministic training with seed {seed}")
    
    # Also set numpy and python seeds
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)

def profile_gpu_usage():
    """Profile GPU usage during training"""
    if not torch.cuda.is_available():
        return
    
    # Enable profiler
    torch.autograd.profiler.profile(
        enabled=True,
        use_cuda=True,
        record_shapes=True,
        with_flops=True
    )
    print("GPU profiling enabled")

def get_gpu_utilization_stats():
    """Get detailed GPU utilization statistics"""
    if not torch.cuda.is_available():
        return {}
    
    stats = {}
    
    # Memory statistics
    stats['memory'] = {
        'allocated': torch.cuda.memory_allocated(),
        'cached': torch.cuda.memory_reserved(),
        'max_allocated': torch.cuda.max_memory_allocated(),
        'max_cached': torch.cuda.max_memory_reserved()
    }
    
    # Device properties
    props = torch.cuda.get_device_properties(0)
    stats['device'] = {
        'name': props.name,
        'major': props.major,
        'minor': props.minor,
        'total_memory': props.total_memory,
        'multi_processor_count': props.multi_processor_count
    }
    
    return stats

# GPU configuration constants
GPU_MEMORY_FRACTION = 0.8  # Use 80% of available GPU memory
CUDNN_BENCHMARK = True     # Enable cuDNN benchmarking
CUDNN_DETERMINISTIC = False # Disable deterministic mode for speed

# Memory optimization settings
MEMORY_CLEANUP_FREQUENCY = 100  # Clean memory every N iterations
GRADIENT_CHECKPOINTING = False  # Enable for very large models

if __name__ == "__main__":
    # Test GPU configuration
    print("Testing GPU Configuration...")
    device, gpu_info = configure_gpu_for_training()
    print(f"Configured device: {device}")
    print(f"GPU info: {gpu_info}")
    
    # Monitor memory
    monitor_gpu_memory()
    
    # Test optimal device
    optimal_device = get_optimal_device()
    print(f"Optimal device: {optimal_device}")
    
    # Cleanup
    cleanup_gpu() 