import numpy as np
import cupy as cp
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import fft as cpu_fft, fftfreq as cpu_fftfreq
from scipy.signal import chirp
import time
import psutil

# Global plotting configuration
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['axes.unicode_minus'] = False

class GPUFFTProcessor:
    """GPU-accelerated FFT processor with both regular and parallel modes"""
    
    def __init__(self, signal_length, num_channels=2, fs=40e9):
        self.N = signal_length
        self.K = num_channels
        self.fs = fs
        self.log2K = int(np.log2(num_channels))
        
        if 2**self.log2K != self.K:
            raise ValueError("num_channels must be 2^N")
        
        # Get GPU information to dynamically adjust stream count
        self.gpu_properties = cp.cuda.runtime.getDeviceProperties(0)
        self.sm_count = self.gpu_properties['multiProcessorCount']
        
        # Dynamic stream count based on GPU SM count and FFT size
        # For small FFT sizes, use fewer streams to reduce overhead
        # For large FFT sizes, use more streams to maximize parallelism
        if self.N < 10000:
            self.num_streams = min(self.sm_count // 2, num_channels, 4)
        elif self.N < 100000:
            self.num_streams = min(self.sm_count, num_channels, 8)
        else:
            self.num_streams = min(self.sm_count * 2, num_channels, 16)
        
        # Create CUDA streams and events for efficient synchronization
        self.streams = [cp.cuda.Stream() for _ in range(self.num_streams)]
        self.events = [cp.cuda.Event() for _ in range(self.num_streams)]
        
        # Warm up GPU to avoid first-time compilation overhead
        self._warmup()
    
    def _warmup(self):
        """Warm up GPU kernels to avoid first-time compilation overhead"""
        try:
            warmup_signal = cp.zeros(1024, dtype=cp.complex128)
            cp.fft.fft(warmup_signal)
            cp.cuda.Stream.null.synchronize()
        except:
            pass
    
    def decompose_signal(self, signal_gpu):
        """Decompose signal into K sub-channels on GPU"""
        sub_signals = []
        for k in range(self.K):
            sub_signal = signal_gpu[k::self.K]
            sub_signals.append(sub_signal)
        return sub_signals
    
    def regular_fft(self, signal_np):
        """Regular FFT processing on GPU (single kernel call)"""
        # Start timing
        start_time = time.time()
        
        # Transfer signal to GPU
        signal_gpu = cp.asarray(signal_np)
        
        # Single FFT kernel call
        fft_result_gpu = cp.fft.fft(signal_gpu)
        
        # Transfer result back to CPU
        fft_result_np = cp.asnumpy(fft_result_gpu)
        
        # End timing
        end_time = time.time()
        
        # Calculate memory usage
        memory_usage = self._calculate_memory_usage(signal_np)
        
        return fft_result_np, end_time - start_time, memory_usage
    
    def parallel_fft(self, signal_np):
        """Parallel FFT processing on GPU with optimized CUDA streams and events"""
        # Start timing
        start_time = time.time()
        
        # Transfer signal to GPU once
        signal_gpu = cp.asarray(signal_np)
        original_length = len(signal_gpu)
        
        # Allocate result arrays in advance
        samples_per_channel = original_length // self.K
        sub_ffts = [cp.zeros(samples_per_channel, dtype=cp.complex128) for _ in range(self.K)]
        
        # Use event-based synchronization instead of explicit stream synchronization
        # Process sub-channels in parallel using streams
        for i in range(self.K):
            # Assign stream based on channel index
            stream_idx = i % self.num_streams
            stream = self.streams[stream_idx]
            
            with stream:
                # Extract sub-channel directly in the stream context
                sub_signal = signal_gpu[i::self.K]
                # Perform FFT
                sub_ffts[i] = cp.fft.fft(sub_signal)
            
            # Record event for this stream when all its channels are processed
            if (i + 1) % self.num_streams == 0 or i == self.K - 1:
                stream.record(self.events[stream_idx])
        
        # Optimized reconstruction using vectorized operations with reduced memory access
        fft_result_gpu = self._reconstruct_fft_highly_optimized(sub_ffts, original_length)
        
        # Transfer result back to CPU
        fft_result_np = cp.asnumpy(fft_result_gpu)
        
        # End timing
        end_time = time.time()
        
        # Calculate memory usage
        memory_usage = self._calculate_memory_usage(signal_np)
        
        return fft_result_np, sub_ffts, end_time - start_time, memory_usage
    
    def _reconstruct_fft(self, sub_ffts, original_length):
        """Reconstruct original signal FFT from K-channel FFT results on GPU"""
        reconstructed = cp.zeros(original_length, dtype=cp.complex128)
        samples_per_channel = original_length // self.K
        
        # Vectorized reconstruction for better GPU performance
        for j in range(self.K):
            # Create index array for phase correction (use integer type)
            k = cp.arange(original_length, dtype=cp.int64)
            phase_correction = cp.exp(-2j * cp.pi * j * k / original_length)
            
            # Calculate indices for each channel (convert to integer)
            idx = (k % samples_per_channel).astype(cp.int64)
            
            # Apply phase correction and accumulate
            reconstructed += sub_ffts[j][idx] * phase_correction
        
        return reconstructed / self.K
    
    def _reconstruct_fft_highly_optimized(self, sub_ffts, original_length):
        """Highly optimized reconstruction with reduced memory access and computation"""
        # Calculate once
        samples_per_channel = original_length // self.K
        
        # Wait for all streams to complete using events
        for event in self.events:
            event.synchronize()
        
        # Pre-allocate result array with correct length
        reconstructed = cp.zeros(original_length, dtype=cp.complex128)
        
        # Precompute indices once (reused for all channels)
        k = cp.arange(original_length, dtype=cp.int64)
        idx = (k % samples_per_channel).astype(cp.int64)
        
        # Precompute common twiddle factor base
        # This avoids recalculating -2j*pi*k/N for each channel
        twiddle_base = -2j * cp.pi * k / original_length
        
        # 优化：使用更高效的批量处理
        # 对于小K值，直接处理；对于大K值，使用批量处理
        if self.K <= 4:
            # 小K值：直接处理所有通道，减少循环开销
            for j in range(self.K):
                # Calculate phase correction using precomputed base
                phase_correction = cp.exp(twiddle_base * j)
                # 使用高级索引，但通过预计算idx优化
                reconstructed += sub_ffts[j][idx] * phase_correction
        else:
            # 大K值：使用批量处理，提高内存访问效率
            # 批量大小根据K值动态调整
            batch_size = min(4, self.K)
            for j in range(0, self.K, batch_size):
                # 批量累加，减少中间变量
                batch_sum = cp.zeros(original_length, dtype=cp.complex128)
                for batch_j in range(batch_size):
                    if j + batch_j < self.K:
                        phase_correction = cp.exp(twiddle_base * (j + batch_j))
                        batch_sum += sub_ffts[j + batch_j][idx] * phase_correction
                reconstructed += batch_sum
        
        # Normalize once at the end
        reconstructed /= self.K
        
        return reconstructed
        
    def _reconstruct_fft_optimized(self, sub_ffts, original_length):
        """Compatibility wrapper for original optimized reconstruction"""
        return self._reconstruct_fft_highly_optimized(sub_ffts, original_length)
    
    def _calculate_memory_usage(self, signal_np):
        """Calculate memory usage in MB"""
        # Signal memory in MB
        signal_memory = signal_np.nbytes / (1024 * 1024)
        
        # Estimate temporary arrays memory
        temp_memory = signal_memory * 2  # For FFT results and phase corrections
        
        # Total GPU memory estimate
        total_gpu_memory = signal_memory + temp_memory
        
        return {
            'signal_memory_mb': signal_memory,
            'gpu_memory_est_mb': total_gpu_memory
        }

class SignalGenerator:
    """Generate test signals"""
    
    def __init__(self, fs=40e9, duration=1e-8):
        self.fs = fs
        self.duration = duration
        self.t = np.arange(0, duration, 1/fs)
        self.N = len(self.t)
    
    def sinusoid_1GHz(self):
        """1 GHz sinusoid signal"""
        return np.sin(2 * np.pi * 1e9 * self.t), "1 GHz Sinusoid"
    
    def lfm_10_18GHz(self):
        """10-18 GHz linear frequency modulation signal"""
        signal = chirp(self.t, 10e9, self.t[-1], 18e9, method='linear')
        return signal, "10-18 GHz LFM"
    
    def qpsk_1Gbps(self):
        """1 Gbps QPSK signal"""
        symbols = np.random.randint(0, 4, int(1e9 * self.duration))
        qpsk_map = {0: 1+1j, 1: 1-1j, 2: -1+1j, 3: -1-1j}
        qpsk_symbols = np.array([qpsk_map[s] for s in symbols]) / np.sqrt(2)
        
        samples_per_symbol = self.N // len(symbols)
        signal = np.repeat(qpsk_symbols, samples_per_symbol)[:self.N]
        carrier = np.exp(2j * np.pi * 5e9 * self.t)
        return signal * carrier, "1 Gbps QPSK"

class PerformanceAnalyzer:
    """Comprehensive performance analysis"""
    
    @staticmethod
    def measure_regular_gpu_fft(signal, processor_gpu):
        """Measure regular GPU FFT performance"""
        return processor_gpu.regular_fft(signal)
    
    @staticmethod
    def measure_parallel_gpu_fft(signal, processor_gpu):
        """Measure parallel GPU FFT performance"""
        return processor_gpu.parallel_fft(signal)
    
    @staticmethod
    def compute_complexity(N, K):
        """Compute theoretical computational complexity"""
        return K * (N/K) * np.log2(N/K)
    
    @staticmethod
    def calculate_error(fft_regular, fft_parallel):
        """Calculate error between regular and parallel GPU FFT results"""
        error = np.mean(np.abs(fft_regular - fft_parallel))
        relative_error = error / np.mean(np.abs(fft_regular))
        return error, relative_error

def benchmark_gpu_performance():
    """Benchmark Regular vs Parallel GPU FFT performance"""
    print("=" * 70)
    print("Regular vs Parallel GPU FFT Performance Benchmark")
    print("=" * 70)
    
    # Test configurations
    fs = 40e9
    duration = 1e-8
    gen = SignalGenerator(fs, duration)
    
    # Create single GPU processor
    processor_gpu = GPUFFTProcessor(gen.N, num_channels=2, fs=fs)
    
    # Test signals
    signals = [
        gen.sinusoid_1GHz(),
        gen.lfm_10_18GHz(),
        gen.qpsk_1Gbps()
    ]
    
    # Results storage
    results = []
    
    for signal, name in signals:
        print(f"\n[Processing] {name}")
        print("-" * 50)
        
        # Regular GPU FFT
        print("  Regular GPU FFT processing...")
        fft_regular, regular_time, regular_memory = PerformanceAnalyzer.measure_regular_gpu_fft(signal, processor_gpu)
        
        # Parallel GPU FFT
        print("  Parallel GPU FFT processing...")
        fft_parallel, sub_ffts_parallel, parallel_time, parallel_memory = PerformanceAnalyzer.measure_parallel_gpu_fft(signal, processor_gpu)
        
        # Calculate errors
        error, relative_error = PerformanceAnalyzer.calculate_error(fft_regular, fft_parallel)
        
        # Calculate computational complexity
        complexity = PerformanceAnalyzer.compute_complexity(gen.N, 2)
        
        # Calculate speedup (parallel vs regular on GPU)
        speedup = regular_time / parallel_time if parallel_time > 0 else float('inf')
        
        # Store results
        results.append({
            'signal_name': name,
            'regular_time_ms': regular_time * 1000,
            'parallel_time_ms': parallel_time * 1000,
            'speedup': speedup,
            'error': error,
            'relative_error': relative_error,
            'complexity': complexity,
            'regular_memory_mb': regular_memory['gpu_memory_est_mb'],
            'parallel_memory_mb': parallel_memory['gpu_memory_est_mb']
        })
        
        # Print results
        print(f"  Regular GPU Time: {regular_time * 1000:.2f} ms")
        print(f"  Parallel GPU Time: {parallel_time * 1000:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x (parallel vs regular)")
        print(f"  Mean Error: {error:.2e}")
        print(f"  Relative Error: {relative_error:.2e}")
        print(f"  Computational Complexity: {complexity:.0f} operations")
        print(f"  Regular GPU Memory: {regular_memory['gpu_memory_est_mb']:.2f} MB")
        print(f"  Parallel GPU Memory: {parallel_memory['gpu_memory_est_mb']:.2f} MB")
    
    # Generate performance report
    generate_performance_report(results)
    
    return results

def generate_performance_report(results):
    """Generate performance report and charts"""
    print("\n" + "=" * 70)
    print("GPU PERFORMANCE COMPARISON REPORT")
    print("=" * 70)
    
    # Table header
    print("\nSignal Name | Regular (ms) | Parallel (ms) | Speedup | Error | Relative Error | Complexity | Memory (MB)")
    print("-" * 120)
    
    # Table data
    for result in results:
        print(f"{result['signal_name']:<12} | {result['regular_time_ms']:>12.2f} | {result['parallel_time_ms']:>12.2f} | {result['speedup']:>7.2f}x | {result['error']:>7.2e} | {result['relative_error']:>15.2e} | {result['complexity']:>11.0f} | {result['parallel_memory_mb']:>10.2f}")
    
    # Plot performance comparison
    plot_performance_comparison(results)
    
    print("\nReport generated successfully!")

def plot_performance_comparison(results):
    """Plot performance comparison charts"""
    signal_names = [r['signal_name'] for r in results]
    regular_times = [r['regular_time_ms'] for r in results]
    parallel_times = [r['parallel_time_ms'] for r in results]
    speedups = [r['speedup'] for r in results]
    memory_usage = [r['parallel_memory_mb'] for r in results]
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Execution Time Comparison
    ax = axes[0, 0]
    x = np.arange(len(signal_names))
    width = 0.35
    
    ax.bar(x - width/2, regular_times, width, label='Regular GPU FFT', alpha=0.7, color='blue')
    ax.bar(x + width/2, parallel_times, width, label='Parallel GPU FFT', alpha=0.7, color='green')
    ax.set_xlabel('Signal Type')
    ax.set_ylabel('Execution Time (ms)')
    ax.set_title('Regular vs Parallel GPU FFT Execution Time')
    ax.set_xticks(x)
    ax.set_xticklabels(signal_names, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Speedup
    ax = axes[0, 1]
    ax.bar(x, speedups, width, alpha=0.7, color='purple')
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Baseline (Speedup = 1)')
    ax.set_xlabel('Signal Type')
    ax.set_ylabel('Speedup (x)')
    ax.set_title('Parallel FFT Speedup Over Regular FFT')
    ax.set_xticks(x)
    ax.set_xticklabels(signal_names, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Memory Usage
    ax = axes[1, 0]
    ax.bar(x, memory_usage, width, alpha=0.7, color='orange')
    ax.set_xlabel('Signal Type')
    ax.set_ylabel('GPU Memory Usage (MB)')
    ax.set_title('Parallel GPU FFT Memory Usage')
    ax.set_xticks(x)
    ax.set_xticklabels(signal_names, rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Computational Complexity
    ax = axes[1, 1]
    complexities = [r['complexity'] for r in results]
    ax.bar(x, complexities, width, alpha=0.7, color='red')
    ax.set_xlabel('Signal Type')
    ax.set_ylabel('Computational Complexity')
    ax.set_title('Theoretical Computational Complexity')
    ax.set_xticks(x)
    ax.set_xticklabels(signal_names, rotation=45)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gpu_parallel_vs_regular_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def memory_analysis():
    """Analyze memory usage patterns"""
    print("\n" + "=" * 70)
    print("MEMORY ANALYSIS")
    print("=" * 70)
    
    # Test different FFT sizes
    fs = 40e9
    duration = 1e-8
    fft_sizes = [2**10, 2**12, 2**14, 2**16, 2**18]
    
    memory_results = []
    
    for size in fft_sizes:
        gen = SignalGenerator(fs, duration)
        gen.N = size
        gen.t = np.arange(0, duration, 1/fs)[:size]
        
        # Generate signal
        signal, _ = gen.sinusoid_1GHz()
        
        # Measure memory
        processor_gpu = GPUFFTProcessor(size, num_channels=2, fs=fs)
        _, _, _, memory_usage = processor_gpu.parallel_fft(signal)
        
        memory_results.append({
            'fft_size': size,
            'signal_memory': memory_usage['signal_memory_mb'],
            'gpu_memory': memory_usage['gpu_memory_est_mb']
        })
        
        print(f"FFT Size: {size:>8} | Signal Memory: {memory_usage['signal_memory_mb']:>8.2f} MB | GPU Memory: {memory_usage['gpu_memory_est_mb']:>8.2f} MB")
    
    # Plot memory scaling
    plot_memory_scaling(memory_results)

def plot_memory_scaling(memory_results):
    """Plot memory scaling with FFT size"""
    fft_sizes = [r['fft_size'] for r in memory_results]
    signal_memory = [r['signal_memory'] for r in memory_results]
    gpu_memory = [r['gpu_memory'] for r in memory_results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(fft_sizes, signal_memory, 'o-', linewidth=2, markersize=8, label='Signal Memory')
    plt.plot(fft_sizes, gpu_memory, 's-', linewidth=2, markersize=8, label='Estimated GPU Memory')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('FFT Size')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Scaling with FFT Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('memory_scaling_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Run GPU parallel vs regular benchmark
    benchmark_gpu_performance()
    
    # Run memory analysis
    memory_analysis()
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE!")
    print("=" * 70)
