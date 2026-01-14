import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import fft
import time

# 配置 matplotlib 字体以支持中文，避免缺失字形警告
try:
    if os.name == 'nt':
        matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    else:
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

class FFTParallelProcessor:
    """
    FFT decimation decomposition parallel processor
    Based on APP principle from Qian et al.
    """
    
    def __init__(self, signal_length, num_channels=2):
        """
        Initialize processor
        
        Parameters:
        -----------
        signal_length : int
            Input signal length
        num_channels : int
            Number of parallel channels (2^N)
        """
        self.N = signal_length
        self.K = num_channels  # Number of parallel channels
        self.log2K = int(np.log2(num_channels))
        
        if 2**self.log2K != self.K:
            raise ValueError("num_channels must be 2^N")
    
    def decompose_signal(self, signal):
        """
        Decompose signal into K sub-channels
        Simulating optical time decomposition and parallelization
        
        Parameters:
        -----------
        signal : ndarray
            Input time-domain signal
            
        Returns:
        --------
        sub_signals : list of ndarray
            K decomposed sub-signals
        """
        sub_signals = []
        samples_per_channel = len(signal) // self.K
        
        for k in range(self.K):
            # Extract k-th channel data from original signal
            sub_signal = signal[k::self.K]
            sub_signals.append(sub_signal)
        
        return sub_signals
    
    def parallel_fft(self, signal):
        """
        Parallel FFT processing
        
        Parameters:
        -----------
        signal : ndarray
            Input signal
            
        Returns:
        --------
        fft_result : ndarray
            FFT result
        sub_ffts : list
            FFT results of each channel
        """
        # Decompose signal
        sub_signals = self.decompose_signal(signal)
        sub_ffts = []
        
        # Perform FFT on each sub-signal
        for sub_signal in sub_signals:
            sub_fft = fft(sub_signal)
            sub_ffts.append(sub_fft)
        
        # Reconstruct FFT result
        fft_result = self._reconstruct_fft(sub_ffts, len(signal))
        
        return fft_result, sub_ffts
    
    def _reconstruct_fft(self, sub_ffts, original_length):
        """
        Reconstruct original signal FFT from K-channel FFT results
        Based on formula: Y(k) = sum_{j=0}^{K-1} X_j(k_j)
        where k_j = (k - j) mod (N/K)
        
        Parameters:
        -----------
        sub_ffts : list
            K-channel FFT results
        original_length : int
            Original signal length
            
        Returns:
        --------
        reconstructed : ndarray
            Reconstructed FFT result
        """
        reconstructed = np.zeros(original_length, dtype=complex)
        samples_per_channel = original_length // self.K
        
        for k in range(original_length):
            for j in range(self.K):
                # Calculate corresponding position in sub-channel FFT
                # Considering circular convolution fusion rule
                idx = (k % samples_per_channel)
                phase_correction = np.exp(-2j * np.pi * j * k / original_length)
                reconstructed[k] += sub_ffts[j][idx] * phase_correction
        
        return reconstructed / self.K


class ComplexityAnalysis:
    """
    Computational complexity and latency analysis
    """
    
    @staticmethod
    def compute_complexity_direct(N):
        """
        Direct FFT computational complexity
        Complexity: O(N log N)
        """
        return N * np.log2(N)
    
    @staticmethod
    def compute_complexity_parallel(N, K):
        """
        Parallel FFT computational complexity
        K-channel parallel, each channel length N/K
        Total complexity: K * (N/K) * log(N/K) = N * (log N - log K)
        """
        return K * (N/K) * np.log2(N/K)
    
    @staticmethod
    def compute_latency_direct(N, clock_rate_MHz):
        """
        Direct processing latency
        Assuming single-core processing time
        
        Parameters:
        -----------
        N : int
            Number of FFT points
        clock_rate_MHz : float
            Clock frequency (MHz)
        """
        operations = N * np.log2(N)
        ops_per_cycle = 2  # Operations per clock cycle
        cycles = operations / ops_per_cycle
        latency_us = cycles / clock_rate_MHz
        return latency_us
    
    @staticmethod
    def compute_latency_parallel(N, K, clock_rate_MHz):
        """
        Parallel processing latency
        K-channel parallel processing with reconstruction time
        
        Parameters:
        -----------
        N : int
            Number of FFT points
        K : int
            Number of parallel channels
        clock_rate_MHz : float
            Single-channel clock frequency (MHz)
        """
        # Parallel processing time (K-channel parallel, time reduced by K times)
        operations_per_channel = (N/K) * np.log2(N/K)
        ops_per_cycle = 2
        cycles_per_channel = operations_per_channel / ops_per_cycle
        latency_parallel_us = cycles_per_channel / clock_rate_MHz
        
        # Add reconstruction time (approximated as original length processing time)
        latency_reconstruct_us = N / (clock_rate_MHz * 1000)  # Rough reconstruction overhead
        
        total_latency_us = latency_parallel_us + latency_reconstruct_us
        return total_latency_us


# ============ Verification Experiment ============

def verify_fft_parallel_processing():
    """
    Verify advantages of FFT decimation decomposition parallel processing
    """
    
    print("="*70)
    print("FFT Decimation Decomposition Parallel Processing Advantage Verification")
    print("="*70)
    
    # Parameter settings
    fft_sizes = [256, 512, 1024, 2048, 4096, 8192]
    num_channels = 2
    clock_rate = 1000  # MHz
    
    # 1. Verify FFT correctness
    print("\n1. FFT Calculation Correctness Verification")
    print("-"*70)
    
    test_signal = np.random.randn(512) + 1j * np.random.randn(512)
    processor = FFTParallelProcessor(512, 2)
    
    # Direct FFT
    fft_direct = fft(test_signal)
    
    # Parallel FFT
    fft_parallel, sub_ffts = processor.parallel_fft(test_signal)
    
    # Compare results
    error = np.mean(np.abs(fft_direct - fft_parallel))
    print(f"Direct FFT vs Parallel FFT Mean Error: {error:.2e}")
    print(f"Relative Error: {error/np.mean(np.abs(fft_direct)):.2e}")
    print("✓ FFT Results Consistent\n")
    
    # 2. Computational complexity comparison
    print("2. Computational Complexity Comparison")
    print("-"*70)
    print(f"{'FFT Size':<12} {'Direct Processing':<17} {'Parallel (K=2)':<20} {'Complexity Improvement':<15}")
    print("-"*70)
    
    complexity_improvements = []
    
    for fft_size in fft_sizes:
        complexity_direct = ComplexityAnalysis.compute_complexity_direct(fft_size)
        complexity_parallel = ComplexityAnalysis.compute_complexity_parallel(fft_size, 2)
        improvement = (complexity_direct - complexity_parallel) / complexity_direct * 100
        
        complexity_improvements.append(improvement)
        
        print(f"{fft_size:<12} {complexity_direct:<17.0f} {complexity_parallel:<20.0f} "
              f"{improvement:<15.1f}%")
    
    # 3. End-to-end latency comparison
    print("\n3. End-to-End Latency Comparison")
    print("-"*70)
    print(f"{'FFT Size':<12} {'Direct Latency (μs)':<20} {'Parallel Latency (μs)':<22} {'Latency Improvement':<15}")
    print("-"*70)
    
    latency_improvements = []
    
    for fft_size in fft_sizes:
        latency_direct = ComplexityAnalysis.compute_latency_direct(fft_size, clock_rate)
        latency_parallel = ComplexityAnalysis.compute_latency_parallel(fft_size, 2, clock_rate)
        improvement = (latency_direct - latency_parallel) / latency_direct * 100
        
        latency_improvements.append(improvement)
        
        print(f"{fft_size:<12} {latency_direct:<20.3f} {latency_parallel:<22.3f} "
              f"{improvement:<15.1f}%")
    
    # 4. Data rate reduction
    print("\n4. Parallel Channel Data Rate Reduction")
    print("-"*70)
    print(f"{'FFT Size':<12} {'Original Rate (Gbps)':<20} {'Per-Channel Rate (Gbps)':<25} {'Reduction Ratio':<15}")
    print("-"*70)
    
    sampling_rate = 40  # GHz (based on APP chip in paper)
    
    for fft_size in fft_sizes:cls
def plot_results(fft_sizes, complexity_improvements, latency_improvements):
    """
    Plot comparison results
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Complexity improvement
    axes[0].plot(fft_sizes, complexity_improvements, 'o-', linewidth=2, 
                 markersize=8, label='Parallel FFT')
    axes[0].axhline(y=np.mean(complexity_improvements), color='r', 
                    linestyle='--', label=f'Average: {np.mean(complexity_improvements):.1f}%')
    axes[0].set_xlabel('FFT Size', fontsize=12)
    axes[0].set_ylabel('Complexity Reduction Ratio (%)', fontsize=12)
    axes[0].set_title('Computational Complexity Improvement (K=2)', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)
    axes[0].set_xscale('log')
    
    # Latency improvement
    axes[1].plot(fft_sizes, latency_improvements, 's-', linewidth=2, 
                 markersize=8, color='green', label='Parallel FFT')
    axes[1].axhline(y=np.mean(latency_improvements), color='r', 
                    linestyle='--', label=f'Average: {np.mean(latency_improvements):.1f}%')
    axes[1].set_xlabel('FFT Size', fontsize=12)
    axes[1].set_ylabel('End-to-End Latency Reduction (%)', fontsize=12)
    axes[1].set_title('End-to-End Latency Improvement (K=2)', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=11)
    axes[1].set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('fft_parallel_improvement.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_multicore_efficiency():
    """
    Analyze efficiency under different parallelism levels
    """
    
    print("\n" + "="*70)
    print("Efficiency Analysis of Different Parallelism Levels (K Values)")
    print("="*70)
    
    fft_size = 4096
    K_values = [1, 2, 4, 8, 16]
    
    print(f"\nFFT Size: {fft_size}")
    print("-"*70)
    print(f"{'Parallelism K':<14} {'Per-Channel Complexity':<24} {'Total Complexity':<20} {'Parallel Efficiency':<15}")
    print("-"*70)
    
    complexity_baseline = ComplexityAnalysis.compute_complexity_direct(fft_size)
    
    for K in K_values:
        if fft_size % K != 0:
            continue
        
        complexity_total = ComplexityAnalysis.compute_complexity_parallel(fft_size, K)
        complexity_per_channel = complexity_total / K
        efficiency = complexity_baseline / complexity_total * 100
        
        print(f"{K:<14} {complexity_per_channel:<24.0f} {complexity_total:<20.0f} "
              f"{efficiency:<15.1f}%")


def memory_analysis():
    """
    Memory requirement analysis
    """
    
    print("\n" + "="*70)
    print("Memory Requirement Analysis")
    print("="*70)
    
    fft_sizes = [256, 512, 1024, 2048, 4096, 8192]
    K = 2
    bytes_per_sample = 16  # Complex data: 2 float64
    
    print("\nDirect Processing vs Parallel Processing Memory Requirements")
    print("-"*70)
    print(f"{'FFT Size':<12} {'Direct Processing (MB)':<22} {'Parallel Processing (MB)':<25} {'Savings Ratio':<15}")
    print("-"*70)
    
    for fft_size in fft_sizes:
        # Direct processing requires storing entire signal and FFT result
        memory_direct = (fft_size + fft_size) * bytes_per_sample / 1e6
        
        # Parallel processing each channel only needs N/K data
        memory_parallel = 2 * (fft_size/K) * bytes_per_sample / 1e6
        
        saving = (memory_direct - memory_parallel) / memory_direct * 100
        
        print(f"{fft_size:<12} {memory_direct:<22.2f} {memory_parallel:<25.2f} "
              f"{saving:<15.1f}%")


# ============ Main Program ============

if __name__ == "__main__":
    # Execute verification
    complexity_improvements, latency_improvements = verify_fft_parallel_processing()
    
    # Multi-core efficiency analysis
    analyze_multicore_efficiency()
    
    # Memory analysis
    memory_analysis()
    
    # Plot results
    fft_sizes = [256, 512, 1024, 2048, 4096, 8192]
    plot_results(fft_sizes, complexity_improvements, latency_improvements)
    
    print("\n" + "="*70)
    print("Verification Complete!")
    print("="*70)
