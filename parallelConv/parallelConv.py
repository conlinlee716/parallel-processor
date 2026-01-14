import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, correlate
import pandas as pd

class ConvolutionParallelProcessor:
    """Parallel convolution processor with decimation decomposition"""
    
    def __init__(self, seq_length, kernel_length, num_channels=2):
        self.N = seq_length
        self.M = kernel_length
        self.K = num_channels
        self.log2K = int(np.log2(num_channels))
        
        if 2**self.log2K != self.K:
            raise ValueError("num_channels must be 2^N")
    
    def decompose_sequence(self, sequence):
        """Decompose sequence into K sub-channels"""
        sub_sequences = []
        for k in range(self.K):
            sub_seq = sequence[k::self.K]
            sub_sequences.append(sub_seq)
        return sub_sequences
    
    def parallel_convolution(self, sequence, kernel):
        """Parallel convolution processing"""
        sub_sequences = self.decompose_sequence(sequence)
        sub_convolutions = []
        
        for sub_seq in sub_sequences:
            sub_conv = convolve(sub_seq, kernel, mode='same')
            sub_convolutions.append(sub_conv)
        
        result = self._reconstruct_convolution(sub_convolutions, len(sequence))
        return result, sub_convolutions
    
    def _reconstruct_convolution(self, sub_convs, original_length):
        """Reconstruct original convolution from K-channel results"""
        reconstructed = np.zeros(original_length)
        samples_per_channel = original_length // self.K
        
        for k in range(original_length):
            channel_idx = k % self.K
            sample_idx = k // self.K
            
            if sample_idx < len(sub_convs[channel_idx]):
                reconstructed[k] = sub_convs[channel_idx][sample_idx]
        
        return reconstructed
    
    def direct_convolution(self, sequence, kernel):
        """Direct convolution without decomposition"""
        return convolve(sequence, kernel, mode='same')


class SignalGenerator:
    """Generate test signals and kernels"""
    
    def __init__(self, fs=40e9):
        self.fs = fs
    
    def generate_signal(self, signal_type, duration=1e-8, length=None):
        """Generate test signal"""
        if length is None:
            length = int(duration * self.fs)
        
        t = np.arange(length) / self.fs
        
        if signal_type == "sinusoid_1GHz":
            signal = np.sin(2 * np.pi * 1e9 * t)
            return signal, "1 GHz Sinusoid"
        
        elif signal_type == "lfm_10_18GHz":
            from scipy.signal import chirp
            signal = chirp(t, 10e9, t[-1], 18e9, method='linear')
            return signal, "10-18 GHz LFM"
        
        elif signal_type == "qpsk_1Gbps":
            symbols = np.random.randint(0, 4, int(1e9 * duration))
            qpsk_map = {0: 1+1j, 1: 1-1j, 2: -1+1j, 3: -1-1j}
            qpsk_symbols = np.array([qpsk_map[s] for s in symbols]) / np.sqrt(2)
            
            samples_per_symbol = length // len(symbols)
            signal = np.repeat(qpsk_symbols, samples_per_symbol)[:length]
            carrier = np.exp(2j * np.pi * 5e9 * t)
            return signal * carrier, "1 Gbps QPSK"
    
    def generate_kernel(self, kernel_type, length=64):
        """Generate convolution kernel"""
        if kernel_type == "lowpass":
            kernel = np.sinc(np.linspace(-2, 2, length))
            kernel = kernel / np.sum(np.abs(kernel))
            return kernel, "Lowpass Filter (Sinc)"
        
        elif kernel_type == "highpass":
            kernel = np.zeros(length)
            kernel[length//2] = 1
            kernel -= np.sinc(np.linspace(-2, 2, length)) / (length/2)
            kernel = kernel / np.sum(np.abs(kernel))
            return kernel, "Highpass Filter"
        
        elif kernel_type == "matched":
            kernel = np.random.randn(length)
            kernel = kernel / np.linalg.norm(kernel)
            return kernel, "Matched Filter"


class ComplexityAnalysis:
    """Convolution complexity and latency analysis"""
    
    @staticmethod
    def compute_complexity_direct(N, M):
        """Direct convolution: O(N*M)"""
        return N * M
    
    @staticmethod
    def compute_complexity_parallel(N, M, K):
        """Parallel convolution: K channels, each (N/K)*M"""
        return K * (N/K) * M
    
    @staticmethod
    def compute_latency_direct(N, M, clock_rate_MHz):
        """Direct convolution latency"""
        operations = N * M
        ops_per_cycle = 2
        cycles = operations / ops_per_cycle
        return cycles / clock_rate_MHz
    
    @staticmethod
    def compute_latency_parallel(N, M, K, clock_rate_MHz):
        """Parallel convolution latency with reconstruction"""
        operations_per_channel = (N/K) * M
        ops_per_cycle = 2
        cycles_per_channel = operations_per_channel / ops_per_cycle
        latency_parallel = cycles_per_channel / clock_rate_MHz
        
        latency_reconstruct = N / (clock_rate_MHz * 1000)
        return latency_parallel + latency_reconstruct


def plot_signal_and_kernel(signal, kernel, signal_name, kernel_name):
    """Plot signal and kernel"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Signal
    ax = axes[0]
    t = np.arange(len(signal))
    ax.plot(t[:min(500, len(signal))], np.real(signal[:min(500, len(signal))]), linewidth=1)
    ax.set_xlabel('Sample Index', fontsize=11)
    ax.set_ylabel('Amplitude', fontsize=11)
    ax.set_title(f'Input Signal: {signal_name}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Kernel
    ax = axes[1]
    ax.plot(kernel, linewidth=1.5, color='orange')
    ax.fill_between(range(len(kernel)), kernel, alpha=0.3, color='orange')
    ax.set_xlabel('Tap Index', fontsize=11)
    ax.set_ylabel('Coefficient', fontsize=11)
    ax.set_title(f'Convolution Kernel: {kernel_name}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'signal_kernel_{signal_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_parallel_convolution_results(sequence, kernel, signal_name, kernel_name, processor):
    """Plot parallel convolution processing results"""
    conv_direct = processor.direct_convolution(sequence, kernel)
    conv_parallel, sub_convs = processor.parallel_convolution(sequence, kernel)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Direct convolution
    ax = axes[0, 0]
    ax.plot(np.real(conv_direct), linewidth=1, label='Direct Conv')
    ax.set_xlabel('Sample Index', fontsize=11)
    ax.set_ylabel('Amplitude', fontsize=11)
    ax.set_title(f'Direct Convolution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Channel 1 convolution
    ax = axes[0, 1]
    ax.plot(np.real(sub_convs[0]), linewidth=1, color='green', label='Channel 1')
    ax.set_xlabel('Sample Index', fontsize=11)
    ax.set_ylabel('Amplitude', fontsize=11)
    ax.set_title(f'Channel 1 Convolution (decimated)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Channel 2 convolution
    ax = axes[1, 0]
    ax.plot(np.real(sub_convs[1]), linewidth=1, color='purple', label='Channel 2')
    ax.set_xlabel('Sample Index', fontsize=11)
    ax.set_ylabel('Amplitude', fontsize=11)
    ax.set_title(f'Channel 2 Convolution (decimated)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Reconstructed convolution
    ax = axes[1, 1]
    ax.plot(np.real(conv_parallel), linewidth=1, color='red', label='Reconstructed')
    ax.plot(np.real(conv_direct), linewidth=1, linestyle='--', alpha=0.7, label='Direct', color='black')
    ax.set_xlabel('Sample Index', fontsize=11)
    ax.set_ylabel('Amplitude', fontsize=11)
    ax.set_title(f'Reconstructed vs Direct Convolution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'parallel_conv_{signal_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Error analysis
    error = np.mean(np.abs(conv_direct - conv_parallel))
    relative_error = error / (np.mean(np.abs(conv_direct)) + 1e-10)
    peak_error = np.max(np.abs(conv_direct - conv_parallel))
    
    print(f"\n{signal_name} + {kernel_name}:")
    print(f"  Mean Error: {error:.2e}")
    print(f"  Relative Error: {relative_error:.2e}")
    print(f"  Peak Error: {peak_error:.2e}")


def plot_complexity_comparison():
    """Plot convolution complexity comparison"""
    seq_lengths = np.array([256, 512, 1024, 2048, 4096, 8192])
    kernel_length = 64
    K = 2
    clock_rate = 1000
    
    complexity_direct = np.array([ComplexityAnalysis.compute_complexity_direct(n, kernel_length) 
                                   for n in seq_lengths])
    complexity_parallel = np.array([ComplexityAnalysis.compute_complexity_parallel(n, kernel_length, K) 
                                    for n in seq_lengths])
    latency_direct = np.array([ComplexityAnalysis.compute_latency_direct(n, kernel_length, clock_rate) 
                               for n in seq_lengths])
    latency_parallel = np.array([ComplexityAnalysis.compute_latency_parallel(n, kernel_length, K, clock_rate) 
                                 for n in seq_lengths])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Complexity comparison
    ax = axes[0, 0]
    ax.semilogy(seq_lengths, complexity_direct, 'o-', label='Direct', linewidth=2, markersize=8)
    ax.semilogy(seq_lengths, complexity_parallel, 's-', label='Parallel (K=2)', linewidth=2, markersize=8)
    ax.set_xlabel('Sequence Length', fontsize=11)
    ax.set_ylabel('Multiplications', fontsize=11)
    ax.set_title('Computational Complexity (Kernel=64)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Complexity improvement
    ax = axes[0, 1]
    improvement = (complexity_direct - complexity_parallel) / complexity_direct * 100
    ax.plot(seq_lengths, improvement, 'o-', color='green', linewidth=2, markersize=8)
    ax.axhline(np.mean(improvement), color='r', linestyle='--', 
               label=f'Mean: {np.mean(improvement):.1f}%')
    ax.set_xlabel('Sequence Length', fontsize=11)
    ax.set_ylabel('Improvement Ratio (%)', fontsize=11)
    ax.set_title('Complexity Reduction (K=2)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Latency comparison
    ax = axes[1, 0]
    ax.semilogy(seq_lengths, latency_direct * 1e6, 'o-', label='Direct', linewidth=2, markersize=8)
    ax.semilogy(seq_lengths, latency_parallel * 1e6, 's-', label='Parallel (K=2)', linewidth=2, markersize=8)
    ax.set_xlabel('Sequence Length', fontsize=11)
    ax.set_ylabel('Latency (μs)', fontsize=11)
    ax.set_title('End-to-End Latency', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Latency improvement
    ax = axes[1, 1]
    latency_improvement = (latency_direct - latency_parallel) / latency_direct * 100
    ax.plot(seq_lengths, latency_improvement, 's-', color='purple', linewidth=2, markersize=8)
    ax.axhline(np.mean(latency_improvement), color='r', linestyle='--', 
               label=f'Mean: {np.mean(latency_improvement):.1f}%')
    ax.set_xlabel('Sequence Length', fontsize=11)
    ax.set_ylabel('Improvement Ratio (%)', fontsize=11)
    ax.set_title('Latency Reduction (K=2)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('conv_complexity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_kernel_length_effect():
    """Plot effect of kernel length on complexity"""
    seq_length = 4096
    kernel_lengths = np.array([16, 32, 64, 128, 256, 512])
    K = 2
    
    complexity_direct = np.array([ComplexityAnalysis.compute_complexity_direct(seq_length, m) 
                                   for m in kernel_lengths])
    complexity_parallel = np.array([ComplexityAnalysis.compute_complexity_parallel(seq_length, m, K) 
                                    for m in kernel_lengths])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Complexity vs kernel length
    ax = axes[0]
    ax.semilogy(kernel_lengths, complexity_direct, 'o-', label='Direct', linewidth=2, markersize=8)
    ax.semilogy(kernel_lengths, complexity_parallel, 's-', label='Parallel (K=2)', linewidth=2, markersize=8)
    ax.set_xlabel('Kernel Length', fontsize=11)
    ax.set_ylabel('Multiplications', fontsize=11)
    ax.set_title(f'Complexity vs Kernel Length (Seq Length={seq_length})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Improvement ratio
    ax = axes[1]
    improvement = (complexity_direct - complexity_parallel) / complexity_direct * 100
    ax.bar(range(len(kernel_lengths)), improvement, color='skyblue', alpha=0.8, edgecolor='black')
    ax.set_xlabel('Kernel Length', fontsize=11)
    ax.set_ylabel('Improvement Ratio (%)', fontsize=11)
    ax.set_title('Complexity Improvement Ratio', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(kernel_lengths)))
    ax.set_xticklabels([str(m) for m in kernel_lengths])
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0, color='black', linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig('kernel_length_effect.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_multicore_convolution_efficiency():
    """Plot multi-core convolution efficiency"""
    seq_length = 4096
    kernel_length = 64
    K_values = np.array([1, 2, 4, 8, 16])
    
    complexity_baseline = ComplexityAnalysis.compute_complexity_direct(seq_length, kernel_length)
    complexity_parallel = np.array([ComplexityAnalysis.compute_complexity_parallel(seq_length, kernel_length, K) 
                                    for K in K_values])
    efficiency = complexity_baseline / complexity_parallel * 100
    per_channel = complexity_parallel / K_values
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Efficiency vs parallelism
    ax = axes[0]
    ax.plot(K_values, efficiency, 'o-', linewidth=2.5, markersize=10, color='darkblue')
    ax.axhline(100, color='r', linestyle='--', alpha=0.5, label='Baseline')
    ax.set_xlabel('Parallelism Level (K)', fontsize=11)
    ax.set_ylabel('Parallel Efficiency (%)', fontsize=11)
    ax.set_title('Convolution Parallel Efficiency', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Complexity distribution
    ax = axes[1]
    width = 0.35
    ax.bar(K_values - width/2, per_channel, width, label='Per-Channel', alpha=0.8)
    ax.bar(K_values + width/2, complexity_parallel, width, label='Total', alpha=0.8)
    ax.set_xlabel('Parallelism Level (K)', fontsize=11)
    ax.set_ylabel('Multiplications', fontsize=11)
    ax.set_title(f'Complexity Distribution (N={seq_length}, M={kernel_length})', 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('conv_multicore_efficiency.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_memory_bandwidth_analysis():
    """Plot memory and bandwidth analysis"""
    seq_lengths = np.array([256, 512, 1024, 2048, 4096, 8192])
    kernel_length = 64
    K = 2
    bytes_per_sample = 8
    
    # Memory footprint
    mem_direct = (seq_lengths + kernel_length) * bytes_per_sample / 1e6
    mem_parallel = (seq_lengths/K + kernel_length) * bytes_per_sample / 1e6
    mem_saving = (mem_direct - mem_parallel) / mem_direct * 100
    
    # Bandwidth (assuming 1GHz access rate)
    bandwidth_direct = seq_lengths * kernel_length / 1e3
    bandwidth_parallel = (seq_lengths/K) * kernel_length / 1e3
    bandwidth_reduction = (bandwidth_direct - bandwidth_parallel) / bandwidth_direct * 100
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Memory comparison
    ax = axes[0, 0]
    ax.plot(seq_lengths, mem_direct, 'o-', label='Direct', linewidth=2, markersize=8)
    ax.plot(seq_lengths, mem_parallel, 's-', label='Parallel (K=2)', linewidth=2, markersize=8)
    ax.set_xlabel('Sequence Length', fontsize=11)
    ax.set_ylabel('Memory (MB)', fontsize=11)
    ax.set_title('Memory Footprint Comparison', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Memory saving
    ax = axes[0, 1]
    ax.bar(range(len(seq_lengths)), mem_saving, color='green', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Sequence Length', fontsize=11)
    ax.set_ylabel('Memory Saving (%)', fontsize=11)
    ax.set_title('Memory Reduction Ratio (K=2)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(seq_lengths)))
    ax.set_xticklabels([str(s) for s in seq_lengths], rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Bandwidth comparison
    ax = axes[1, 0]
    ax.semilogy(seq_lengths, bandwidth_direct, 'o-', label='Direct', linewidth=2, markersize=8)
    ax.semilogy(seq_lengths, bandwidth_parallel, 's-', label='Parallel (K=2)', linewidth=2, markersize=8)
    ax.set_xlabel('Sequence Length', fontsize=11)
    ax.set_ylabel('Data Access (GB/s)', fontsize=11)
    ax.set_title('Memory Bandwidth Requirement', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Bandwidth reduction
    ax = axes[1, 1]
    ax.bar(range(len(seq_lengths)), bandwidth_reduction, color='purple', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Sequence Length', fontsize=11)
    ax.set_ylabel('Bandwidth Reduction (%)', fontsize=11)
    ax.set_title('Bandwidth Reduction Ratio (K=2)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(seq_lengths)))
    ax.set_xticklabels([str(s) for s in seq_lengths], rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('conv_memory_bandwidth.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("=" * 70)
    print("Parallel Convolution Processing Analysis")
    print("=" * 70)
    
    # Setup
    fs = 40e9
    duration = 1e-8
    seq_length = int(duration * fs)
    kernel_length = 64
    
    gen = SignalGenerator(fs)
    processor = ConvolutionParallelProcessor(seq_length, kernel_length, num_channels=2)
    
    # Test signals and kernels
    signals = ["sinusoid_1GHz", "lfm_10_18GHz", "qpsk_1Gbps"]
    kernels = ["lowpass", "highpass", "matched"]
    
    print("\n[1] Signal and Kernel Processing...")
    for signal_type in signals:
        signal, signal_name = gen.generate_signal(signal_type, duration, seq_length)
        kernel, kernel_name = gen.generate_kernel("lowpass", kernel_length)
        
        print(f"\nProcessing: {signal_name} + {kernel_name}")
        plot_signal_and_kernel(signal, kernel, signal_name, kernel_name)
        plot_parallel_convolution_results(signal, kernel, signal_name, kernel_name, processor)
    
    # Complexity analysis
    print("\n[2] Plotting complexity analysis...")
    plot_complexity_comparison()
    
    print("\n[3] Plotting kernel length effect...")
    plot_kernel_length_effect()
    
    print("\n[4] Plotting multi-core efficiency...")
    plot_multicore_convolution_efficiency()
    
    print("\n[5] Plotting memory and bandwidth analysis...")
    plot_memory_bandwidth_analysis()
    
    print("\n" + "=" * 70)
    print("Parallel Convolution Analysis Complete!")
    print("=" * 70)
