import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import chirp
import pandas as pd

# Global plotting configuration: Arial font, fontsize 20
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['axes.unicode_minus'] = False

class FFTParallelProcessor:
    """FFT decimation decomposition parallel processor"""
    
    def __init__(self, signal_length, num_channels=2, fs=40e9):
        self.N = signal_length
        self.K = num_channels
        self.fs = fs
        self.log2K = int(np.log2(num_channels))
        
        if 2**self.log2K != self.K:
            raise ValueError("num_channels must be 2^N")
    
    def decompose_signal(self, signal):
        """Decompose signal into K sub-channels"""
        sub_signals = []
        for k in range(self.K):
            sub_signal = signal[k::self.K]
            sub_signals.append(sub_signal)
        return sub_signals
    
    def parallel_fft(self, signal):
        """Parallel FFT processing"""
        sub_signals = self.decompose_signal(signal)
        sub_ffts = [fft(sub) for sub in sub_signals]
        fft_result = self._reconstruct_fft(sub_ffts, len(signal))
        return fft_result, sub_ffts
    
    def _reconstruct_fft(self, sub_ffts, original_length):
        """Reconstruct original signal FFT from K-channel FFT results"""
        reconstructed = np.zeros(original_length, dtype=complex)
        samples_per_channel = original_length // self.K
        
        for k in range(original_length):
            for j in range(self.K):
                idx = (k % samples_per_channel)
                phase_correction = np.exp(-2j * np.pi * j * k / original_length)
                reconstructed[k] += sub_ffts[j][idx] * phase_correction
        
        return reconstructed / self.K


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


class ComplexityAnalysis:
    """Computational complexity and latency analysis"""
    
    @staticmethod
    def compute_complexity_direct(N):
        return N * np.log2(N)
    
    @staticmethod
    def compute_complexity_parallel(N, K):
        return K * (N/K) * np.log2(N/K)
    
    @staticmethod
    def compute_latency_direct(N, clock_rate_MHz):
        operations = N * np.log2(N)
        cycles = operations / 2
        return cycles / clock_rate_MHz
    
    @staticmethod
    def compute_latency_parallel(N, K, clock_rate_MHz):
        operations_per_channel = (N/K) * np.log2(N/K)
        cycles_per_channel = operations_per_channel / 2
        latency_parallel = cycles_per_channel / clock_rate_MHz
        latency_reconstruct = N / (clock_rate_MHz * 1000)
        return latency_parallel + latency_reconstruct


def plot_complexity_analysis():
    """Plot computational complexity comparison"""
    fft_sizes = np.array([256, 512, 1024, 2048, 4096, 8192])
    K = 2
    clock_rate = 1000
    
    complexity_direct = np.array([ComplexityAnalysis.compute_complexity_direct(n) for n in fft_sizes])
    complexity_parallel = np.array([ComplexityAnalysis.compute_complexity_parallel(n, K) for n in fft_sizes])
    latency_direct = np.array([ComplexityAnalysis.compute_latency_direct(n, clock_rate) for n in fft_sizes])
    latency_parallel = np.array([ComplexityAnalysis.compute_latency_parallel(n, K, clock_rate) for n in fft_sizes])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Complexity comparison
    ax = axes[0, 0]
    ax.semilogy(fft_sizes, complexity_direct, 'o-', label='Direct', linewidth=2, markersize=8)
    ax.semilogy(fft_sizes, complexity_parallel, 's-', label='Parallel (K=2)', linewidth=2, markersize=8)
    ax.set_xlabel('FFT Size', fontsize=20)
    ax.set_ylabel('Operations', fontsize=20)
    ax.set_title('Computational Complexity', fontsize=20, fontweight='bold')
    ax.legend(fontsize=20)
    ax.grid(True, alpha=0.3)
    
    # Complexity improvement ratio
    ax = axes[0, 1]
    improvement = (complexity_direct - complexity_parallel) / complexity_direct * 100
    ax.plot(fft_sizes, improvement, 'o-', color='green', linewidth=2, markersize=8)
    ax.axhline(np.mean(improvement), color='r', linestyle='--', label=f'Mean: {np.mean(improvement):.1f}%')
    ax.set_xlabel('FFT Size', fontsize=20)
    ax.set_ylabel('Improvement Ratio (%)', fontsize=20)
    ax.set_title('Complexity Reduction', fontsize=20, fontweight='bold')
    ax.legend(fontsize=20)
    ax.grid(True, alpha=0.3)
    
    # Latency comparison
    ax = axes[1, 0]
    ax.semilogy(fft_sizes, latency_direct * 1e6, 'o-', label='Direct', linewidth=2, markersize=8)
    ax.semilogy(fft_sizes, latency_parallel * 1e6, 's-', label='Parallel (K=2)', linewidth=2, markersize=8)
    ax.set_xlabel('FFT Size', fontsize=20)
    ax.set_ylabel('Latency (μs)', fontsize=20)
    ax.set_title('End-to-End Latency', fontsize=20, fontweight='bold')
    ax.legend(fontsize=20)
    ax.grid(True, alpha=0.3)
    
    # Latency improvement ratio
    ax = axes[1, 1]
    latency_improvement = (latency_direct - latency_parallel) / latency_direct * 100
    ax.plot(fft_sizes, latency_improvement, 's-', color='purple', linewidth=2, markersize=8)
    ax.axhline(np.mean(latency_improvement), color='r', linestyle='--', label=f'Mean: {np.mean(latency_improvement):.1f}%')
    ax.set_xlabel('FFT Size', fontsize=20)
    ax.set_ylabel('Improvement Ratio (%)', fontsize=20)
    ax.set_title('Latency Reduction', fontsize=20, fontweight='bold')
    ax.legend(fontsize=20)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('complexity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_signal_spectrum(signal, name, fs=40e9):
    """Plot signal and its FFT spectrum"""
    fft_result = fft(signal)
    freq = fftfreq(len(signal), 1/fs)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Time domain
    ax = axes[0]
    t = np.arange(len(signal)) / fs * 1e9
    ax.plot(t[:min(500, len(signal))], np.real(signal[:min(500, len(signal))]))
    ax.set_xlabel('Time (ns)', fontsize=20)
    ax.set_ylabel('Amplitude', fontsize=20)
    ax.set_title(f'{name} - Time Domain', fontsize=20, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Frequency domain
    ax = axes[1]
    ax.plot(freq[:len(freq)//2] / 1e9, 20 * np.log10(np.abs(fft_result[:len(fft_result)//2]) + 1e-10))
    ax.set_xlabel('Frequency (GHz)', fontsize=20)
    ax.set_ylabel('Magnitude (dB)', fontsize=20)
    ax.set_title(f'{name} - Frequency Domain', fontsize=20, fontweight='bold')
    ax.set_xlim([0, 20])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'signal_{name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_parallel_processing(signal, signal_name, processor):
    """Plot parallel FFT processing results"""
    fft_direct = fft(signal)
    fft_parallel, sub_ffts = processor.parallel_fft(signal)
    
    freq = fftfreq(len(signal), 1/processor.fs)
    freq_sub = fftfreq(len(sub_ffts[0]), 2/processor.fs)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Direct FFT
    ax = axes[0, 0]
    ax.semilogy(freq[:len(freq)//2] / 1e9, np.abs(fft_direct[:len(fft_direct)//2]) + 1e-10)
    ax.set_xlabel('Frequency (GHz)', fontsize=20)
    ax.set_ylabel('Magnitude (log)', fontsize=20)
    ax.set_title(f'{signal_name} - Direct FFT', fontsize=20, fontweight='bold')
    ax.set_xlim([0, 20])
    ax.grid(True, alpha=0.3)
    
    # Channel 1 FFT
    ax = axes[0, 1]
    ax.semilogy(freq_sub[:len(freq_sub)//2] / 1e9, np.abs(sub_ffts[0][:len(sub_ffts[0])//2]) + 1e-10)
    ax.set_xlabel('Frequency (GHz)', fontsize=20)
    ax.set_ylabel('Magnitude (log)', fontsize=20)
    ax.set_title(f'Channel 1 FFT', fontsize=20, fontweight='bold')
    ax.set_xlim([0, 20])
    ax.grid(True, alpha=0.3)
    
    # Channel 2 FFT
    ax = axes[1, 0]
    ax.semilogy(freq_sub[:len(freq_sub)//2] / 1e9, np.abs(sub_ffts[1][:len(sub_ffts[1])//2]) + 1e-10)
    ax.set_xlabel('Frequency (GHz)', fontsize=20)
    ax.set_ylabel('Magnitude (log)', fontsize=20)
    ax.set_title(f'Channel 2 FFT', fontsize=20, fontweight='bold')
    ax.set_xlim([0, 20])
    ax.grid(True, alpha=0.3)
    
    # Reconstructed FFT
    ax = axes[1, 1]
    ax.semilogy(freq[:len(freq)//2] / 1e9, np.abs(fft_parallel[:len(fft_parallel)//2]) + 1e-10)
    ax.set_xlabel('Frequency (GHz)', fontsize=20)
    ax.set_ylabel('Magnitude (log)', fontsize=20)
    ax.set_title(f'{signal_name} - Reconstructed FFT', fontsize=20, fontweight='bold')
    ax.set_xlim([0, 20])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'parallel_processing_{signal_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Error analysis
    error = np.mean(np.abs(fft_direct - fft_parallel))
    relative_error = error / np.mean(np.abs(fft_direct))
    print(f"\n{signal_name}:")
    print(f"  Mean Error: {error:.2e}")
    print(f"  Relative Error: {relative_error:.2e}")


def plot_multicore_efficiency():
    """Plot multi-core efficiency analysis"""
    fft_size = 4096
    K_values = np.array([1, 2, 4, 8, 16])
    
    complexity_baseline = ComplexityAnalysis.compute_complexity_direct(fft_size)
    complexity_parallel = np.array([ComplexityAnalysis.compute_complexity_parallel(fft_size, K) for K in K_values])
    efficiency = complexity_baseline / complexity_parallel * 100
    per_channel = complexity_parallel / K_values
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Efficiency vs Parallelism
    ax = axes[0]
    ax.plot(K_values, efficiency, 'o-', linewidth=2, markersize=10, color='darkblue')
    ax.axhline(100, color='r', linestyle='--', alpha=0.5, label='Baseline')
    ax.set_xlabel('Parallelism Level (K)', fontsize=11)
    ax.set_ylabel('Parallel Efficiency (%)', fontsize=11)
    ax.set_title('Multi-Core Efficiency Analysis', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Per-channel complexity
    ax = axes[1]
    width = 0.35
    ax.bar(K_values - width/2, per_channel, width, label='Per-Channel Complexity', alpha=0.8)
    ax.bar(K_values + width/2, complexity_parallel, width, label='Total Complexity', alpha=0.8)
    ax.set_xlabel('Parallelism Level (K)', fontsize=11)
    ax.set_ylabel('Operations', fontsize=11)
    ax.set_title('Complexity Distribution (FFT Size=4096)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('multicore_efficiency.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_memory_analysis():
    """Plot memory requirement analysis"""
    fft_sizes = np.array([256, 512, 1024, 2048, 4096, 8192])
    K = 2
    bytes_per_sample = 16
    
    memory_direct = (fft_sizes + fft_sizes) * bytes_per_sample / 1e6
    memory_parallel = 2 * (fft_sizes/K) * bytes_per_sample / 1e6
    saving = (memory_direct - memory_parallel) / memory_direct * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Memory comparison
    ax = axes[0]
    ax.plot(fft_sizes, memory_direct, 'o-', label='Direct Processing', linewidth=2, markersize=8)
    ax.plot(fft_sizes, memory_parallel, 's-', label='Parallel Processing (K=2)', linewidth=2, markersize=8)
    ax.set_xlabel('FFT Size', fontsize=11)
    ax.set_ylabel('Memory (MB)', fontsize=11)
    ax.set_title('Memory Requirement Comparison', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Saving ratio
    ax = axes[1]
    ax.bar(range(len(fft_sizes)), saving, color='green', alpha=0.7)
    ax.set_xlabel('FFT Size', fontsize=11)
    ax.set_ylabel('Memory Savings (%)', fontsize=11)
    ax.set_title('Memory Savings Ratio (K=2)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(fft_sizes)))
    ax.set_xticklabels([str(s) for s in fft_sizes])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('memory_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("=" * 70)
    print("FFT Parallel Processing Analysis")
    print("=" * 70)
    
    # Generate signals
    fs = 40e9
    duration = 1e-8
    gen = SignalGenerator(fs, duration)
    processor = FFTParallelProcessor(gen.N, num_channels=2, fs=fs)
    
    # Signal 1: 1 GHz Sinusoid
    print("\n[1] Processing 1 GHz Sinusoid...")
    signal_1ghz, name_1ghz = gen.sinusoid_1GHz()
    plot_signal_spectrum(signal_1ghz, name_1ghz, fs)
    plot_parallel_processing(signal_1ghz, name_1ghz, processor)
    
    # Signal 2: 10-18 GHz LFM
    print("\n[2] Processing 10-18 GHz LFM...")
    signal_lfm, name_lfm = gen.lfm_10_18GHz()
    plot_signal_spectrum(signal_lfm, name_lfm, fs)
    plot_parallel_processing(signal_lfm, name_lfm, processor)
    
    # Signal 3: 1 Gbps QPSK
    print("\n[3] Processing 1 Gbps QPSK...")
    signal_qpsk, name_qpsk = gen.qpsk_1Gbps()
    plot_signal_spectrum(signal_qpsk, name_qpsk, fs)
    plot_parallel_processing(signal_qpsk, name_qpsk, processor)
    
    # Analysis plots
    print("\n[4] Plotting complexity analysis...")
    plot_complexity_analysis()
    
    print("\n[5] Plotting multi-core efficiency...")
    plot_multicore_efficiency()
    
    print("\n[6] Plotting memory analysis...")
    plot_memory_analysis()
    
    print("\n" + "=" * 70)
    print("Analysis Complete! All figures saved.")
    print("=" * 70)