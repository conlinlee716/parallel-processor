import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, chirp
from scipy.fft import fft, ifft
import warnings
warnings.filterwarnings('ignore')

class PulseCompressionParallelProcessor:
    """Parallel pulse compression using FFT-based approach"""
    
    def __init__(self, pulse_length, waveform_length, num_channels=2):
        self.N = pulse_length
        self.L = waveform_length
        self.K = num_channels
    
    def direct_pulse_compression(self, signal, ref_waveform):
        """Direct pulse compression using FFT"""
        # Pad to same length for FFT
        pad_length = len(signal) + len(ref_waveform) - 1
        sig_padded = np.pad(signal, (0, len(ref_waveform) - 1), mode='constant')
        ref_padded = np.pad(ref_waveform, (0, len(signal) - 1), mode='constant')
        
        # FFT-based correlation
        sig_fft = fft(sig_padded, n=pad_length)
        ref_fft = fft(np.conj(ref_padded[::-1]), n=pad_length)
        compressed = ifft(sig_fft * ref_fft)
        
        return np.abs(compressed[:len(signal)])
    
    def parallel_pulse_compression(self, signal, ref_waveform):
        """Parallel pulse compression with correct reconstruction"""
        # Step 1: Decompose signal into K channels
        sub_signals = [signal[k::self.K] for k in range(self.K)]
        N_sub = len(sub_signals[0])
        L_ref = len(ref_waveform)
        
        # Step 2: Parallel FFT on each sub-signal
        pad_length = N_sub + L_ref - 1
        sub_ffts = []
        for sub_sig in sub_signals:
            sub_padded = np.pad(sub_sig, (0, L_ref - 1), mode='constant')
            sub_fft = fft(sub_padded, n=pad_length)
            sub_ffts.append(sub_fft)
        
        # Step 3: Reference waveform FFT (shared)
        ref_padded = np.pad(ref_waveform, (0, N_sub - 1), mode='constant')
        ref_fft = fft(np.conj(ref_padded[::-1]), n=pad_length)
        
        # Step 4: Parallel frequency domain multiplication
        sub_products = [sub_ffts[k] * ref_fft for k in range(self.K)]
        
        # Step 5: Parallel IFFT
        sub_compressed = [ifft(sub_products[k])[:N_sub] for k in range(self.K)]
        
        # Step 6: Reconstruct by interleaving sub-channel results
        reconstructed = np.zeros(len(signal), dtype=complex)
        for k in range(len(signal)):
            channel_idx = k % self.K
            sample_idx = k // self.K
            if sample_idx < len(sub_compressed[channel_idx]):
                reconstructed[k] = sub_compressed[channel_idx][sample_idx]
        
        return np.abs(reconstructed), sub_compressed


class SignalGenerator:
    """Generate LFM signals"""
    
    def __init__(self, fs=40e9):
        self.fs = fs
    
    def generate_lfm_waveform(self, f0=10e9, f1=18e9, duration=1e-8):
        """Generate reference LFM waveform"""
        t = np.arange(int(duration * self.fs)) / self.fs
        waveform = chirp(t, f0, t[-1], f1, method='linear')
        return waveform, t
    
    def generate_received_signal(self, waveform, delay=1e-9, snr_db=20, num_targets=1):
        """Generate received signal with targets"""
        length = int(2 * delay * self.fs + len(waveform))
        signal = np.zeros(length, dtype=complex)
        
        for idx in range(num_targets):
            target_delay = delay + idx * 2e-9
            target_idx = int(target_delay * self.fs)
            
            if target_idx + len(waveform) <= length:
                amplitude = 1.0 / (1 + 0.3 * idx)
                signal[target_idx:target_idx + len(waveform)] += amplitude * waveform
        
        snr_linear = 10**(snr_db / 10)
        noise_power = np.mean(np.abs(signal)**2) / snr_linear
        noise = np.sqrt(noise_power/2) * (np.random.randn(length) + 1j * np.random.randn(length))
        signal += noise
        
        return signal, np.arange(length) / self.fs


class ComplexityAnalysis:
    """Complexity analysis"""
    
    @staticmethod
    def compute_complexity_direct(N, L):
        """FFT-based: O((N+L)*log(N+L))"""
        return (N + L) * np.log2(N + L)
    
    @staticmethod
    def compute_complexity_parallel(N, L, K):
        """Parallel: K*((N/K+L)*log(N/K+L)) + L*log(N+L)"""
        return K * ((N/K + L) * np.log2(N/K + L)) + L * np.log2(N + L)
    
    @staticmethod
    def compute_latency_direct(N, L, clock_rate_MHz):
        ops = (N + L) * np.log2(N + L)
        return (ops / 2) / clock_rate_MHz
    
    @staticmethod
    def compute_latency_parallel(N, L, K, clock_rate_MHz):
        ops = K * ((N/K + L) * np.log2(N/K + L)) + L * np.log2(N + L)
        return (ops / 2) / clock_rate_MHz


def plot_results(signal, ref_waveform, processor, fs):
    """Plot comprehensive results"""
    # Compute compressions
    compressed_direct = processor.direct_pulse_compression(signal, ref_waveform)
    compressed_parallel, sub_compressed = processor.parallel_pulse_compression(signal, ref_waveform)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Direct compression
    ax = axes[0, 0]
    ax.plot(np.abs(compressed_direct), linewidth=1.5, color='blue')
    peaks_direct = np.argsort(compressed_direct)[-3:]
    ax.scatter(peaks_direct, compressed_direct[peaks_direct], color='r', s=100, marker='*', zorder=5)
    ax.set_xlabel('Sample Index', fontsize=11)
    ax.set_ylabel('Magnitude', fontsize=11)
    ax.set_title('Direct Pulse Compression (FFT)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Channel 1
    ax = axes[0, 1]
    ax.plot(np.abs(sub_compressed[0]), linewidth=1.5, color='green')
    ax.set_xlabel('Sample Index (Ch1)', fontsize=11)
    ax.set_ylabel('Magnitude', fontsize=11)
    ax.set_title('Channel 1 Result (decimated)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Channel 2
    ax = axes[1, 0]
    ax.plot(np.abs(sub_compressed[1]), linewidth=1.5, color='purple')
    ax.set_xlabel('Sample Index (Ch2)', fontsize=11)
    ax.set_ylabel('Magnitude', fontsize=11)
    ax.set_title('Channel 2 Result (decimated)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Comparison
    ax = axes[1, 1]
    ax.plot(np.abs(compressed_direct), linewidth=1.5, linestyle='--', 
            alpha=0.7, color='blue', label='Direct')
    ax.plot(np.abs(compressed_parallel), linewidth=1.5, color='red', label='Parallel')
    peaks_parallel = np.argsort(np.abs(compressed_parallel))[-3:]
    ax.scatter(peaks_parallel, np.abs(compressed_parallel)[peaks_parallel], 
               color='orange', s=100, marker='*', zorder=5)
    ax.set_xlabel('Sample Index', fontsize=11)
    ax.set_ylabel('Magnitude', fontsize=11)
    ax.set_title('Reconstructed vs Direct', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pulse_compression_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Error analysis
    error = np.mean(np.abs(compressed_direct - np.abs(compressed_parallel)))
    relative_error = error / (np.mean(np.abs(compressed_direct)) + 1e-10)
    peak_error = np.max(np.abs(compressed_direct - np.abs(compressed_parallel)))
    
    print(f"\nPulse Compression Analysis:")
    print(f"  Mean Error: {error:.2e}")
    print(f"  Relative Error: {relative_error:.2e}")
    print(f"  Peak Error: {peak_error:.2e}")
    print(f"  Direct Peak Magnitude: {np.max(compressed_direct):.4f}")
    print(f"  Parallel Peak Magnitude: {np.max(np.abs(compressed_parallel)):.4f}")
    print(f"  Peak Alignment Error: {np.abs(np.max(compressed_direct) - np.max(np.abs(compressed_parallel))):.2e}")


def plot_complexity():
    """Plot complexity analysis"""
    pulse_lengths = np.array([512, 1024, 2048, 4096, 8192, 16384])
    waveform_length = 256
    K = 2
    clock_rate = 1000
    
    complexity_direct = np.array([ComplexityAnalysis.compute_complexity_direct(n, waveform_length) 
                                   for n in pulse_lengths])
    complexity_parallel = np.array([ComplexityAnalysis.compute_complexity_parallel(n, waveform_length, K) 
                                    for n in pulse_lengths])
    latency_direct = np.array([ComplexityAnalysis.compute_latency_direct(n, waveform_length, clock_rate) 
                               for n in pulse_lengths])
    latency_parallel = np.array([ComplexityAnalysis.compute_latency_parallel(n, waveform_length, K, clock_rate) 
                                 for n in pulse_lengths])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Complexity
    ax = axes[0, 0]
    ax.semilogy(pulse_lengths, complexity_direct, 'o-', label='Direct', linewidth=2, markersize=8)
    ax.semilogy(pulse_lengths, complexity_parallel, 's-', label='Parallel (K=2)', linewidth=2, markersize=8)
    ax.set_xlabel('Pulse Length', fontsize=11)
    ax.set_ylabel('Operations', fontsize=11)
    ax.set_title('Computational Complexity (L=256)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Complexity improvement
    ax = axes[0, 1]
    improvement = (complexity_direct - complexity_parallel) / complexity_direct * 100
    ax.plot(pulse_lengths, improvement, 'o-', color='green', linewidth=2, markersize=8)
    ax.axhline(np.mean(improvement), color='r', linestyle='--', 
               label=f'Mean: {np.mean(improvement):.1f}%')
    ax.set_xlabel('Pulse Length', fontsize=11)
    ax.set_ylabel('Improvement (%)', fontsize=11)
    ax.set_title('Complexity Reduction (K=2)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Latency
    ax = axes[1, 0]
    ax.semilogy(pulse_lengths, latency_direct * 1e6, 'o-', label='Direct', linewidth=2, markersize=8)
    ax.semilogy(pulse_lengths, latency_parallel * 1e6, 's-', label='Parallel (K=2)', linewidth=2, markersize=8)
    ax.set_xlabel('Pulse Length', fontsize=11)
    ax.set_ylabel('Latency (μs)', fontsize=11)
    ax.set_title('End-to-End Latency', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Latency improvement
    ax = axes[1, 1]
    latency_improvement = (latency_direct - latency_parallel) / latency_direct * 100
    ax.plot(pulse_lengths, latency_improvement, 's-', color='purple', linewidth=2, markersize=8)
    ax.axhline(np.mean(latency_improvement), color='r', linestyle='--', 
               label=f'Mean: {np.mean(latency_improvement):.1f}%')
    ax.set_xlabel('Pulse Length', fontsize=11)
    ax.set_ylabel('Improvement (%)', fontsize=11)
    ax.set_title('Latency Reduction (K=2)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('compression_complexity.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_waveform_effect():
    """Plot waveform length effect"""
    pulse_length = 8192
    waveform_lengths = np.array([32, 64, 128, 256, 512, 1024])
    K = 2
    
    complexity_direct = np.array([ComplexityAnalysis.compute_complexity_direct(pulse_length, w) 
                                   for w in waveform_lengths])
    complexity_parallel = np.array([ComplexityAnalysis.compute_complexity_parallel(pulse_length, w, K) 
                                    for w in waveform_lengths])
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    ax.semilogy(waveform_lengths, complexity_direct, 'o-', label='Direct', linewidth=2, markersize=8)
    ax.semilogy(waveform_lengths, complexity_parallel, 's-', label='Parallel (K=2)', linewidth=2, markersize=8)
    ax.set_xlabel('Waveform Length', fontsize=11)
    ax.set_ylabel('Operations', fontsize=11)
    ax.set_title(f'Complexity vs Waveform Length (N={pulse_length})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('waveform_effect.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_multicore():
    """Plot multi-core efficiency"""
    pulse_length = 8192
    waveform_length = 256
    K_values = np.array([1, 2, 4, 8])
    
    complexity_baseline = ComplexityAnalysis.compute_complexity_direct(pulse_length, waveform_length)
    complexity_parallel = np.array([ComplexityAnalysis.compute_complexity_parallel(pulse_length, waveform_length, K) 
                                    for K in K_values])
    efficiency = complexity_baseline / complexity_parallel * 100
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    ax.plot(K_values, efficiency, 'o-', linewidth=2.5, markersize=10, color='darkblue')
    ax.axhline(100, color='r', linestyle='--', alpha=0.5, label='Baseline')
    ax.set_xlabel('Parallelism Level (K)', fontsize=11)
    ax.set_ylabel('Parallel Efficiency (%)', fontsize=11)
    ax.set_title('Multi-Core Efficiency', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('multicore_efficiency.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("=" * 70)
    print("Parallel Pulse Compression (FFT-based)")
    print("=" * 70)
    
    # Setup
    fs = 40e9
    duration = 1e-8
    
    gen = SignalGenerator(fs)
    
    # Generate signals
    print("\n[1] Generating LFM Waveform...")
    ref_waveform, t_ref = gen.generate_lfm_waveform(10e9, 18e9, duration)
    
    print("[2] Generating Received Signal with Targets...")
    received_signal, t_received = gen.generate_received_signal(ref_waveform, delay=1e-9, 
                                                                snr_db=20, num_targets=3)
    
    # Create processor with actual signal/waveform lengths
    processor = PulseCompressionParallelProcessor(len(received_signal), len(ref_waveform), num_channels=2)
    
    print("[3] Parallel Pulse Compression...")
    plot_results(received_signal, ref_waveform, processor, fs)
    
    print("[4] Complexity Analysis...")
    plot_complexity()
    plot_waveform_effect()
    plot_multicore()
    
    print("\n" + "=" * 70)
    print("Analysis Complete - 4 figures generated!")
    print("=" * 70)
