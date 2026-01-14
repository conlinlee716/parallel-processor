import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.linalg import eigh
import warnings
warnings.filterwarnings('ignore')

class BeamformingParallelProcessor:
    """Parallel beamforming processor with decimation decomposition"""
    
    def __init__(self, seq_length, num_channels=2, num_arrays=8, num_sources=2):
        self.N = seq_length
        self.K = num_channels  # parallel channels
        self.M = num_arrays    # antenna array elements
        self.P = num_sources   # number of sources
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
    
    def direct_doa_estimation(self, signal_matrix, array_response, angles):
        """Direct DOA estimation using MUSIC algorithm"""
        L = signal_matrix.shape[1]
        R = signal_matrix @ signal_matrix.conj().T / L
        
        eigvals, eigvecs = eigh(R)
        noise_subspace = eigvecs[:, :self.M-self.P]
        
        spectrum = np.zeros(len(angles))
        for i, theta in enumerate(angles):
            a = array_response(theta, self.M)
            spectrum[i] = 1.0 / (np.abs(a.conj().T @ noise_subspace @ noise_subspace.conj().T @ a) + 1e-10)
        
        return spectrum, np.argsort(-spectrum)[:self.P]
    
    def parallel_doa_estimation(self, signal_matrix, array_response, angles):
        """Parallel DOA estimation"""
        L = signal_matrix.shape[1]
        
        # Decompose along time axis
        decomposed_signals = []
        for k in range(self.K):
            sub_signal = signal_matrix[:, k::self.K]
            decomposed_signals.append(sub_signal)
        
        # Process each channel
        sub_spectra = []
        for sub_signal in decomposed_signals:
            L_sub = sub_signal.shape[1]
            R_sub = sub_signal @ sub_signal.conj().T / L_sub
            
            eigvals, eigvecs = eigh(R_sub)
            noise_subspace = eigvecs[:, :self.M-self.P]
            
            spectrum = np.zeros(len(angles))
            for i, theta in enumerate(angles):
                a = array_response(theta, self.M)
                spectrum[i] = 1.0 / (np.abs(a.conj().T @ noise_subspace @ noise_subspace.conj().T @ a) + 1e-10)
            
            sub_spectra.append(spectrum)
        
        # Reconstruct
        reconstructed_spectrum = self._reconstruct_spectrum(sub_spectra)
        
        return reconstructed_spectrum, np.argsort(-reconstructed_spectrum)[:self.P], sub_spectra
    
    def _reconstruct_spectrum(self, sub_spectra):
        """Reconstruct spectrum from sub-channel results"""
        num_angles = len(sub_spectra[0])
        reconstructed = np.zeros(num_angles)
        
        for i in range(num_angles):
            values = [sub_spectra[k][i] for k in range(self.K)]
            reconstructed[i] = np.mean(values)
        
        return reconstructed


class SignalGenerator:
    """Generate test signals for beamforming"""
    
    def __init__(self, fs=40e9, M=8):
        self.fs = fs
        self.M = M  # number of antennas
        self.c = 3e8  # speed of light
        self.wavelength = self.c / 10e9  # wavelength at 10 GHz
        self.d = self.wavelength / 2  # antenna spacing
    
    def generate_signal(self, signal_type, duration=1e-8, num_sources=2, source_angles=None):
        """Generate signal from sources at specific angles"""
        length = int(duration * self.fs)
        t = np.arange(length) / self.fs
        
        if source_angles is None:
            source_angles = np.array([15, 45])  # degrees
        
        # Generate source signals
        source_signals = []
        if signal_type == "sinusoid_1GHz":
            for _ in range(num_sources):
                source = np.sin(2 * np.pi * 1e9 * t)
                source_signals.append(source)
            signal_name = "1 GHz Sinusoid Sources"
        
        elif signal_type == "lfm_10_18GHz":
            from scipy.signal import chirp
            for _ in range(num_sources):
                source = chirp(t, 10e9, t[-1], 18e9, method='linear')
                source_signals.append(source)
            signal_name = "10-18 GHz LFM Sources"
        
        elif signal_type == "qpsk_1Gbps":
            for _ in range(num_sources):
                symbols = np.random.randint(0, 4, int(1e9 * duration))
                qpsk_map = {0: 1+1j, 1: 1-1j, 2: -1+1j, 3: -1-1j}
                qpsk_symbols = np.array([qpsk_map[s] for s in symbols]) / np.sqrt(2)
                samples_per_symbol = length // len(symbols)
                source = np.repeat(qpsk_symbols, samples_per_symbol)[:length]
                carrier = np.exp(2j * np.pi * 5e9 * t)
                source_signals.append(source * carrier)
            signal_name = "1 Gbps QPSK Sources"
        
        # Generate array response matrix
        signal_matrix = self._generate_array_signal(source_signals, source_angles)
        
        return signal_matrix, source_angles, signal_name
    
    def _generate_array_signal(self, source_signals, source_angles):
        """Generate signals at array elements"""
        signal_matrix = np.zeros((self.M, len(source_signals[0])), dtype=complex)
        
        for m in range(self.M):
            for angle_idx, angle in enumerate(source_angles):
                # Steering vector element
                phase = 2 * np.pi * m * self.d * np.sin(np.radians(angle)) / self.wavelength
                steering_element = np.exp(1j * phase)
                signal_matrix[m, :] += steering_element * source_signals[angle_idx]
        
        # Add noise
        noise = 0.1 * (np.random.randn(self.M, len(source_signals[0])) + 
                       1j * np.random.randn(self.M, len(source_signals[0])))
        signal_matrix += noise
        
        return signal_matrix
    
    def array_response(self, angle, M):
        """Steering vector for given angle"""
        a = np.zeros(M, dtype=complex)
        for m in range(M):
            phase = 2 * np.pi * m * self.d * np.sin(np.radians(angle)) / self.wavelength
            a[m] = np.exp(1j * phase)
        return a


class ComplexityAnalysis:
    """DOA estimation complexity and latency analysis"""
    
    @staticmethod
    def compute_complexity_direct(N, M, P):
        """Direct MUSIC: O(M^3 + N*M^2)"""
        # Covariance: N*M^2, Eigendecomposition: M^3
        return N * (M**2) + M**3
    
    @staticmethod
    def compute_complexity_parallel(N, M, P, K):
        """Parallel MUSIC: K channels with N/K samples each"""
        # Each channel: (N/K)*M^2 + M^3
        return K * ((N/K) * (M**2) + M**3)
    
    @staticmethod
    def compute_latency_direct(N, M, P, clock_rate_MHz):
        """Direct DOA latency"""
        operations = N * (M**2) + M**3
        ops_per_cycle = 2
        cycles = operations / ops_per_cycle
        return cycles / clock_rate_MHz
    
    @staticmethod
    def compute_latency_parallel(N, M, P, K, clock_rate_MHz):
        """Parallel DOA latency"""
        operations_per_channel = (N/K) * (M**2) + M**3
        ops_per_cycle = 2
        cycles_per_channel = operations_per_channel / ops_per_cycle
        latency_parallel = cycles_per_channel / clock_rate_MHz
        
        latency_reconstruct = M**2 / (clock_rate_MHz * 1000)
        return latency_parallel + latency_reconstruct


def plot_signal_and_array(signal_matrix, source_angles, signal_name, M):
    """Plot received signal at array elements"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Time domain signals
    ax = axes[0]
    for m in range(min(4, M)):
        ax.plot(np.real(signal_matrix[m, :500]), alpha=0.7, label=f'Element {m+1}')
    ax.set_xlabel('Sample Index', fontsize=11)
    ax.set_ylabel('Amplitude', fontsize=11)
    ax.set_title(f'Array Received Signals: {signal_name}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Source angles
    ax = axes[1]
    ax.bar(range(len(source_angles)), source_angles, color='skyblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Source Index', fontsize=11)
    ax.set_ylabel('Angle (degrees)', fontsize=11)
    ax.set_title(f'Source DOA: {source_angles.astype(int)}°', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(source_angles)))
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'beamforming_signal_{signal_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_parallel_doa_results(signal_matrix, source_angles, signal_name, processor, gen):
    """Plot parallel DOA estimation results"""
    angles = np.arange(-90, 91, 1)
    
    # Direct DOA
    spectrum_direct, doa_direct = processor.direct_doa_estimation(signal_matrix, gen.array_response, angles)
    
    # Parallel DOA
    spectrum_parallel, doa_parallel, sub_spectra = processor.parallel_doa_estimation(
        signal_matrix, gen.array_response, angles)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Direct DOA spectrum
    ax = axes[0, 0]
    ax.plot(angles, spectrum_direct, linewidth=2, color='blue')
    for angle in source_angles:
        ax.axvline(angle, color='r', linestyle='--', alpha=0.7, linewidth=2)
    ax.set_xlabel('Angle (degrees)', fontsize=11)
    ax.set_ylabel('Spectrum Magnitude', fontsize=11)
    ax.set_title('Direct DOA Estimation (MUSIC)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-90, 90])
    
    # Channel 1 spectrum
    ax = axes[0, 1]
    ax.plot(angles, sub_spectra[0], linewidth=2, color='green')
    for angle in source_angles:
        ax.axvline(angle, color='r', linestyle='--', alpha=0.7, linewidth=2)
    ax.set_xlabel('Angle (degrees)', fontsize=11)
    ax.set_ylabel('Spectrum Magnitude', fontsize=11)
    ax.set_title('Channel 1 DOA Spectrum (decimated)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-90, 90])
    
    # Channel 2 spectrum
    ax = axes[1, 0]
    ax.plot(angles, sub_spectra[1], linewidth=2, color='purple')
    for angle in source_angles:
        ax.axvline(angle, color='r', linestyle='--', alpha=0.7, linewidth=2)
    ax.set_xlabel('Angle (degrees)', fontsize=11)
    ax.set_ylabel('Spectrum Magnitude', fontsize=11)
    ax.set_title('Channel 2 DOA Spectrum (decimated)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-90, 90])
    
    # Reconstructed spectrum
    ax = axes[1, 1]
    ax.plot(angles, spectrum_direct, linewidth=2, linestyle='--', alpha=0.7, 
            color='blue', label='Direct')
    ax.plot(angles, spectrum_parallel, linewidth=2, color='red', label='Reconstructed')
    for angle in source_angles:
        ax.axvline(angle, color='k', linestyle=':', alpha=0.5, linewidth=2)
    ax.set_xlabel('Angle (degrees)', fontsize=11)
    ax.set_ylabel('Spectrum Magnitude', fontsize=11)
    ax.set_title('Reconstructed vs Direct DOA Spectrum', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-90, 90])
    
    plt.tight_layout()
    plt.savefig(f'parallel_doa_{signal_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Error analysis
    error = np.mean(np.abs(spectrum_direct - spectrum_parallel))
    relative_error = error / (np.mean(np.abs(spectrum_direct)) + 1e-10)
    peak_error = np.max(np.abs(spectrum_direct - spectrum_parallel))
    
    print(f"\n{signal_name}:")
    print(f"  Mean Error: {error:.2e}")
    print(f"  Relative Error: {relative_error:.2e}")
    print(f"  Peak Error: {peak_error:.2e}")
    print(f"  Direct DOA: {angles[doa_direct[:2]]}°")
    print(f"  Parallel DOA: {angles[doa_parallel[:2]]}°")


def plot_complexity_comparison():
    """Plot DOA complexity comparison"""
    seq_lengths = np.array([256, 512, 1024, 2048, 4096, 8192])
    M = 8  # antenna elements
    P = 2  # sources
    K = 2
    clock_rate = 1000
    
    complexity_direct = np.array([ComplexityAnalysis.compute_complexity_direct(n, M, P) 
                                   for n in seq_lengths])
    complexity_parallel = np.array([ComplexityAnalysis.compute_complexity_parallel(n, M, P, K) 
                                    for n in seq_lengths])
    latency_direct = np.array([ComplexityAnalysis.compute_latency_direct(n, M, P, clock_rate) 
                               for n in seq_lengths])
    latency_parallel = np.array([ComplexityAnalysis.compute_latency_parallel(n, M, P, K, clock_rate) 
                                 for n in seq_lengths])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Complexity comparison
    ax = axes[0, 0]
    ax.semilogy(seq_lengths, complexity_direct, 'o-', label='Direct', linewidth=2, markersize=8)
    ax.semilogy(seq_lengths, complexity_parallel, 's-', label='Parallel (K=2)', linewidth=2, markersize=8)
    ax.set_xlabel('Snapshot Length', fontsize=11)
    ax.set_ylabel('Operations', fontsize=11)
    ax.set_title('DOA Computational Complexity (M=8, P=2)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Complexity improvement
    ax = axes[0, 1]
    improvement = (complexity_direct - complexity_parallel) / complexity_direct * 100
    ax.plot(seq_lengths, improvement, 'o-', color='green', linewidth=2, markersize=8)
    ax.axhline(np.mean(improvement), color='r', linestyle='--', 
               label=f'Mean: {np.mean(improvement):.1f}%')
    ax.set_xlabel('Snapshot Length', fontsize=11)
    ax.set_ylabel('Improvement Ratio (%)', fontsize=11)
    ax.set_title('Complexity Reduction (K=2)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Latency comparison
    ax = axes[1, 0]
    ax.semilogy(seq_lengths, latency_direct * 1e6, 'o-', label='Direct', linewidth=2, markersize=8)
    ax.semilogy(seq_lengths, latency_parallel * 1e6, 's-', label='Parallel (K=2)', linewidth=2, markersize=8)
    ax.set_xlabel('Snapshot Length', fontsize=11)
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
    ax.set_xlabel('Snapshot Length', fontsize=11)
    ax.set_ylabel('Improvement Ratio (%)', fontsize=11)
    ax.set_title('Latency Reduction (K=2)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('doa_complexity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_antenna_elements_effect():
    """Plot effect of antenna elements on complexity"""
    seq_length = 4096
    antenna_counts = np.array([4, 8, 16, 32, 64])
    P = 2
    K = 2
    
    complexity_direct = np.array([ComplexityAnalysis.compute_complexity_direct(seq_length, m, P) 
                                   for m in antenna_counts])
    complexity_parallel = np.array([ComplexityAnalysis.compute_complexity_parallel(seq_length, m, P, K) 
                                    for m in antenna_counts])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Complexity vs antenna elements
    ax = axes[0]
    ax.semilogy(antenna_counts, complexity_direct, 'o-', label='Direct', linewidth=2, markersize=8)
    ax.semilogy(antenna_counts, complexity_parallel, 's-', label='Parallel (K=2)', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Antenna Elements', fontsize=11)
    ax.set_ylabel('Operations', fontsize=11)
    ax.set_title(f'Complexity vs Antenna Count (N={seq_length})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Improvement ratio
    ax = axes[1]
    improvement = (complexity_direct - complexity_parallel) / complexity_direct * 100
    ax.bar(range(len(antenna_counts)), improvement, color='skyblue', alpha=0.8, edgecolor='black')
    ax.set_xlabel('Number of Antenna Elements', fontsize=11)
    ax.set_ylabel('Improvement Ratio (%)', fontsize=11)
    ax.set_title('Complexity Improvement with Parallelism', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(antenna_counts)))
    ax.set_xticklabels([str(m) for m in antenna_counts])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('antenna_elements_effect.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_multicore_doa_efficiency():
    """Plot multi-core DOA efficiency"""
    seq_length = 4096
    M = 8
    P = 2
    K_values = np.array([1, 2, 4, 8])
    
    complexity_baseline = ComplexityAnalysis.compute_complexity_direct(seq_length, M, P)
    complexity_parallel = np.array([ComplexityAnalysis.compute_complexity_parallel(seq_length, M, P, K) 
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
    ax.set_title('DOA Parallel Efficiency', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Complexity distribution
    ax = axes[1]
    width = 0.35
    ax.bar(K_values - width/2, per_channel, width, label='Per-Channel', alpha=0.8)
    ax.bar(K_values + width/2, complexity_parallel, width, label='Total', alpha=0.8)
    ax.set_xlabel('Parallelism Level (K)', fontsize=11)
    ax.set_ylabel('Operations', fontsize=11)
    ax.set_title(f'Complexity Distribution (N={seq_length}, M={M})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('doa_multicore_efficiency.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_resolution_accuracy():
    """Plot angular resolution and accuracy"""
    seq_lengths = np.array([256, 512, 1024, 2048, 4096, 8192])
    M = 8
    
    # Rayleigh resolution limit (simplified)
    resolution = 50.0 / seq_lengths  # in degrees
    
    # SNR effect on accuracy
    snr_db = np.linspace(0, 30, 31)
    accuracy_vs_snr = 1.0 / (1.0 + 10**(snr_db/10))  # simplified accuracy model
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Resolution vs snapshot length
    ax = axes[0]
    ax.semilogy(seq_lengths, resolution, 'o-', linewidth=2, markersize=8, color='darkblue')
    ax.fill_between(seq_lengths, resolution*0.8, resolution*1.2, alpha=0.2)
    ax.set_xlabel('Snapshot Length', fontsize=11)
    ax.set_ylabel('Angular Resolution (degrees)', fontsize=11)
    ax.set_title('Rayleigh Resolution vs Snapshot Length', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Accuracy vs SNR
    ax = axes[1]
    ax.plot(snr_db, accuracy_vs_snr, 'o-', linewidth=2, markersize=6, color='green')
    ax.set_xlabel('SNR (dB)', fontsize=11)
    ax.set_ylabel('Angle Error Probability', fontsize=11)
    ax.set_title('DOA Accuracy vs SNR', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('doa_resolution_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("=" * 70)
    print("Parallel Beamforming (DOA Estimation) Analysis")
    print("=" * 70)
    
    # Setup
    fs = 40e9
    duration = 1e-8
    seq_length = int(duration * fs)
    M = 8  # antenna elements
    P = 2  # sources
    
    gen = SignalGenerator(fs, M)
    processor = BeamformingParallelProcessor(seq_length, num_channels=2, 
                                            num_arrays=M, num_sources=P)
    
    # Test signals
    signals = ["sinusoid_1GHz", "lfm_10_18GHz", "qpsk_1Gbps"]
    
    print("\n[1] Signal and Array Processing...")
    for signal_type in signals:
        signal_matrix, source_angles, signal_name = gen.generate_signal(signal_type, duration, P)
        
        print(f"\nProcessing: {signal_name}")
        plot_signal_and_array(signal_matrix, source_angles, signal_name, M)
        plot_parallel_doa_results(signal_matrix, source_angles, signal_name, processor, gen)
    
    # Complexity analysis
    print("\n[2] Plotting complexity analysis...")
    plot_complexity_comparison()
    
    print("\n[3] Plotting antenna elements effect...")
    plot_antenna_elements_effect()
    
    print("\n[4] Plotting multi-core efficiency...")
    plot_multicore_doa_efficiency()
    
    print("\n[5] Plotting resolution and accuracy...")
    plot_resolution_accuracy()
    
    print("\n" + "=" * 70)
    print("Parallel Beamforming Analysis Complete!")
    print("=" * 70)
