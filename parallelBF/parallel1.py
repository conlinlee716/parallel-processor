import numpy as np
import matplotlib.pyplot as plt

class ParallelBeamformer:
    """Parallel beamforming processor with decimation decomposition"""
    
    def __init__(self, num_elements=4, wavelength=1.0, element_spacing=0.5, num_channels=2):
        self.num_elements = num_elements
        self.wavelength = wavelength
        self.element_spacing = element_spacing
        self.array_positions = np.arange(num_elements) * element_spacing
        self.K = num_channels
    
    def steering_vector(self, theta):
        """Compute steering vector"""
        theta_rad = np.radians(theta)
        phase = 2 * np.pi * self.array_positions * np.sin(theta_rad) / self.wavelength
        return np.exp(1j * phase)
    
    def decompose_data(self, X):
        """Decompose received data into K channels by time decimation"""
        sub_data = []
        for k in range(self.K):
            sub_x = X[:, k::self.K]
            sub_data.append(sub_x)
        return sub_data
    
    def covariance_matrix(self, X_sub):
        """Compute covariance matrix from sub-channel data"""
        return (X_sub @ np.conj(X_sub.T)) / X_sub.shape[1]
    
    def mvdr_beamformer(self, R, theta_desired):
        """Compute MVDR beamformer weights"""
        a = self.steering_vector(theta_desired)
        try:
            R_inv = np.linalg.inv(R)
        except:
            R_inv = np.linalg.pinv(R)
        
        numerator = R_inv @ a
        denominator = np.conj(a).T @ numerator
        w = numerator / (denominator + 1e-10)
        return w
    
    def estimate_doa_mvdr(self, R, search_range=None, resolution=0.5):
        """DOA estimation using MVDR spectrum"""
        if search_range is None:
            search_range = np.arange(-90, 91, resolution)
        
        spectrum = np.zeros(len(search_range))
        for i, theta in enumerate(search_range):
            a = self.steering_vector(theta)
            try:
                R_inv = np.linalg.inv(R)
            except:
                R_inv = np.linalg.pinv(R)
            
            denominator = np.conj(a).T @ R_inv @ a
            spectrum[i] = 1.0 / (np.abs(denominator) + 1e-10)
        
        spectrum_norm = spectrum / np.max(spectrum)
        spectrum_db = 10 * np.log10(spectrum_norm + 1e-10)
        
        peak_idx = np.argmax(spectrum_db)
        estimated_doa = search_range[peak_idx]
        
        return estimated_doa, spectrum_db, search_range
    
    def beampattern(self, R, theta_desired, theta_range=None):
        """Compute beampattern"""
        if theta_range is None:
            theta_range = np.linspace(-90, 90, 361)
        
        w = self.mvdr_beamformer(R, theta_desired)
        power = np.zeros(len(theta_range))
        
        for i, theta in enumerate(theta_range):
            a = self.steering_vector(theta)
            power[i] = np.abs(np.conj(w).T @ a) ** 2
        
        power_norm = power / np.max(power)
        power_db = 10 * np.log10(power_norm + 1e-10)
        
        return theta_range, power_db, w
    
    def parallel_beamforming(self, X, theta_desired, search_resolution=0.5):
        """
        Parallel beamforming: decompose into K channels, process independently
        """
        sub_data = self.decompose_data(X)
        
        sub_doas = []
        sub_spectra = []
        sub_covs = []
        
        # Channel-wise processing
        for sub_x in sub_data:
            R_sub = self.covariance_matrix(sub_x)
            sub_covs.append(R_sub)
            
            doa_est, spectrum, search_range = self.estimate_doa_mvdr(R_sub, resolution=search_resolution)
            sub_doas.append(doa_est)
            sub_spectra.append(spectrum)
        
        # Fusion: average covariance and DOA
        reconstructed_doa = np.mean(sub_doas)
        R_avg = np.mean(sub_covs, axis=0)
        
        theta_range, power_db, w = self.beampattern(R_avg, reconstructed_doa)
        
        return {
            'reconstructed_doa': reconstructed_doa,
            'sub_doas': sub_doas,
            'sub_spectra': sub_spectra,
            'search_range': search_range,
            'beampattern': power_db,
            'theta_range': theta_range,
            'weights': w,
            'R_parallel': R_avg
        }
    
    def direct_beamforming(self, X, theta_desired, search_resolution=0.5):
        """Direct beamforming without decomposition"""
        R = (X @ np.conj(X.T)) / X.shape[1]
        
        doa_est, spectrum, search_range = self.estimate_doa_mvdr(R, resolution=search_resolution)
        theta_range, power_db, w = self.beampattern(R, theta_desired)
        
        return {
            'doa': doa_est,
            'spectrum': spectrum,
            'search_range': search_range,
            'beampattern': power_db,
            'theta_range': theta_range,
            'weights': w,
            'R_direct': R
        }


class BeamformingComplexityAnalysis:
    """
    Complexity analysis for beamforming operations
    
    Key stages:
    1. Covariance matrix computation: O(M^2 * N_snapshots)
    2. Matrix inversion: O(M^3)
    3. DOA estimation (MVDR search): O(N_search * M^2)
    4. Beamforming weights: O(M^2)
    """
    
    @staticmethod
    def covariance_complexity(M, N_snapshots):
        """Covariance matrix computation: R = X @ X^H / N"""
        return M * M * N_snapshots
    
    @staticmethod
    def matrix_inversion_complexity(M):
        """Matrix inversion using LU decomposition: ~2.67*M^3"""
        return int(2.67 * (M ** 3))
    
    @staticmethod
    def doa_search_complexity(M, N_search):
        """DOA estimation: O(N_search * M^2) for MVDR spectral search"""
        return N_search * (M ** 2)
    
    @staticmethod
    def beamforming_complexity(M):
        """Beamforming weight computation: O(M^2)"""
        return M * M
    
    @staticmethod
    def total_complexity_direct(M, N_search):
        """
        Direct processing (single channel)
        Total = Matrix_inv + DOA_search + Beamforming
        (Covariance computation is data-dependent, analyzed separately)
        """
        return (BeamformingComplexityAnalysis.matrix_inversion_complexity(M) +
                BeamformingComplexityAnalysis.doa_search_complexity(M, N_search) +
                BeamformingComplexityAnalysis.beamforming_complexity(M))
    
    @staticmethod
    def total_complexity_parallel(M, N_search, K=2):
        """
        Parallel processing (K channels, independent DOA estimation per channel)
        
        Each channel performs:
        - Matrix inversion: O(M^3)
        - DOA search: O(N_search * M^2)
        
        Final stage (fusion):
        - Average covariance: O(M^2)
        - Select DOA: O(K)
        - Beamforming: O(M^2)
        
        Total = K * (Matrix_inv + DOA_search) + 2*M^2 + Beamforming
        """
        return (K * (BeamformingComplexityAnalysis.matrix_inversion_complexity(M) +
                     BeamformingComplexityAnalysis.doa_search_complexity(M, N_search)) +
                2 * (M ** 2) +
                BeamformingComplexityAnalysis.beamforming_complexity(M))
    
    @staticmethod
    def compute_latency(complexity, clock_rate_MHz, ops_per_cycle=2):
        """Compute latency from complexity"""
        cycles = complexity / ops_per_cycle
        return cycles / clock_rate_MHz
    
    @staticmethod
    def memory_footprint(M, N_snapshots, K=1):
        """
        Memory footprint analysis:
        - Covariance matrix: M^2 * 16 bytes (complex128)
        - Data buffer: M * N_snapshots/K * 16 bytes (parallel reduces by 1/K)
        """
        cov_size = (M ** 2) * 16 / 1e6
        data_size = (M * N_snapshots / K) * 16 / 1e6
        return cov_size + data_size
    
    @staticmethod
    def memory_bandwidth(M, N_search, N_snapshots, K=1):
        """Memory bandwidth requirement during processing"""
        data_access = M * N_snapshots / K
        return data_access * 16 / 1e9  # GB


def signal_generator(num_elements=4, num_snapshots=200000, fs=40e9, f_signal=11e9, 
                     true_doa=-30, snr_db=15, c=3e8):
    """Generate received signal"""
    wavelength = c / f_signal
    element_spacing = wavelength / 2
    
    beamformer = ParallelBeamformer(num_elements=num_elements, 
                                   wavelength=wavelength,
                                   element_spacing=element_spacing)
    
    t = np.arange(num_snapshots) / fs
    signal = np.exp(1j * 2 * np.pi * f_signal * t)
    
    a = beamformer.steering_vector(true_doa)
    signal_power = 10 ** (snr_db / 20)
    X = np.outer(a, signal) * signal_power
    
    noise = (np.random.randn(num_elements, num_snapshots) + 
             1j * np.random.randn(num_elements, num_snapshots)) / np.sqrt(2)
    X += noise
    
    return X, beamformer, true_doa, wavelength, element_spacing


def plot_doa_beampattern_comparison(result_direct, result_parallel, true_doa):
    """Plot DOA and beampattern comparison (2x2 subplots)"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Direct DOA spectrum
    ax = axes[0, 0]
    ax.plot(result_direct['search_range'], result_direct['spectrum'], 'b-', linewidth=2.5, label='Direct')
    ax.axvline(true_doa, color='red', linestyle='--', linewidth=2.5, label=f'True DOA: {true_doa}°')
    ax.scatter(result_direct['doa'], np.max(result_direct['spectrum']),
               color='green', s=200, marker='o', label=f"Est: {result_direct['doa']:.1f}°", 
               zorder=5, edgecolors='darkgreen', linewidths=2)
    ax.set_xlabel('Angle (°)', fontsize=11, fontweight='bold')
    ax.set_ylabel('MVDR Spectrum (dB)', fontsize=11, fontweight='bold')
    ax.set_title('Direct DOA Estimation', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(-90, 90)
    
    # Parallel DOA spectrum (both channels)
    ax = axes[0, 1]
    ax.plot(result_parallel['search_range'], result_parallel['sub_spectra'][0], 'g-', 
            linewidth=2.5, label='Channel 1')
    ax.plot(result_parallel['search_range'], result_parallel['sub_spectra'][1], 'purple', 
            linewidth=2.5, label='Channel 2')
    ax.axvline(true_doa, color='red', linestyle='--', linewidth=2.5, label=f'True DOA: {true_doa}°')
    ax.scatter(result_parallel['sub_doas'][0], np.max(result_parallel['sub_spectra'][0]),
               color='darkgreen', s=150, marker='^', zorder=5, edgecolors='darkgreen', linewidth=1.5)
    ax.scatter(result_parallel['sub_doas'][1], np.max(result_parallel['sub_spectra'][1]),
               color='indigo', s=150, marker='v', zorder=5, edgecolors='indigo', linewidth=1.5)
    ax.set_xlabel('Angle (°)', fontsize=11, fontweight='bold')
    ax.set_ylabel('MVDR Spectrum (dB)', fontsize=11, fontweight='bold')
    ax.set_title('Parallel DOA Estimation (K=2)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(-90, 90)
    
    # Direct beampattern
    ax = axes[1, 0]
    ax.plot(result_direct['theta_range'], result_direct['beampattern'], 'b-', linewidth=2.5, label='Direct')
    ax.axvline(true_doa, color='red', linestyle='--', linewidth=2.5, label=f'Desired: {true_doa}°')
    ax.set_xlabel('Angle (°)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Gain (dB)', fontsize=11, fontweight='bold')
    ax.set_title('Direct Beampattern', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(-90, 90)
    ax.set_ylim(np.min(result_direct['beampattern']), 5)
    
    # Parallel beampattern
    ax = axes[1, 1]
    ax.plot(result_parallel['theta_range'], result_parallel['beampattern'], 'r-', 
            linewidth=2.5, label='Parallel (K=2)', zorder=2)
    ax.plot(result_direct['theta_range'], result_direct['beampattern'], 'b--', 
            linewidth=2, alpha=0.6, label='Direct', zorder=1)
    ax.axvline(true_doa, color='red', linestyle='--', linewidth=2.5, label=f'Desired: {true_doa}°', zorder=0)
    ax.set_xlabel('Angle (°)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Gain (dB)', fontsize=11, fontweight='bold')
    ax.set_title('Beampattern Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(-90, 90)
    ax.set_ylim(np.min(result_direct['beampattern']), 5)
    
    plt.tight_layout()
    plt.savefig('01_beamforming_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    doa_error_direct = np.abs(result_direct['doa'] - true_doa)
    doa_error_parallel = np.abs(result_parallel['reconstructed_doa'] - true_doa)
    
    print("\n" + "="*70)
    print("DOA Estimation Results")
    print("="*70)
    print(f"Direct DOA: {result_direct['doa']:.2f}° | Error: {doa_error_direct:.2f}°")
    print(f"Parallel DOA: {result_parallel['reconstructed_doa']:.2f}° | Error: {doa_error_parallel:.2f}°")
    print(f"  └─ Channel 1 DOA: {result_parallel['sub_doas'][0]:.2f}°")
    print(f"  └─ Channel 2 DOA: {result_parallel['sub_doas'][1]:.2f}°")
    print("="*70)


def plot_complexity_analysis():
    """
    Complexity analysis for beamforming operations
    
    Analysis includes:
    1. Matrix inversion: O(M^3)
    2. DOA search: O(N_search * M^2)
    3. Beamforming: O(M^2)
    
    Direct: Single processing chain
    Parallel (K=2): Each channel independently computes DOA
    """
    num_elements = np.array([4, 8, 16, 32])
    n_search = 181  # Search range: -90° to 90° with 0.5° resolution
    clock_rate = 1000  # MHz
    
    comp_direct = np.array([BeamformingComplexityAnalysis.total_complexity_direct(M, n_search) 
                            for M in num_elements])
    comp_parallel = np.array([BeamformingComplexityAnalysis.total_complexity_parallel(M, n_search, K=2) 
                             for M in num_elements])
    
    latency_direct = np.array([BeamformingComplexityAnalysis.compute_latency(c, clock_rate) 
                              for c in comp_direct])
    latency_parallel = np.array([BeamformingComplexityAnalysis.compute_latency(c, clock_rate) 
                               for c in comp_parallel])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ========== Subplot 1: Computational Complexity Comparison ==========
    ax = axes[0, 0]
    ax.semilogy(num_elements, comp_direct, 'o-', label='Direct (K=1)', 
                linewidth=2.5, markersize=10, color='#2E86AB')
    ax.semilogy(num_elements, comp_parallel, 's-', label='Parallel (K=2)', 
                linewidth=2.5, markersize=10, color='#A23B72')
    ax.set_xlabel('Number of Array Elements (M)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Total Operations (log scale)', fontsize=11, fontweight='bold')
    ax.set_title('Computational Complexity: DOA + Beamforming', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.set_xticks(num_elements)
    
    # ========== Subplot 2: Complexity Reduction Ratio ==========
    ax = axes[0, 1]
    reduction = (comp_direct - comp_parallel) / comp_direct * 100
    colors_reduction = ['#F18F01' if r < 0 else '#06A77D' for r in reduction]
    bars = ax.bar(range(len(num_elements)), reduction, color=colors_reduction, 
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Number of Array Elements (M)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Complexity Change (%)', fontsize=11, fontweight='bold')
    ax.set_title('Parallel vs Direct Complexity', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(num_elements)))
    ax.set_xticklabels([str(m) for m in num_elements])
    ax.axhline(0, color='black', linewidth=1)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, reduction)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:+.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10, fontweight='bold')
    
    # ========== Subplot 3: End-to-End Latency Comparison ==========
    ax = axes[1, 0]
    ax.semilogy(num_elements, latency_direct * 1e6, 'o-', label='Direct (K=1)', 
                linewidth=2.5, markersize=10, color='#2E86AB')
    ax.semilogy(num_elements, latency_parallel * 1e6, 's-', label='Parallel (K=2)', 
                linewidth=2.5, markersize=10, color='#A23B72')
    ax.set_xlabel('Number of Array Elements (M)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Latency (μs, log scale)', fontsize=11, fontweight='bold')
    ax.set_title('End-to-End Processing Latency', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.set_xticks(num_elements)
    
    # ========== Subplot 4: Latency Reduction Ratio ==========
    ax = axes[1, 1]
    latency_reduction = (latency_direct - latency_parallel) / latency_direct * 100
    colors_latency = ['#F18F01' if r < 0 else '#06A77D' for r in latency_reduction]
    bars = ax.bar(range(len(num_elements)), latency_reduction, color=colors_latency, 
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Number of Array Elements (M)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Latency Change (%)', fontsize=11, fontweight='bold')
    ax.set_title('Latency Reduction (K=2)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(num_elements)))
    ax.set_xticklabels([str(m) for m in num_elements])
    ax.axhline(0, color='black', linewidth=1)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, latency_reduction)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:+.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('02_complexity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*70)
    print("Complexity Analysis (DOA Estimation + Beamforming)")
    print("="*70)
    print(f"Search Range: -90° to +90° (Resolution: 0.5°, N_search = {n_search})")
    print(f"Clock Rate: {clock_rate} MHz, Ops/Cycle: 2")
    print()
    for i, M in enumerate(num_elements):
        print(f"M={M:2d} elements:")
        print(f"  Direct:  {comp_direct[i]:15,.0f} ops | Latency: {latency_direct[i]*1e6:8.2f} μs")
        print(f"  Parallel:{comp_parallel[i]:15,.0f} ops | Latency: {latency_parallel[i]*1e6:8.2f} μs")
        print(f"  Change:  {reduction[i]:+6.1f}% | Latency change: {latency_reduction[i]:+6.1f}%")
        print()
    print("="*70)


def plot_memory_analysis():
    """
    Memory footprint analysis:
    - Covariance matrix: M^2 complex128 = M^2 * 16 bytes
    - Data buffer: M * N_snapshots/K * 16 bytes
    
    Parallel processing reduces data buffer by factor K
    """
    num_elements = np.array([4, 8, 16, 32, 64])
    num_snapshots = 200000
    
    mem_direct = np.array([BeamformingComplexityAnalysis.memory_footprint(M, num_snapshots, K=1) 
                          for M in num_elements])
    mem_parallel = np.array([BeamformingComplexityAnalysis.memory_footprint(M, num_snapshots, K=2) 
                            for M in num_elements])
    
    mem_saving = (mem_direct - mem_parallel) / mem_direct * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ========== Subplot 1: Memory Footprint Comparison ==========
    ax = axes[0]
    ax.plot(num_elements, mem_direct, 'o-', label='Direct (K=1)', 
            linewidth=2.5, markersize=10, color='#2E86AB')
    ax.plot(num_elements, mem_parallel, 's-', label='Parallel (K=2)', 
            linewidth=2.5, markersize=10, color='#A23B72')
    ax.fill_between(num_elements, mem_direct, mem_parallel, alpha=0.2, color='green')
    ax.set_xlabel('Number of Array Elements (M)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Memory Footprint (MB)', fontsize=11, fontweight='bold')
    ax.set_title('Memory Footprint Comparison (N_snapshots=200k)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(num_elements)
    
    # ========== Subplot 2: Memory Saving Ratio ==========
    ax = axes[1]
    bars = ax.bar(range(len(num_elements)), mem_saving, color='#06A77D', 
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Number of Array Elements (M)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Memory Saving (%)', fontsize=11, fontweight='bold')
    ax.set_title('Memory Reduction (K=2)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(num_elements)))
    ax.set_xticklabels([str(m) for m in num_elements])
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, mem_saving)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('03_memory_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*70)
    print("Memory Analysis (Covariance + Data Buffer)")
    print("="*70)
    print(f"Number of Snapshots: {num_snapshots:,}")
    print(f"Data Type: complex128 (16 bytes per element)")
    print()
    for i, M in enumerate(num_elements):
        print(f"M={M:2d} elements:")
        print(f"  Direct:  {mem_direct[i]:8.3f} MB")
        print(f"  Parallel:{mem_parallel[i]:8.3f} MB")
        print(f"  Saving:  {mem_saving[i]:6.1f}%")
        print()
    print("="*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PARALLEL BEAMFORMING PROCESSING ANALYSIS")
    print("="*70)
    
    np.random.seed(42)
    
    # ============================================================
    # [1] Generate test signal
    # ============================================================
    print("\n[1/4] Generating test signal...")
    X, beamformer, true_doa, wavelength, element_spacing = signal_generator(
        num_elements=4, num_snapshots=200000, true_doa=-30, snr_db=15)
    
    print(f"  ✓ Array elements: 4")
    print(f"  ✓ True DOA: {true_doa}°")
    print(f"  ✓ Wavelength: {wavelength*1e3:.4f} mm")
    print(f"  ✓ Element spacing: {element_spacing*1e3:.4f} mm")
    print(f"  ✓ Snapshots: 200,000")
    
    # ============================================================
    # [2] Direct beamforming
    # ============================================================
    print("\n[2/4] Processing with direct beamforming...")
    result_direct = beamformer.direct_beamforming(X, true_doa)
    print(f"  ✓ DOA estimation complete")
    print(f"  ✓ Beamforming complete")
    
    # ============================================================
    # [3] Parallel beamforming (K=2)
    # ============================================================
    print("\n[3/4] Processing with parallel beamforming (K=2)...")
    result_parallel = beamformer.parallel_beamforming(X, true_doa)
    print(f"  ✓ Channel 1 processed")
    print(f"  ✓ Channel 2 processed")
    print(f"  ✓ Results fused")
    
    # ============================================================
    # [4] Generate analysis plots
    # ============================================================
    print("\n[4/4] Generating analysis plots...")
    
    print("\n  Plot 1: DOA & Beampattern Comparison...")
    plot_doa_beampattern_comparison(result_direct, result_parallel, true_doa)
    
    print("\n  Plot 2: Complexity Analysis...")
    plot_complexity_analysis()
    
    print("\n  Plot 3: Memory Analysis...")
    plot_memory_analysis()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("Generated 3 analysis figures:")
    print("  ✓ 01_beamforming_comparison.png (2×2 subplots)")
    print("  ✓ 02_complexity_analysis.png (2×2 subplots)")
    print("  ✓ 03_memory_analysis.png (1×2 subplots)")
    print("="*70 + "\n")
