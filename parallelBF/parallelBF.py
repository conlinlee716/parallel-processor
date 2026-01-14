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
        """Decompose received data into K channels"""
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
        """Parallel beamforming processing"""
        sub_data = self.decompose_data(X)
        
        sub_doas = []
        sub_spectra = []
        sub_covs = []
        
        for sub_x in sub_data:
            R_sub = self.covariance_matrix(sub_x)
            sub_covs.append(R_sub)
            
            doa_est, spectrum, search_range = self.estimate_doa_mvdr(R_sub, resolution=search_resolution)
            sub_doas.append(doa_est)
            sub_spectra.append(spectrum)
        
        reconstructed_doa = np.mean(sub_doas)
        R_avg = (sub_covs[0] + sub_covs[1]) / 2
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
    """Complexity analysis for beamforming operations"""
    
    @staticmethod
    def doa_estimation_complexity(M, N_search, K=1):
        """DOA estimation: O(N_search * M^2)"""
        return K * N_search * (M ** 2)
    
    @staticmethod
    def beamforming_complexity(M, K=1):
        """Beamforming: O(M^3) for matrix inversion"""
        return K * (M ** 3)
    
    @staticmethod
    def total_complexity(M, N_search, K=1):
        """Total DOA + beamforming complexity"""
        return BeamformingComplexityAnalysis.doa_estimation_complexity(M, N_search, K) + \
               BeamformingComplexityAnalysis.beamforming_complexity(M, K)
    
    @staticmethod
    def compute_latency(complexity, clock_rate_MHz, ops_per_cycle=2):
        """Compute latency from complexity"""
        cycles = complexity / ops_per_cycle
        return cycles / clock_rate_MHz
    
    @staticmethod
    def memory_footprint(M, N_snapshots, K=1):
        """Memory for covariance matrix and data buffer"""
        cov_size = (M ** 2) * 16 / 1e6
        data_size = (M * N_snapshots / K) * 16 / 1e6
        return cov_size + data_size


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
    """Plot DOA and beampattern comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Direct DOA spectrum
    ax = axes[0, 0]
    ax.plot(result_direct['search_range'], result_direct['spectrum'], 'b-', linewidth=2.5)
    ax.axvline(true_doa, color='red', linestyle='--', linewidth=2, label=f'True DOA: {true_doa}°')
    ax.scatter(result_direct['doa'], np.max(result_direct['spectrum']),
               color='green', s=200, marker='o', label=f"Est: {result_direct['doa']:.1f}°", zorder=5)
    ax.set_xlabel('Angle (°)', fontsize=11)
    ax.set_ylabel('Power (dB)', fontsize=11)
    ax.set_title('Direct DOA Estimation', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xlim(-90, 90)
    
    # Parallel DOA spectrum
    ax = axes[0, 1]
    ax.plot(result_parallel['search_range'], result_parallel['sub_spectra'][0], 'g-', 
            linewidth=2.5, label='Channel 1')
    ax.plot(result_parallel['search_range'], result_parallel['sub_spectra'][1], 'purple', 
            linewidth=2.5, label='Channel 2')
    ax.axvline(true_doa, color='red', linestyle='--', linewidth=2, label=f'True DOA: {true_doa}°')
    ax.scatter(result_parallel['sub_doas'][0], np.max(result_parallel['sub_spectra'][0]),
               color='darkgreen', s=150, marker='^', zorder=5)
    ax.scatter(result_parallel['sub_doas'][1], np.max(result_parallel['sub_spectra'][1]),
               color='indigo', s=150, marker='v', zorder=5)
    ax.set_xlabel('Angle (°)', fontsize=11)
    ax.set_ylabel('Power (dB)', fontsize=11)
    ax.set_title('Parallel DOA Estimation (K=2)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xlim(-90, 90)
    
    # Direct beampattern
    ax = axes[1, 0]
    ax.plot(result_direct['theta_range'], result_direct['beampattern'], 'b-', linewidth=2.5)
    ax.axvline(true_doa, color='red', linestyle='--', linewidth=2, label=f'Desired: {true_doa}°')
    ax.set_xlabel('Angle (°)', fontsize=11)
    ax.set_ylabel('Gain (dB)', fontsize=11)
    ax.set_title('Direct Beampattern', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xlim(-90, 90)
    
    # Parallel beampattern
    ax = axes[1, 1]
    ax.plot(result_parallel['theta_range'], result_parallel['beampattern'], 'r-', 
            linewidth=2.5, label='Reconstructed')
    ax.plot(result_direct['theta_range'], result_direct['beampattern'], 'b--', 
            linewidth=2, alpha=0.6, label='Direct')
    ax.axvline(true_doa, color='red', linestyle='--', linewidth=2, label=f'Desired: {true_doa}°')
    ax.set_xlabel('Angle (°)', fontsize=11)
    ax.set_ylabel('Gain (dB)', fontsize=11)
    ax.set_title('Parallel vs Direct Beampattern', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xlim(-90, 90)
    
    plt.tight_layout()
    plt.savefig('beamforming_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    doa_error = np.abs(result_direct['doa'] - true_doa)
    parallel_error = np.abs(result_parallel['reconstructed_doa'] - true_doa)
    
    print(f"\nDirect DOA: {result_direct['doa']:.2f}° (Error: {doa_error:.2f}°)")
    print(f"Parallel DOA: {result_parallel['reconstructed_doa']:.2f}° (Error: {parallel_error:.2f}°)")
    print(f"Channel 1 DOA: {result_parallel['sub_doas'][0]:.2f}°")
    print(f"Channel 2 DOA: {result_parallel['sub_doas'][1]:.2f}°")


def plot_complexity_comparison():
    """Plot complexity analysis"""
    num_elements = np.array([4, 8, 16, 32])
    n_search = 181
    clock_rate = 1000
    
    comp_direct = np.array([BeamformingComplexityAnalysis.total_complexity(M, n_search, K=1) 
                            for M in num_elements])
    comp_parallel = np.array([BeamformingComplexityAnalysis.total_complexity(M, n_search, K=2) 
                             for M in num_elements])
    
    latency_direct = np.array([BeamformingComplexityAnalysis.compute_latency(c, clock_rate) 
                              for c in comp_direct])
    latency_parallel = np.array([BeamformingComplexityAnalysis.compute_latency(c, clock_rate) 
                               for c in comp_parallel])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Complexity
    ax = axes[0, 0]
    ax.semilogy(num_elements, comp_direct, 'o-', label='Direct (K=1)', linewidth=2.5, markersize=8)
    ax.semilogy(num_elements, comp_parallel, 's-', label='Parallel (K=2)', linewidth=2.5, markersize=8)
    ax.set_xlabel('Number of Elements', fontsize=11)
    ax.set_ylabel('Operations', fontsize=11)
    ax.set_title('Computational Complexity (DOA + Beamforming)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Complexity improvement
    ax = axes[0, 1]
    improvement = (comp_direct - comp_parallel) / comp_direct * 100
    ax.bar(range(len(num_elements)), improvement, color='green', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of Elements', fontsize=11)
    ax.set_ylabel('Improvement (%)', fontsize=11)
    ax.set_title('Complexity Reduction (K=2)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(num_elements)))
    ax.set_xticklabels([str(m) for m in num_elements])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Latency
    ax = axes[1, 0]
    ax.semilogy(num_elements, latency_direct * 1e6, 'o-', label='Direct', linewidth=2.5, markersize=8)
    ax.semilogy(num_elements, latency_parallel * 1e6, 's-', label='Parallel', linewidth=2.5, markersize=8)
    ax.set_xlabel('Number of Elements', fontsize=11)
    ax.set_ylabel('Latency (μs)', fontsize=11)
    ax.set_title('End-to-End Latency', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Latency improvement
    ax = axes[1, 1]
    latency_imp = (latency_direct - latency_parallel) / latency_direct * 100
    ax.bar(range(len(num_elements)), latency_imp, color='purple', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of Elements', fontsize=11)
    ax.set_ylabel('Improvement (%)', fontsize=11)
    ax.set_title('Latency Reduction (K=2)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(num_elements)))
    ax.set_xticklabels([str(m) for m in num_elements])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('beamforming_complexity.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_memory_analysis():
    """Plot memory footprint analysis"""
    num_elements = np.array([4, 8, 16, 32, 64])
    num_snapshots = 200000
    
    mem_direct = np.array([BeamformingComplexityAnalysis.memory_footprint(M, num_snapshots, K=1) 
                          for M in num_elements])
    mem_parallel = np.array([BeamformingComplexityAnalysis.memory_footprint(M, num_snapshots, K=2) 
                            for M in num_elements])
    
    mem_saving = (mem_direct - mem_parallel) / mem_direct * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Memory comparison
    ax = axes[0]
    ax.plot(num_elements, mem_direct, 'o-', label='Direct (K=1)', linewidth=2.5, markersize=8)
    ax.plot(num_elements, mem_parallel, 's-', label='Parallel (K=2)', linewidth=2.5, markersize=8)
    ax.set_xlabel('Number of Elements', fontsize=11)
    ax.set_ylabel('Memory (MB)', fontsize=11)
    ax.set_title('Memory Footprint Comparison', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Memory saving
    ax = axes[1]
    ax.bar(range(len(num_elements)), mem_saving, color='orange', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of Elements', fontsize=11)
    ax.set_ylabel('Memory Saving (%)', fontsize=11)
    ax.set_title('Memory Reduction (K=2)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(num_elements)))
    ax.set_xticklabels([str(m) for m in num_elements])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('beamforming_memory.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_accuracy_vs_snr():
    """Plot DOA estimation accuracy vs SNR"""
    num_elements = 4
    num_trials = 10
    snr_values = np.array([0, 5, 10, 15, 20, 25])
    true_doa = -30
    
    error_direct = []
    error_parallel = []
    
    print("    Computing SNR sensitivity analysis...")
    for snr in snr_values:
        errs_d = []
        errs_p = []
        
        for trial in range(num_trials):
            X, beamformer, _, _, _ = signal_generator(num_elements=num_elements, snr_db=snr, true_doa=true_doa)
            
            result_d = beamformer.direct_beamforming(X, true_doa)
            result_p = beamformer.parallel_beamforming(X, true_doa)
            
            errs_d.append(np.abs(result_d['doa'] - true_doa) + 1e-6)
            errs_p.append(np.abs(result_p['reconstructed_doa'] - true_doa) + 1e-6)
        
        error_direct.append(np.mean(errs_d))
        error_parallel.append(np.mean(errs_p))
    
    error_direct = np.array(error_direct)
    error_parallel = np.array(error_parallel)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Error vs SNR
    ax = axes[0]
    ax.semilogy(snr_values, error_direct, 'o-', label='Direct', linewidth=2.5, markersize=8)
    ax.semilogy(snr_values, error_parallel, 's-', label='Parallel (K=2)', linewidth=2.5, markersize=8)
    ax.set_xlabel('SNR (dB)', fontsize=11)
    ax.set_ylabel('DOA Error (°)', fontsize=11)
    ax.set_title('DOA Estimation Accuracy vs SNR', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Error difference
    ax = axes[1]
    error_diff = np.abs(error_direct - error_parallel) / (error_direct + 1e-10) * 100
    ax.plot(snr_values, error_diff, 'o-', color='red', linewidth=2.5, markersize=8)
    ax.set_xlabel('SNR (dB)', fontsize=11)
    ax.set_ylabel('Error Difference (%)', fontsize=11)
    ax.set_title('Performance Gap (Parallel vs Direct)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('beamforming_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_complexity_breakdown():
    """Plot complexity breakdown by task"""
    num_elements = np.array([4, 8, 16, 32])
    n_search = 181
    
    doa_comp = np.array([BeamformingComplexityAnalysis.doa_estimation_complexity(M, n_search) 
                        for M in num_elements])
    bf_comp = np.array([BeamformingComplexityAnalysis.beamforming_complexity(M) 
                       for M in num_elements])
    total_comp = doa_comp + bf_comp
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Complexity breakdown
    ax = axes[0]
    width = 0.35
    x_pos = np.arange(len(num_elements))
    ax.bar(x_pos - width/2, doa_comp, width, label='DOA Estimation', alpha=0.8, edgecolor='black')
    ax.bar(x_pos + width/2, bf_comp, width, label='Beamforming', alpha=0.8, edgecolor='black')
    ax.set_xlabel('Number of Elements', fontsize=11)
    ax.set_ylabel('Operations', fontsize=11)
    ax.set_title('Complexity Breakdown by Task', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(m) for m in num_elements])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Percentage contribution
    ax = axes[1]
    doa_pct = doa_comp / total_comp * 100
    bf_pct = bf_comp / total_comp * 100
    x_pos = np.arange(len(num_elements))
    ax.bar(x_pos, doa_pct, label='DOA Estimation', alpha=0.8, edgecolor='black')
    ax.bar(x_pos, bf_pct, bottom=doa_pct, label='Beamforming', alpha=0.8, edgecolor='black')
    ax.set_xlabel('Number of Elements', fontsize=11)
    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.set_title('Computational Load Distribution', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(m) for m in num_elements])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('beamforming_breakdown.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_performance_summary():
    """Plot overall performance metrics"""
    num_elements = np.array([4, 8, 16, 32])
    n_search = 181
    clock_rate = 1000
    
    comp_direct = np.array([BeamformingComplexityAnalysis.total_complexity(M, n_search, K=1) 
                            for M in num_elements])
    comp_parallel = np.array([BeamformingComplexityAnalysis.total_complexity(M, n_search, K=2) 
                             for M in num_elements])
    
    latency_direct = np.array([BeamformingComplexityAnalysis.compute_latency(c, clock_rate) * 1e6
                              for c in comp_direct])
    latency_parallel = np.array([BeamformingComplexityAnalysis.compute_latency(c, clock_rate) * 1e6
                               for c in comp_parallel])
    
    comp_improve = (comp_direct - comp_parallel) / comp_direct * 100
    latency_improve = (latency_direct - latency_parallel) / latency_direct * 100
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Complexity improvement
    ax = axes[0, 0]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    bars = ax.bar(range(len(num_elements)), comp_improve, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_xlabel('Number of Elements', fontsize=11)
    ax.set_ylabel('Improvement (%)', fontsize=11)
    ax.set_title('Complexity Reduction', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(num_elements)))
    ax.set_xticklabels([str(m) for m in num_elements])
    ax.grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Latency improvement
    ax = axes[0, 1]
    bars = ax.bar(range(len(num_elements)), latency_improve, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_xlabel('Number of Elements', fontsize=11)
    ax.set_ylabel('Improvement (%)', fontsize=11)
    ax.set_title('Latency Reduction', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(num_elements)))
    ax.set_xticklabels([str(m) for m in num_elements])
    ax.grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Latency comparison
    ax = axes[1, 0]
    x_pos = np.arange(len(num_elements))
    width = 0.35
    ax.bar(x_pos - width/2, latency_direct, width, label='Direct', alpha=0.8, edgecolor='black')
    ax.bar(x_pos + width/2, latency_parallel, width, label='Parallel (K=2)', alpha=0.8, edgecolor='black')
    ax.set_xlabel('Number of Elements', fontsize=11)
    ax.set_ylabel('Latency (μs)', fontsize=11)
    ax.set_title('End-to-End Latency Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(m) for m in num_elements])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Performance ratio
    ax = axes[1, 1]
    speedup = comp_direct / comp_parallel
    ax.plot(num_elements, speedup, 'o-', linewidth=3, markersize=10, color='darkblue')
    ax.axhline(2, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Linear speedup (K=2)')
    ax.set_xlabel('Number of Elements', fontsize=11)
    ax.set_ylabel('Speedup (×)', fontsize=11)
    ax.set_title('Parallel Speedup', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(num_elements)
    
    plt.tight_layout()
    plt.savefig('beamforming_summary.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("=" * 70)
    print("Parallel Beamforming Processing Analysis")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Generate signals
    print("\n[1] Generating test signal...")
    X, beamformer, true_doa, wavelength, element_spacing = signal_generator(
        num_elements=4, num_snapshots=200000, true_doa=-30, snr_db=15)
    
    print(f"Array elements: 4")
    print(f"True DOA: {true_doa}°")
    print(f"Wavelength: {wavelength*1e3:.4f} mm")
    print(f"Element spacing: {element_spacing*1e3:.4f} mm")
    
    # Process
    print("\n[2] Direct beamforming...")
    result_direct = beamformer.direct_beamforming(X, true_doa)
    
    print("\n[3] Parallel beamforming (K=2)...")
    result_parallel = beamformer.parallel_beamforming(X, true_doa)
    
    # Plots
    print("\n[4] Plotting DOA and beampattern...")
    plot_doa_beampattern_comparison(result_direct, result_parallel, true_doa)
    
    print("\n[5] Plotting complexity analysis...")
    plot_complexity_comparison()
    
    print("\n[6] Plotting memory analysis...")
    plot_memory_analysis()
    
    print("\n[7] Plotting accuracy vs SNR...")
    plot_accuracy_vs_snr()
    
    print("\n[8] Plotting complexity breakdown...")
    plot_complexity_breakdown()
    
    print("\n[9] Plotting performance summary...")
    plot_performance_summary()
    
    print("\n" + "=" * 70)
    print("Parallel Beamforming Analysis Complete!")
    print("Generated 7 analysis figures:")
    print("  - beamforming_comparison.png")
    print("  - beamforming_complexity.png")
    print("  - beamforming_memory.png")
    print("  - beamforming_accuracy.png")
    print("  - beamforming_breakdown.png")
    print("  - beamforming_summary.png")
    print("=" * 70)
