import numpy as np
import matplotlib.pyplot as plt
import warnings

# 设置全局绘图风格
warnings.filterwarnings('ignore', category=UserWarning)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['axes.unicode_minus'] = False

class ParallelBeamformer:
    """Parallel beamforming processor with time-domain decimation"""
    
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
        """Parallel beamforming: decompose into K channels, process independently"""
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
    """波束成形复杂度分析 - 时序降速分解"""
    
    @staticmethod
    def matrix_inversion_ops(M):
        """矩阵求逆 (LU分解): ~2.67*M³"""
        return int(2.67 * (M ** 3))
    
    @staticmethod
    def doa_search_ops(M, N_search):
        """DOA搜索 (MVDR谱): O(N_search * M²)"""
        return N_search * (M ** 2)
    
    @staticmethod
    def beamforming_ops(M):
        """波束成形权重: O(M²)"""
        return M * M
    
    @staticmethod
    def covariance_ops(M, N_snapshots):
        """协方差计算: O(M² × N_snapshots)"""
        return M * M * int(N_snapshots)
    
    @staticmethod
    def total_ops_direct(M, N_search, N_snapshots):
        """直接处理 (K=1)"""
        return (BeamformingComplexityAnalysis.covariance_ops(M, N_snapshots) +
                BeamformingComplexityAnalysis.matrix_inversion_ops(M) +
                BeamformingComplexityAnalysis.doa_search_ops(M, N_search))
    
    @staticmethod
    def total_ops_parallel(M, N_search, N_snapshots, K=2):
        """并行处理 (K通道)"""
        cov_per_ch = BeamformingComplexityAnalysis.covariance_ops(M, N_snapshots / K)
        inv_ops = BeamformingComplexityAnalysis.matrix_inversion_ops(M)
        search_ops = BeamformingComplexityAnalysis.doa_search_ops(M, N_search)
        
        per_channel = cov_per_ch + inv_ops + search_ops
        fusion_ops = 2 * (M ** 2)
        
        return K * per_channel + fusion_ops
    
    @staticmethod
    def memory_ops(M, N_snapshots, K=1):
        """内存占用"""
        cov_mem = (M ** 2) * 16 / 1e6
        data_mem = (M * N_snapshots / K) * 16 / 1e6
        return cov_mem + data_mem
    
    @staticmethod
    def latency_us(ops, clock_MHz=1000, ops_per_cycle=2):
        """延时计算"""
        cycles = ops / ops_per_cycle
        return cycles / clock_MHz


def signal_generator(num_elements=4, num_snapshots=200000, fs=40e9, f_signal=11e9, 
                     true_doa=-30, snr_db=15, c=3e8):
    """生成接收信号"""
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
    """绘制DOA与波束图对比 (2×3 子图 + 极坐标)"""
    fig = plt.figure(figsize=(20, 10))
    
    # ===== (0,0): Direct DOA spectrum =====
    ax = plt.subplot(2, 3, 1)
    ax.plot(result_direct['search_range'], result_direct['spectrum'], 'b-', linewidth=2.5, label='Direct')
    ax.axvline(true_doa, color='red', linestyle='--', linewidth=2.5, label=f'True: {true_doa}°')
    ax.scatter(result_direct['doa'], np.max(result_direct['spectrum']),
               color='green', s=200, marker='o', label=f"Est: {result_direct['doa']:.1f}°", 
               zorder=5, edgecolors='darkgreen', linewidths=2)
    ax.set_xlabel('Angle (°)', fontweight='bold')
    ax.set_ylabel('MVDR Spectrum (dB)', fontweight='bold')
    ax.set_title('Direct DOA Estimation', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=16)
    ax.set_xlim(-90, 90)
    
    # ===== (0,1): Parallel DOA spectrum =====
    ax = plt.subplot(2, 3, 2)
    ax.plot(result_parallel['search_range'], result_parallel['sub_spectra'][0], 'g-', 
            linewidth=2.5, label='Channel 1')
    ax.plot(result_parallel['search_range'], result_parallel['sub_spectra'][1], 'purple', 
            linewidth=2.5, label='Channel 2')
    ax.axvline(true_doa, color='red', linestyle='--', linewidth=2.5, label=f'True: {true_doa}°')
    ax.scatter(result_parallel['sub_doas'][0], np.max(result_parallel['sub_spectra'][0]),
               color='darkgreen', s=150, marker='^', zorder=5)
    ax.scatter(result_parallel['sub_doas'][1], np.max(result_parallel['sub_spectra'][1]),
               color='indigo', s=150, marker='v', zorder=5)
    ax.set_xlabel('Angle (°)', fontweight='bold')
    ax.set_ylabel('MVDR Spectrum (dB)', fontweight='bold')
    ax.set_title('Parallel DOA Estimation (K=2)', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=16)
    ax.set_xlim(-90, 90)
    
    # ===== (0,2): Beampattern Polar (Direct) =====
    ax = plt.subplot(2, 3, 3, projection='polar')
    theta_rad = np.radians(result_direct['theta_range'])
    power_linear = 10 ** (result_direct['beampattern'] / 10)
    ax.plot(theta_rad, power_linear, 'b-', linewidth=2.5, label='Direct')
    
    # 在极坐标中画 true_doa 的射线
    true_doa_rad = np.radians(true_doa)
    r_max = power_linear.max()
    ax.plot([true_doa_rad, true_doa_rad], [0, r_max * 1.1], 'r--', linewidth=2.5, label=f'True: {true_doa}°')
    
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_title('Direct Beampattern (Polar)', fontweight='bold', pad=20)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=14, loc='upper right')
    
    # ===== (1,0): Direct beampattern (Cartesian) =====
    ax = plt.subplot(2, 3, 4)
    ax.plot(result_direct['theta_range'], result_direct['beampattern'], 'b-', linewidth=2.5, label='Direct')
    ax.axvline(true_doa, color='red', linestyle='--', linewidth=2.5, label=f'Desired: {true_doa}°')
    ax.set_xlabel('Angle (°)', fontweight='bold')
    ax.set_ylabel('Gain (dB)', fontweight='bold')
    ax.set_title('Direct Beampattern (Cartesian)', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=16)
    ax.set_xlim(-90, 90)
    ax.set_ylim(np.min(result_direct['beampattern']), 5)
    
    # ===== (1,1): Beampattern comparison (Cartesian) =====
    ax = plt.subplot(2, 3, 5)
    ax.plot(result_parallel['theta_range'], result_parallel['beampattern'], 'r-', 
            linewidth=2.5, label='Parallel (K=2)', zorder=2)
    ax.plot(result_direct['theta_range'], result_direct['beampattern'], 'b--', 
            linewidth=2, alpha=0.6, label='Direct', zorder=1)
    ax.axvline(true_doa, color='red', linestyle='--', linewidth=2.5, label=f'Desired: {true_doa}°', zorder=0)
    ax.set_xlabel('Angle (°)', fontweight='bold')
    ax.set_ylabel('Gain (dB)', fontweight='bold')
    ax.set_title('Beampattern Comparison (Cartesian)', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=16)
    ax.set_xlim(-90, 90)
    ax.set_ylim(np.min(result_direct['beampattern']), 5)
    
    # ===== (1,2): Beampattern Polar (Parallel) =====
    ax = plt.subplot(2, 3, 6, projection='polar')
    theta_rad = np.radians(result_parallel['theta_range'])
    power_linear = 10 ** (result_parallel['beampattern'] / 10)
    ax.plot(theta_rad, power_linear, 'r-', linewidth=2.5, label='Parallel (K=2)')
    
    # 在极坐标中画 true_doa 的射线
    true_doa_rad = np.radians(true_doa)
    r_max = power_linear.max()
    ax.plot([true_doa_rad, true_doa_rad], [0, r_max * 1.1], 'r--', linewidth=2.5, label=f'True: {true_doa}°')
    
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_title('Parallel Beampattern (Polar)', fontweight='bold', pad=20)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=14, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('01_beamforming_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    doa_error_direct = np.abs(result_direct['doa'] - true_doa)
    doa_error_parallel = np.abs(result_parallel['reconstructed_doa'] - true_doa)
    
    print("\n" + "="*80)
    print("DOA ESTIMATION RESULTS")
    print("="*80)
    print(f"Direct DOA:    {result_direct['doa']:7.2f}° | Error: {doa_error_direct:6.2f}°")
    print(f"Parallel DOA:  {result_parallel['reconstructed_doa']:7.2f}° | Error: {doa_error_parallel:6.2f}°")
    print(f"  └─ Ch.1 DOA: {result_parallel['sub_doas'][0]:7.2f}°")
    print(f"  └─ Ch.2 DOA: {result_parallel['sub_doas'][1]:7.2f}°")
    print("="*80)


def plot_beamforming_complexity_analysis(N_snapshots_range=None, M=4, N_search=181, K_values=[1, 2], clock_MHz=1000):
    """波束成形复杂度分析 (2×2 = 4个子图)"""
    if N_snapshots_range is None:
        N_snapshots_range = np.array([1e3, 5e3, 10e3, 50e3, 100e3, 200e3, 500e3, 1e6])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {1: 'black', 2: 'steelblue', 4: 'red', 8: 'green'}
    
    # ===== (0,0): Computational Complexity vs Snapshots =====
    ax = axes[0, 0]
    for K in K_values:
        ops = np.array([BeamformingComplexityAnalysis.total_ops_direct(M, N_search, int(n)) if K == 1 
                       else BeamformingComplexityAnalysis.total_ops_parallel(M, N_search, int(n), K) 
                       for n in N_snapshots_range])
        label_txt = 'Direct (K=1)' if K == 1 else f'Parallel (K={K})'
        ax.loglog(N_snapshots_range, ops, 'o-', color=colors.get(K, 'gray'), label=label_txt, linewidth=2.5, markersize=8)
    ax.set_title('Computational Complexity vs Snapshots', fontweight='bold')
    ax.set_xlabel('Number of Snapshots (N)', fontweight='bold')
    ax.set_ylabel('Operations (log scale)', fontweight='bold')
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.legend(fontsize=16, loc='upper left')
    
    # ===== (0,1): Speedup Factor vs Snapshots =====
    ax = axes[0, 1]
    for K in K_values[1:]:
        speedup = []
        for n in N_snapshots_range:
            ops_direct = BeamformingComplexityAnalysis.total_ops_direct(M, N_search, int(n))
            ops_parallel = BeamformingComplexityAnalysis.total_ops_parallel(M, N_search, int(n), K)/K
            speedup.append(ops_direct / ops_parallel) 
        ax.plot(N_snapshots_range, speedup, 'o-', color=colors.get(K, 'gray'), 
               label=f'K={K}', linewidth=2.5, markersize=8)
    ax.axhline(1, color='gray', ls='--', alpha=0.5, linewidth=1.5)
    ax.set_title('Parallel Speedup Ratio vs Snapshots', fontweight='bold')
    ax.set_xlabel('Number of Snapshots (N)', fontweight='bold')
    ax.set_ylabel('Speedup (×)', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=16)
    ax.set_xscale('log')
    
    # ===== (1,0): End-to-End Latency vs Snapshots =====
    ax = axes[1, 0]
    for K in K_values:
        ops_array = np.array([BeamformingComplexityAnalysis.total_ops_direct(M, N_search, int(n)) if K == 1 
                             else BeamformingComplexityAnalysis.total_ops_parallel(M, N_search, int(n), K)/K 
                             for n in N_snapshots_range])
        lat = np.array([BeamformingComplexityAnalysis.latency_us(o, clock_MHz) for o in ops_array])
        label_txt = 'Direct (K=1)' if K == 1 else f'Parallel (K={K})'
        ax.loglog(N_snapshots_range, lat, 'o-', color=colors.get(K, 'gray'), label=label_txt, linewidth=2.5, markersize=8)
    ax.set_title('End-to-End Latency vs Snapshots', fontweight='bold')
    ax.set_xlabel('Number of Snapshots (N)', fontweight='bold')
    ax.set_ylabel('Latency (μs, log scale)', fontweight='bold')
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.legend(fontsize=16, loc='upper left')
    
    # ===== (1,1): Memory Footprint vs Snapshots =====
    ax = axes[1, 1]
    for K in K_values:
        mem = np.array([BeamformingComplexityAnalysis.memory_ops(M, int(n), K) for n in N_snapshots_range])
        label_txt = 'Direct (K=1)' if K == 1 else f'Parallel (K={K})'
        ax.loglog(N_snapshots_range, mem, 's-', color=colors.get(K, 'gray'), label=label_txt, linewidth=2.5, markersize=8)
    ax.fill_between(N_snapshots_range, 
                    np.array([BeamformingComplexityAnalysis.memory_ops(M, int(n), 1) for n in N_snapshots_range]),
                    np.array([BeamformingComplexityAnalysis.memory_ops(M, int(n), 2) for n in N_snapshots_range]),
                    alpha=0.2, color='green')
    ax.set_title('Memory Footprint vs Snapshots', fontweight='bold')
    ax.set_xlabel('Number of Snapshots (N)', fontweight='bold')
    ax.set_ylabel('Memory (MB, log scale)', fontweight='bold')
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.legend(fontsize=16, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('02_complexity_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*80)
    print("COMPLEXITY ANALYSIS - TIME-DOMAIN POLYPHASE DECOMPOSITION")
    print("="*80)
    print(f"Fixed Configuration: M=4 (array elements), N_search=181 (DOA search points)")
    print(f"Variable: Snapshots N from {int(N_snapshots_range[0]):,} to {int(N_snapshots_range[-1]):,}")
    print(f"Clock Rate: {clock_MHz} MHz, Ops/Cycle: 2")
    print("="*80)


def plot_beamforming_memory_analysis(N_snapshots_range=None, M=4, K_values=[1, 2]):
    """波束成形内存分析 (1×2 = 2个子图)"""
    if N_snapshots_range is None:
        N_snapshots_range = np.array([1e3, 5e3, 10e3, 50e3, 100e3, 200e3, 500e3, 1e6])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {1: 'black', 2: 'steelblue', 4: 'red', 8: 'green'}
    
    # ===== (0): Memory Footprint vs Snapshots =====
    ax = axes[0]
    for K in K_values:
        mem = np.array([BeamformingComplexityAnalysis.memory_ops(M, int(n), K) for n in N_snapshots_range])
        label_txt = 'Direct (K=1)' if K == 1 else f'Parallel (K={K})'
        ax.loglog(N_snapshots_range, mem, 'o-', color=colors.get(K, 'gray'), label=label_txt, 
                 linewidth=2.5, markersize=8)
    ax.fill_between(N_snapshots_range, 
                    np.array([BeamformingComplexityAnalysis.memory_ops(M, int(n), 1) for n in N_snapshots_range]),
                    np.array([BeamformingComplexityAnalysis.memory_ops(M, int(n), 2) for n in N_snapshots_range]),
                    alpha=0.2, color='green')
    ax.set_title('Memory Footprint vs Snapshots', fontweight='bold')
    ax.set_xlabel('Number of Snapshots (N)', fontweight='bold')
    ax.set_ylabel('Memory (MB, log scale)', fontweight='bold')
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.legend(fontsize=16, loc='upper left')
    
    # ===== (1): Memory Saving % vs Snapshots =====
    ax = axes[1]
    mem_direct = np.array([BeamformingComplexityAnalysis.memory_ops(M, int(n), 1) for n in N_snapshots_range])
    mem_parallel = np.array([BeamformingComplexityAnalysis.memory_ops(M, int(n), 2) for n in N_snapshots_range])
    saving = (mem_direct - mem_parallel) / mem_direct * 100
    
    ax.plot(N_snapshots_range, saving, 'o-', color=colors[2], linewidth=2.5, markersize=10, label='Memory Saving')
    ax.axhline(50, color='red', ls='--', linewidth=2, alpha=0.7, label='50% (K=2 bound)')
    ax.fill_between(N_snapshots_range, saving, 50, alpha=0.2, color='green')
    ax.set_title('Memory Reduction (K=2)', fontweight='bold')
    ax.set_xlabel('Number of Snapshots (N)', fontweight='bold')
    ax.set_ylabel('Memory Saving (%)', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=16)
    ax.set_xscale('log')
    ax.set_ylim([0, 60])
    
    plt.tight_layout()
    plt.savefig('03_memory_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*80)
    print("MEMORY ANALYSIS - TIME-DOMAIN DECIMATION")
    print("="*80)
    print(f"Fixed: M=4 elements")
    print(f"Variable: Snapshots N from {int(N_snapshots_range[0]):,} to {int(N_snapshots_range[-1]):,}")
    print("="*80)


def print_performance_summary(N_snapshots_values=[1e3, 10e3, 100e3, 1e6], M=4, N_search=181, K_values=[1, 2]):
    """打印性能统计表"""
    print("\n" + "="*130)
    print("BEAMFORMING PERFORMANCE SUMMARY - TIME-DOMAIN POLYPHASE DECOMPOSITION (M=4 FIXED)")
    print("="*130)
    print(f"Configuration: M=4 elements, N_search={N_search}, Clock=1000 MHz, Ops/Cycle=2")
    print("="*130)
    
    for N in N_snapshots_values:
        N_int = int(N)
        print(f"\n┌{'─'*124}┐")
        print(f"│ Snapshots N = {N_int:>10,d}{' '*99}│")
        print(f"├{'─'*124}┤")
        print(f"│ {'K':>3s} │ {'Operations':>20s} │ {'Latency (μs)':>15s} │ {'Memory (MB)':>15s} │ {'Speedup':>10s} │")
        print(f"├{'─'*124}┤")
        
        ops_direct = BeamformingComplexityAnalysis.total_ops_direct(M, N_search, N_int)
        lat_direct = BeamformingComplexityAnalysis.latency_us(ops_direct)
        mem_direct = BeamformingComplexityAnalysis.memory_ops(M, N_int, 1)
        
        print(f"│   1 │ {ops_direct:>20.3e} │ {lat_direct:>15.6f} │ {mem_direct:>15.3f} │ {'1.00x':>10s} │")
        
        for K in K_values[1:]:
            ops_parallel = BeamformingComplexityAnalysis.total_ops_parallel(M, N_search, N_int, K)
            lat_parallel = BeamformingComplexityAnalysis.latency_us(ops_parallel)
            mem_parallel = BeamformingComplexityAnalysis.memory_ops(M, N_int, K)
            speedup = ops_direct / ops_parallel
            
            print(f"│   {K} │ {ops_parallel:>20.3e} │ {lat_parallel:>15.6f} │ {mem_parallel:>15.3f} │ {speedup:>10.2f}x │")
        
        print(f"└{'─'*124}┘")
    
    print("\n" + "="*130 + "\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("PARALLEL BEAMFORMING - TIME-DOMAIN POLYPHASE DECOMPOSITION")
    print("="*80)
    
    np.random.seed(42)
    
    # ============================================================
    # [1] Generate test signal
    # ============================================================
    print("\n[1/5] Generating test signal...")
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
    print("\n[2/5] Processing with direct beamforming...")
    result_direct = beamformer.direct_beamforming(X, true_doa)
    print(f"  ✓ DOA estimation complete")
    print(f"  ✓ Beamforming complete")
    
    # ============================================================
    # [3] Parallel beamforming (K=2)
    # ============================================================
    print("\n[3/5] Processing with parallel beamforming (K=2)...")
    result_parallel = beamformer.parallel_beamforming(X, true_doa)
    print(f"  ✓ Channel 1 processed (N/2 snapshots)")
    print(f"  ✓ Channel 2 processed (N/2 snapshots)")
    print(f"  ✓ Results fused")
    
    # ============================================================
    # [4] Generate analysis plots
    # ============================================================
    print("\n[4/5] Generating DOA & Beampattern comparison (with polar plots)...")
    plot_doa_beampattern_comparison(result_direct, result_parallel, true_doa)
    
    print("\n[5/5] Generating complexity and memory analysis...")
    plot_beamforming_complexity_analysis(
        N_snapshots_range=np.array([1e3, 5e3, 10e3, 50e3, 100e3, 200e3, 500e3, 1e6]), 
        M=4, N_search=181, K_values=[1, 2])
    
    plot_beamforming_memory_analysis(
        N_snapshots_range=np.array([1e3, 5e3, 10e3, 50e3, 100e3, 200e3, 500e3, 1e6]), 
        M=4, K_values=[1, 2])
    
    # ============================================================
    # [5] Performance summary table
    # ============================================================
    print_performance_summary(N_snapshots_values=[1e3, 10e3, 100e3, 1e6], M=4, N_search=181, K_values=[1, 2])
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("Generated 3 analysis figures:")
    print("  ✓ 01_beamforming_comparison.png (2×3 subplots + Polar views)")
    print("  ✓ 02_complexity_analysis.png (2×2 subplots)")
    print("  ✓ 03_memory_analysis.png (1×2 subplots)")
    print("="*80 + "\n")
