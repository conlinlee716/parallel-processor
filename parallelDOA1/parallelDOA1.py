import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.linalg import inv
from scipy.signal import hilbert, chirp
import warnings

# 全局绘图配置
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['axes.unicode_minus'] = False

# =========================
# Parallel Beamforming (MVDR) with coherent fusion
# =========================

class BeamformingParallelProcessor:
    def __init__(self, seq_length, num_channels=2, num_arrays=16, num_sources=2,
                 diag_load=1e-3, fb_avg=True, hybrid_alpha=0.85):
        self.N = seq_length
        self.K = num_channels
        self.M = num_arrays
        self.P = num_sources
        self.diag_load = diag_load
        self.fb_avg = fb_avg
        self.hybrid_alpha = hybrid_alpha
        if int(np.log2(num_channels)) != np.log2(num_channels):
            raise ValueError("num_channels must be 2^N")

    def _forward_backward(self, R):
        if not self.fb_avg:
            return R
        J = np.fliplr(np.eye(self.M))
        R_fb = 0.5 * (R + J @ R.conj() @ J)
        return R_fb

    def _covariance(self, X):
        X = X - X.mean(axis=1, keepdims=True)
        L = X.shape[1]
        R = X @ X.conj().T / max(L, 1)
        R = self._forward_backward(R)
        lam = self.diag_load * (np.trace(R).real / self.M)
        R = R + lam * np.eye(self.M)
        return R

    def _normalize_a(self, a):
        nrm = np.linalg.norm(a)
        return a / (nrm + 1e-12)

    def _mvdr_spectrum_from_Rinv(self, R_inv, array_response, angles):
        spec = np.zeros(len(angles))
        for i, th in enumerate(angles):
            a = self._normalize_a(array_response(th, self.M))
            denom = np.real(a.conj().T @ R_inv @ a)
            spec[i] = 1.0 / (denom + 1e-12)
        return spec

    def _bartlett_spectrum_from_R(self, R, array_response, angles):
        spec = np.zeros(len(angles))
        for i, th in enumerate(angles):
            a = self._normalize_a(array_response(th, self.M))
            spec[i] = np.real(a.conj().T @ R @ a)
        return spec

    def _mvdr_bartlett_hybrid(self, mvdr_spec, bart_spec):
        b = bart_spec / (np.max(bart_spec) + 1e-12)
        m = mvdr_spec / (np.max(mvdr_spec) + 1e-12)
        return self.hybrid_alpha * m + (1 - self.hybrid_alpha) * b

    def compute_mvdr_weights(self, R_inv, steering_angle, array_response):
        a0 = self._normalize_a(array_response(steering_angle, self.M))
        w = R_inv @ a0
        denom = a0.conj().T @ w
        w = w / (denom + 1e-12)  # 指向角处增益=1
        return w

    def beam_pattern_from_weights(self, w, array_response, angles, steer_angle):
        B = np.zeros(len(angles))
        a0 = self._normalize_a(array_response(steer_angle, self.M))
        norm0 = np.abs(w.conj().T @ a0)**2 + 1e-12
        for i, th in enumerate(angles):
            a = self._normalize_a(array_response(th, self.M))
            B[i] = np.abs(w.conj().T @ a)**2 / norm0
        return B

    def direct_doa_estimation(self, signal_matrix, array_response, angles):
        R = self._covariance(signal_matrix)
        R_inv = inv(R)
        mvdr_spec = self._mvdr_spectrum_from_Rinv(R_inv, array_response, angles)
        bart_spec = self._bartlett_spectrum_from_R(R, array_response, angles)
        spec = self._mvdr_bartlett_hybrid(mvdr_spec, bart_spec)
        doa_idx = np.argsort(-spec)[:self.P]
        doa_angles = angles[doa_idx]
        return spec, doa_idx, doa_angles

    def parallel_doa_estimation(self, signal_matrix, array_response, angles):
        R_list, Rinv_list, L_list = [], [], []
        for k in range(self.K):
            Xk = signal_matrix[:, k::self.K]
            Lk = Xk.shape[1]
            Rk = self._covariance(Xk)
            Rinv_k = inv(Rk)
            R_list.append(Rk)
            Rinv_list.append(Rinv_k)
            L_list.append(Lk)

        L_arr = np.array(L_list, dtype=float)
        w_ch = L_arr / (np.sum(L_arr) + 1e-12)

        Rinv_recon = np.zeros_like(Rinv_list[0])
        R_recon = np.zeros_like(R_list[0])
        for wk, Rinv_k, Rk in zip(w_ch, Rinv_list, R_list):
            Rinv_recon += wk * Rinv_k
            R_recon += wk * Rk

        sub_specs = []
        for Rinv_k, Rk in zip(Rinv_list, R_list):
            mvdr_k = self._mvdr_spectrum_from_Rinv(Rinv_k, array_response, angles)
            bart_k = self._bartlett_spectrum_from_R(Rk, array_response, angles)
            sub_specs.append(self._mvdr_bartlett_hybrid(mvdr_k, bart_k))

        mvdr_rec = self._mvdr_spectrum_from_Rinv(Rinv_recon, array_response, angles)
        bart_rec = self._bartlett_spectrum_from_R(R_recon, array_response, angles)
        reconstructed_spectrum = self._mvdr_bartlett_hybrid(mvdr_rec, bart_rec)

        doa_idx = np.argsort(-reconstructed_spectrum)[:self.P]
        doa_angles = angles[doa_idx]
        return reconstructed_spectrum, doa_idx, doa_angles, sub_specs

# =========================
# Array signal generator (center-referenced steering + analytic signals)
# =========================
class SignalGenerator:
    def __init__(self, fs=40e9, M=8, fc=10e9):
        self.fs = fs
        self.M = M
        self.c = 3e8
        self.fc = fc
        self.wavelength = self.c / self.fc
        self.d = self.wavelength / 2
        self.rng = np.random.default_rng(2024)

    def generate_signal(self, signal_type, duration=100e-9, num_sources=1,
                        source_angles=None, source_powers=None, noise_std=0.08):
        L = int(duration * self.fs)
        t = np.arange(L) / self.fs
        if source_angles is None:
            source_angles = np.array([45])
        if source_powers is None:
            source_powers = np.ones(len(source_angles))

        sigs = []
        if signal_type == "sinusoid_1GHz":
            base = np.sin(2*np.pi*1e9*t)
            base_ana = hilbert(base)
            for p in source_powers:
                sigs.append(np.sqrt(p) * base_ana)
            name = "1 GHz Sinusoid Sources"
        elif signal_type == "lfm_10_18GHz":
            f0, f1 = 10e9, 18e9
            base = chirp(t, f0, t[-1], f1, method='linear')
            base_ana = hilbert(base)
            for p in source_powers:
                sigs.append(np.sqrt(p) * base_ana)
            name = "10-18 GHz LFM Sources"
        else:
            raise ValueError("signal_type must be 'sinusoid_1GHz' or 'lfm_10_18GHz'")

        X = self._array_mix(sigs, source_angles)
        noise = noise_std * (self.rng.standard_normal((self.M, L)) + 1j*self.rng.standard_normal((self.M, L)))
        X = X + noise
        return X, np.array(source_angles), name, np.array(source_powers)

    def _array_mix(self, source_signals, source_angles):
        L = len(source_signals[0])
        X = np.zeros((self.M, L), dtype=complex)
        m_center = (self.M - 1) / 2.0
        for m in range(self.M):
            for s, th in enumerate(source_angles):
                phase = 2*np.pi*(m - m_center)*self.d*np.sin(np.radians(th))/self.wavelength
                a_m = np.exp(1j*phase)
                X[m, :] += a_m * source_signals[s]
        return X

    def array_response(self, angle, M):
        a = np.zeros(M, dtype=complex)
        m_center = (M - 1) / 2.0
        for m in range(M):
            phase = 2*np.pi*(m - m_center)*self.d*np.sin(np.radians(angle))/self.wavelength
            a[m] = np.exp(1j*phase)
        return a / (np.linalg.norm(a) + 1e-12)

# =========================
# Complexity & Latency Analysis for MVDR + Coherent Fusion
# =========================
class ComplexityAnalysis:
    @staticmethod
    def ops_covariance(N_snap, M):
        return M**2 * N_snap
    @staticmethod
    def ops_fb_avg(M):
        return 2 * M**2
    @staticmethod
    def ops_inversion(M):
        return (2.0/3.0) * M**3
    @staticmethod
    def ops_spectrum_eval(M, A):
        return A * (M**2 + M)
    @staticmethod
    def ops_fusion(M, K):
        return 2 * K * (M**2)
    @staticmethod
    def compute_ops_direct(N, M, A):
        ops_cov = ComplexityAnalysis.ops_covariance(N, M) + ComplexityAnalysis.ops_fb_avg(M)
        ops_inv = ComplexityAnalysis.ops_inversion(M)
        ops_spec = ComplexityAnalysis.ops_spectrum_eval(M, A) * 2
        return ops_cov + ops_inv + ops_spec
    @staticmethod
    def compute_ops_parallel(N, M, K, A):
        N_sub = N // K
        ops_cov_ch = ComplexityAnalysis.ops_covariance(N_sub, M) + ComplexityAnalysis.ops_fb_avg(M)
        ops_inv_ch = ComplexityAnalysis.ops_inversion(M)
        ops_channels_total = K * (ops_cov_ch + ops_inv_ch)
        ops_fus = ComplexityAnalysis.ops_fusion(M, K)
        ops_spec_final = ComplexityAnalysis.ops_spectrum_eval(M, A) * 2
        return ops_channels_total + ops_fus + ops_spec_final
    @staticmethod
    def latency_direct(N, M, A, clock_rate_GOPS):
        ops = ComplexityAnalysis.compute_ops_direct(N, M, A)
        return ops / (clock_rate_GOPS * 1e9)
    @staticmethod
    def latency_parallel(N, M, K, A, clock_rate_GOPS, parallel_cores=None):
        if parallel_cores is None:
            parallel_cores = K
        N_sub = N // K
        ops_cov_ch = ComplexityAnalysis.ops_covariance(N_sub, M) + ComplexityAnalysis.ops_fb_avg(M)
        ops_inv_ch = ComplexityAnalysis.ops_inversion(M)
        ops_ch = ops_cov_ch + ops_inv_ch
        batches = int(np.ceil(K / float(parallel_cores)))
        ops_channels_parallel = batches * ops_ch
        ops_serial = ComplexityAnalysis.ops_fusion(M, K) + ComplexityAnalysis.ops_spectrum_eval(M, A) * 2
        ops_total = ops_channels_parallel + ops_serial
        return ops_total / (clock_rate_GOPS * 1e9)

# =========================
# Plotting: Complexity and Latency Advantages
# =========================
def plot_complexity_and_latency():
    seq_lengths = np.array([256, 512, 1024, 2048, 4096, 8192])
    M = 8
    A = 181
    K = 2
    clock_rate_GOPS = 50.0
    cores = 2

    ops_direct = np.array([ComplexityAnalysis.compute_ops_direct(N, M, A) for N in seq_lengths])
    ops_parallel = np.array([ComplexityAnalysis.compute_ops_parallel(N, M, K, A) for N in seq_lengths])

    lat_direct = np.array([ComplexityAnalysis.latency_direct(N, M, A, clock_rate_GOPS) for N in seq_lengths])
    lat_parallel = np.array([ComplexityAnalysis.latency_parallel(N, M, K, A, clock_rate_GOPS, cores) for N in seq_lengths])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.semilogy(seq_lengths, ops_direct, 'o-', label='Direct', linewidth=2, markersize=8)
    ax.semilogy(seq_lengths, ops_parallel, 's-', label=f'Parallel (K={K})', linewidth=2, markersize=8)
    ax.set_xlabel('Snapshot Length N')
    ax.set_ylabel('Operations (log scale)')
    ax.set_title(f'Computational Complexity (M={M}, A={A})')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    improvement_ops = (ops_direct - ops_parallel) / ops_direct * 100
    ax.plot(seq_lengths, improvement_ops, 'o-', color='green', linewidth=2, markersize=8)
    ax.axhline(np.mean(improvement_ops), color='r', linestyle='--', label=f'Mean: {np.mean(improvement_ops):.1f}%')
    ax.set_xlabel('Snapshot Length N')
    ax.set_ylabel('Reduction (%)')
    ax.set_title('Complexity Reduction with Parallel Fusion')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.semilogy(seq_lengths, lat_direct * 1e6, 'o-', label='Direct', linewidth=2, markersize=8)
    ax.semilogy(seq_lengths, lat_parallel * 1e6, 's-', label=f'Parallel (K={K}, cores={cores})', linewidth=2, markersize=8)
    ax.set_xlabel('Snapshot Length N')
    ax.set_ylabel('Latency (μs, log scale)')
    ax.set_title('End-to-End Latency')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    improvement_lat = (lat_direct - lat_parallel) / lat_direct * 100
    ax.plot(seq_lengths, improvement_lat, 's-', color='purple', linewidth=2, markersize=8)
    ax.axhline(np.mean(improvement_lat), color='r', linestyle='--', label=f'Mean: {np.mean(improvement_lat):.1f}%')
    ax.set_xlabel('Snapshot Length N')
    ax.set_ylabel('Reduction (%)')
    ax.set_title('Latency Reduction with Parallel Fusion')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('doa_parallel_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_antenna_elements_effect():
    seq_length = 4096
    antenna_counts = np.array([4, 8, 16, 32, 64])
    A = 181
    K = 2
    clock_rate_GOPS = 50.0
    cores = 2

    ops_direct = np.array([ComplexityAnalysis.compute_ops_direct(seq_length, m, A) for m in antenna_counts])
    ops_parallel = np.array([ComplexityAnalysis.compute_ops_parallel(seq_length, m, K, A) for m in antenna_counts])

    lat_direct = np.array([ComplexityAnalysis.latency_direct(seq_length, m, A, clock_rate_GOPS) for m in antenna_counts])
    lat_parallel = np.array([ComplexityAnalysis.latency_parallel(seq_length, m, K, A, clock_rate_GOPS, cores) for m in antenna_counts])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.semilogy(antenna_counts, ops_direct, 'o-', label='Direct', linewidth=2, markersize=8)
    ax.semilogy(antenna_counts, ops_parallel, 's-', label=f'Parallel (K={K})', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Antenna Elements M')
    ax.set_ylabel('Operations (log scale)')
    ax.set_title(f'Complexity vs M (N={seq_length}, A={A})')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    improvement = (ops_direct - ops_parallel) / ops_direct * 100
    ax.bar(range(len(antenna_counts)), improvement, color='skyblue', alpha=0.8, edgecolor='black')
    ax.set_xlabel('M'); ax.set_ylabel('Reduction (%)')
    ax.set_title('Complexity Improvement with Parallelism')
    ax.set_xticks(range(len(antenna_counts)))
    ax.set_xticklabels([str(m) for m in antenna_counts])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('antenna_elements_effect.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_multicore_efficiency():
    seq_length = 4096
    M = 8
    A = 181
    K_values = np.array([1, 2, 4, 8])
    clock_rate_GOPS = 50.0

    baseline = ComplexityAnalysis.compute_ops_direct(seq_length, M, A)
    parallel_ops = np.array([ComplexityAnalysis.compute_ops_parallel(seq_length, M, K, A) for K in K_values])
    efficiency = baseline / parallel_ops * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(K_values, efficiency, 'o-', linewidth=2.5, markersize=10, color='darkblue')
    ax.axhline(100, color='r', linestyle='--', alpha=0.5, label='Direct baseline')
    ax.set_xlabel('Parallelism Level K')
    ax.set_ylabel('Parallel Efficiency (%)')
    ax.set_title('DOA Parallel Efficiency (Ops baseline/direct)')
    ax.grid(True, alpha=0.3); ax.legend()

    latencies = np.array([ComplexityAnalysis.latency_parallel(seq_length, M, K, A, clock_rate_GOPS, parallel_cores=K)
                          for K in K_values])
    lat_direct = ComplexityAnalysis.latency_direct(seq_length, M, A, clock_rate_GOPS)

    ax = axes[1]
    ax.plot(K_values, latencies * 1e6, 's-', linewidth=2.5, markersize=10, color='purple', label='Parallel (μs)')
    ax.axhline(lat_direct * 1e6, color='gray', linestyle='--', label='Direct (μs)')
    ax.set_xlabel('Parallelism Level K'); ax.set_ylabel('Latency (μs)')
    ax.set_title(f'Latency vs Parallelism (N={seq_length}, M={M}, A={A})')
    ax.grid(True, alpha=0.3); ax.legend()

    plt.tight_layout()
    plt.savefig('doa_multicore_efficiency.png', dpi=300, bbox_inches='tight')
    plt.show()

# =========================
# Spectrum + Beampattern demo
# =========================
def plot_parallel_doa_results(signal_matrix, source_angles, signal_name, processor, gen):
    angles = np.arange(-90, 91, 1)
    spec_d, idx_d, doa_d = processor.direct_doa_estimation(signal_matrix, gen.array_response, angles)
    spec_p, idx_p, doa_p, sub_specs = processor.parallel_doa_estimation(signal_matrix, gen.array_response, angles)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax = axes[0, 0]
    ax.plot(angles, spec_d, lw=2, label='Direct (Hybrid MVDR)')
    for th in source_angles: ax.axvline(th, color='r', ls='--', alpha=0.7)
    ax.set_xlabel('Angle (deg)'); ax.set_ylabel('Spectrum')
    ax.set_title('Direct DOA'); ax.set_xlim([-90, 90]); ax.grid(True); ax.legend()

    ax = axes[0, 1]
    ax.plot(angles, sub_specs[0], lw=2, color='green', label='Ch1 Hybrid')
    for th in source_angles: ax.axvline(th, color='r', ls='--', alpha=0.7)
    ax.set_title('Channel 1'); ax.set_xlim([-90, 90]); ax.grid(True); ax.legend()

    ax = axes[1, 0]
    ax.plot(angles, sub_specs[1], lw=2, color='purple', label='Ch2 Hybrid')
    for th in source_angles: ax.axvline(th, color='r', ls='--', alpha=0.7)
    ax.set_title('Channel 2'); ax.set_xlim([-90, 90]); ax.grid(True); ax.legend()

    ax = axes[1, 1]
    ax.plot(angles, spec_d, lw=2, ls='--', label='Direct')
    ax.plot(angles, spec_p, lw=2, label='Reconstructed (Coherent Fusion)')
    for th in source_angles: ax.axvline(th, color='k', ls=':', alpha=0.6)
    ax.set_title('Reconstructed vs Direct'); ax.set_xlim([-90, 90]); ax.grid(True); ax.legend()

    # ===== 波束图 =====
    angles_bp = np.linspace(-90, 90, 721)
    angles_bp_rad = np.deg2rad(angles_bp)
    R = processor._covariance(signal_matrix)
    R_inv = inv(R)
    steer_angle = float(source_angles[0])

    w_direct = processor.compute_mvdr_weights(R_inv, steer_angle, gen.array_response)

    Rinv_list, L_list = [], []
    for k in range(processor.K):
        Xk = signal_matrix[:, k::processor.K]
        Rk = processor._covariance(Xk)
        Rinv_k = inv(Rk)
        Rinv_list.append(Rinv_k)
        L_list.append(Xk.shape[1])
    L_arr = np.array(L_list, dtype=float)
    w_coeff = L_arr / (np.sum(L_arr) + 1e-12)
    Rinv_recon = np.zeros_like(Rinv_list[0])
    for wk, Rinv_k in zip(w_coeff, Rinv_list):
        Rinv_recon += wk * Rinv_k
    w_parallel = processor.compute_mvdr_weights(Rinv_recon, steer_angle, gen.array_response)

    B_direct = processor.beam_pattern_from_weights(w_direct, gen.array_response, angles_bp, steer_angle)
    B_parallel = processor.beam_pattern_from_weights(w_parallel, gen.array_response, angles_bp, steer_angle)

    fig.delaxes(axes[1, 1])
    axb = fig.add_subplot(2, 2, 4, projection='polar')
    axb.plot(angles_bp_rad, 10*np.log10(B_direct + 1e-12), lw=2, label='Direct MVDR', color='blue')
    axb.plot(angles_bp_rad, 10*np.log10(B_parallel + 1e-12), lw=2, label='Parallel (Fused) MVDR', color='red')
    for th in source_angles:
        axb.plot(np.deg2rad([th, th]), [-60, 6], ls='--', color='k', alpha=0.6)
    axb.set_theta_zero_location('N')
    axb.set_theta_direction(-1)
    try:
        axb.set_thetamin(-90); axb.set_thetamax(90)
    except Exception:
        pass
    axb.set_rlim(-60, 6)
    axb.set_title(f'Beampattern (steer {steer_angle}°)', va='bottom')
    axb.grid(True, alpha=0.3)
    axb.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig(f'parallel_doa_{signal_name.replace(" ", "_")}_with_beampattern.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"{signal_name}: Direct DOA {doa_d}, Parallel DOA {doa_p}")

# =========================
# Demo
# =========================
def demo_spectra():
    fs = 40e9
    duration = 100e-9
    seq_length = int(duration * fs)
    M = 8
    P = 1
    gen = SignalGenerator(fs, M, fc=10e9)
    proc = BeamformingParallelProcessor(seq_length, num_channels=2,
                                        num_arrays=M, num_sources=P,
                                        diag_load=1e-3, fb_avg=True, hybrid_alpha=0.85)

    for sig_type in ["sinusoid_1GHz", "lfm_10_18GHz"]:
        X, src_angles, name, _ = gen.generate_signal(sig_type, duration,
                                                     num_sources=1,
                                                     source_angles=np.array([45]),
                                                     source_powers=np.array([1.0]),
                                                     noise_std=0.08)
        plot_parallel_doa_results(X, src_angles, name, proc, gen)

if __name__ == "__main__":
    demo_spectra()
    plot_complexity_and_latency()
    plot_antenna_elements_effect()
    plot_multicore_efficiency()
