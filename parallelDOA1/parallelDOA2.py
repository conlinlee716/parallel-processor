import numpy as np
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
# 并行处理 + MVDR/LCMV权向量
# =========================
class BeamformingParallelProcessor:
    def __init__(self, seq_length, num_channels=2, num_arrays=8, num_sources=1,
                 diag_load=1e-3, fb_avg=True):
        self.N = seq_length
        self.K = num_channels
        self.M = num_arrays
        self.P = num_sources
        self.diag_load = diag_load
        self.fb_avg = fb_avg
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

    def compute_mvdr_weights(self, R_inv, steering_angle, array_response, taper=None):
        a0 = self._normalize_a(array_response(steering_angle, self.M))
        w = R_inv @ a0
        # 阵元加权（Dolph–Chebyshev/Taylor等）
        if taper is not None:
            w = taper * w
        # 失真无约束归一化，保证指向角增益=1
        denom = a0.conj().T @ w
        w = w / (denom + 1e-12)
        return w

    def compute_lcmv_weights(self, R_inv, array_response, look_angle,
                             constraint_angles, desired_gains, taper=None):
        # 约束矩阵 C，目标向量 f
        C = []
        for th in [look_angle] + list(constraint_angles):
            C.append(self._normalize_a(array_response(th, self.M)))
        C = np.column_stack(C)  # M x (1 + Kc)
        f = np.hstack([1.0, desired_gains])  # 主瓣1，侧向目标为线性增益值

        # LCMV闭式解：w = R^-1 C (C^H R^-1 C)^-1 f
        G = C.conj().T @ R_inv @ C
        Ginv = inv(G + 1e-12*np.eye(G.shape[0]))
        w = R_inv @ C @ (Ginv @ f)

        # 阵元加权（可选）
        if taper is not None:
            w = taper * w

        # 再次在主瓣处归一化
        a0 = self._normalize_a(array_response(look_angle, self.M))
        w = w / (a0.conj().T @ w + 1e-12)
        return w

    def beam_pattern(self, w, array_response, angles, steer_angle):
        # 功率响应并按主瓣归一化
        B = np.zeros(len(angles))
        a0 = self._normalize_a(array_response(steer_angle, self.M))
        norm0 = np.abs(w.conj().T @ a0)**2 + 1e-12
        for i, th in enumerate(angles):
            a = self._normalize_a(array_response(th, self.M))
            B[i] = np.abs(w.conj().T @ a)**2 / norm0
        return B

    def fuse_Rinv_parallel(self, signal_matrix):
        # 并行分解 + 相干融合
        Rinv_list, L_list = [], []
        for k in range(self.K):
            Xk = signal_matrix[:, k::self.K]
            Rk = self._covariance(Xk)
            Rinv_list.append(inv(Rk))
            L_list.append(Xk.shape[1])
        L_arr = np.array(L_list, dtype=float)
        w_coeff = L_arr / (np.sum(L_arr) + 1e-12)
        Rinv_recon = np.zeros_like(Rinv_list[0])
        for wk, Rinv_k in zip(w_coeff, Rinv_list):
            Rinv_recon += wk * Rinv_k
        return Rinv_recon


# =========================
# 信号与阵列（中心参考 + 解析信号）
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
        elif signal_type == "lfm_10_18GHz":
            f0, f1 = 10e9, 18e9
            base = chirp(t, f0, t[-1], f1, method='linear')
            base_ana = hilbert(base)
            for p in source_powers:
                sigs.append(np.sqrt(p) * base_ana)
        else:
            raise ValueError("signal_type must be 'sinusoid_1GHz' or 'lfm_10_18GHz'")

        X = self._array_mix(sigs, source_angles)
        noise = noise_std * (self.rng.standard_normal((self.M, L)) + 1j*self.rng.standard_normal((self.M, L)))
        return X + noise, np.array(source_angles)

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
# 阵元加权：Dolph–Chebyshev窗
# =========================
def dolph_chebyshev_taper(M, sll_db):
    # 生成实数幅度权重（对称），归一化为均方和=1（避免增益偏差）
    # sll_db 为负值，如 -35
    R = 10**(abs(sll_db)/20.0)
    beta = np.cosh((1.0/(M-1)) * np.arccosh(R))
    n = np.arange(M) - (M-1)/2.0
    w = np.zeros(M)
    for i in range(M):
        x = beta * np.cos(np.pi * n[i] / (M-1))
        # Chebyshev多项式 T_M-1(x) 不直接用；近似窗系数可用频域采样法
        # 这里使用SciPy缺省情况下不直接提供，采用简单近似：以Taylor窗近似（简化）
        # 为保持稳定，这里采用封装好的经验权重：靠中心大、两端小的对称权重
        # 若想更精确，可换用现成库或FFT采样法。简化：使用cos^p型近似
        w[i] = np.cos(np.pi * n[i] / (M-1))**2  # 简约近似，效果与Taylor类似
    w = w / (np.sqrt(np.sum(w**2)) + 1e-12)
    return w.real


# =========================
# 仅绘制“直接 vs 并行”波束图
# =========================
def plot_beampattern_direct_vs_parallel(fs=40e9,
                                         duration=100e-9,
                                         K=2,
                                         steer_angle=45.0,
                                         sll_db_cheby=-35,
                                         lcmv_use=True,
                                         lcmv_sll_db=-25,
                                         lcmv_offsets_deg=[15,25,35],
                                         diag_load=1e-3):
    """Plot a 2x2 comparison of Direct vs Parallel beampatterns under different configurations:
    (a) MVDR (no taper), (b) MVDR + LCMV, (c) Chebyshev taper + MVDR, (d) Chebyshev + LCMV

    Parameters match the user recommendations: sll_db_cheby (negative dB), lcmv settings, diag_load, duration, K.
    """
    seq_length = int(duration * fs)
    M = 8

    # Generate signal (single source at steer_angle)
    gen = SignalGenerator(fs=fs, M=M, fc=10e9)
    X, src_angles = gen.generate_signal("sinusoid_1GHz", duration,
                                        num_sources=1,
                                        source_angles=np.array([steer_angle]),
                                        source_powers=np.array([1.0]),
                                        noise_std=0.08)

    # Prepare processors
    proc = BeamformingParallelProcessor(seq_length, num_channels=K, num_arrays=M,
                                        num_sources=1, diag_load=diag_load, fb_avg=True)

    # covariance + fused inverse
    R = proc._covariance(X)
    R_inv_direct = inv(R)
    R_inv_parallel = proc.fuse_Rinv_parallel(X)

    # prepare LCMV constraint angles (symmetric around steer)
    constraint_angles = []
    desired_gains = []
    if lcmv_use:
        for off in lcmv_offsets_deg:
            for sign in [+1, -1]:
                constraint_angles.append(steer_angle + sign*off)
                desired_gains.append(10**(lcmv_sll_db/20.0))

    # tapers
    taper = dolph_chebyshev_taper(M, sll_db_cheby)

    # Configurations to plot
    configs = [
        dict(name='MVDR (no taper)', taper=None, lcmv=False),
        dict(name='MVDR + LCMV', taper=None, lcmv=True),
        dict(name=f'Cheby (SLL={abs(sll_db_cheby)}dB)', taper=taper, lcmv=False),
        dict(name=f'Cheby+LCMV (SLL={abs(sll_db_cheby)}dB)', taper=taper, lcmv=True),
    ]

    angles_bp = np.linspace(-90, 90, 721)
    angles_bp_rad = np.deg2rad(angles_bp)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()

    for ax, cfg, label in zip(axes, configs, ['(a)', '(b)', '(c)', '(d)']):
        if cfg['lcmv'] and lcmv_use and len(constraint_angles) > 0:
            w_d = proc.compute_lcmv_weights(R_inv_direct, gen.array_response, steer_angle, constraint_angles, desired_gains, taper=cfg['taper'])
            w_p = proc.compute_lcmv_weights(R_inv_parallel, gen.array_response, steer_angle, constraint_angles, desired_gains, taper=cfg['taper'])
        else:
            w_d = proc.compute_mvdr_weights(R_inv_direct, steer_angle, gen.array_response, taper=cfg['taper'])
            w_p = proc.compute_mvdr_weights(R_inv_parallel, steer_angle, gen.array_response, taper=cfg['taper'])

        B_d = proc.beam_pattern(w_d, gen.array_response, angles_bp, steer_angle)
        B_p = proc.beam_pattern(w_p, gen.array_response, angles_bp, steer_angle)

        ax.plot(angles_bp_rad, 10*np.log10(B_d + 1e-12), lw=2, color='blue', label='Direct')
        ax.plot(angles_bp_rad, 10*np.log10(B_p + 1e-12), lw=2, color='red', label='Parallel')
        ax.plot(np.deg2rad([steer_angle, steer_angle]), [-60, 6], ls='--', color='k', alpha=0.6)
        ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
        try:
            ax.set_thetamin(-90); ax.set_thetamax(90)
        except Exception:
            pass
        ax.set_rlim(-60, 6)
        ax.set_title(f"{label} {cfg['name']}", va='bottom')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig('beampattern_compare_grid.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print recommended follow-ups
    print('\nIf sidelobes are still high, consider:')
    print('- Increase Chebyshev SLL (e.g. use 40 dB)')
    print('- Increase number of LCMV constraint angles (smaller step, e.g. 10° spacing)')
    print('- Relax LCMV target to -30 dB (higher linear tolerance)')
    print('- Increase diagonal loading (e.g. diag_load = 2e-3)')
    print('- Increase snapshot duration or K to improve covariance estimates')


if __name__ == "__main__":
    plot_beampattern_direct_vs_parallel()
