import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, convolve
import warnings

# 配置全局绘图字体为 Arial，统一字号为 20，并抑制字体警告
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

# ==================== 基础函数 ====================
def matched_filter_from_ref(ref):
    return np.conj(ref[::-1]).astype(np.complex128)

def polyphase_pulse_compress_detailed(x, ref, K):
    """K-way parallel pulse compression with detailed intermediate results"""
    x = np.asarray(x, dtype=np.complex128)
    h = matched_filter_from_ref(np.asarray(ref, dtype=np.complex128))
    N, L = len(x), len(h)
    
    if K <= 0 or N == 0 or L == 0:
        return np.zeros(N + L - 1, dtype=np.complex128), {}
    
    # ===== Step 1: Polyphase decomposition =====
    x_branches = [x[j::K] for j in range(K)]
    h_branches = [h[j::K] for j in range(K)]
    
    # ===== Step 2: K×K branch convolutions =====
    branch_convs = {}  # Store all sub-convolutions
    out_len = (N + L - 1 + K - 1) // K + 1
    y_phase = [np.zeros(out_len, dtype=np.complex128) for _ in range(K)]
    
    for i in range(K):
        for j in range(K):
            r = (i - j) % K
            g_ij = h_branches[r]
            if g_ij.size == 0:
                continue
            
            # Low-rate convolution
            c_ij = convolve(x_branches[j], g_ij)
            branch_convs[f'c_{i}{j}'] = c_ij
            
            # One-tap delay if i < j
            if i < j:
                c_ij = np.concatenate([np.zeros(1, dtype=np.complex128), c_ij])
            
            # Accumulate to phase i
            if len(c_ij) > len(y_phase[i]):
                y_phase[i] = np.pad(y_phase[i], (0, len(c_ij) - len(y_phase[i])))
            y_phase[i][:len(c_ij)] += c_ij
    
    # ===== Step 3: Phase interleaving (reconstruction) =====
    y = np.zeros(N + L - 1, dtype=np.complex128)
    for i in range(K):
        n_idx = i + np.arange(len(y_phase[i])) * K
        valid = n_idx < len(y)
        y[n_idx[valid]] += y_phase[i][valid]
    
    # Return with intermediate results
    intermediate = {
        'x_branches': x_branches,
        'h_branches': h_branches,
        'branch_convs': branch_convs,
        'y_phase': y_phase,
    }
    return y, intermediate

def gen_lfm(fs, f0, f1, duration):
    t = np.arange(int(duration * fs)) / fs
    s = chirp(t, f0=f0, t1=t[-1], f1=f1, method='linear')
    return s.astype(np.complex128), t

def gen_received(signal_tx, fs, delays_s, amplitudes, snr_db=30):
    Ntx = len(signal_tx)
    max_delay = max(delays_s) if len(delays_s) else 0.0
    N = int(np.ceil(max_delay * fs)) + Ntx
    x = np.zeros(N, dtype=np.complex128)
    for a, d in zip(amplitudes, delays_s):
        k = int(np.round(d * fs))
        if k + Ntx <= N:
            x[k:k+Ntx] += a * signal_tx
    snr_lin = 10**(snr_db/10)
    p_sig = np.mean(np.abs(x)**2) + 1e-12
    p_n = p_sig / snr_lin
    noise = np.sqrt(p_n/2) * (np.random.randn(N) + 1j*np.random.randn(N))
    return x + noise

# ==================== 详细过程可视化 ====================
def plot_pulse_compress_detailed(x, ref, K=2, fs=40e9):
    """
    展示脉冲压缩的完整多相处理过程：
    - 输入与滤波器的多相分解
    - K×K 子通道卷积结果
    - 相位累积
    - 交织重构与对比

    参数:
    --------
    fs : float
        采样率（Hz），默认 40e9 (40 GHz)
    """
    h = matched_filter_from_ref(ref)
    N, L = len(x), len(h)
    
    # 直接脉冲压缩（基准）
    y_direct = convolve(x, h)
    
    # 多相并行脉冲压缩
    y_poly, inter = polyphase_pulse_compress_detailed(x, ref, K)
    
    # 对齐长度并生成时间轴（秒和纳秒）
    P = min(len(y_direct), len(y_poly))
    y_direct, y_poly = y_direct[:P], y_poly[:P]
    t_full = np.arange(P) / fs
    t_full_ns = t_full * 1e9
    
    # 统计误差
    mse = np.mean(np.abs(y_direct - y_poly)**2)
    mae = np.mean(np.abs(y_direct - y_poly))
    max_err = np.max(np.abs(y_direct - y_poly))
    
    x_branches = inter['x_branches']
    h_branches = inter['h_branches']
    branch_convs = inter['branch_convs']
    y_phase = inter['y_phase']
    
    # ===== Figure 1: Input & Filter Decomposition =====
    fig1, axes1 = plt.subplots(K, 2, figsize=(14, 4*K))
    if K == 1:
        axes1 = axes1.reshape(1, -1)
    
    for k in range(K):
        # Input branch k
        t_xb = (k + np.arange(len(x_branches[k])) * K) / fs * 1e9  # ns
        axes1[k, 0].plot(t_xb, np.abs(x_branches[k]), lw=1.5, color='steelblue', label=f'Input x_{k}')
        axes1[k, 0].set_title(f'Input Branch {k} (length={len(x_branches[k])})', fontsize=20, fontweight='bold')
        axes1[k, 0].set_xlabel('Time (ns)')
        axes1[k, 0].set_ylabel('Magnitude')
        axes1[k, 0].grid(True, alpha=0.3)
        axes1[k, 0].legend()
        
        # Filter branch k
        t_hb = (k + np.arange(len(h_branches[k])) * K) / fs * 1e9  # ns
        axes1[k, 1].plot(t_hb, np.abs(h_branches[k]), lw=1.5, color='darkorange', label=f'Filter h_{k}')
        axes1[k, 1].set_title(f'Filter Branch {k} (length={len(h_branches[k])})', fontsize=20, fontweight='bold')
        axes1[k, 1].set_xlabel('Time (ns)')
        axes1[k, 1].set_ylabel('Magnitude')
        axes1[k, 1].grid(True, alpha=0.3)
        axes1[k, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f'pulse_compress_01_decomposition_K{K}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ===== Figure 2: K×K Sub-channel Convolutions =====
    fig2, axes2 = plt.subplots(K, K, figsize=(14, 4*K))
    if K == 1:
        axes2 = axes2.reshape(1, 1)
    elif K == 2:
        if axes2.ndim == 1:
            axes2 = axes2.reshape(-1, 1)
    
    for i in range(K):
        for j in range(K):
            key = f'c_{i}{j}'
            c_ij = branch_convs.get(key, np.array([]))
            
            ax = axes2[i, j] if K > 1 else axes2
            if c_ij.size > 0:
                t_cij = (i + np.arange(len(c_ij)) * K) / fs * 1e9  # ns
                ax.plot(t_cij, np.abs(c_ij), lw=1, color='green', alpha=0.8)
            ax.set_title(f'c_{i}{j}: x_{j} * h_{(i-j)%K}  (len={len(c_ij)})', fontsize=20, fontweight='bold')
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel('Magnitude')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'pulse_compress_02_branch_convs_K{K}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ===== Figure 3: Phase Accumulation & Interleaving =====
    fig3, axes3 = plt.subplots(K+1, 1, figsize=(14, 3*(K+1)))
    if K == 1:
        axes3 = [axes3]
    
    # 显示每个相位的累积结果
    for i in range(K):
        t_phase = (i + np.arange(len(y_phase[i])) * K) / fs * 1e9  # ns
        axes3[i].plot(t_phase, np.abs(y_phase[i]), lw=1.5, color='purple', alpha=0.8, label=f'y_phase[{i}]')
        axes3[i].set_title(f'Phase {i} Accumulated Result (length={len(y_phase[i])})', fontsize=20, fontweight='bold')
        axes3[i].set_xlabel('Time (ns)')
        axes3[i].set_ylabel('Magnitude')
        axes3[i].grid(True, alpha=0.3)
        axes3[i].legend()
    
    # 重构结果
    axes3[K].plot(t_full_ns, np.abs(y_poly), lw=1.5, color='red', alpha=0.7, label='Reconstructed (Polyphase)')
    axes3[K].plot(t_full_ns, np.abs(y_direct), lw=1, ls='--', color='black', alpha=0.6, label='Direct Convolution')
    axes3[K].set_title('Phase Interleaving Result vs Direct Convolution', fontsize=20, fontweight='bold')
    axes3[K].set_xlabel('Time (ns)')
    axes3[K].set_ylabel('Magnitude')
    axes3[K].grid(True, alpha=0.3)
    axes3[K].legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'pulse_compress_03_reconstruction_K{K}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ===== Figure 4: Overall Comparison & Error Analysis =====
    fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))
    
    # 子图1：直接脉冲压缩
    axes4[0, 0].plot(t_full_ns, np.abs(y_direct), lw=1.5, color='steelblue', label='Direct')
    axes4[0, 0].set_title('Direct Convolution Result', fontsize=20, fontweight='bold')
    axes4[0, 0].set_xlabel('Time (ns)')
    axes4[0, 0].set_ylabel('Magnitude')
    axes4[0, 0].grid(True, alpha=0.3)
    axes4[0, 0].legend()
    
    # 子图2：并行重构
    axes4[0, 1].plot(t_full_ns, np.abs(y_poly), lw=1.5, color='red', label=f'Polyphase K={K}')
    axes4[0, 1].set_title('Polyphase Reconstruction Result', fontsize=20, fontweight='bold')
    axes4[0, 1].set_xlabel('Time (ns)')
    axes4[0, 1].set_ylabel('Magnitude')
    axes4[0, 1].grid(True, alpha=0.3)
    axes4[0, 1].legend()
    
    # 子图3：叠加对比
    axes4[1, 0].plot(t_full_ns, np.abs(y_direct), lw=1.5, alpha=0.7, label='Direct', color='steelblue')
    axes4[1, 0].plot(t_full_ns, np.abs(y_poly), lw=1, ls='--', alpha=0.8, label=f'Polyphase K={K}', color='red')
    axes4[1, 0].set_title('Direct vs Polyphase', fontsize=20, fontweight='bold')
    axes4[1, 0].set_xlabel('Time (ns)')
    axes4[1, 0].set_ylabel('Magnitude')
    axes4[1, 0].grid(True, alpha=0.3)
    axes4[1, 0].legend()
    
    # 子图4：误差与统计
    err = np.abs(y_direct - y_poly)
    axes4[1, 1].semilogy(t_full_ns, err + 1e-16, lw=1.5, color='green', label='Absolute Error')
    axes4[1, 1].axhline(mse, color='orange', ls='--', linewidth=2, label=f'MSE={mse:.3e}')
    axes4[1, 1].axhline(mae, color='purple', ls='--', linewidth=2, label=f'MAE={mae:.3e}')
    axes4[1, 1].set_title('Error Analysis (log scale)', fontsize=20, fontweight='bold')
    axes4[1, 1].set_xlabel('Time (ns)')
    axes4[1, 1].set_ylabel('Error Magnitude')
    axes4[1, 1].grid(True, alpha=0.3, which='both')
    axes4[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f'pulse_compress_04_comparison_K{K}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ===== 打印统计信息 =====
    print(f"\n{'='*70}")
    print(f"Polyphase Pulse Compression Analysis (K={K})")
    print(f"{'='*70}")
    print(f"Input signal length N: {N}")
    print(f"Matched filter length L: {L}")
    print(f"Output length N+L-1: {N+L-1}")
    print(f"\nMultiphase Decomposition:")
    for k in range(K):
        print(f"  Branch {k}: x_{k} len={len(x_branches[k])}, h_{k} len={len(h_branches[k])}")
    print(f"\nError Metrics:")
    print(f"  MSE: {mse:.6e}")
    print(f"  MAE: {mae:.6e}")
    print(f"  Max Error: {max_err:.6e}")
    print(f"  Relative Error: {mae / (np.mean(np.abs(y_direct)) + 1e-12) * 100:.6e}%")
    print(f"{'='*70}\n")
    
    return {
        'y_direct': y_direct,
        'y_poly': y_poly,
        'mse': mse,
        'mae': mae,
        'max_err': max_err
    }

# ==================== 复杂度分析 ====================
class ComplexityAnalyzer:
    @staticmethod
    def compute_ops(N, L, K):
        return K * (N // K) * L

    @staticmethod
    def compute_memory(N, L, K, bytes_per_sample=16):
        mem_input = (N // K) * K * bytes_per_sample / 1e6
        mem_filter = (L // K) * K * bytes_per_sample / 1e6
        mem_output = (N + L - 1) * bytes_per_sample / 1e6
        return mem_input + mem_filter + mem_output

    @staticmethod
    def compute_latency_us(N, L, K, clock_MHz=1000, is_poly=False):
        ops = K * (N // K) * L
        if is_poly:
            ops_per_ch = (N // K) * L
            t_compute = ops_per_ch / (clock_MHz * 1e6)
            t_recon = (N + L) / (clock_MHz * 1e6)
            return (t_compute + t_recon) * 1e6
        else:
            return (ops / (clock_MHz * 1e6)) * 1e6

def plot_complexity_analysis(N_range=None, L=256, K_values=[1, 2, 4, 8], clock_MHz=1000):
    if N_range is None:
        N_range = np.array([512, 1024, 2048, 4096, 8192, 16384])
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    colors = {1: 'black', 2: 'blue', 4: 'red', 8: 'green'}
    
    ax = axes[0, 0]
    for K in K_values:
        ops = np.array([ComplexityAnalyzer.compute_ops(n, L, K) for n in N_range])
        ax.loglog(N_range, ops, 'o-', color=colors.get(K, 'gray'), label=f'K={K}', linewidth=2)
    ax.set_title(f'Computational Complexity (L={L})'); ax.set_xlabel('Input Length N'); ax.set_ylabel('MACs')
    ax.grid(True, alpha=0.3, which='both'); ax.legend()
    
    ax = axes[0, 1]
    speedup = []
    for N in N_range:
        ops_direct = ComplexityAnalyzer.compute_ops(N, L, 1)
        speedups_N = [ops_direct / ComplexityAnalyzer.compute_ops(N, L, K) for K in K_values]
        speedup.append(speedups_N)
    speedup = np.array(speedup)
    for i, N in enumerate(N_range):
        ax.plot(K_values, speedup[i], 'o-', label=f'N={N}', linewidth=1.5)
    ax.set_title('Speedup Factor'); ax.set_xlabel('K'); ax.set_ylabel('Speedup')
    ax.grid(True, alpha=0.3); ax.legend(); ax.set_xscale('log')
    
    ax = axes[0, 2]
    for K in K_values:
        lat = np.array([ComplexityAnalyzer.compute_latency_us(n, L, K, clock_MHz, is_poly=K>1) for n in N_range])
        ax.loglog(N_range, lat, 'o-', color=colors.get(K, 'gray'), label=f'K={K}', linewidth=2)
    ax.set_title(f'End-to-End Latency'); ax.set_xlabel('Input Length N'); ax.set_ylabel('Latency (us)')
    ax.grid(True, alpha=0.3, which='both'); ax.legend()
    
    ax = axes[1, 0]
    for K in K_values:
        mem = np.array([ComplexityAnalyzer.compute_memory(n, L, K)/K for n in N_range])
        ax.loglog(N_range, mem, 's-', color=colors.get(K, 'gray'), label=f'K={K}', linewidth=2)
    ax.set_title(f'Memory Footprint'); ax.set_xlabel('Input Length N'); ax.set_ylabel('Memory (MB)')
    ax.grid(True, alpha=0.3, which='both'); ax.legend()
    
    ax = axes[1, 1]
    mem_direct = np.array([ComplexityAnalyzer.compute_memory(n, L, 1) for n in N_range])
    for K in K_values[1:]:
        mem_poly = np.array([ComplexityAnalyzer.compute_memory(n, L, K)/K for n in N_range])
        saving = (mem_direct - mem_poly) / mem_direct * 100
        ax.plot(N_range, saving, 'o-', color=colors.get(K, 'gray'), label=f'K={K}', linewidth=2)
    ax.set_title('Memory Saving (%)'); ax.set_xlabel('Input Length N'); ax.set_ylabel('Saving (%)')
    ax.grid(True, alpha=0.3); ax.legend()
    
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = """CONFIGURATION
============
Filter Length: L = 256
Clock Rate: 1000 MHz
Bytes/Sample: 16 (complex128)

KEY INSIGHTS
============
- K-way parallel: K*K branch convs
- Ops: O(N*L), scalable parallel
- Latency: algo L-1, compute ~L-1/K
- Memory: reduced ~K per branch
- Speedup: limited by hardware

SPEEDUP BOUND
=============
Ideal: K (if K cores available)
Practical: 0.7-0.9x K"""
    ax.text(0.05, 0.95, summary_text, fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('complexity_analysis_comprehensive.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_kernel_length_effect(N=4096, L_range=None, K_values=[1, 2, 4]):
    if L_range is None:
        L_range = np.array([32, 64, 128, 256, 512, 1024])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {1: 'black', 2: 'blue', 4: 'red'}
    
    ax = axes[0]
    for K in K_values:
        ops = np.array([ComplexityAnalyzer.compute_ops(N, L, K) for L in L_range])
        ax.loglog(L_range, ops, 'o-', color=colors.get(K, 'gray'), label=f'K={K}', linewidth=2)
    ax.set_title(f'Complexity vs Filter Length (N={N})'); ax.set_xlabel('L'); ax.set_ylabel('MACs')
    ax.grid(True, alpha=0.3, which='both'); ax.legend()
    
    ax = axes[1]
    ops_base = np.array([ComplexityAnalyzer.compute_ops(N, L, 1) for L in L_range])
    for K in K_values[1:]:
        ops_poly = np.array([ComplexityAnalyzer.compute_ops(N, L, K) for L in L_range])
        eff = ops_base / ops_poly
        ax.plot(L_range, eff, 'o-', color=colors.get(K, 'gray'), label=f'K={K}', linewidth=2)
    ax.set_title('Speedup vs Filter Length'); ax.set_xlabel('L'); ax.set_ylabel('Speedup')
    ax.axhline(1, color='gray', ls='--', alpha=0.5); ax.grid(True, alpha=0.3)
    ax.legend(); ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('kernel_length_effect.png', dpi=150, bbox_inches='tight')
    plt.show()

# ==================== 主演示 ====================
def main():
    print("=" * 70)
    print("Polyphase Pulse Compression - Detailed Process Analysis")
    print("=" * 70)
    
    # ===== 生成 10-18 GHz 线性调频信号（时长 100 ns） =====
    print("\n[1] Generating 10-18 GHz LFM Signal (duration = 100 ns)...")
    fs = 40e9  # 40 GHz 采样率
    duration = 100e-9  # 100 ns
    f0, f1 = 10e9, 18e9
    ref, _ = gen_lfm(fs, f0, f1, duration)
    
    # ===== 生成接收信号（多个延迟目标） =====
    print("[2] Generating Received Signal with Multiple Targets...")
    # delays_s = [0.15e-9, 0.42e-9, 0.78e-9]
    # amplitudes = [1.0, 0.7, 0.5]
    delays_s = [0.15e-9]
    amplitudes = [1.0]
    x = gen_received(ref, fs, delays_s, amplitudes, snr_db=30)
    
    # ===== 不同 K 值的脉冲压缩展示 =====
    print("[3] Polyphase Pulse Compression Process Visualization...")
    for K in [2, 4]:
        print(f"\n  Processing with K={K}...")
        plot_pulse_compress_detailed(x, ref, K=K, fs=fs)
    
    # ===== 复杂度分析 =====
    print("\n[4] Complexity Analysis...")
    plot_complexity_analysis(N_range=np.array([512, 1024, 2048, 4096, 8192]), L=256, K_values=[1, 2, 4, 8])
    plot_kernel_length_effect(N=4096, L_range=np.array([32, 64, 128, 256, 512]), K_values=[1, 2, 4])
    
    # ===== 性能统计 =====
    print("\n[5] Performance Summary...")
    N_test = 4096
    L_test = 256
    summary = []
    for K in [1, 2, 4, 8]:
        ops = ComplexityAnalyzer.compute_ops(N_test, L_test, K)
        mem = ComplexityAnalyzer.compute_memory(N_test, L_test, K)
        lat = ComplexityAnalyzer.compute_latency_us(N_test, L_test, K, is_poly=K>1)
        summary.append({'K': K, 'MACs': ops, 'Memory(MB)': mem, 'Latency(us)': lat})
    
    print("\n" + "="*70)
    print(f"Configuration: N={N_test}, L={L_test}, Clock=1000 MHz")
    print("="*70)
    print(f"{'K':3s} {'MACs':15s} {'Memory(MB)':15s} {'Latency(us)':15s}")
    print("-"*70)
    for s in summary:
        print(f"{s['K']:3d} {s['MACs']:15.2e} {s['Memory(MB)']:15.3f} {s['Latency(us)']:15.4f}")
    print("="*70)

if __name__ == "__main__":
    main()
