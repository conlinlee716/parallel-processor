import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.signal import chirp

# Global plotting configuration: Arial font, fontsize 20
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['axes.unicode_minus'] = False

class ConvolutionParallelProcessor:
    """Parallel convolution using polyphase (K=2)"""

    def __init__(self, seq_length, kernel_length, num_channels=2, mode='same'):
        assert num_channels == 2, "This demo implements K=2 polyphase only."
        self.N = seq_length
        self.M = kernel_length
        self.K = num_channels
        self.mode = mode  # 'same' is used in comparisons

    @staticmethod
    def _center_same_from_full(y_full, N, M):
        start = (M - 1) // 2
        return y_full[start:start + N]

    def direct_convolution(self, x, h):
        y_full = convolve(x, h, mode='full')
        return self._center_same_from_full(y_full, len(x), len(h))

    def parallel_convolution(self, x, h):
        # polyphase split
        x0, x1 = x[0::2], x[1::2]
        he, ho = h[0::2], h[1::2]

        # low-rate sub-convolutions (full)
        c00 = convolve(x0, he, mode='full')  # even-even
        c11 = convolve(x1, ho, mode='full')  # odd-odd
        c01 = convolve(x0, ho, mode='full')  # even-odd
        c10 = convolve(x1, he, mode='full')  # odd-even

        # reconstruct full-rate output (length N+M-1)
        L_full = len(x) + len(h) - 1
        L00 = len(c00); L11 = len(c11); L01 = len(c01); L10 = len(c10)

        # y_even_low[n’] = c00[n’] + c11[n’-1]
        L_even_low = max(L00, L11 + 1)
        y_even_low = np.zeros(L_even_low, dtype=np.result_type(x, h))
        y_even_low[:L00] += c00
        if L11 > 0:
            y_even_low[1:1+L11] += c11

        # y_odd_low[n’] = c01[n’] + c10[n’]
        L_odd_low = max(L01, L10)
        y_odd_low = np.zeros(L_odd_low, dtype=np.result_type(x, h))
        if L01 > 0:
            y_odd_low[:L01] += c01
        if L10 > 0:
            y_odd_low[:L10] += c10

        # interleave to full-rate
        y_full = np.zeros(L_full, dtype=np.result_type(x, h))
        even_positions = np.arange(0, 2*L_even_low, 2)
        even_positions = even_positions[even_positions < L_full]
        y_full[even_positions] = y_even_low[:len(even_positions)]
        odd_positions = np.arange(1, 2*L_odd_low+1, 2)
        odd_positions = odd_positions[odd_positions < L_full]
        y_full[odd_positions] = y_odd_low[:len(odd_positions)]

        # match 'same'
        y_same = self._center_same_from_full(y_full, len(x), len(h))

        sub_convs = {
            'c00_x0_he': c00,
            'c11_x1_ho': c11,
            'c01_x0_ho': c01,
            'c10_x1_he': c10,
            'y_even_low': y_even_low,
            'y_odd_low': y_odd_low
        }
        return y_same, sub_convs


def gen_signal(signal_type, fs=40e9, duration=1e-8, N=None):
    if N is None:
        N = int(fs * duration)
    t = np.arange(N) / fs
    if signal_type == "sinusoid_1GHz":
        x = np.sin(2*np.pi*1e9*t)
        name = "1 GHz Sinusoid"
    elif signal_type == "lfm_10_18GHz":
        x = chirp(t, 10e9, t[-1], 18e9, method='linear')
        name = "10-18 GHz LFM"
    elif signal_type == "qpsk_1Gbps":
        n_sym = max(1, int(1e9 * duration))
        syms = np.random.randint(0, 4, n_sym)
        map4 = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
        s = map4[syms]
        sps = max(1, N // n_sym)
        base = np.repeat(s, sps)[:N]
        carrier = np.exp(2j*np.pi*5e9*t)
        x = base * carrier
        name = "1 Gbps QPSK"
    else:
        raise ValueError("unknown signal_type")
    return x, name

def gen_kernel(kernel_type="lowpass", M=64):
    if kernel_type == "lowpass":
        h = np.sinc(np.linspace(-2, 2, M))
        h /= np.sum(np.abs(h)) + 1e-12
        name = "Lowpass (Sinc)"
    elif kernel_type == "highpass":
        h = np.zeros(M); h[M//2] = 1.0
        h -= np.sinc(np.linspace(-2, 2, M)) / (M/2)
        h /= np.sum(np.abs(h)) + 1e-12
        name = "Highpass"
    elif kernel_type == "matched":
        h = np.random.randn(M); h /= np.linalg.norm(h) + 1e-12
        name = "Matched"
    else:
        raise ValueError("unknown kernel_type")
    return h, name


def plot_parallel_conv(sequence, kernel, signal_name, kernel_name, proc, fs=40e9):
    """Plot signals using time-based x-axis (ns).

    sequence: full-rate sequence (length N)
    sub-convolutions are plotted with low-rate time (fs/2)
    """
    y_dir = proc.direct_convolution(sequence, kernel)
    y_par, subs = proc.parallel_convolution(sequence, kernel)

    # time axes
    N = len(sequence)
    t_full_ns = np.arange(N) / fs * 1e9  # ns

    # low-rate time axes (sub-sampled by K=2)
    fs_low = fs / 2.0
    t_c00_ns = np.arange(len(subs['c00_x0_he'])) / fs_low * 1e9
    t_c11_ns = np.arange(len(subs['c11_x1_ho'])) / fs_low * 1e9
    t_c01_ns = np.arange(len(subs['c01_x0_ho'])) / fs_low * 1e9
    t_c10_ns = np.arange(len(subs['c10_x1_he'])) / fs_low * 1e9

    # accumulated low-rate signals
    t_even_low_ns = np.arange(len(subs['y_even_low'])) / fs_low * 1e9
    t_odd_low_ns = np.arange(len(subs['y_odd_low'])) / fs_low * 1e9

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(t_full_ns, np.real(y_dir), lw=1, label='Direct')
    axes[0, 0].set_title('Direct Convolution'); axes[0, 0].grid(True, alpha=0.3); axes[0, 0].legend()

    axes[0, 1].plot(t_c00_ns, np.real(subs['c00_x0_he']), lw=1, label='c00: x0*he')
    axes[0, 1].plot(t_c11_ns, np.real(subs['c11_x1_ho']), lw=1, label='c11: x1*ho')
    axes[0, 1].set_title('Low-rate Convs (even path terms)'); axes[0, 1].grid(True, alpha=0.3); axes[0, 1].legend()

    axes[1, 0].plot(t_c01_ns, np.real(subs['c01_x0_ho']), lw=1, label='c01: x0*ho')
    axes[1, 0].plot(t_c10_ns, np.real(subs['c10_x1_he']), lw=1, label='c10: x1*he')
    axes[1, 0].set_title('Low-rate Convs (odd path terms)'); axes[1, 0].grid(True, alpha=0.3); axes[1, 0].legend()

    axes[1, 1].plot(t_full_ns, np.real(y_par), lw=1.2, color='red', label='Reconstructed')
    axes[1, 1].plot(t_full_ns, np.real(y_dir), lw=1.0, ls='--', color='black', alpha=0.7, label='Direct')
    axes[1, 1].set_title('Reconstructed vs Direct (same)'); axes[1, 1].grid(True, alpha=0.3); axes[1, 1].legend()

    # set labels to time
    for ax in axes.ravel():
        ax.set_xlabel('Time (ns)'); ax.set_ylabel('Amplitude')

    plt.tight_layout()
    plt.savefig(f'parallel_conv_polyphase_{signal_name.replace(" ", "_")}_{kernel_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

    err = np.mean(np.abs(y_dir - y_par))
    rel = err / (np.mean(np.abs(y_dir)) + 1e-12)
    print(f"{signal_name} + {kernel_name}: mean error={err:.3e}, rel={rel:.3e}")


class ComplexityAnalysis:
    @staticmethod
    def compute_complexity_direct(N, M):
        return N * M  # MACs

    @staticmethod
    def compute_complexity_parallel(N, M, K):
        return K * (N / K) * M  # total MACs same as direct (ignoring recon)

    @staticmethod
    def compute_latency_direct(N, M, clock_rate_MHz):
        ops = N * M
        cycles = ops / 2.0
        return cycles / clock_rate_MHz

    @staticmethod
    def compute_latency_parallel(N, M, K, clock_rate_MHz):
        ops_ch = (N / K) * M
        cycles_ch = ops_ch / 2.0
        t_parallel = cycles_ch / clock_rate_MHz
        t_recon = N / (clock_rate_MHz * 1000.0)  # simple recon overhead
        return t_parallel + t_recon


def plot_complexity_comparison(seq_lengths=None, kernel_length=64, K=2, clock_rate_MHz=1000):
    if seq_lengths is None:
        seq_lengths = np.array([256, 512, 1024, 2048, 4096, 8192])
    cd = np.array([ComplexityAnalysis.compute_complexity_direct(n, kernel_length) for n in seq_lengths])
    cp = np.array([ComplexityAnalysis.compute_complexity_parallel(n, kernel_length, K) for n in seq_lengths])
    ld = np.array([ComplexityAnalysis.compute_latency_direct(n, kernel_length, clock_rate_MHz) for n in seq_lengths])
    lp = np.array([ComplexityAnalysis.compute_latency_parallel(n, kernel_length, K, clock_rate_MHz) for n in seq_lengths])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].semilogy(seq_lengths, cd, 'o-', label='Direct'); axes[0, 0].semilogy(seq_lengths, cp, 's-', label=f'Parallel (K={K})')
    axes[0, 0].set_title('Computational Complexity'); axes[0, 0].set_xlabel('Sequence Length'); axes[0, 0].set_ylabel('MACs'); axes[0, 0].grid(True, alpha=0.3); axes[0, 0].legend()

    imp_c = (cd - cp) / cd * 100
    axes[0, 1].plot(seq_lengths, imp_c, 'o-', color='green'); axes[0, 1].axhline(np.mean(imp_c), color='r', ls='--', label=f'Mean {np.mean(imp_c):.1f}%')
    axes[0, 1].set_title('Complexity Reduction'); axes[0, 1].set_xlabel('Sequence Length'); axes[0, 1].set_ylabel('Reduction (%)'); axes[0, 1].grid(True, alpha=0.3); axes[0, 1].legend()

    axes[1, 0].semilogy(seq_lengths, ld*1e6, 'o-', label='Direct'); axes[1, 0].semilogy(seq_lengths, lp*1e6, 's-', label=f'Parallel (K={K})')
    axes[1, 0].set_title('End-to-End Latency'); axes[1, 0].set_xlabel('Sequence Length'); axes[1, 0].set_ylabel('Latency (μs)'); axes[1, 0].grid(True, alpha=0.3); axes[1, 0].legend()

    imp_l = (ld - lp) / ld * 100
    axes[1, 1].plot(seq_lengths, imp_l, 's-', color='purple'); axes[1, 1].axhline(np.mean(imp_l), color='r', ls='--', label=f'Mean {np.mean(imp_l):.1f}%')
    axes[1, 1].set_title('Latency Reduction'); axes[1, 1].set_xlabel('Sequence Length'); axes[1, 1].set_ylabel('Reduction (%)'); axes[1, 1].grid(True, alpha=0.3); axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('conv_complexity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_kernel_length_effect(seq_length=4096, kernel_lengths=None, K=2, clock_rate_MHz=1000):
    if kernel_lengths is None:
        kernel_lengths = np.array([16, 32, 64, 128, 256, 512])
    cd = np.array([ComplexityAnalysis.compute_complexity_direct(seq_length, m) for m in kernel_lengths])
    cp = np.array([ComplexityAnalysis.compute_complexity_parallel(seq_length, m, K) for m in kernel_lengths])
    ld = np.array([ComplexityAnalysis.compute_latency_direct(seq_length, m, clock_rate_MHz) for m in kernel_lengths])
    lp = np.array([ComplexityAnalysis.compute_latency_parallel(seq_length, m, K, clock_rate_MHz) for m in kernel_lengths])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].semilogy(kernel_lengths, cd, 'o-', label='Direct'); axes[0].semilogy(kernel_lengths, cp, 's-', label=f'Parallel (K={K})')
    axes[0].set_title(f'Complexity vs Kernel Length (N={seq_length})'); axes[0].set_xlabel('Kernel Length'); axes[0].set_ylabel('MACs'); axes[0].grid(True, alpha=0.3); axes[0].legend()

    imp = (cd - cp) / cd * 100
    axes[1].bar(range(len(kernel_lengths)), imp, color='skyblue', edgecolor='black')
    axes[1].set_title('Complexity Reduction (%)'); axes[1].set_xlabel('Kernel Length'); axes[1].set_ylabel('Reduction (%)')
    axes[1].set_xticks(range(len(kernel_lengths))); axes[1].set_xticklabels([str(m) for m in kernel_lengths])
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('kernel_length_effect.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_multicore_convolution_efficiency(seq_length=4096, kernel_length=64, K_values=None):
    if K_values is None:
        K_values = np.array([1, 2, 4, 8, 16])
    base = ComplexityAnalysis.compute_complexity_direct(seq_length, kernel_length)
    c_par = np.array([ComplexityAnalysis.compute_complexity_parallel(seq_length, kernel_length, K) for K in K_values])
    eff = base / c_par * 100
    per_ch = c_par / K_values

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(K_values, eff, 'o-', color='darkblue'); axes[0].axhline(100, color='r', ls='--', alpha=0.5, label='Baseline')
    axes[0].set_title('Parallel Efficiency'); axes[0].set_xlabel('K'); axes[0].set_ylabel('Efficiency (%)'); axes[0].grid(True, alpha=0.3); axes[0].legend()

    width = 0.35
    axes[1].bar(K_values - width/2, per_ch, width, label='Per-Channel', alpha=0.8)
    axes[1].bar(K_values + width/2, c_par, width, label='Total', alpha=0.8)
    axes[1].set_title(f'Complexity Distribution (N={seq_length}, M={kernel_length})')
    axes[1].set_xlabel('K'); axes[1].set_ylabel('MACs'); axes[1].grid(True, alpha=0.3, axis='y'); axes[1].legend()

    plt.tight_layout()
    plt.savefig('conv_multicore_efficiency.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_memory_bandwidth_analysis(seq_lengths=None, kernel_length=64, K=2, bytes_per_sample=8):
    if seq_lengths is None:
        seq_lengths = np.array([256, 512, 1024, 2048, 4096, 8192])
    mem_direct = (seq_lengths + kernel_length) * bytes_per_sample / 1e6
    mem_parallel = (seq_lengths / K + kernel_length) * bytes_per_sample / 1e6
    mem_save = (mem_direct - mem_parallel) / mem_direct * 100

    bw_direct = seq_lengths * kernel_length / 1e3
    bw_parallel = (seq_lengths / K) * kernel_length / 1e3
    bw_reduce = (bw_direct - bw_parallel) / bw_direct * 100

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(seq_lengths, mem_direct, 'o-', label='Direct'); axes[0, 0].plot(seq_lengths, mem_parallel, 's-', label=f'Parallel (K={K})')
    axes[0, 0].set_title('Memory Footprint'); axes[0, 0].set_xlabel('Sequence Length'); axes[0, 0].set_ylabel('Memory (MB)'); axes[0, 0].grid(True, alpha=0.3); axes[0, 0].set_xscale('log'); axes[0, 0].legend()

    axes[0, 1].bar(range(len(seq_lengths)), mem_save, color='green', edgecolor='black')
    axes[0, 1].set_title('Memory Saving (%)'); axes[0, 1].set_xlabel('Sequence Length'); axes[0, 1].set_ylabel('Saving (%)')
    axes[0, 1].set_xticks(range(len(seq_lengths))); axes[0, 1].set_xticklabels([str(s) for s in seq_lengths], rotation=45); axes[0, 1].grid(True, alpha=0.3, axis='y')

    axes[1, 0].semilogy(seq_lengths, bw_direct, 'o-', label='Direct'); axes[1, 0].semilogy(seq_lengths, bw_parallel, 's-', label=f'Parallel (K={K})')
    axes[1, 0].set_title('Memory Bandwidth'); axes[1, 0].set_xlabel('Sequence Length'); axes[1, 0].set_ylabel('Access (GB/s)'); axes[1, 0].grid(True, alpha=0.3); axes[1, 0].legend()

    axes[1, 1].bar(range(len(seq_lengths)), bw_reduce, color='purple', edgecolor='black')
    axes[1, 1].set_title('Bandwidth Reduction (%)'); axes[1, 1].set_xlabel('Sequence Length'); axes[1, 1].set_ylabel('Reduction (%)')
    axes[1, 1].set_xticks(range(len(seq_lengths))); axes[1, 1].set_xticklabels([str(s) for s in seq_lengths], rotation=45); axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('conv_memory_bandwidth.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("=" * 70)
    print("Parallel Convolution (Polyphase K=2) Verification")
    print("=" * 70)

    fs = 40e9
    duration = 1e-8
    N = int(fs * duration)
    M = 64

    proc = ConvolutionParallelProcessor(N, M, num_channels=2)

    signals = ["sinusoid_1GHz", "lfm_10_18GHz", "qpsk_1Gbps"]
    kernel_types = ["lowpass"]  # 关键图像，避免过多输出；可改为多种核

    for s in signals:
        x, sname = gen_signal(s, fs=fs, duration=duration, N=N)
        for kt in kernel_types:
            h, hname = gen_kernel(kt, M)
            plot_parallel_conv(x, h, sname, hname, proc, fs=fs)
    plot_multicore_convolution_efficiency(seq_length=4096, kernel_length=64, K_values=np.array([1, 2, 4, 8, 16]))
    plot_memory_bandwidth_analysis(seq_lengths=np.array([256, 512, 1024, 2048, 4096, 8192]), kernel_length=64, K=2, bytes_per_sample=8)
