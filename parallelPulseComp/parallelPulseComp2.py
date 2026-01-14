import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp

def matched_filter_from_ref(ref):
    return np.conj(ref[::-1]).astype(np.complex128)

def polyphase_pulse_compress(x, ref, K):
    """
    基于polyphase的并行脉冲压缩。
    等价于 y = np.convolve(x, matched_filter_from_ref(ref))
    x: 接收信号 (complex)
    ref: 参考脉冲 (complex)
    K: 并行相位数（正整数）
    返回: y (complex), 长度 N + L - 1
    """
    x = np.asarray(x, dtype=np.complex128)
    h = matched_filter_from_ref(np.asarray(ref, dtype=np.complex128))
    N = len(x)
    L = len(h)
    if K <= 0:
        raise ValueError("K must be positive")
    if N == 0 or L == 0:
        return np.zeros(N + L - 1, dtype=np.complex128)

    # 输入抽取为K路
    x_branches = [x[j::K] for j in range(K)]

    # 每个相位的累加缓冲
    out_len_per_phase = (N + L - 1 + K - 1) // K + 1
    y_phase = [np.zeros(out_len_per_phase, dtype=np.complex128) for _ in range(K)]

    # KxK 分支：跨相位子滤波器 + 一拍对齐
    for i in range(K):
        for j in range(K):
            r = (i - j) % K
            g_ij = h[r::K]  # 子滤波器
            if g_ij.size == 0:
                continue
            c = np.convolve(x_branches[j], g_ij)
            # i<j 时做一拍延时
            if i < j:
                c = np.concatenate([np.zeros(1, dtype=np.complex128), c])
            # 累加到相位 i
            if len(c) > len(y_phase[i]):
                grow = len(c) - len(y_phase[i])
                y_phase[i] = np.pad(y_phase[i], (0, grow))
            y_phase[i][:len(c)] += c

    # 交织回整速
    y = np.zeros(N + L - 1, dtype=np.complex128)
    for i in range(K):
        yi = y_phase[i]
        n_idx = i + np.arange(len(yi)) * K
        valid = n_idx < len(y)
        y[n_idx[valid]] += yi[valid]
    return y

# =============== 演示与一致性校验 ===============
def generate_lfm(fs, f0, f1, duration):
    t = np.arange(int(duration * fs)) / fs
    s = chirp(t, f0=f0, t1=t[-1], f1=f1, method='linear')
    return s.astype(np.complex128), t

def generate_received(signal_tx, fs, delays_s, amplitudes, snr_db=30):
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

def demo():
    # 用较小采样率演示，避免超大FFT
    fs = 5e6
    duration = 1e-3
    f0, f1 = 0.5e6, 1.2e6
    ref, _ = generate_lfm(fs, f0, f1, duration)
    h = matched_filter_from_ref(ref)

    delays_s = [0.15e-3, 0.42e-3, 0.78e-3]
    amplitudes = [1.0, 0.7, 0.5]
    x = generate_received(ref, fs, delays_s, amplitudes, snr_db=25)

    # 直接脉冲压缩（基准）
    y_direct = np.convolve(x, h)

    # polyphase 并行脉冲压缩
    K = 4
    y_poly = polyphase_pulse_compress(x, ref, K)

    # 一致性
    P = min(len(y_direct), len(y_poly))
    y_direct = y_direct[:P]
    y_poly = y_poly[:P]
    mse = np.mean(np.abs(y_direct - y_poly)**2)
    mae = np.mean(np.abs(y_direct - y_poly))
    print(f"Polyphase parity check: K={K}, MSE={mse:.3e}, MAE={mae:.3e}")

    plt.figure(figsize=(12,5))
    plt.plot(np.abs(y_direct), label='Direct', linewidth=1.6)
    plt.plot(np.abs(y_poly), '--', label=f'Polyphase K={K}', linewidth=1.2)
    plt.title('Pulse Compression Magnitude: Direct vs Polyphase')
    plt.xlabel('Sample Index')
    plt.ylabel('Magnitude')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    idx_peak = np.argmax(np.abs(y_direct))
    w = 600
    a = max(0, idx_peak - w)
    b = min(P, idx_peak + w)
    plt.figure(figsize=(10,4))
    plt.plot(np.abs(y_direct[a:b]), label='Direct')
    plt.plot(np.abs(y_poly[a:b]), '--', label=f'Polyphase K={K}')
    plt.title('Zoom around strongest peak')
    plt.xlabel('Sample Index')
    plt.ylabel('Magnitude')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    demo()
