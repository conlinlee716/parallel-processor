import numpy as np
import matplotlib.pyplot as plt

def fft_cost(n, c=1.0):
    # 相对复杂度：c * n * log2(n)
    return c * n * np.log2(max(n, 2))

def direct_complexity_and_latency(N, L, rops=1e9, c=1.0):
    """
    直接整段FFT脉冲压缩的复杂度与延时（相对评估）
    rops: 处理吞吐率（ops/s），只是用来把复杂度换算为时间
    返回: ops_total, time_total (秒)
    """
    P = N + L - 1
    # 2*FFT + 1*IFFT + 频域乘法（P次复乘）
    ops_fft = 3 * fft_cost(P, c=c)
    ops_mul = P
    ops_total = ops_fft + ops_mul
    time_total = ops_total / rops
    return ops_total, time_total

def overlap_save_complexity_and_latency(N, L, M, K=1, rops=1e9, c=1.0):
    """
    Overlap-Save分块并行的复杂度与延时（墙钟与首输出）
    K: 并行核数，假设理想均分负载
    返回: ops_total, wall_time_total (秒), first_output_latency (秒), n_blocks
    """
    assert M >= L, "M must be >= L"
    B = M - (L - 1)  # 每块有效样点
    P = N + L - 1    # 线性卷积输出长度
    n_blocks = int(np.ceil(P / B))
    # 每块复杂度：2FFT + 1IFFT + 乘法
    ops_block = 3 * fft_cost(M, c=c) + M
    ops_total = n_blocks * ops_block
    # 时间
    T_block = ops_block / rops
    wall_time_total = np.ceil(n_blocks / K) * T_block
    first_output_latency = T_block  # 首块完成即可输出有效样点（OS丢弃前L-1样点）
    return ops_total, wall_time_total, first_output_latency, n_blocks

def plot_complexity_latency(ref_len, L, rops=5e9, c=1.0):
    """
    生成3张图：复杂度、墙钟时间、首输出延时
    rops: 假设的处理吞吐率（ops/s）；增大它会整体降低时间数值，但曲线形状不变
    """
    # 输入长度范围（可根据你的应用修改）
    Ns = np.array([2**k for k in range(10, 17)])  # 1024 到 131072
    M_choices = [2*L, 4*L]  # 两个块长选择
    K_values = [1, 2, 4, 8]

    # 图1：复杂度对比（相对ops）
    plt.figure(figsize=(10, 5))
    direct_ops = []
    os_ops_dict = {M: [] for M in M_choices}
    for N in Ns:
        d_ops, _ = direct_complexity_and_latency(N, L, rops=rops, c=c)
        direct_ops.append(d_ops)
        for M in M_choices:
            os_ops, _, _, _ = overlap_save_complexity_and_latency(N, L, M, K=1, rops=rops, c=c)
            os_ops_dict[M].append(os_ops)
    plt.semilogy(Ns, direct_ops, 'o-', label='Direct FFT (full-length)', linewidth=2)
    for M in M_choices:
        plt.semilogy(Ns, os_ops_dict[M], 's--', label=f'Overlap-Save (M={M})', linewidth=2)
    plt.xlabel('Input length N')
    plt.ylabel('Relative operations (ops)')
    plt.title('Pulse Compression Complexity vs Input Length')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 图2：墙钟处理时间（不同并行核数K）
    plt.figure(figsize=(10, 5))
    for K in K_values:
        wall_times = []
        for N in Ns:
            # 选一个合理的M示例（4*L），实际可按内存/吞吐调整
            _, wall_t, _, _ = overlap_save_complexity_and_latency(N, L, M=4*L, K=K, rops=rops, c=c)
            wall_times.append(wall_t * 1e3)  # ms
        plt.plot(Ns, wall_times, marker='o', label=f'Overlap-Save (K={K})', linewidth=2)
    # 直接FFT墙钟（不能分块并行，但FFT库内部也会并行；此处画成一条基准）
    direct_wall = []
    for N in Ns:
        _, t = direct_complexity_and_latency(N, L, rops=rops, c=c)
        direct_wall.append(t * 1e3)
    plt.plot(Ns, direct_wall, 'k--', label='Direct FFT baseline', linewidth=2)
    plt.xlabel('Input length N')
    plt.ylabel('Wall-clock time (ms)')
    plt.title('Wall-clock Processing Time vs Input Length')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 图3：首输出端到端延时（Overlap-Save流水线特性）
    plt.figure(figsize=(10, 5))
    for K in K_values:
        first_lat = []
        for N in Ns:
            _, _, first_t, _ = overlap_save_complexity_and_latency(N, L, M=4*L, K=K, rops=rops, c=c)
            first_lat.append(first_t * 1e3)  # ms
        plt.plot(Ns, first_lat, marker='s', label=f'Overlap-Save First Output (K={K})', linewidth=2)
    # 直接FFT首输出延时≈整段处理时长（一次性FFT需全量）
    direct_first = []
    for N in Ns:
        _, t = direct_complexity_and_latency(N, L, rops=rops, c=c)
        direct_first.append(t * 1e3)
    plt.plot(Ns, direct_first, 'k--', label='Direct FFT First Output', linewidth=2)
    plt.xlabel('Input length N')
    plt.ylabel('First-output latency (ms)')
    plt.title('End-to-end First-output Latency vs Input Length')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 结合你前面主函数的设置获取L（滤波器长度）
    # 这里简单复用一次以获得L
    fs = 40e9
    duration = 1e-8
    f0, f1 = 10e9, 18e9
    from scipy.signal import chirp
    t = np.arange(int(duration * fs)) / fs
    ref = chirp(t, f0=f0, t1=t[-1], f1=f1, method='linear').astype(np.complex128)
    L = len(ref)

    # 假设处理吞吐率 rops（ops/s），用于把复杂度换算为时间
    # 例如 5e9 ops/s（5 GOPS）仅用于绘图尺度；实际应以你的平台参数替换
    plot_complexity_latency(ref_len=len(ref), L=L, rops=5e9, c=1.0)
