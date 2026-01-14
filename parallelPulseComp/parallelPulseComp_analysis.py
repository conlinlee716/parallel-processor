import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp
from scipy.fft import fft, ifft

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
    # AWGN
    snr_lin = 10**(snr_db/10)
    p_sig = np.mean(np.abs(x)**2) + 1e-12
    p_n = p_sig / snr_lin
    noise = np.sqrt(p_n/2) * (np.random.randn(N) + 1j*np.random.randn(N))
    return x + noise

def matched_filter_from_ref(ref):
    h = np.conj(ref[::-1])
    return h.astype(np.complex128)

def pulse_compress_direct(x, h):
    # Linear convolution via full-length FFT
    P = len(x) + len(h) - 1
    X = fft(x, n=P)
    H = fft(h, n=P)
    y = ifft(X * H)
    return y  # length P

def overlap_save_fft(x, h, M=None):
    L = len(h)
    if M is None:
        # choose a power-of-two block length >= L, with some headroom
        M = 1 << int(np.ceil(np.log2(max(L, 2*L))))
    if M < L:
        raise ValueError("Block length M must be >= filter length L.")
    B = M - (L - 1)  # valid samples per block
    # Precompute H at length M
    H = fft(h, n=M)
    # Pad x with L-1 zeros at front
    x_pad = np.concatenate([np.zeros(L-1, dtype=np.complex128), x])
    n_blocks = int(np.ceil((len(x_pad) - (L-1)) / B))
    # Prepare blocks [n_blocks, M]
    blocks = np.zeros((n_blocks, M), dtype=np.complex128)
    for i in range(n_blocks):
        start = i * B
        end = start + M
        # safe slice
        blk = x_pad[start:min(end, len(x_pad))]
        blocks[i, :len(blk)] = blk
    # Batch FFT/IFFT (parallelizable along axis=1)
    Xb = fft(blocks, n=M, axis=1)
    Yb = ifft(Xb * H[None, :], axis=1)
    # Discard first L-1 from each block, keep B valid
    y_valid = Yb[:, L-1:]
    y_os = y_valid.reshape(-1)[:len(x) + L - 1]  # trim to linear conv length
    return y_os

def main():
    fs = 40e9
    duration = 1e-8
    f0, f1 = 10e9, 18e9

    # Reference waveform and matched filter
    ref, t = generate_lfm(fs, f0, f1, duration)
    h = matched_filter_from_ref(ref)

    # Received signal with multiple targets
    delays_s = [1.0e-9, 3.0e-9, 6.0e-9]      # target delays
    amplitudes = [1.0, 0.7, 0.5]             # target amplitudes
    x = generate_received(ref, fs, delays_s, amplitudes, snr_db=25)

    # Direct pulse compression (ground truth)
    y_direct = pulse_compress_direct(x, h)

    # Parallelizable overlap-save FFT pulse compression
    # Choose block length M (power of two >= L, larger M -> fewer blocks)
    L = len(h)
    M = 1 << int(np.ceil(np.log2(4 * L)))    # e.g., 4x filter length
    y_os = overlap_save_fft(x, h, M=M)

    # Metrics
    mag_direct = np.abs(y_direct)
    mag_os = np.abs(y_os)
    # Align lengths (should already be equal)
    P = min(len(mag_direct), len(mag_os))
    mag_direct = mag_direct[:P]
    mag_os = mag_os[:P]

    mse = np.mean((mag_direct - mag_os)**2)
    mae = np.mean(np.abs(mag_direct - mag_os))
    peak_direct = np.max(mag_direct)
    peak_os = np.max(mag_os)

    print("Results:")
    print(f"  Block length M: {M}, filter length L: {L}")
    print(f"  Output length: {P}")
    print(f"  MSE: {mse:.3e}, MAE: {mae:.3e}")
    print(f"  Peak (direct): {peak_direct:.4f}, Peak (overlap-save): {peak_os:.4f}")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(mag_direct, label='Direct FFT (linear conv)', linewidth=1.5, alpha=0.8)
    plt.plot(mag_os, label='Overlap-Save FFT (parallelizable)', linewidth=1.2, linestyle='--')
    plt.title('Pulse Compression Result (Magnitude)')
    plt.xlabel('Sample Index')
    plt.ylabel('Magnitude')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Optional: zoom around the strongest peak
    idx_peak = np.argmax(mag_direct)
    w = 500
    a = max(0, idx_peak - w)
    b = min(P, idx_peak + w)
    plt.figure(figsize=(10, 4))
    plt.plot(mag_direct[a:b], label='Direct')
    plt.plot(mag_os[a:b], label='Overlap-Save', linestyle='--')
    plt.title('Zoom around strongest peak')
    plt.xlabel('Sample Index')
    plt.ylabel('Magnitude')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
