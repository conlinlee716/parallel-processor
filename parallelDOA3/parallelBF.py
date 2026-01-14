import numpy as np
from scipy.signal import get_window, resample_poly
import matplotlib.pyplot as plt

class WidebanBeamformerParallel:
    """宽带MVDR波束成形并行实现"""
    
    def __init__(self, M, fs, fc=None, num_subbands=8, fft_len=512):
        """
        M: 阵元数
        fs: 采样率
        fc: 中心频率（用于TTD对准），若None则用nyquist
        num_subbands: 子带数
        fft_len: STFT长度
        """
        self.M = M
        self.fs = fs
        self.fc = fc if fc else fs / 2
        self.K = num_subbands
        self.fft_len = fft_len
        self.freq_bins = np.fft.rfftfreq(fft_len, 1/fs)
        
        # 子带中心频率
        self.subband_freqs = self.freq_bins[1:self.fft_len//2:max(1, len(self.freq_bins)//self.K)][:self.K]
        
    def stft_analysis(self, x, hop_len=None):
        """
        x: (M, N) 时域信号
        return: (M, K, T) STFT频域，K是频点数，T是时间帧数
        """
        if hop_len is None:
            hop_len = self.fft_len // 2
        M, N = x.shape
        window = get_window('hann', self.fft_len)
        
        # 逐阵元做STFT
        num_frames = (N - self.fft_len) // hop_len + 1
        X = np.zeros((M, self.fft_len//2+1, num_frames), dtype=complex)
        
        for m in range(M):
            for t in range(num_frames):
                start = t * hop_len
                frame = x[m, start:start+self.fft_len] * window
                X[m, :, t] = np.fft.rfft(frame, self.fft_len)
        return X, hop_len
    
    def subband_select(self, X):
        """
        从STFT X中提取K个子带
        return: (M, K, T) 子带复信号
        """
        M, num_freqs, T = X.shape
        bin_indices = np.linspace(0, num_freqs-1, self.K, dtype=int)
        return X[:, bin_indices, :], bin_indices
    
    def estimate_cov(self, x_subband):
        """
        估计子带协方差
        x_subband: (M, T) 单个子带信号
        return: (M, M) 协方差矩阵
        """
        x_subband = x_subband - np.mean(x_subband, axis=1, keepdims=True)
        R = (x_subband @ x_subband.conj().T) / x_subband.shape[1]
        return R
    
    def regularize_cov(self, R, alpha=0.01):
        """
        对角加载与前后向平均
        """
        # 前后向平均（对ULA）
        R = (R + np.fliplr(np.conj(R)).T) / 2
        # 对角加载
        R = R + alpha * np.trace(R) / self.M * np.eye(self.M)
        return R
    
    def mvdr_weight(self, R, a, fc=None):
        """
        计算MVDR权向量
        R: (M, M) 协方差
        a: (M,) 导向向量
        return: (M,) 权向量
        """
        try:
            Rinv = np.linalg.inv(R)
        except:
            Rinv = np.linalg.pinv(R)
        w = Rinv @ a / (a.conj() @ Rinv @ a)
        return w
    
    def steer_vector(self, theta, f, c=343):
        """
        导向向量
        theta: 方向角(rad)
        f: 频率
        c: 声速
        """
        d = c / (2 * f) if f > 0 else c / (2 * self.fc)  # 相邻阵元间距
        k = 2 * np.pi * f / c
        m = np.arange(self.M)
        return np.exp(1j * k * d * m * np.sin(theta))
    
    def ttd_focusing(self, x_k, f_k, theta, c=343):
        """
        真时延(TTD)对准到参考角度
        x_k: (M, T) 子带信号
        f_k: 子带频率
        theta: 目标角度
        return: (M, T) 对准后信号
        """
        d = c / (2 * self.fc)  # 基于中心频率的阵元间距
        # 计算各阵元相对延时
        tau = -d * np.arange(self.M) * np.sin(theta) / c
        # 近似用相移替代分数延时
        phase = 2 * np.pi * f_k * tau[:, np.newaxis]
        x_focused = x_k * np.exp(1j * phase)
        return x_focused
    
    def process_wideband(self, x, theta_target, c=343):
        """
        完整宽带MVDR流程
        x: (M, N) 输入信号
        theta_target: 目标方向(rad)
        return: output (M, N), spectrum
        """
        # 1. STFT分析
        X, hop_len = self.stft_analysis(x)
        
        # 2. 子带提取
        X_sub, bin_idx = self.subband_select(X)
        
        # 3. 各子带并行处理
        weights_sub = []
        y_sub = []
        
        for k in range(self.K):
            x_k = X_sub[:, k, :]  # (M, T)
            f_k = self.subband_freqs[k]
            
            # TTD对准
            x_focused = self.ttd_focusing(x_k, f_k, theta_target, c)
            
            # 协方差估计与正则化
            R_k = self.estimate_cov(x_focused)
            R_k = self.regularize_cov(R_k, alpha=0.02)
            
            # 导向向量（对准后用参考频率）
            a_k = self.steer_vector(theta_target, self.fc, c)
            
            # MVDR权
            w_k = self.mvdr_weight(R_k, a_k, self.fc)
            weights_sub.append(w_k)
            
            # 波束输出
            y_k = w_k.conj() @ x_focused  # (T,)
            y_sub.append(y_k)
        
        # 4. 相干融合
        y_fused = np.mean(np.array(y_sub), axis=0)  # (T,)
        
        # 5. iSTFT重构
        X_recon = np.zeros_like(X)
        for k in range(self.K):
            X_recon[:, bin_idx[k], :] = weights_sub[k][:, np.newaxis] * X_sub[:, k, :]
        
        output = self.istft_synthesis(X_recon, hop_len)
        
        return output, np.abs(y_fused)
    
    def istft_synthesis(self, X, hop_len):
        """iSTFT重构"""
        M, num_freqs, T = X.shape
        window = get_window('hann', self.fft_len)
        output_len = (T - 1) * hop_len + self.fft_len
        output = np.zeros((M, output_len))
        
        for m in range(M):
            for t in range(T):
                frame = np.fft.irfft(X[m, :, t], self.fft_len).real
                frame *= window
                start = t * hop_len
                output[m, start:start+self.fft_len] += frame
        
        return output
    
    def compute_spectrum(self, x, theta_range, c=343):
        """
        计算DOA谱（全角度扫描）
        """
        X, _ = self.stft_analysis(x)
        X_sub, bin_idx = self.subband_select(X)
        
        P = np.zeros(len(theta_range))
        
        for k in range(self.K):
            x_k = X_sub[:, k, :]
            f_k = self.subband_freqs[k]
            R_k = self.estimate_cov(x_k)
            R_k = self.regularize_cov(R_k, alpha=0.02)
            Rinv = np.linalg.inv(R_k + 1e-6*np.eye(self.M))
            
            for idx, theta in enumerate(theta_range):
                a = self.steer_vector(theta, f_k, c)
                P[idx] += 1.0 / np.abs(a.conj() @ Rinv @ a)
        
        return P / self.K

# ============ 测试与演示 ============
if __name__ == "__main__":
    # 参数
    M = 16  # 阵元数
    fs = 16000  # 采样率
    fc = 4000  # 中心频率
    N = 8000  # 样本数
    SNR = 10  # dB
    
    # 生成测试信号
    c = 343  # 声速
    theta_true = np.pi / 6  # 30度
    theta_jammer = -np.pi / 4  # -45度
    
    # 目标信号
    t = np.arange(N) / fs
    s_target = np.sin(2 * np.pi * fc * t) + 0.3 * np.sin(2 * np.pi * (fc-500) * t)
    
    # 导向向量
    d = c / (2 * fc)
    a_target = np.exp(1j * 2 * np.pi * fc / c * d * np.arange(M) * np.sin(theta_true))
    a_jammer = np.exp(1j * 2 * np.pi * fc / c * d * np.arange(M) * np.sin(theta_jammer))
    
    # 构造输入信号（阵列观测）
    x = np.outer(a_target, s_target).real  # (M, N)
    
    # 加干扰
    s_jammer = 2 * np.sin(2 * np.pi * (fc+800) * t)
    x += 1.5 * np.outer(a_jammer, s_jammer).real
    
    # 加噪声
    noise = np.random.randn(M, N) * 0.2
    x += noise
    
    # 波束成形
    bf = WidebanBeamformerParallel(M, fs, fc=fc, num_subbands=8, fft_len=512)
    
    # 处理
    y_output, y_waveform = bf.process_wideband(x, theta_true, c=c)
    
    # DOA谱
    theta_scan = np.linspace(-np.pi/2, np.pi/2, 181)
    P_spectrum = bf.compute_spectrum(x, theta_scan, c=c)
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0,0].plot(t[:1000], s_target[:1000], 'b-', label='Target')
    axes[0,0].set_title('Target Signal')
    axes[0,0].legend()
    axes[0,0].grid()
    
    axes[0,1].plot(np.arange(M), np.abs(x[:, 100]), 'ro-')
    axes[0,1].set_title('Array Snapshot @ sample 100')
    axes[0,1].set_xlabel('Element Index')
    axes[0,1].grid()
    
    axes[1,0].plot(t[:2000], y_waveform[:2000], 'g-')
    axes[1,0].set_title('Beamformed Output (TTD+MVDR)')
    axes[1,0].grid()
    
    axes[1,1].plot(np.degrees(theta_scan), 10*np.log10(P_spectrum+1e-10), 'b-', linewidth=2)
    axes[1,1].axvline(np.degrees(theta_true), color='r', linestyle='--', label=f'Target {np.degrees(theta_true):.1f}°')
    axes[1,1].axvline(np.degrees(theta_jammer), color='orange', linestyle='--', label=f'Jammer {np.degrees(theta_jammer):.1f}°')
    axes[1,1].set_title('DOA Spectrum (dB)')
    axes[1,1].set_xlabel('Angle (degrees)')
    axes[1,1].set_ylabel('Power (dB)')
    axes[1,1].legend()
    axes[1,1].grid()
    axes[1,1].set_xlim(-90, 90)
    
    plt.tight_layout()
    plt.show()
