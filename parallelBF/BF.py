import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Use Arial font
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
rcParams['font.family'] = 'sans-serif'
rcParams['axes.unicode_minus'] = False

class MVDRBeamformer:
    def __init__(self, num_elements=4, wavelength=1.0, element_spacing=0.5):
        """
        初始化MVDR波束成形器
        
        参数:
            num_elements: 阵元数量
            wavelength: 波长
            element_spacing: 阵元间距（波长的倍数）
        """
        self.num_elements = num_elements
        self.wavelength = wavelength
        self.element_spacing = element_spacing
        self.array_positions = np.arange(num_elements) * element_spacing
        
    def steering_vector(self, theta):
        """
        计算导向向量
        
        参数:
            theta: 入射角（度）
        返回:
            导向向量
        """
        theta_rad = np.radians(theta)
        phase = 2 * np.pi * self.array_positions * np.sin(theta_rad) / self.wavelength
        return np.exp(1j * phase)
    
    def mvdr_beamformer(self, R, theta_desired):
        """
        计算MVDR波束成形权重
        
        参数:
            R: 协方差矩阵 (num_elements x num_elements)
            theta_desired: 期望方向（度）
        返回:
            波束成形权重
        """
        a = self.steering_vector(theta_desired)
        
        # 计算逆矩阵
        try:
            R_inv = np.linalg.inv(R)
        except:
            # 如果矩阵奇异，使用伪逆
            R_inv = np.linalg.pinv(R)
        
        # MVDR权重: w = (R^-1 * a) / (a^H * R^-1 * a)
        numerator = R_inv @ a
        denominator = np.conj(a).T @ numerator
        w = numerator / (denominator + 1e-10)
        
        return w
    
    def beampattern(self, R, theta_desired, theta_range=None):
        """
        计算波束图
        
        参数:
            R: 协方差矩阵
            theta_desired: 期望方向（度）
            theta_range: 角度范围
        返回:
            角度数组, 归一化功率数组
        """
        if theta_range is None:
            theta_range = np.linspace(-90, 90, 361)
        
        w = self.mvdr_beamformer(R, theta_desired)
        
        power = np.zeros(len(theta_range))
        for i, theta in enumerate(theta_range):
            a = self.steering_vector(theta)
            power[i] = np.abs(np.conj(w).T @ a) ** 2
        
        # 归一化
        power_norm = power / np.max(power)
        power_db = 10 * np.log10(power_norm + 1e-10)
        
        return theta_range, power_db, w
    
    def estimate_doa(self, R, search_range=None, resolution=0.5):
        """
        DOA估计（扫描法）
        
        参数:
            R: 协方差矩阵
            search_range: 搜索角度范围
            resolution: 搜索分辨率（度）
        返回:
            DOA角度数组, 对应的MVDR谱
        """
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
        
        # 归一化
        spectrum_norm = spectrum / np.max(spectrum)
        spectrum_db = 10 * np.log10(spectrum_norm + 1e-10)
        
        return search_range, spectrum_db


def simulate_signals():
    """
    模拟信号场景
    
    返回:
        - 接收数据矩阵
        - 协方差矩阵
        - 真实DOA
        - beamformer对象
    """
    # 系统参数
    fs = 40e9  # 采样率 40GHz
    f_signal = 11e9  # 信号频率 11GHz
    c = 3e8  # 光速
    
    num_elements = 4
    num_snapshots = 200000
    snr_db = 15
    
    # 真实DOA
    true_doa = -30  # 度
    
    # 波长
    wavelength = c / f_signal
    
    # 阵元间距（半波长）
    element_spacing = wavelength / 2
    
    print(f"系统采样率: {fs/1e9:.2f} GHz")
    print(f"信号频率: {f_signal/1e9:.2f} GHz")
    print(f"波长: {wavelength*1e3:.4f} mm")
    print(f"阵元间距: {element_spacing*1e3:.4f} mm")
    
    # 时间向量
    t = np.arange(num_snapshots) / fs
    
    # 生成11GHz正弦信号（复数形式）
    signal = np.exp(1j * 2 * np.pi * f_signal * t)
    
    # 创建波束成形器
    beamformer = MVDRBeamformer(num_elements=num_elements, 
                                wavelength=wavelength,
                                element_spacing=element_spacing)
    
    # 导向向量
    a = beamformer.steering_vector(true_doa)
    
    # 接收数据：信号功率
    signal_power = 10 ** (snr_db / 20)
    X = np.outer(a, signal) * signal_power
    
    # 加入噪声
    noise = (np.random.randn(num_elements, num_snapshots) + 
             1j * np.random.randn(num_elements, num_snapshots)) / np.sqrt(2)
    X += noise
    
    # 计算协方差矩阵
    R = (X @ np.conj(X.T)) / num_snapshots
    
    return X, R, true_doa, beamformer


def main():
    # 仿真信号
    print("="*60)
    print("4阵元MVDR波束成形仿真")
    print("="*60)
    print()
    
    X, R, true_doa, beamformer = simulate_signals()
    
    print(f"\n真实DOA: {true_doa}°")
    print(f"阵元数量: {beamformer.num_elements}")
    print("="*60)
    
    print("\n进行DOA估计...")
    
    # DOA估计
    search_angles, spectrum_db = beamformer.estimate_doa(R, resolution=0.5)
    
    # 找到峰值
    peaks_idx = []
    threshold = np.max(spectrum_db) - 3  # 3dB下降
    
    for i in range(1, len(spectrum_db) - 1):
        if (spectrum_db[i] > spectrum_db[i-1] and 
            spectrum_db[i] > spectrum_db[i+1] and 
            spectrum_db[i] > threshold):
            peaks_idx.append(i)
    
    if len(peaks_idx) > 0:
        estimated_doa = search_angles[peaks_idx[0]]
        print(f"估计DOA: {estimated_doa:.2f}°")
        print(f"DOA估计误差: {np.abs(estimated_doa - true_doa):.2f}°")
    else:
        estimated_doa = None
        print("未检测到信号")
    
    # 计算波束图
    theta_range = np.linspace(-90, 90, 361)
    theta_bf, power_db, w = beamformer.beampattern(R, true_doa, theta_range)
    
    # 创建图表 - 仅显示3个图
    fig = plt.figure(figsize=(15, 4.5))
    
    # Figure 1: DOA Spectrum (MVDR)
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(search_angles, spectrum_db, 'b-', linewidth=2, label='MVDR Spectrum')
    ax1.axvline(true_doa, color='red', linestyle='--', linewidth=2, label=f'True DOA: {true_doa}°')
    if estimated_doa is not None:
        idx = np.argmin(np.abs(search_angles - estimated_doa))
        ax1.scatter(estimated_doa, spectrum_db[idx],
                   color='green', s=200, marker='o', label=f'Estimated DOA: {estimated_doa:.1f}°', 
                   zorder=5, edgecolors='darkgreen', linewidths=2)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlabel('Angle (°)', fontsize=20, fontweight='bold', fontname='Arial')
    ax1.set_ylabel('Power (dB)', fontsize=20, fontweight='bold', fontname='Arial')
    ax1.set_title('DOA Estimate (MVDR)', fontsize=18, fontweight='bold', fontname='Arial')
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.legend(fontsize=12, loc='upper right')
    ax1.set_xlim(-90, 90)
    
    # Figure 2: Beampattern (Cartesian)
    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(theta_bf, power_db, 'b-', linewidth=2)
    ax2.axvline(true_doa, color='red', linestyle='--', linewidth=2, label=f'True DOA: {true_doa}°')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlabel('Angle (°)', fontsize=20, fontweight='bold', fontname='Arial')
    ax2.set_ylabel('Gain (dB)', fontsize=20, fontweight='bold', fontname='Arial')
    ax2.set_title('Beampattern (Cartesian)', fontsize=18, fontweight='bold', fontname='Arial')
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.set_xlim(-90, 90)
    ax2.set_ylim(np.min(power_db), 5)
    ax2.legend(fontsize=12, loc='upper right')
    
    # Figure 3: Beampattern (Polar)
    ax3 = plt.subplot(1, 3, 3, projection='polar')
    theta_rad = np.radians(theta_bf)
    power_linear = 10 ** (power_db / 10)
    ax3.plot(theta_rad, power_linear, 'b-', linewidth=2)
    ax3.axvline(np.radians(true_doa), color='red', linestyle='--', linewidth=2, label=f'True DOA: {true_doa}°')
    ax3.set_theta_zero_location('N')
    ax3.set_theta_direction(-1)
    # show only upper half
    ax3.set_thetamin(-90)
    ax3.set_thetamax(90)
    ax3.set_title('Beampattern (Polar)', fontsize=18, fontweight='bold', pad=20, fontname='Arial')
    ax3.tick_params(axis='both', which='major', labelsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=12, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('mvdr_beamformer_4elements.png', dpi=150, bbox_inches='tight')
    print("\n✓ 图表已保存为 'mvdr_beamformer_4elements.png'")
    plt.show()
    
    # 打印详细信息
    print("\n" + "="*60)
    print("MVDR波束成形仿真结果汇总")
    print("="*60)
    print(f"系统采样率: 40 GHz")
    print(f"信号频率: 11 GHz")
    print(f"波长: {beamformer.wavelength*1e3:.4f} mm")
    print(f"阵元数量: {beamformer.num_elements}")
    print(f"阵元间距: {beamformer.element_spacing*1e3:.4f} mm")
    print(f"真实DOA: {true_doa}°")
    if estimated_doa is not None:
        print(f"估计DOA: {estimated_doa:.2f}°")
        print(f"估计误差: {np.abs(estimated_doa - true_doa):.2f}°")
    print("="*60)


if __name__ == "__main__":
    np.random.seed(42)
    main()
