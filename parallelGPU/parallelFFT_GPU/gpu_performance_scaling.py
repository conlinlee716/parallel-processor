import numpy as np
import cupy as cp
import time
import matplotlib.pyplot as plt
from scipy.fft import fft as cpu_fft

# 导入现有的处理器类
from parallelFFT import FFTParallelProcessor as CPUFFTParallelProcessor
from parallelFFT_GPU import GPUFFTProcessor

class GPUScalingTester:
    """测试GPU性能随数据规模的变化"""
    
    def __init__(self):
        self.fs = 40e9
        self.duration = 1e-8
    
    def generate_signal(self, length):
        """生成指定长度的测试信号"""
        t = np.arange(0, self.duration, 1/self.fs)[:length]
        # 生成复合信号：1GHz正弦波 + 5GHz正弦波
        signal = np.sin(2 * np.pi * 1e9 * t) + 0.5 * np.sin(2 * np.pi * 5e9 * t)
        return signal, t
    
    def test_performance_scaling(self, fft_sizes):
        """测试不同FFT大小下的性能"""
        results = []
        
        print("\n" + "=" * 70)
        print("GPU PERFORMANCE SCALING TEST")
        print("=" * 70)
        print("FFT Size | Regular (ms) | Parallel (ms) | Speedup | Relative Error")
        print("-" * 70)
        
        for size in fft_sizes:
            # 生成信号
            signal, _ = self.generate_signal(size)
            
            # 创建单个GPU处理器
            processor_gpu = GPUFFTProcessor(size, num_channels=2, fs=self.fs)
            
            # 普通GPU FFT性能测试
            time_regular_list = []
            fft_regular_list = []
            
            # 运行3次，取平均值
            for _ in range(3):
                start_regular = time.time()
                fft_regular, regular_time, _ = processor_gpu.regular_fft(signal)
                time_regular_list.append(regular_time)
                fft_regular_list.append(fft_regular)
            
            time_regular = np.mean(time_regular_list)
            # 使用最后一次运行的FFT结果进行误差计算，避免平均导致的精度损失
            fft_regular = fft_regular_list[-1]
            
            # 并行GPU FFT性能测试
            time_parallel_list = []
            fft_parallel_list = []
            
            # 运行3次，取平均值
            for _ in range(3):
                start_parallel = time.time()
                fft_parallel, _, parallel_time, _ = processor_gpu.parallel_fft(signal)
                time_parallel_list.append(parallel_time)
                fft_parallel_list.append(fft_parallel)
            
            time_parallel = np.mean(time_parallel_list)
            # 使用最后一次运行的FFT结果进行误差计算
            fft_parallel = fft_parallel_list[-1]
            
            # 计算误差 - 使用更稳定的计算方法
            # 避免除零错误，使用L2范数计算相对误差
            diff = fft_regular - fft_parallel
            error = np.mean(np.abs(diff))
            # 使用L2范数计算相对误差，更稳定
            norm_regular = np.linalg.norm(fft_regular)
            norm_diff = np.linalg.norm(diff)
            relative_error = norm_diff / (norm_regular + 1e-12)  # 添加小常数避免除零
            
            # 计算加速比（并行 vs 普通）
            speedup = time_regular / time_parallel if time_parallel > 0 else float('inf')
            
            # 存储结果
            results.append({
                'fft_size': size,
                'regular_time_ms': time_regular * 1000,
                'parallel_time_ms': time_parallel * 1000,
                'speedup': speedup,
                'error': error,
                'relative_error': relative_error
            })
            
            # 打印结果
            print(f"{size:9} | {time_regular*1000:13.2f} | {time_parallel*1000:13.2f} | {speedup:7.2f}x | {relative_error:14.2e}")
        
        return results
    
    def plot_scaling_results(self, results):
        """绘制性能缩放结果"""
        fft_sizes = [r['fft_size'] for r in results]
        regular_times = [r['regular_time_ms'] for r in results]
        parallel_times = [r['parallel_time_ms'] for r in results]
        speedups = [r['speedup'] for r in results]
        
        # 创建图表
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # 执行时间对比
        ax1 = axes[0]
        ax1.loglog(fft_sizes, regular_times, 'o-', linewidth=2, markersize=8, label='Regular GPU FFT')
        ax1.loglog(fft_sizes, parallel_times, 's-', linewidth=2, markersize=8, label='Parallel GPU FFT')
        ax1.set_xlabel('FFT Size')
        ax1.set_ylabel('Execution Time (ms)')
        ax1.set_title('Execution Time vs FFT Size (Regular vs Parallel GPU FFT)')
        ax1.legend()
        ax1.grid(True, alpha=0.3, which='both')
        
        # 加速比
        ax2 = axes[1]
        ax2.semilogx(fft_sizes, speedups, 'o-', linewidth=2, markersize=8, color='purple')
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Baseline (Speedup = 1)')
        ax2.set_xlabel('FFT Size')
        ax2.set_ylabel('Speedup (x)')
        ax2.set_title('Parallel GPU FFT Speedup vs FFT Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        plt.savefig('gpu_scaling_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_error_analysis(self, results):
        """绘制误差分析图表"""
        fft_sizes = [r['fft_size'] for r in results]
        relative_errors = [r['relative_error'] for r in results]
        
        # 过滤掉异常大的误差值（可能是数值问题）
        valid_indices = [i for i, err in enumerate(relative_errors) if err < 1.0 and err > 0]
        if not valid_indices:
            print("警告: 所有相对误差值都异常，可能存在问题")
            # 仍然绘制，但添加警告
            valid_indices = range(len(relative_errors))
        
        valid_sizes = [fft_sizes[i] for i in valid_indices]
        valid_errors = [relative_errors[i] for i in valid_indices]
        
        plt.figure(figsize=(10, 6))
        
        # 绘制有效数据点
        if valid_sizes:
            plt.loglog(valid_sizes, valid_errors, 'o-', linewidth=2, markersize=8, color='red', label='相对误差')
        
        # 如果有一些异常值，用不同颜色标记
        invalid_indices = [i for i in range(len(relative_errors)) if i not in valid_indices]
        if invalid_indices:
            invalid_sizes = [fft_sizes[i] for i in invalid_indices]
            invalid_errors = [relative_errors[i] for i in invalid_indices]
            plt.loglog(invalid_sizes, invalid_errors, 'x', markersize=10, color='orange', label='异常值（需检查）')
        
        plt.xlabel('FFT Size', fontsize=14)
        plt.ylabel('Relative Error', fontsize=14)
        plt.title('数值精度 vs FFT大小 (并行FFT vs 常规FFT)', fontsize=14)
        plt.grid(True, alpha=0.3, which='both')
        plt.legend()
        
        # 添加参考线：典型数值误差范围
        if valid_errors:
            min_error = min(valid_errors)
            max_error = max(valid_errors)
            if max_error > min_error * 10:
                # 如果误差范围很大，添加说明
                plt.text(0.05, 0.95, f'误差范围: {min_error:.2e} - {max_error:.2e}', 
                        transform=plt.gca().transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('gpu_error_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n误差分析图表已保存: gpu_error_analysis.png")
        print(f"有效数据点: {len(valid_sizes)}, 异常值: {len(invalid_indices)}")
        if valid_errors:
            print(f"相对误差范围: {min(valid_errors):.2e} - {max(valid_errors):.2e}")
        plt.show()

if __name__ == "__main__":
    # 测试不同的FFT大小（从较小到较大）- 扩展范围
    fft_sizes = [
        100,    # 非常小的FFT
        500,    # 小FFT
        1000,   # 原始范围起点
        5000,   # 中间值
        10000,  # 中间值
        50000,  # 中间值
        100000, # 中间值
        500000, # 原始范围终点
        1000000,# 大FFT
        2000000 # 非常大的FFT
    ]
    
    # 创建测试器
    tester = GPUScalingTester()
    
    # 运行性能测试
    results = tester.test_performance_scaling(fft_sizes)
    
    # 绘制结果
    tester.plot_scaling_results(results)
    tester.plot_error_analysis(results)
    
    print("\n" + "=" * 70)
    print("SCALING TEST COMPLETE!")
    print("=" * 70)