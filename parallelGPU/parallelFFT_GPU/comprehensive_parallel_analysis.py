"""
全面的并行FFT性能分析
分析不同通道数K=2,4,8,16下的性能表现，讨论是否能达到K倍加速
"""
import numpy as np
import cupy as cp
import time
import matplotlib.pyplot as plt
from parallelFFT_GPU import GPUFFTProcessor
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

class ComprehensiveParallelAnalyzer:
    """全面的并行FFT性能分析器"""
    
    def __init__(self):
        self.fs = 40e9
        self.duration = 1e-8
    
    def generate_signal(self, length):
        """生成指定长度的测试信号"""
        t = np.arange(0, self.duration, 1/self.fs)[:length]
        # 生成复合信号：1GHz正弦波 + 5GHz正弦波
        signal = np.sin(2 * np.pi * 1e9 * t) + 0.5 * np.sin(2 * np.pi * 5e9 * t)
        return signal
    
    def test_multiple_channels(self, fft_size, channel_counts=[2, 4, 8, 16]):
        """测试不同通道数K下的性能"""
        print(f"\n{'='*80}")
        print(f"测试FFT大小: {fft_size}, 通道数: {channel_counts}")
        print(f"{'='*80}")
        
        signal = self.generate_signal(fft_size)
        results = []
        
        # 首先测试常规GPU FFT作为基准
        processor_regular = GPUFFTProcessor(fft_size, num_channels=2, fs=self.fs)
        regular_times = []
        for _ in range(5):  # 多次运行取平均
            _, regular_time, _ = processor_regular.regular_fft(signal)
            regular_times.append(regular_time)
        baseline_time = np.mean(regular_times)
        baseline_std = np.std(regular_times)
        
        print(f"\n基准 (Regular GPU FFT):")
        print(f"  平均时间: {baseline_time*1000:.3f} ms (std: {baseline_std*1000:.3f} ms)")
        
        # 测试不同通道数
        for K in channel_counts:
            if fft_size % K != 0:
                print(f"\n跳过 K={K} (FFT大小 {fft_size} 不能被 {K} 整除)")
                continue
            
            print(f"\n测试 K={K}:")
            print("-" * 60)
            
            try:
                processor = GPUFFTProcessor(fft_size, num_channels=K, fs=self.fs)
                
                # 并行FFT性能测试
                parallel_times = []
                fft_parallel_list = []
                fft_regular_list = []
                
                # 多次运行取平均
                for _ in range(5):
                    # 并行FFT
                    fft_parallel, _, parallel_time, _ = processor.parallel_fft(signal)
                    parallel_times.append(parallel_time)
                    fft_parallel_list.append(fft_parallel)
                    
                    # 常规FFT用于误差计算
                    fft_regular, _, _ = processor.regular_fft(signal)
                    fft_regular_list.append(fft_regular)
                
                avg_parallel_time = np.mean(parallel_times)
                std_parallel_time = np.std(parallel_times)
                
                # 计算误差（使用最后一次运行的结果）
                fft_parallel_avg = np.mean(fft_parallel_list, axis=0)
                fft_regular_avg = np.mean(fft_regular_list, axis=0)
                
                error = np.mean(np.abs(fft_regular_avg - fft_parallel_avg))
                relative_error = error / (np.mean(np.abs(fft_regular_avg)) + 1e-12)
                
                # 计算加速比
                speedup_vs_baseline = baseline_time / avg_parallel_time if avg_parallel_time > 0 else 0
                theoretical_speedup = K  # 理论上的K倍加速
                efficiency = speedup_vs_baseline / theoretical_speedup * 100  # 效率百分比
                
                # 分析各个阶段的时间
                # 分解阶段（估算）
                decompose_time = 0  # 分解是内存操作，很快
                
                # FFT阶段（估算）：K个并行FFT，每个大小为N/K
                # 由于并行执行，实际时间约等于单个FFT的时间
                single_fft_size = fft_size // K
                # 重建阶段时间（估算）：总时间 - FFT时间
                # 重建时间 = 总时间 - 分解时间 - FFT时间
                
                results.append({
                    'K': K,
                    'fft_size': fft_size,
                    'baseline_time_ms': baseline_time * 1000,
                    'parallel_time_ms': avg_parallel_time * 1000,
                    'parallel_time_std_ms': std_parallel_time * 1000,
                    'speedup': speedup_vs_baseline,
                    'theoretical_speedup': theoretical_speedup,
                    'efficiency_percent': efficiency,
                    'relative_error': relative_error,
                    'error': error,
                    'single_fft_size': single_fft_size
                })
                
                print(f"  并行时间: {avg_parallel_time*1000:.3f} ms (std: {std_parallel_time*1000:.3f} ms)")
                print(f"  加速比: {speedup_vs_baseline:.2f}x (理论: {theoretical_speedup}x, 效率: {efficiency:.1f}%)")
                print(f"  相对误差: {relative_error:.2e}")
                print(f"  单个子通道FFT大小: {single_fft_size}")
                
            except Exception as e:
                print(f"  错误: {e}")
                continue
        
        return results
    
    def analyze_scaling_behavior(self, fft_sizes, channel_counts=[2, 4, 8, 16]):
        """分析不同FFT大小和通道数下的缩放行为"""
        all_results = []
        
        for fft_size in fft_sizes:
            results = self.test_multiple_channels(fft_size, channel_counts)
            all_results.extend(results)
        
        return all_results
    
    def plot_comprehensive_analysis(self, results):
        """绘制全面的分析图表"""
        if not results:
            print("没有结果可绘制")
            return
        
        # 按K值分组
        results_by_K = {}
        for r in results:
            K = r['K']
            if K not in results_by_K:
                results_by_K[K] = []
            results_by_K[K].append(r)
        
        # 创建图表
        fig = plt.figure(figsize=(18, 12))
        
        # 1. 执行时间对比（不同K值）
        ax1 = plt.subplot(2, 3, 1)
        for K in sorted(results_by_K.keys()):
            data = results_by_K[K]
            fft_sizes = [r['fft_size'] for r in data]
            times = [r['parallel_time_ms'] for r in data]
            ax1.loglog(fft_sizes, times, 'o-', linewidth=2, markersize=6, label=f'K={K}')
        
        # 添加基准线
        baseline_data = results_by_K.get(2, [])
        if baseline_data:
            baseline_fft_sizes = [r['fft_size'] for r in baseline_data]
            baseline_times = [r['baseline_time_ms'] for r in baseline_data]
            ax1.loglog(baseline_fft_sizes, baseline_times, 'k--', linewidth=2, label='Baseline (Regular)')
        
        ax1.set_xlabel('FFT Size')
        ax1.set_ylabel('Execution Time (ms)')
        ax1.set_title('执行时间 vs FFT大小 (不同K值)')
        ax1.legend()
        ax1.grid(True, alpha=0.3, which='both')
        
        # 2. 加速比 vs FFT大小
        ax2 = plt.subplot(2, 3, 2)
        for K in sorted(results_by_K.keys()):
            data = results_by_K[K]
            fft_sizes = [r['fft_size'] for r in data]
            speedups = [r['speedup'] for r in data]
            theoretical = [r['theoretical_speedup'] for r in data]
            ax2.semilogx(fft_sizes, speedups, 'o-', linewidth=2, markersize=6, label=f'K={K} (实际)')
            ax2.semilogx(fft_sizes, theoretical, '--', linewidth=1, alpha=0.5, label=f'K={K} (理论)')
        
        ax2.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
        ax2.set_xlabel('FFT Size')
        ax2.set_ylabel('Speedup (x)')
        ax2.set_title('加速比 vs FFT大小')
        ax2.legend()
        ax2.grid(True, alpha=0.3, which='both')
        
        # 3. 效率 vs FFT大小
        ax3 = plt.subplot(2, 3, 3)
        for K in sorted(results_by_K.keys()):
            data = results_by_K[K]
            fft_sizes = [r['fft_size'] for r in data]
            efficiencies = [r['efficiency_percent'] for r in data]
            ax3.semilogx(fft_sizes, efficiencies, 'o-', linewidth=2, markersize=6, label=f'K={K}')
        
        ax3.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='100% 效率')
        ax3.set_xlabel('FFT Size')
        ax3.set_ylabel('Efficiency (%)')
        ax3.set_title('并行效率 vs FFT大小')
        ax3.legend()
        ax3.grid(True, alpha=0.3, which='both')
        ax3.set_ylim([0, 120])
        
        # 4. 加速比 vs K值（固定FFT大小）
        ax4 = plt.subplot(2, 3, 4)
        # 按FFT大小分组
        results_by_size = {}
        for r in results:
            size = r['fft_size']
            if size not in results_by_size:
                results_by_size[size] = []
            results_by_size[size].append(r)
        
        for size in sorted(results_by_size.keys())[::2]:  # 每隔一个显示，避免太拥挤
            data = sorted(results_by_size[size], key=lambda x: x['K'])
            Ks = [r['K'] for r in data]
            speedups = [r['speedup'] for r in data]
            theoretical = [r['theoretical_speedup'] for r in data]
            ax4.plot(Ks, speedups, 'o-', linewidth=2, markersize=6, label=f'N={size} (实际)')
            ax4.plot(Ks, theoretical, '--', linewidth=1, alpha=0.5, label=f'N={size} (理论)')
        
        ax4.set_xlabel('Number of Channels (K)')
        ax4.set_ylabel('Speedup (x)')
        ax4.set_title('加速比 vs 通道数K')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks([2, 4, 8, 16])
        
        # 5. 相对误差 vs FFT大小
        ax5 = plt.subplot(2, 3, 5)
        for K in sorted(results_by_K.keys()):
            data = results_by_K[K]
            fft_sizes = [r['fft_size'] for r in data]
            errors = [r['relative_error'] for r in data]
            ax5.loglog(fft_sizes, errors, 'o-', linewidth=2, markersize=6, label=f'K={K}')
        
        ax5.set_xlabel('FFT Size')
        ax5.set_ylabel('Relative Error')
        ax5.set_title('数值精度 vs FFT大小')
        ax5.legend()
        ax5.grid(True, alpha=0.3, which='both')
        
        # 6. 效率 vs K值（固定FFT大小）
        ax6 = plt.subplot(2, 3, 6)
        for size in sorted(results_by_size.keys())[::2]:
            data = sorted(results_by_size[size], key=lambda x: x['K'])
            Ks = [r['K'] for r in data]
            efficiencies = [r['efficiency_percent'] for r in data]
            ax6.plot(Ks, efficiencies, 'o-', linewidth=2, markersize=6, label=f'N={size}')
        
        ax6.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='100% 效率')
        ax6.set_xlabel('Number of Channels (K)')
        ax6.set_ylabel('Efficiency (%)')
        ax6.set_title('并行效率 vs 通道数K')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_xticks([2, 4, 8, 16])
        ax6.set_ylim([0, 120])
        
        plt.tight_layout()
        plt.savefig('comprehensive_parallel_analysis.png', dpi=300, bbox_inches='tight')
        print("\n图表已保存: comprehensive_parallel_analysis.png")
        plt.show()
    
    def generate_analysis_report(self, results):
        """生成分析报告"""
        print("\n" + "="*80)
        print("并行FFT性能分析报告")
        print("="*80)
        
        # 按K值分组分析
        results_by_K = {}
        for r in results:
            K = r['K']
            if K not in results_by_K:
                results_by_K[K] = []
            results_by_K[K].append(r)
        
        print("\n关键发现:")
        print("-" * 80)
        
        for K in sorted(results_by_K.keys()):
            data = results_by_K[K]
            avg_speedup = np.mean([r['speedup'] for r in data])
            avg_efficiency = np.mean([r['efficiency_percent'] for r in data])
            avg_error = np.mean([r['relative_error'] for r in data])
            
            print(f"\nK={K}:")
            print(f"  平均加速比: {avg_speedup:.2f}x (理论: {K}x)")
            print(f"  平均效率: {avg_efficiency:.1f}%")
            print(f"  平均相对误差: {avg_error:.2e}")
            
            # 分析为什么无法达到K倍加速
            print(f"  分析:")
            if avg_efficiency < 50:
                print(f"    - 效率较低，可能原因:")
                print(f"      1. 重建阶段开销过大（需要合并K个通道的结果）")
                print(f"      2. 子通道FFT太小，GPU利用率不足")
                print(f"      3. 内存访问模式不优化")
            elif avg_efficiency < 80:
                print(f"    - 效率中等，主要瓶颈:")
                print(f"      1. 重建阶段的开销限制了加速比")
                print(f"      2. 随着K增加，重建复杂度O(K*N)增加")
            else:
                print(f"    - 效率较高，接近理论值")
        
        print("\n" + "="*80)
        print("结论:")
        print("-" * 80)
        print("1. 并行FFT无法完全达到K倍加速的主要原因:")
        print("   - 重建阶段需要O(K*N)的计算，这是串行操作")
        print("   - 随着K增加，重建开销线性增长")
        print("   - 子通道FFT大小N/K减小，可能导致GPU利用率下降")
        print("   - 内存访问模式：重建需要访问所有子通道结果")
        print("\n2. 优化建议:")
        print("   - 优化重建算法，使用更高效的向量化操作")
        print("   - 对于大FFT，K=2或4可能更合适")
        print("   - 对于小FFT，增加K可能反而降低性能")
        print("   - 考虑使用自定义CUDA kernel优化重建阶段")
        
        # 保存结果到JSON
        with open('parallel_analysis_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("\n结果已保存到: parallel_analysis_results.json")

if __name__ == "__main__":
    analyzer = ComprehensiveParallelAnalyzer()
    
    # 测试不同的FFT大小和通道数
    fft_sizes = [
        1000,      # 小FFT
        10000,     # 中等FFT
        100000,    # 大FFT
        1000000,   # 很大FFT
    ]
    
    channel_counts = [2, 4, 8, 16]
    
    print("开始全面的并行FFT性能分析...")
    results = analyzer.analyze_scaling_behavior(fft_sizes, channel_counts)
    
    # 绘制分析图表
    analyzer.plot_comprehensive_analysis(results)
    
    # 生成报告
    analyzer.generate_analysis_report(results)
    
    print("\n分析完成!")
