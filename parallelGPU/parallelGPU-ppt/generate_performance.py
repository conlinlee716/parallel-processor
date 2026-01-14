import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 禁用字体警告
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.text')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('GPU DSP系统性能优化对比', fontsize=16, fontweight='bold', y=0.995)

# ===== 子图1: FFT加速倍数 =====
ax1 = axes[0, 0]
methods = ['CuPy\n(基准)', 'GPU\n(优化)']
speedups = [1, 115.8]
colors = ['#cccccc', '#2E86AB']

bars1 = ax1.bar(methods, speedups, color=colors, edgecolor='black', linewidth=2, width=0.6)
ax1.set_ylabel('加速倍数 (×)', fontsize=11, fontweight='bold')
ax1.set_title('FFT性能加速', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 130)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# 添加数值标签
for i, (bar, val) in enumerate(zip(bars1, speedups)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}x', ha='center', va='bottom', fontsize=11, fontweight='bold')

# ===== 子图2: 脉冲压缩加速倍数 =====
ax2 = axes[0, 1]
pulse_methods = ['CPU\n(基准)', 'GPU\n(基础)', 'GPU\n(优化)']
pulse_speedups = [1, 38.2, 104]
pulse_colors = ['#cccccc', '#F18F01', '#C73E1D']

bars2 = ax2.bar(pulse_methods, pulse_speedups, color=pulse_colors, edgecolor='black', 
               linewidth=2, width=0.6)
ax2.set_ylabel('加速倍数 (×)', fontsize=11, fontweight='bold')
ax2.set_title('脉冲压缩性能加速', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 120)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

for bar, val in zip(bars2, pulse_speedups):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')

# ===== 子图3: 流式处理吞吐量对比 =====
ax3 = axes[1, 0]
modes = ['简单流式', '批处理\n(Batch=5)', '批处理\n(Batch=10)']
throughputs = [20.2, 55.5, 52.8]
mode_colors = ['#F6AE2D', '#6A994E', '#386641']

bars3 = ax3.bar(modes, throughputs, color=mode_colors, edgecolor='black', 
               linewidth=2, width=0.6)
ax3.set_ylabel('吞吐量 (Msps)', fontsize=11, fontweight='bold')
ax3.set_title('流式处理吞吐量对比', fontsize=12, fontweight='bold')
ax3.set_ylim(0, 70)
ax3.grid(axis='y', alpha=0.3, linestyle='--')

# 添加最优标记
for i, (bar, val) in enumerate(zip(bars3, throughputs)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    if i == 1:  # 标记最优
        ax3.text(bar.get_x() + bar.get_width()/2., height + 3,
                '★ 最优', ha='center', fontsize=10, color='red', fontweight='bold')

# ===== 子图4: 优化前后对比雷达图 =====
ax4 = axes[1, 1]

categories = ['FFT\n加速', '脉冲压缩\n加速', '流式处理\n吞吐', 'Streams\n并行度', '显存\n效率']
before = [30, 15, 1.4, 1.33, 0.4]  # 优化前
after = [115.8, 104, 55.5, 1.40, 0.8]  # 优化后

# 归一化（便于对比）
before_norm = [min(v/100, 1) for v in before]
after_norm = [min(v/100, 1) for v in after]

N = len(categories)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
before_norm += before_norm[:1]
after_norm += after_norm[:1]
angles += angles[:1]

ax4 = plt.subplot(2, 2, 4, projection='polar')
ax4.plot(angles, before_norm, 'o-', linewidth=2, label='优化前', color='#cccccc')
ax4.fill(angles, before_norm, alpha=0.25, color='#cccccc')
ax4.plot(angles, after_norm, 'o-', linewidth=2, label='优化后', color='#2E86AB')
ax4.fill(angles, after_norm, alpha=0.25, color='#2E86AB')

ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(categories, fontsize=9)
ax4.set_ylim(0, 1)
ax4.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
ax4.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
ax4.grid(True, linestyle='--', alpha=0.5)
ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
ax4.set_title('性能指标综合对比\n(归一化)', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('02_performance_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ 图表2已保存: 02_performance_comparison.png")
plt.close()
