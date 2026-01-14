import matplotlib.pyplot as plt
import numpy as np
import warnings

# 禁用所有字体相关警告
warnings.filterwarnings('ignore', category=UserWarning)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('算法优化与性能收益分析', fontsize=16, fontweight='bold', y=0.995)

# ===== 子图1：优化步骤与性能累积收益 =====
ax1 = axes[0, 0]

steps = ['基础\nCuPy', '批处理', 'Kernel\n融合', 'CUDA\nStreams', 'Pinned\nMemory']
speedups = [1, 2.75, 3.2, 3.5, 3.8]
cumulative_speedups = np.cumprod([1, 2.75, 1.16, 1.09, 1.09])

x_pos = np.arange(len(steps))
width = 0.35

bars1 = ax1.bar(x_pos - width/2, speedups, width, label='单步改进', 
               color='#A7C6DA', edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x_pos + width/2, cumulative_speedups, width, label='累积改进', 
               color='#2E86AB', edgecolor='black', linewidth=1.5)

ax1.set_ylabel('加速倍数 (×)', fontsize=11, fontweight='bold')
ax1.set_title('优化步骤性能收益累积', fontsize=12, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(steps, fontsize=9)
ax1.legend(fontsize=9, loc='upper left')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim(0, 4.5)

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}x', ha='center', va='bottom', fontsize=8, fontweight='bold')

# ===== 子图2：优化策略的有效性分布 =====
ax2 = axes[0, 1]

strategies = ['批处理', '异步传输', 'Kernel融合', 'Stream并行', '内存预分配']
effectiveness = [2.75, 0.95, 1.16, 1.09, 1.12]  # 相对于基础的倍数
colors_strategies = ['#6BCB77', '#FF6B6B', '#4ECDC4', '#FFD93D', '#95E1D3']

# 排序
sorted_indices = np.argsort(effectiveness)[::-1]
sorted_strategies = [strategies[i] for i in sorted_indices]
sorted_effectiveness = [effectiveness[i] for i in sorted_indices]
sorted_colors = [colors_strategies[i] for i in sorted_indices]

bars = ax2.barh(sorted_strategies, sorted_effectiveness, color=sorted_colors, 
               edgecolor='black', linewidth=1.5)

ax2.set_xlabel('效能倍数 (×)', fontsize=11, fontweight='bold')
ax2.set_title('优化策略有效性对比', fontsize=12, fontweight='bold')
ax2.set_xlim(0, 3.5)
ax2.grid(axis='x', alpha=0.3, linestyle='--')

# 添加标签
for i, (bar, val) in enumerate(zip(bars, sorted_effectiveness)):
    ax2.text(val + 0.05, bar.get_y() + bar.get_height()/2,
            f'{val:.2f}x', ha='left', va='center', fontsize=9, fontweight='bold')
    
    # 标记最优
    if i == 0:
        ax2.text(val/2, bar.get_y() + bar.get_height()/2,
                '★', ha='center', va='center', fontsize=14, color='white', fontweight='bold')

# ===== 子图3：时间成本分解 =====
ax3 = axes[1, 0]

stages = ['H2D传输\n(1-2us)', 'FFT\n(1-2ms)', '脉冲压缩\n(2-5ms)', 
         '降噪\n(0.5-1ms)', 'D2H传输\n(1-2us)']
times = [1.5, 1.5, 3.5, 0.75, 1.5]
colors_time = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#6BCB77']

# 堆积柱状图
cumsum = np.cumsum([0] + times[:-1])
bars = ax3.barh(range(1), [sum(times)], color='#DDD', alpha=0.3, edgecolor='none')

for i, (time, color) in enumerate(zip(times, colors_time)):
    ax3.barh(0, time, left=sum(times[:i]), color=color, edgecolor='black', linewidth=1.5)
    # 标签
    mid_pos = sum(times[:i]) + time/2
    ax3.text(mid_pos, 0, f'{time:.1f}ms\n{time/sum(times)*100:.0f}%', 
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')

ax3.set_ylim(-0.5, 1.5)
ax3.set_xlim(0, sum(times)*1.1)
ax3.set_ylabel('')
ax3.set_xlabel('总处理时间 (ms)', fontsize=11, fontweight='bold')
ax3.set_title('单个样本处理时间分解', fontsize=12, fontweight='bold')
ax3.set_yticks([])

# 总时间标注
total_time = sum(times)
ax3.text(total_time + 0.3, 0, f'总计: {total_time:.1f}ms\n吞吐: {1000/total_time:.0f} samples/s',
        ha='left', va='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffcc', edgecolor='#ff9900'))

# ===== 子图4：ROI分析（投入产出比） =====
ax4 = axes[1, 1]

optimization_types = ['批处理\n实现', '异步\n传输', 'Kernel\n融合', 'Stream\n调度', '内存\n优化']
implementation_effort = [2, 3, 5, 3, 2]  # 实现难度(1-10)
performance_gain = [2.75, 0.95, 1.16, 1.09, 1.12]  # 性能收益
roi = [pg/ie for pg, ie in zip(performance_gain, implementation_effort)]

# 气泡图
scatter = ax4.scatter(implementation_effort, performance_gain, s=[r*500 for r in roi], 
                     c=roi, cmap='RdYlGn', alpha=0.7, edgecolor='black', linewidth=2)

# 标签
for i, txt in enumerate(optimization_types):
    ax4.annotate(txt, (implementation_effort[i], performance_gain[i]),
                ha='center', va='center', fontsize=8, fontweight='bold')

ax4.set_xlabel('实现难度 (相对)', fontsize=11, fontweight='bold')
ax4.set_ylabel('性能收益 (×)', fontsize=11, fontweight='bold')
ax4.set_title('优化方案ROI分析\n(气泡大小表示ROI)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, linestyle='--')

# 添加参考线
ax4.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='基准线')
ax4.axvline(x=3, color='gray', linestyle='--', alpha=0.5)

# 最优区域标记
ax4.fill_between([0, 5], 1, 3, alpha=0.1, color='green', label='高ROI区域')

ax4.legend(loc='lower right', fontsize=8)
ax4.set_xlim(0, 7)
ax4.set_ylim(0, 3.5)

# 添加colorbar
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('ROI值', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('06_optimization_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ 图表6已保存: 06_optimization_analysis.png")
plt.close()
