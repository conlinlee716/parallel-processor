import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon
import numpy as np
import warnings

# 禁用所有字体相关警告
warnings.filterwarnings('ignore', category=UserWarning)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(14, 10))

# 创建两个子图
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)

# ===== 上图：数据流向 =====
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 6)
ax1.axis('off')
ax1.text(5, 5.7, '数据流向与处理流程', fontsize=14, fontweight='bold', ha='center')

# 输入数据
input_node = FancyBboxPatch((0.5, 4), 1.2, 0.7, boxstyle="round,pad=0.05",
                           edgecolor='#264653', facecolor='#E9F5F9', linewidth=2)
ax1.add_patch(input_node)
ax1.text(1.1, 4.35, '输入数据', fontsize=9, fontweight='bold', ha='center')

# H2D传输
h2d_node = FancyBboxPatch((2, 4), 1.2, 0.7, boxstyle="round,pad=0.05",
                         edgecolor='#F18F01', facecolor='#FFCBA4', linewidth=2)
ax1.add_patch(h2d_node)
ax1.text(2.6, 4.35, 'H2D传输', fontsize=9, fontweight='bold', ha='center')

# GPU预处理
preproc_node = FancyBboxPatch((3.5, 4), 1.2, 0.7, boxstyle="round,pad=0.05",
                             edgecolor='#6A994E', facecolor='#D4EBD9', linewidth=2)
ax1.add_patch(preproc_node)
ax1.text(4.1, 4.35, 'GPU预处理', fontsize=9, fontweight='bold', ha='center')

# GPU FFT
fft_node = FancyBboxPatch((5, 4), 1.2, 0.7, boxstyle="round,pad=0.05",
                         edgecolor='#C73E1D', facecolor='#F6AE2D', linewidth=2)
ax1.add_patch(fft_node)
ax1.text(5.6, 4.35, 'GPU-FFT', fontsize=9, fontweight='bold', ha='center')

# GPU脉冲压缩
pulse_node = FancyBboxPatch((6.5, 4), 1.2, 0.7, boxstyle="round,pad=0.05",
                           edgecolor='#C73E1D', facecolor='#F6AE2D', linewidth=2)
ax1.add_patch(pulse_node)
ax1.text(7.1, 4.35, 'GPU脉冲压缩', fontsize=9, fontweight='bold', ha='center')

# D2H传输
d2h_node = FancyBboxPatch((8, 4), 1.2, 0.7, boxstyle="round,pad=0.05",
                         edgecolor='#6A994E', facecolor='#D4EBD9', linewidth=2)
ax1.add_patch(d2h_node)
ax1.text(8.6, 4.35, 'D2H传输', fontsize=9, fontweight='bold', ha='center')

# 输出结果
output_node = FancyBboxPatch((9.3, 4), 0.4, 0.7, boxstyle="round,pad=0.05",
                            edgecolor='#2A9D8F', facecolor='#80B5A3', linewidth=2)
ax1.add_patch(output_node)
ax1.text(9.5, 4.35, '输出', fontsize=8, fontweight='bold', ha='center')

# 连接箭头
nodes_x = [1.7, 2.6, 3.5, 4.1, 5.6, 7.1, 8.6, 9.3]
for i in range(len(nodes_x)-1):
    arrow = FancyArrowPatch((nodes_x[i]+0.55, 4.35), (nodes_x[i+1]-0.05, 4.35),
                           arrowstyle='->', mutation_scale=20, linewidth=2.5, color='#555555')
    ax1.add_patch(arrow)

# 成本标注
costs = ['100ns', '1-2us', '50-100us', '1-2ms', '2-5ms', '1-2us']
cost_y = 3.2
for i, cost in enumerate(costs):
    ax1.text(2.1 + i*1.1, cost_y, cost, fontsize=7, ha='center', 
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))

# 内存使用情况
mem_text = [
    '主机内存: ~800MB',
    'GPU全局显存: ~2GB',
    'GPU共享内存: ~96KB/块',
    'Pinned内存: 100MB'
]

ax1.text(0.5, 2.2, '内存使用情况', fontsize=10, fontweight='bold')
for i, mem in enumerate(mem_text):
    ax1.text(0.7, 1.9-i*0.3, f'• {mem}', fontsize=8, ha='left')

# ===== 下图：内存分层与优化策略 =====
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 7)
ax2.axis('off')
ax2.text(5, 6.7, '显存分层结构与优化策略', fontsize=14, fontweight='bold', ha='center')

# 内存层级
mem_levels = [
    {
        'name': 'L1 Cache',
        'size': '96KB/SM',
        'speed': '~4TB/s',
        'y': 5.5,
        'color': '#FF6B6B',
        'x': 0.5
    },
    {
        'name': 'L2 Cache',
        'size': '5-6MB',
        'speed': '~2TB/s',
        'y': 4.5,
        'color': '#4ECDC4',
        'x': 0.5
    },
    {
        'name': '共享内存',
        'size': '96KB/块',
        'speed': '~1.5TB/s',
        'y': 5.5,
        'color': '#45B7D1',
        'x': 2.5
    },
    {
        'name': '全局显存',
        'size': '100GB+',
        'speed': '~350GB/s',
        'y': 3.5,
        'color': '#FFA07A',
        'x': 2.5
    },
    {
        'name': 'Pinned内存',
        'size': '100MB',
        'speed': '~50GB/s',
        'y': 2,
        'color': '#FFD93D',
        'x': 5
    },
    {
        'name': '主机内存',
        'size': '~800MB',
        'speed': '~40GB/s',
        'y': 2,
        'color': '#6BCB77',
        'x': 7.5
    }
]

for mem in mem_levels:
    # 内存框
    mem_box = FancyBboxPatch((mem['x'], mem['y']), 2, 0.7, boxstyle="round,pad=0.05",
                            edgecolor='black', facecolor=mem['color'], linewidth=1.5)
    ax2.add_patch(mem_box)
    
    ax2.text(mem['x']+1, mem['y']+0.5, mem['name'], fontsize=9, fontweight='bold', ha='center')
    ax2.text(mem['x']+1, mem['y']+0.2, f"Size: {mem['size']}", fontsize=7, ha='center')
    ax2.text(mem['x']+1, mem['y']+0.05, f"Speed: {mem['speed']}", fontsize=7, ha='center', style='italic')

# 优化策略标注
strategies = [
    {'text': '[✓] 充分利用缓存\n局部性优化', 'x': 0.5, 'y': 0.8, 'color': '#FFE5E5'},
    {'text': '[✓] 预分配GPU缓冲\n减少碎片化', 'x': 3, 'y': 0.8, 'color': '#E5F5FF'},
    {'text': '[✓] 异步H2D/D2H\n重叠传输', 'x': 6, 'y': 0.8, 'color': '#E5FFE5'},
]

for strategy in strategies:
    box = FancyBboxPatch((strategy['x'], strategy['y']), 2.8, 0.6, boxstyle="round,pad=0.05",
                        edgecolor='#555', facecolor=strategy['color'], linewidth=1.5)
    ax2.add_patch(box)
    ax2.text(strategy['x']+1.4, strategy['y']+0.3, strategy['text'], fontsize=7.5, 
            ha='center', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('05_data_flow_memory.png', dpi=300, bbox_inches='tight', facecolor='white')
print("[✓] 图表5已保存: 05_data_flow_memory.png")
plt.close()
