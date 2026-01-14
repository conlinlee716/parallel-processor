import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Wedge
import numpy as np
import warnings

# 禁用所有字体相关警告
warnings.filterwarnings('ignore', category=UserWarning)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# 标题
ax.text(5, 9.5, 'GPU硬件部署与算子映射架构', fontsize=18, fontweight='bold', ha='center')

# ===== 左侧：CPU主机 =====
# CPU框
cpu_box = FancyBboxPatch((0.3, 6.5), 2.5, 2, boxstyle="round,pad=0.1",
                        edgecolor='#264653', facecolor='#E9F5F9', linewidth=2)
ax.add_patch(cpu_box)
ax.text(1.55, 8.2, 'CPU主机', fontsize=11, fontweight='bold', ha='center')

cpu_modules = [
    '应用程序',
    'CUDA Runtime',
    'Host Memory',
    '数据编排'
]

for i, module in enumerate(cpu_modules):
    y_pos = 7.8 - i*0.35
    ax.text(0.5, y_pos, f'• {module}', fontsize=8, ha='left')

# ===== 中间：PCIe总线 =====
pcie_width = 0.5
pcie_box = FancyBboxPatch((3.1, 6.5), pcie_width, 2, boxstyle="round,pad=0.02",
                         edgecolor='#E63946', facecolor='#FFE5E5', linewidth=2)
ax.add_patch(pcie_box)
ax.text(3.35, 8.2, 'PCIe\nx16', fontsize=9, fontweight='bold', ha='center', va='center')

# 数据流箭头
arrow_h2d = FancyArrowPatch((2.8, 7.8), (3.1, 7.8), arrowstyle='->', 
                           mutation_scale=25, linewidth=3, color='#F18F01')
ax.add_patch(arrow_h2d)
ax.text(2.95, 8.05, 'H2D', fontsize=8, ha='center', color='#F18F01', fontweight='bold')

arrow_d2h = FancyArrowPatch((3.1, 7.2), (2.8, 7.2), arrowstyle='->', 
                           mutation_scale=25, linewidth=3, color='#6A994E')
ax.add_patch(arrow_d2h)
ax.text(2.95, 6.95, 'D2H', fontsize=8, ha='center', color='#6A994E', fontweight='bold')

# ===== 右侧：GPU设备 =====
gpu_box = FancyBboxPatch((3.7, 6.5), 5.8, 2, boxstyle="round,pad=0.1",
                        edgecolor='#2E86AB', facecolor='#A7C6DA', linewidth=3)
ax.add_patch(gpu_box)
ax.text(6.6, 8.2, 'GPU设备(RTX/A100)', fontsize=11, fontweight='bold', ha='center')

# GPU内部结构
# 全局显存
globalmem_box = FancyBboxPatch((3.9, 7.3), 1.2, 0.6, boxstyle="round,pad=0.05",
                              edgecolor='#457B9D', facecolor='#D6E9F8', linewidth=1.5)
ax.add_patch(globalmem_box)
ax.text(4.5, 7.6, '全局显存', fontsize=9, fontweight='bold', ha='center')
ax.text(4.5, 7.35, '100GB+', fontsize=7, ha='center', style='italic')

# Pinned内存缓冲
pinned_box = FancyBboxPatch((3.9, 6.6), 1.2, 0.6, boxstyle="round,pad=0.05",
                           edgecolor='#457B9D', facecolor='#E8F5F2', linewidth=1.5)
ax.add_patch(pinned_box)
ax.text(4.5, 6.9, 'Pinned缓冲', fontsize=9, fontweight='bold', ha='center')
ax.text(4.5, 6.65, '100MB', fontsize=7, ha='center', style='italic')

# SM块群
sm_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
sm_positions = [(5.2, 7.5), (5.2, 6.8), (6.4, 7.5), (6.4, 6.8)]

for i, (x, y) in enumerate(sm_positions):
    sm_box = FancyBboxPatch((x, y), 0.9, 0.5, boxstyle="round,pad=0.03",
                           edgecolor='black', facecolor=sm_colors[i], linewidth=1.5)
    ax.add_patch(sm_box)
    ax.text(x+0.45, y+0.25, f'SM{i}', fontsize=8, fontweight='bold', ha='center', va='center')

# ===== 第二层：CUDA Streams与Kernel映射 =====
y_streams = 5.5

ax.text(5, y_streams+0.5, 'CUDA Streams与Kernel执行', fontsize=12, fontweight='bold', 
       ha='center', bbox=dict(boxstyle='round', facecolor='#fff9e6', alpha=0.8))

# 三个流
stream_y = y_streams - 0.3
for stream_id in range(3):
    x_offset = stream_id * 3.2
    
    # 流框
    stream_box = FancyBboxPatch((0.5+x_offset, stream_y-1.2), 3, 1.2, 
                               boxstyle="round,pad=0.05",
                               edgecolor='#264653', facecolor='#E9F5F9', linewidth=1.5)
    ax.add_patch(stream_box)
    ax.text(2+x_offset, stream_y-0.1, f'Stream {stream_id}', fontsize=9, fontweight='bold', ha='center')
    
    # 任务序列
    tasks = ['H2D', 'FFT', 'Filter', 'IFFT', 'D2H']
    for task_idx, task in enumerate(tasks):
        task_x = 0.7 + x_offset + task_idx * 0.55
        task_y = stream_y - 0.7
        
        task_box = Rectangle((task_x, task_y), 0.5, 0.35, 
                            edgecolor='black', facecolor='#CCFFCC', linewidth=1)
        ax.add_patch(task_box)
        ax.text(task_x+0.25, task_y+0.175, task, fontsize=6, ha='center', va='center')

# ===== 第三层：算子到硬件的映射 =====
y_mapping = 2.5

ax.text(5, y_mapping+0.6, '算子映射与执行策略', fontsize=12, fontweight='bold', 
       ha='center', bbox=dict(boxstyle='round', facecolor='#fff9e6', alpha=0.8))

mapping_data = [
    {
        'name': 'FFT算子',
        'gpu_part': 'cuFFT库',
        'blocks': 'SM群组',
        'threads': '1024线程',
        'color': '#FFD93D',
        'x': 0.5
    },
    {
        'name': '滤波算子',
        'gpu_part': 'Kernel融合',
        'blocks': 'SM群组',
        'threads': '256线程',
        'color': '#6BCB77',
        'x': 3.5
    },
    {
        'name': '脉冲压缩',
        'gpu_part': 'Overlap-Save',
        'blocks': '批处理',
        'threads': '动态',
        'color': '#FF6B6B',
        'x': 6.5
    }
]

for mapping in mapping_data:
    x = mapping['x']
    
    # 算子框
    algo_box = FancyBboxPatch((x, y_mapping-0.3), 2.8, 0.4, boxstyle="round,pad=0.02",
                             edgecolor='black', facecolor=mapping['color'], linewidth=1.5)
    ax.add_patch(algo_box)
    ax.text(x+1.4, y_mapping-0.1, mapping['name'], fontsize=9, fontweight='bold', ha='center')
    
    # 详细信息
    details = [
        f"GPU: {mapping['gpu_part']}",
        f"执行: {mapping['blocks']}",
        f"线程: {mapping['threads']}"
    ]
    
    for i, detail in enumerate(details):
        ax.text(x+0.1, y_mapping-0.7-i*0.25, detail, fontsize=7, ha='left',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# ===== 性能监控与反馈 =====
y_monitor = 0.8

monitor_box = FancyBboxPatch((0.3, y_monitor-0.3), 9.4, 0.6, boxstyle="round,pad=0.05",
                            edgecolor='#457B9D', facecolor='#D6E9F8', linewidth=2)
ax.add_patch(monitor_box)

monitor_text = '[监控] 性能监控：GPU占用率 92% | 显存用量 8.5GB | 执行时间 <10ms | 吞吐量 55.5 Msps'
ax.text(5, y_monitor, monitor_text, fontsize=9, ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('04_gpu_deployment.png', dpi=300, bbox_inches='tight', facecolor='white')
print("[✓] 图表4已保存: 04_gpu_deployment.png")
plt.close()
