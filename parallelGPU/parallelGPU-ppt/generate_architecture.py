import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import warnings

# 禁用所有字体相关警告
warnings.filterwarnings('ignore', category=UserWarning)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False



fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# 标题
ax.text(5, 9.5, 'GPU DSP信号处理系统架构', fontsize=20, fontweight='bold', 
        ha='center', va='top')

# ===== 输入层 =====
input_box = FancyBboxPatch((0.5, 8), 2, 0.8, boxstyle="round,pad=0.1", 
                           edgecolor='#2E86AB', facecolor='#A7C6DA', linewidth=2)
ax.add_patch(input_box)
ax.text(1.5, 8.4, '信号输入', fontsize=11, fontweight='bold', ha='center')

# ===== 主处理单元 =====
y_pos = 6.5

# 数据预处理
preprocess_box = FancyBboxPatch((0.2, y_pos), 2.2, 0.7, boxstyle="round,pad=0.05",
                                edgecolor='#F18F01', facecolor='#FFCBA4', linewidth=2)
ax.add_patch(preprocess_box)
ax.text(1.3, y_pos+0.35, '1. 数据预处理', fontsize=10, fontweight='bold', ha='center')

# FFT处理
fft_box = FancyBboxPatch((3, y_pos), 2.2, 0.7, boxstyle="round,pad=0.05",
                         edgecolor='#C73E1D', facecolor='#F6AE2D', linewidth=2)
ax.add_patch(fft_box)
ax.text(4.1, y_pos+0.35, '2. FFT变换', fontsize=10, fontweight='bold', ha='center')

# 脉冲压缩
pulse_box = FancyBboxPatch((5.8, y_pos), 2.2, 0.7, boxstyle="round,pad=0.05",
                          edgecolor='#6A994E', facecolor='#BC4749', linewidth=2)
ax.add_patch(pulse_box)
ax.text(6.9, y_pos+0.35, '3. 脉冲压缩', fontsize=10, fontweight='bold', ha='center')

# 波束成形
beam_box = FancyBboxPatch((7.6, y_pos), 2.2, 0.7, boxstyle="round,pad=0.05",
                         edgecolor='#386641', facecolor='#9DB4C4', linewidth=2)
ax.add_patch(beam_box)
ax.text(8.7, y_pos+0.35, '4. 波束成形', fontsize=10, fontweight='bold', ha='center')

# ===== GPU优化层 =====
y_opt = 5

# 显存管理
mem_box = FancyBboxPatch((0.2, y_opt), 2.2, 0.6, boxstyle="round,pad=0.05",
                         edgecolor='#264653', facecolor='#E9F5F9', linewidth=1.5)
ax.add_patch(mem_box)
ax.text(1.3, y_opt+0.3, '显存管理', fontsize=9, ha='center')
ax.text(1.3, y_opt+0.05, 'Pinned Memory', fontsize=7, ha='center', style='italic')

# CUDA Streams
stream_box = FancyBboxPatch((3, y_opt), 2.2, 0.6, boxstyle="round,pad=0.05",
                           edgecolor='#264653', facecolor='#E9F5F9', linewidth=1.5)
ax.add_patch(stream_box)
ax.text(4.1, y_opt+0.3, 'CUDA Streams', fontsize=9, ha='center')
ax.text(4.1, y_opt+0.05, '并行调度', fontsize=7, ha='center', style='italic')

# Kernel融合
kernel_box = FancyBboxPatch((5.8, y_opt), 2.2, 0.6, boxstyle="round,pad=0.05",
                           edgecolor='#264653', facecolor='#E9F5F9', linewidth=1.5)
ax.add_patch(kernel_box)
ax.text(6.9, y_opt+0.3, 'Kernel融合', fontsize=9, ha='center')
ax.text(6.9, y_opt+0.05, '减少Launch开销', fontsize=7, ha='center', style='italic')

# 流式处理
stream_proc_box = FancyBboxPatch((7.6, y_opt), 2.2, 0.6, boxstyle="round,pad=0.05",
                                edgecolor='#264653', facecolor='#E9F5F9', linewidth=1.5)
ax.add_patch(stream_proc_box)
ax.text(8.7, y_opt+0.3, '流式处理', fontsize=9, ha='center')
ax.text(8.7, y_opt+0.05, '批处理优化', fontsize=7, ha='center', style='italic')

# ===== 箭头连接 =====
# 输入 -> 预处理
arrow1 = FancyArrowPatch((1.5, 8), (1.3, 7.2), arrowstyle='->', 
                        mutation_scale=20, linewidth=2, color='#555555')
ax.add_patch(arrow1)

# 预处理 -> FFT
arrow2 = FancyArrowPatch((2.4, 6.85), (3, 6.85), arrowstyle='->', 
                        mutation_scale=20, linewidth=2, color='#555555')
ax.add_patch(arrow2)

# FFT -> 脉冲压缩
arrow3 = FancyArrowPatch((5.2, 6.85), (5.8, 6.85), arrowstyle='->', 
                        mutation_scale=20, linewidth=2, color='#555555')
ax.add_patch(arrow3)

# 脉冲压缩 -> 波束成形
arrow4 = FancyArrowPatch((8, 6.85), (7.6, 6.85), arrowstyle='->', 
                        mutation_scale=20, linewidth=2, color='#555555')
ax.add_patch(arrow4)

# 处理单元 -> 优化层
for x in [1.3, 4.1, 6.9, 8.7]:
    arrow = FancyArrowPatch((x, y_pos), (x, y_opt+0.6), arrowstyle='->', 
                           mutation_scale=15, linewidth=1.5, color='#888888',
                           linestyle='dashed')
    ax.add_patch(arrow)

# ===== 输出层 =====
output_box = FancyBboxPatch((3.5, 3.5), 3, 0.8, boxstyle="round,pad=0.1",
                           edgecolor='#2A9D8F', facecolor='#80B5A3', linewidth=2)
ax.add_patch(output_box)
ax.text(5, 3.9, '处理结果输出', fontsize=11, fontweight='bold', ha='center')

arrow_output = FancyArrowPatch((4.1, y_opt), (5, 4.3), arrowstyle='->', 
                             mutation_scale=20, linewidth=2, color='#555555')
ax.add_patch(arrow_output)

# ===== 性能指标框 =====
perf_box = FancyBboxPatch((0.5, 0.5), 4, 2.3, boxstyle="round,pad=0.1",
                         edgecolor='#E63946', facecolor='#FFF0F3', linewidth=2)
ax.add_patch(perf_box)
ax.text(2.5, 2.6, '性能指标', fontsize=11, fontweight='bold', ha='center')

perf_text = [
    'FFT加速: 83.8x - 115.8x',
    '脉冲压缩: 38.2x - 104x',
    '流式处理: 55.5 Msps',
    '并行度: 1.40x'
]
for i, text in enumerate(perf_text):
    ax.text(1, 2.2-i*0.4, f'[✓] {text}', fontsize=9, ha='left')

# ===== 部署配置框 =====
deploy_box = FancyBboxPatch((5.5, 0.5), 4, 2.3, boxstyle="round,pad=0.1",
                           edgecolor='#457B9D', facecolor='#D6E9F8', linewidth=2)
ax.add_patch(deploy_box)
ax.text(7.5, 2.6, '部署配置', fontsize=11, fontweight='bold', ha='center')

deploy_text = [
    'GPU: NVIDIA RTX/A100',
    'Pinned Memory: 100MB',
    'CUDA Streams: 3个',
    'Batch Size: 5-10'
]
for i, text in enumerate(deploy_text):
    ax.text(6, 2.2-i*0.4, f'[•] {text}', fontsize=9, ha='left')

plt.tight_layout()
plt.savefig('01_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
print("[✓] 图表1已保存: 01_architecture.png")
plt.close()
