# generate_all_figures.py
"""
生成完整的PPT所需的所有图表
"""

import os
import sys

print("\n" + "="*70)
print("GPU DSP系统 - 完整PPT图表生成器")
print("="*70 + "\n")

# 确保输出目录存在
os.makedirs('ppt_figures', exist_ok=True)
os.chdir('ppt_figures')

print("[1/6] 生成系统架构图...")
# 这里粘贴图表1的代码
exec(open('../generate_architecture.py', encoding='utf-8').read())

print("[2/6] 生成性能对比图...")
# 这里粘贴图表2的代码
exec(open('../generate_performance.py', encoding='utf-8').read())

print("[3/6] 生成处理流程图...")
# 这里粘贴图表3的代码
exec(open('../generate_pipeline.py', encoding='utf-8').read())

print("[4/6] 生成GPU部署图...")
# 这里粘贴图表4的代码
exec(open('../generate_deployment.py', encoding='utf-8').read())

print("[5/6] 生成数据流图...")
# 这里粘贴图表5的代码
exec(open('../generate_dataflow.py', encoding='utf-8').read())

print("[6/6] 生成优化分析图...")
# 这里粘贴图表6的代码
exec(open('../generate_optimization.py', encoding='utf-8').read())

print("\n" + "="*70)
print("✓ 所有图表已生成完成！")
print("="*70)
print("\n生成的图表文件：")
for i in range(1, 7):
    filename = f'0{i}_*.png'
    print(f"  ✓ {filename}")

print("\n📊 PPT准备就绪！")
print("   可在以下目录找到所有图表：./ppt_figures/")
