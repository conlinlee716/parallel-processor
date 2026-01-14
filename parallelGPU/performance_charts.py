import matplotlib.pyplot as plt
import numpy as np

# Chart 1: GPU Acceleration Performance Comparison
def plot_gpu_performance():
    # Performance data
    operations = ['FFT', 'Convolution', 'Pulse Compression']
    cpu_time = [37.54, 47.64, 93.15]
    gpu_time = [0.45, 4.90, 2.44]
    speedup = np.array(cpu_time) / np.array(gpu_time)

    # Create chart
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar chart: Time comparison
    x = np.arange(len(operations))
    width = 0.35

    bar1 = ax1.bar(x - width/2, cpu_time, width, label='CPU Time(ms)', color='#3498db')
    bar2 = ax1.bar(x + width/2, gpu_time, width, label='GPU Time(ms)', color='#2ecc71')

    # Set left Y-axis
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_xlabel('Operation Type', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(operations, fontsize=11)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Line chart: Speedup ratio
    ax2 = ax1.twinx()
    line = ax2.plot(x, speedup, 'ro-', linewidth=2, label='Speedup(x)')
    ax2.set_ylabel('Speedup (x)', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right')

    # Add value labels
    for i, (bar_cpu, bar_gpu, sp) in enumerate(zip(bar1, bar2, speedup)):
        height_cpu = bar_cpu.get_height()
        height_gpu = bar_gpu.get_height()
        ax1.text(bar_cpu.get_x() + bar_cpu.get_width()/2., height_cpu, 
                 f'{height_cpu:.2f}', ha='center', va='bottom', fontsize=10)
        ax1.text(bar_gpu.get_x() + bar_gpu.get_width()/2., height_gpu, 
                 f'{height_gpu:.2f}', ha='center', va='bottom', fontsize=10)
        ax2.text(i, sp, f'{sp:.1f}', ha='center', va='bottom', color='red', fontsize=10)

    plt.title('GPU Acceleration Performance Comparison (1M Samples)', fontsize=14)
    plt.tight_layout()
    plt.savefig('gpu_performance.png', dpi=300, bbox_inches='tight')
    # plt.show()  # Skip showing the plot to avoid GUI issues

# Chart 2: Kernel Fusion Performance Comparison
def plot_kernel_fusion():
    # Data
    methods = ['Separate Ops(4 steps)', 'Simple Fused Kernel', 'Complex Fused Kernel']
    times = [1.52, 1.09, 1.59]
    performance_improvement = [(times[0]-t)/times[0]*100 for t in times]

    # 创建图表
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar chart: Time
    x = np.arange(len(methods))
    bars = ax1.bar(x, times, color=['#95a5a6', '#2ecc71', '#f39c12'], alpha=0.8)

    # Set left Y-axis
    ax1.set_ylabel('Processing Time (ms)', fontsize=12)
    ax1.set_xlabel('Processing Method', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=11, rotation=15)
    ax1.grid(True, alpha=0.3, axis='y')

    # Line chart: Performance improvement
    ax2 = ax1.twinx()
    line = ax2.plot(x, performance_improvement, 'ro-', linewidth=2, label='Improvement(%)')
    ax2.set_ylabel('Performance Improvement (%)', fontsize=12, color='red')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Add value labels
    for i, (bar, time, sp) in enumerate(zip(bars, times, performance_improvement)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02, 
                 f'{time:.3f}ms', ha='center', va='bottom', fontsize=10)
        ax2.text(i, sp + (1 if sp > 0 else -1), 
                 f'{sp:+.1f}%', ha='center', va='bottom', color='red', fontsize=10)

    plt.title('Kernel Fusion Performance Optimization', fontsize=14)
    plt.tight_layout()
    plt.savefig('kernel_fusion.png', dpi=300, bbox_inches='tight')
    # plt.show()  # Skip showing the plot to avoid GUI issues

# Chart 3: Synchronous vs Asynchronous Transfer Performance
def plot_async_performance():
    # Data
    task_types = ['Single Task', 'Multi-task(3 streams)']
    sync_time = [63.32, 343.73]
    async_time = [146.31, 245.87]

    # Calculate performance difference
    speedup_single = (sync_time[0] - async_time[0]) / sync_time[0] * 100
    speedup_multi = (sync_time[1] - async_time[1]) / sync_time[1] * 100

    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(task_types))
    width = 0.35

    # Draw bar chart
    bars1 = ax.bar(x - width/2, sync_time, width, label='Synchronous Transfer', color='#3498db')
    bars2 = ax.bar(x + width/2, async_time, width, label='Asynchronous Transfer', color='#9b59b6')

    # Set axes
    ax.set_ylabel('Processing Time (ms)', fontsize=12)
    ax.set_xlabel('Task Type', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(task_types, fontsize=11)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        
        # Synchronous transfer value
        ax.text(bar1.get_x() + bar1.get_width()/2., height1, 
                 f'{height1:.2f}ms', ha='center', va='bottom', fontsize=10)
        
        # Asynchronous transfer value and performance difference
        ax.text(bar2.get_x() + bar2.get_width()/2., height2, 
                 f'{height2:.2f}ms', ha='center', va='bottom', fontsize=10)
        
        # Performance difference
        speedup = speedup_single if i == 0 else speedup_multi
        diff_text = f'{speedup:+.1f}%'
        y_pos = max(height1, height2) + 5
        ax.text(x[i], y_pos, diff_text, ha='center', va='bottom', 
                color='red' if speedup > 0 else 'blue', fontsize=10, fontweight='bold')

    plt.title('Synchronous vs Asynchronous Transfer Performance', fontsize=14)
    plt.tight_layout()
    plt.savefig('async_performance.png', dpi=300, bbox_inches='tight')
    # plt.show()  # Skip showing the plot to avoid GUI issues

# Main function
if __name__ == "__main__":
    print("Generating GPU acceleration performance comparison chart...")
    plot_gpu_performance()
    
    print("Generating kernel fusion performance comparison chart...")
    plot_kernel_fusion()
    
    print("Generating asynchronous transfer performance comparison chart...")
    plot_async_performance()
    
    print("All charts have been generated successfully!")
