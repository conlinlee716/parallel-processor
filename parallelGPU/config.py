"""配置文件"""

CONFIG = {
    # FFT 配置
    'fft_size': 2048,
    'hop_size': 1024,
    
    # 显存配置
    'pinned_mem_size': 100 * 1024 * 1024,  # 100 MB
    'gpu_id': 0,
    
    # 处理配置
    'num_workers': 3,
    'batch_size': 1024,
    
    # 算法配置
    'num_beams': 8,
    'element_spacing': 0.5,  # 半波长
}
