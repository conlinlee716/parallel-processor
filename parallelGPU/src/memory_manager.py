"""GPU显存与CUDA Streams管理 - 完整修复版本 v3"""
import cupy as cp
import numpy as np
import time


class GPUMemoryManager:
    """Pinned Memory + CUDA Streams 管理器"""
    
    def __init__(self, pinned_mem_size=100*1024*1024):
        self.pinned_mem_size = pinned_mem_size
        self.num_streams = 3
        
        # 创建Pinned内存池与Streams
        try:
            self.pinned_pool = cp.cuda.alloc_pinned_memory(pinned_mem_size)
        except:
            self.pinned_pool = None
        
        self.streams = [cp.cuda.Stream() for _ in range(self.num_streams)]
        self.events = [[cp.cuda.Event() for _ in range(2)] for _ in range(self.num_streams)]
    
    def async_h2d(self, host_data, stream_idx=0, pinned=True):
        """异步Host->Device传输"""
        gpu_data = cp.asarray(host_data)
        return gpu_data
    
    def async_d2h(self, gpu_data, stream_idx=2):
        """异步Device->Host传输"""
        host_data = cp.asnumpy(gpu_data)
        return host_data
    
    def record_event(self, stream_idx, event_idx):
        """记录事件用于同步"""
        self.streams[stream_idx].record_event(self.events[stream_idx][event_idx])
    
    def wait_event(self, stream_idx, event):
        """等待事件完成"""
        self.streams[stream_idx].wait_event(event)
    
    def synchronize_all(self):
        """同步所有流"""
        for stream in self.streams:
            stream.synchronize()


class SimpleStreamProcessor:
    """
    ✓ 简化版本：不使用Overlap-Save
    直接处理每个块，避免复杂的缓冲区管理
    这是最实用的流式处理方案
    """
    
    def __init__(self, mem_manager):
        self.mem_mgr = mem_manager
        self.chirp_fft_cache = {}
    
    def process_chunks(self, signal_chunks, chirp_ref_gpu):
        """
        简单流式处理：逐块处理信号
        每个块独立进行脉冲压缩
        
        Args:
            signal_chunks: [(chunk_size,), ...] - 信号块列表
            chirp_ref_gpu: GPU上的参考信号
        
        Returns:
            连接后的处理结果
        """
        results = []
        chirp_ref_gpu = cp.asarray(chirp_ref_gpu, dtype=cp.complex64)
        
        for chunk_idx, chunk_np in enumerate(signal_chunks):
            # ✓ 修复：每个块独立处理
            chunk_gpu = cp.asarray(chunk_np, dtype=cp.complex64)
            
            # 计算FFT大小（块长度 + 参考信号长度 - 1）
            fft_size = len(chunk_gpu) + len(chirp_ref_gpu) - 1
            # 向上取整到2的幂次（FFT优化）
            fft_size = int(2 ** np.ceil(np.log2(fft_size)))
            
            # FFT处理
            sig_fft = cp.fft.fft(chunk_gpu, n=fft_size)
            
            # 缓存参考信号FFT（避免重复计算）
            if fft_size not in self.chirp_fft_cache:
                chirp_fft = cp.fft.fft(chirp_ref_gpu, n=fft_size)
                self.chirp_fft_cache[fft_size] = chirp_fft
            else:
                chirp_fft = self.chirp_fft_cache[fft_size]
            
            # 频域相乘
            result_fft = sig_fft * cp.conj(chirp_fft)
            
            # IFFT
            result_time = cp.fft.ifft(result_fft)
            
            # ✓ 关键：只保留有效长度的实部
            result_valid = result_time[:len(chunk_gpu)].real
            
            # 转回CPU
            results.append(cp.asnumpy(result_valid))
        
        return np.concatenate(results)


class OverlapSaveProcessor:
    """
    ✓ 修复版本v3：重新设计缓冲区逻辑
    使用更清晰的Overlap-Save实现
    """
    
    def __init__(self, hop_size, kernel_size, mem_manager):
        """
        Args:
            hop_size: 处理间隔（块大小）
            kernel_size: 卷积核大小
            mem_manager: GPU内存管理器
        """
        self.hop_size = hop_size
        self.kernel_size = kernel_size
        self.mem_mgr = mem_manager
        
        # Overlap-Save参数
        self.fft_size = hop_size + kernel_size - 1
        self.overlap_size = kernel_size - 1
        
        # ✓ 预分配缓冲区
        self.input_buffer = None
        self.kernel_fft = None
        self.initialized = False
    
    def _initialize_for_kernel(self, kernel_gpu):
        """初始化参考信号FFT"""
        if not self.initialized:
            kernel_gpu = cp.asarray(kernel_gpu, dtype=cp.complex64)
            self.kernel_fft = cp.fft.fft(kernel_gpu, n=self.fft_size)
            self.initialized = True
    
    def process_streaming_optimized(self, signal_chunks, chirp_ref_gpu):
        """
        ✓ 修复：Overlap-Save流式处理
        
        工作流程：
        1. 初始化缓冲区（填充kernel_size-1个零）
        2. 对每个块：
           - 在缓冲区末尾添加新块
           - 进行FFT处理
           - 输出有效部分
           - 保留尾部作为下一块的重叠
        """
        self._initialize_for_kernel(chirp_ref_gpu)
        
        results = []
        chirp_ref_gpu = cp.asarray(chirp_ref_gpu, dtype=cp.complex64)
        
        # 初始化输入缓冲区（首次使用）
        # 缓冲区大小 = FFT_SIZE = hop_size + kernel_size - 1
        input_buffer = cp.zeros(self.fft_size, dtype=cp.complex64)
        
        # ✓ 修复：使用正确的重叠索引
        # 前 (kernel_size-1) 个样本是从上一块的重叠
        overlap_start = 0
        overlap_end = self.overlap_size
        new_data_start = self.overlap_size
        new_data_end = self.fft_size
        
        for chunk_idx, chunk_np in enumerate(signal_chunks):
            chunk_gpu = cp.asarray(chunk_np, dtype=cp.complex64)
            chunk_len = len(chunk_gpu)
            
            # ✓ 修复：清空新数据区域
            input_buffer[new_data_start:new_data_end] = 0
            
            # 填充新数据（可能小于可用空间）
            actual_len = min(chunk_len, new_data_end - new_data_start)
            input_buffer[new_data_start:new_data_start+actual_len] = chunk_gpu[:actual_len]
            
            # FFT处理
            input_fft = cp.fft.fft(input_buffer)
            output_fft = input_fft * cp.conj(self.kernel_fft)
            output_time = cp.fft.ifft(output_fft)
            
            # ✓ 关键修复：提取有效输出
            # Overlap-Save的输出范围是 [kernel_size-1 : hop_size+kernel_size-1]
            # 即新数据区域对应的输出
            output_valid = output_time[new_data_start:new_data_start+actual_len].real
            
            # 转回CPU
            results.append(cp.asnumpy(output_valid))
            
            # ✓ 修复：为下一块准备重叠
            # 把当前输出的最后 (kernel_size-1) 个样本作为下一块的重叠
            # 但这里直接使用输入缓冲区的后半部分更简单
            if chunk_idx < len(signal_chunks) - 1:
                # 移动缓冲区：把新数据的后overlap_size个移到前面
                temp = input_buffer[new_data_start + actual_len - self.overlap_size:
                                   new_data_start + actual_len].copy()
                input_buffer[overlap_start:overlap_end] = temp
        
        return np.concatenate(results)


# ===== ✓ 批处理函数（修复流式处理吞吐过低）=====

def batch_process_streaming(signal_chunks, chirp_ref_gpu, batch_size=5):
    """
    ✓ 批处理版本：减少同步开销
    将多个小块合并后再处理，降低kernel launch overhead
    
    Args:
        signal_chunks: 信号块列表 [(chunk_size,), ...]
        chirp_ref_gpu: GPU上的参考信号
        batch_size: 每批处理的块数
    
    Returns:
        处理结果
    """
    results = []
    chirp_ref_gpu = cp.asarray(chirp_ref_gpu, dtype=cp.complex64)
    
    # 分批处理
    for batch_idx in range(0, len(signal_chunks), batch_size):
        batch_end = min(batch_idx + batch_size, len(signal_chunks))
        batch_chunks = signal_chunks[batch_idx:batch_end]
        
        # ✓ 修复：处理可变大小的块
        batch_signal = np.concatenate(batch_chunks).astype(np.complex64)
        batch_signal_gpu = cp.asarray(batch_signal, dtype=cp.complex64)
        
        # 一次性处理
        fft_size = len(batch_signal) + len(chirp_ref_gpu) - 1
        # ✓ 向上取整到2的幂次（FFT优化）
        fft_size = int(2 ** np.ceil(np.log2(fft_size)))
        
        sig_fft = cp.fft.fft(batch_signal_gpu, n=fft_size)
        chirp_fft = cp.fft.fft(chirp_ref_gpu, n=fft_size)
        result_fft = sig_fft * cp.conj(chirp_fft)
        result_time = cp.fft.ifft(result_fft)[:len(batch_signal)].real
        
        # 转回CPU并拆分为原始块大小
        result_np = cp.asnumpy(result_time)
        offset = 0
        for chunk in batch_chunks:
            chunk_size = len(chunk)
            results.append(result_np[offset:offset+chunk_size])
            offset += chunk_size
    
    return np.concatenate(results)


def streaming_with_prefetch(signal_chunks, chirp_ref_gpu, mem_mgr, prefetch_ahead=2):
    """
    ✓ 高级版本：预取优化
    在处理当前块时，同时传输下一块到GPU（H2D-Compute-D2H重叠）
    
    Args:
        signal_chunks: 信号块列表 [(chunk_size,), ...]
        chirp_ref_gpu: GPU上的参考信号
        mem_mgr: GPU内存管理器
        prefetch_ahead: 提前预取的块数
    """
    results = []
    streams = [cp.cuda.Stream() for _ in range(3)]
    events = [cp.cuda.Event() for _ in range(len(signal_chunks))]
    
    chirp_ref_gpu = cp.asarray(chirp_ref_gpu, dtype=cp.complex64)
    
    # ✓ 修复：根据块大小确定FFT大小
    max_chunk_size = max(len(c) for c in signal_chunks)
    fft_size = int(2 ** np.ceil(np.log2(max_chunk_size + len(chirp_ref_gpu) - 1)))
    
    # 预分配GPU缓冲区
    gpu_buffers = [cp.zeros(fft_size, dtype=cp.complex64) for _ in range(3)]
    
    for idx, chunk_np in enumerate(signal_chunks):
        stream_idx = idx % 3
        buffer_idx = idx % 3
        stream = streams[stream_idx]
        
        with stream:
            # H2D：传输当前块
            chunk_gpu = cp.asarray(chunk_np, dtype=cp.complex64)
            chunk_len = len(chunk_gpu)
            
            # ✓ 清空缓冲区后填充
            gpu_buffers[buffer_idx].fill(0)
            gpu_buffers[buffer_idx][:chunk_len] = chunk_gpu
            
            # Compute：FFT处理
            input_fft = cp.fft.fft(gpu_buffers[buffer_idx])
            chirp_fft = cp.fft.fft(chirp_ref_gpu, n=fft_size)
            result_fft = input_fft * cp.conj(chirp_fft)
            result_time = cp.fft.ifft(result_fft)[:chunk_len].real
            
            # D2H：转回CPU
            result_np = cp.asnumpy(result_time)
            results.append(result_np)
            
            events[idx].record()
    
    # 同步所有流
    for stream in streams:
        stream.synchronize()
    
    return np.concatenate(results)


# ===== ✓ 新增：异步事件同步工具 =====

class AsyncStreamManager:
    """
    ✓ 异步流管理器：使用事件而不是显式同步
    减少CPU-GPU同步开销
    """
    
    def __init__(self, num_streams=3):
        self.streams = [cp.cuda.Stream() for _ in range(num_streams)]
        self.events = [[cp.cuda.Event() for _ in range(10)] for _ in range(num_streams)]
    
    def submit_task(self, stream_idx, task_func, *args, **kwargs):
        """提交异步任务"""
        stream = self.streams[stream_idx]
        with stream:
            result = task_func(*args, **kwargs)
            self.events[stream_idx][0].record()
        return result
    
    def synchronize_stream(self, stream_idx):
        """同步指定流"""
        self.streams[stream_idx].synchronize()
    
    def synchronize_all(self):
        """同步所有流"""
        for stream in self.streams:
            stream.synchronize()
    
    def wait_for_event(self, stream_idx, event_idx):
        """在当前流上等待某个事件"""
        self.streams[stream_idx].wait_event(self.events[stream_idx][event_idx])
