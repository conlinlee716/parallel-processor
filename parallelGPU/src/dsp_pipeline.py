"""GPU并行DSP管线 - 修复版本"""
import cupy as cp
import numpy as np
from src.memory_manager import GPUMemoryManager, OverlapSaveProcessor
from src.cuda_kernels import OptimizedKernelWrapper


class DSPGPUPipeline:
    """完整GPU DSP处理管线"""
    
    def __init__(self, config):
        self.config = config
        self.mem_mgr = GPUMemoryManager(config['pinned_mem_size'])
        self.cuda_kernels = OptimizedKernelWrapper()
        
        # 初始化流处理器
        self.overlap_processor = OverlapSaveProcessor(
            config['fft_size'],
            config['hop_size'],
            self.mem_mgr
        )
    
    def pulse_compress(self, signal_gpu, chirp_ref_gpu, block_size=2048):
        """
        脉冲压缩：信号与参考信号的匹配滤波
        使用FFT * conj(FFT) * IFFT实现
        
        ✓ 修复版本：正确处理数据类型
        """
        N = len(signal_gpu)
        M = len(chirp_ref_gpu)
        padded_size = N + M - 1
        
        # ✓ 确保数据类型是complex64
        signal_gpu = cp.asarray(signal_gpu, dtype=cp.complex64)
        chirp_ref_gpu = cp.asarray(chirp_ref_gpu, dtype=cp.complex64)
        
        # FFT（零填充）
        sig_fft = cp.fft.fft(signal_gpu, n=padded_size)
        ref_fft = cp.fft.fft(chirp_ref_gpu, n=padded_size)
        
        # 频域相乘（使用融合kernel或直接乘法）
        result_fft = sig_fft * cp.conj(ref_fft)
        
        # IFFT
        result = cp.fft.ifft(result_fft)
        
        # ✓ 关键修复：取实部（虚部为数值误差）
        return result[:N].real
    
    def pulse_compress_fused(self, signal_gpu, chirp_ref_gpu):
        """
        ✓ 融合版本：使用自定义kernel
        在单个kernel中完成窗函数、相位校正等操作
        """
        N = len(signal_gpu)
        M = len(chirp_ref_gpu)
        padded_size = N + M - 1
        
        signal_gpu = cp.asarray(signal_gpu, dtype=cp.complex64)
        chirp_ref_gpu = cp.asarray(chirp_ref_gpu, dtype=cp.complex64)
        
        # FFT
        sig_fft = cp.fft.fft(signal_gpu, n=padded_size)
        ref_fft = cp.fft.fft(chirp_ref_gpu, n=padded_size)
        
        # 融合：复数乘法 + 窗函数 + 相位校正
        window = cp.hann(N, dtype=cp.float32)
        phases = cp.linspace(0, 2*np.pi, N, dtype=cp.float32)
        
        result_fft = sig_fft * cp.conj(ref_fft)
        result_time = cp.fft.ifft(result_fft)[:N]
        
        # 使用融合kernel应用窗和相位
        result = self.cuda_kernels.fused_pulse_compress(
            result_time, chirp_ref_gpu, window, phases
        )
        
        return result.real
    
    def beamform_mvdr(self, signal_matrix_gpu, num_beams=1):
        """
        MVDR（最小方差无失真响应）波束成形
        signal_matrix_gpu: (num_elements, num_samples)
        
        公式：
        R = X * X^H / N
        w = R^{-1} * a / (a^H * R^{-1} * a)
        beam = w^H * X
        """
        num_elements, num_samples = signal_matrix_gpu.shape
        
        # 步骤1：计算协方差矩阵 R = X * X^H / N
        R = cp.dot(signal_matrix_gpu, cp.conj(signal_matrix_gpu.T)) / num_samples
        
        # 步骤2：加正则化（数值稳定性，Diagonal Loading）
        alpha = 0.01 * cp.trace(R) / num_elements
        R_reg = R + alpha * cp.eye(num_elements, dtype=R.dtype)
        
        # 步骤3：计算协方差矩阵的逆
        R_inv = cp.linalg.inv(R_reg)
        
        # 步骤4：生成导向向量并计算波束权重
        beams = []
        for beam_idx in range(num_beams):
            # 简单导向向量（假设均匀直线阵，波束指向 0°）
            a = cp.exp(1j * 2 * np.pi * np.arange(num_elements) * beam_idx / (num_elements + 1e-8))
            a = a[:, cp.newaxis]  # (num_elements, 1)
            
            # 波束权重：w = R^{-1} * a / (a^H * R^{-1} * a)
            numerator = cp.dot(R_inv, a)
            denominator = cp.dot(cp.conj(a.T), numerator)
            w = numerator / (cp.abs(denominator) + 1e-8)
            
            # 波束输出：y = w^H * X
            beam = cp.dot(cp.conj(w.T), signal_matrix_gpu).squeeze()
            beams.append(beam)
        
        return cp.concatenate(beams, axis=0)
    
    def process_end2end(self, signal_np, chirp_ref_np, method='direct'):
        """
        端到端处理：信号输入 -> 脉冲压缩/波束成形 -> 输出
        
        ✓ 修复版本：确保返回NumPy数组
        """
        # 传输到GPU（确保数据类型）
        signal_gpu = cp.asarray(signal_np, dtype=cp.complex64)
        chirp_ref_gpu = cp.asarray(chirp_ref_np, dtype=cp.complex64)
        
        if method == 'overlap_save':
            # 使用Overlap-Save流式处理
            num_chunks = (len(signal_gpu) + self.config['hop_size'] - 1) // self.config['hop_size']
            chunks = [
                signal_gpu[i*self.config['hop_size']:(i+1)*self.config['hop_size']]
                for i in range(num_chunks)
            ]
            result = self.overlap_processor.process_streaming_optimized(chunks, chirp_ref_gpu)
        else:
            # 直接方法（一次性计算）
            result_gpu = self.pulse_compress(signal_gpu, chirp_ref_gpu)
            result = cp.asnumpy(result_gpu)  # ✓ 转回NumPy
        
        return result
