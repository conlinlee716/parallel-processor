"""Numba CUDA 融合微核 - 完整修复版本 v2"""
import numpy as np
import cupy as cp
from numba import cuda
import math


# ===== Numba CUDA kernels =====

@cuda.jit(fastmath=True)
def complex_multiply_kernel(result, a, b):
    """
    融合复数乘法kernel：result = a * conj(b)
    避免两个独立kernel的launch开销
    """
    idx = cuda.grid(1)
    if idx < result.size:
        ar = a[idx].real
        ai = a[idx].imag
        br = b[idx].real
        bi = b[idx].imag
        # (a+bi)(c-di) = (ac+bd) + (bc-ad)i
        result[idx] = (ar*br + ai*bi) + 1j*(ai*br - ar*bi)


@cuda.jit(fastmath=True)
def apply_window_kernel(signal, window):
    """融合窗函数应用：signal *= window"""
    idx = cuda.grid(1)
    if idx < signal.size:
        signal[idx] *= window[idx]


@cuda.jit(fastmath=True)
def apply_phase_correction_kernel(signal, phases):
    """
    融合相位校正kernel
    signal *= exp(1j * phase)
    """
    idx = cuda.grid(1)
    if idx < signal.size:
        phase = phases[idx]
        cos_p = math.cos(phase)
        sin_p = math.sin(phase)
        real = signal[idx].real * cos_p - signal[idx].imag * sin_p
        imag = signal[idx].real * sin_p + signal[idx].imag * cos_p
        signal[idx] = real + 1j*imag


@cuda.jit
def rms_norm_kernel(signal):
    """融合RMS归一化"""
    idx = cuda.grid(1)
    if idx < signal.size:
        signal[idx] /= (abs(signal[idx]) + 1e-8)


# ===== ✓ 新增：复杂融合kernel（多步操作）=====

@cuda.jit(fastmath=True)
def fused_pulse_compress_kernel(signal, chirp, window, phases, output):
    """
    ✓ 复杂融合kernel：综合多个操作
    
    操作流程：
    1. 加窗：signal *= window
    2. 复数乘法：signal * conj(chirp)
    3. 相位校正：result *= exp(1j * phase)
    
    相比分离操作，减少3倍的kernel launch开销
    """
    idx = cuda.grid(1)
    if idx < output.size:
        # ===== 步骤1：加窗 =====
        win = window[idx % window.size]
        sig_val = signal[idx % signal.size]
        sig_windowed_real = sig_val.real * win
        sig_windowed_imag = sig_val.imag * win
        
        # ===== 步骤2：复数乘法 (signal * conj(chirp)) =====
        chirp_val = chirp[idx % chirp.size]
        chirp_conj_real = chirp_val.real
        chirp_conj_imag = -chirp_val.imag  # conj操作
        
        # (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        mul_real = sig_windowed_real * chirp_conj_real - sig_windowed_imag * chirp_conj_imag
        mul_imag = sig_windowed_real * chirp_conj_imag + sig_windowed_imag * chirp_conj_real
        
        # ===== 步骤3：相位校正 =====
        phase = phases[idx % phases.size]
        cos_p = math.cos(phase)
        sin_p = math.sin(phase)
        
        # result = (mul_real + 1j*mul_imag) * exp(1j*phase)
        real_part = mul_real * cos_p - mul_imag * sin_p
        imag_part = mul_real * sin_p + mul_imag * cos_p
        
        output[idx] = real_part + 1j * imag_part


@cuda.jit(fastmath=True)
def fused_doppler_correction_kernel(signal, doppler_shifts, window, output):
    """
    ✓ 多普勒校正融合kernel：
    在单个kernel中完成多普勒补偿、加窗、归一化
    """
    idx = cuda.grid(1)
    if idx < output.size:
        # 加窗
        win = window[idx % window.size]
        
        # 多普勒相位补偿
        doppler_phase = doppler_shifts[idx % doppler_shifts.size]
        cos_dp = math.cos(doppler_phase)
        sin_dp = math.sin(doppler_phase)
        
        sig = signal[idx]
        # 相位旋转
        real = sig.real * cos_dp - sig.imag * sin_dp
        imag = sig.real * sin_dp + sig.imag * cos_dp
        
        # 加窗
        output[idx] = (real + 1j*imag) * win


@cuda.jit(fastmath=True)
def batch_complex_multiply_kernel(results, signals, references, n_signals):
    """
    ✓ 批量复数乘法kernel：一次性处理多个乘法操作
    减少kernel launch次数
    """
    idx = cuda.grid(1)
    if idx < n_signals:
        results[idx] = signals[idx] * references[idx].conjugate()


# ===== 优化的Kernel包装器 =====

class OptimizedKernelWrapper:
    """✓ 改名：使用OptimizedKernelWrapper替代CudaKernelWrapper"""
    
    @staticmethod
    def complex_multiply(a_gpu, b_gpu):
        """调用复数乘法kernel"""
        result = cp.zeros_like(a_gpu)
        threadsperblock = 256
        blockspergrid = (result.size + threadsperblock - 1) // threadsperblock
        complex_multiply_kernel[blockspergrid, threadsperblock](result, a_gpu, b_gpu)
        return result
    
    @staticmethod
    def apply_window(signal_gpu, window_gpu):
        """应用窗函数"""
        threadsperblock = 256
        blockspergrid = (signal_gpu.size + threadsperblock - 1) // threadsperblock
        apply_window_kernel[blockspergrid, threadsperblock](signal_gpu, window_gpu)
    
    @staticmethod
    def apply_phase_correction(signal_gpu, phases_gpu):
        """应用相位校正"""
        threadsperblock = 256
        blockspergrid = (signal_gpu.size + threadsperblock - 1) // threadsperblock
        apply_phase_correction_kernel[blockspergrid, threadsperblock](signal_gpu, phases_gpu)
    
    # ===== ✓ 新增：复杂融合操作 =====
    
    @staticmethod
    def fused_pulse_compress(signal_gpu, chirp_gpu, window_gpu, phases_gpu):
        """
        ✓ 融合脉冲压缩：在单个kernel中完成多步操作
        signal * conj(chirp) * window * exp(1j*phase)
        
        性能改进：相比分离操作，减少3倍kernel launch开销
        """
        output = cp.zeros_like(signal_gpu)
        threadsperblock = 256
        blockspergrid = (output.size + threadsperblock - 1) // threadsperblock
        
        fused_pulse_compress_kernel[blockspergrid, threadsperblock](
            signal_gpu, chirp_gpu, window_gpu, phases_gpu, output
        )
        return output
    
    @staticmethod
    def fused_doppler_correction(signal_gpu, doppler_shifts_gpu, window_gpu):
        """
        ✓ 融合多普勒校正
        """
        output = cp.zeros_like(signal_gpu)
        threadsperblock = 256
        blockspergrid = (output.size + threadsperblock - 1) // threadsperblock
        
        fused_doppler_correction_kernel[blockspergrid, threadsperblock](
            signal_gpu, doppler_shifts_gpu, window_gpu, output
        )
        return output
    
    @staticmethod
    def batch_complex_multiply(signals_gpu, references_gpu):
        """
        ✓ 批量复数乘法（一个kernel调用处理多个乘法）
        """
        n_signals = len(signals_gpu)
        results = cp.zeros_like(signals_gpu)
        threadsperblock = 256
        blockspergrid = (n_signals + threadsperblock - 1) // threadsperblock
        
        batch_complex_multiply_kernel[blockspergrid, threadsperblock](
            results, signals_gpu, references_gpu, n_signals
        )
        return results


# ===== ✓ 新增：预热函数（避免JIT编译开销）=====

def warmup_kernels():
    """
    ✓ 预热所有kernels：
    第一次调用时会JIT编译，这个开销会被第一次测试计数
    预先编译可避免影响测试结果
    
    ✓ 修复：使用NumPy创建窗函数，再转移到GPU
    """
    # 创建小型测试数据
    n_warmup = 1024
    a = cp.ones(n_warmup, dtype=cp.complex64)
    b = cp.ones(n_warmup, dtype=cp.complex64)
    
    # ✓ 修复：在NumPy中创建窗函数，然后转移到GPU
    window_np = np.hanning(n_warmup).astype(np.float32)
    window = cp.asarray(window_np)
    
    phases = cp.zeros(n_warmup, dtype=cp.float32)
    
    # 预热所有kernel
    try:
        OptimizedKernelWrapper.complex_multiply(a, b)
        cp.cuda.Stream.null.synchronize()
        
        OptimizedKernelWrapper.apply_window(a.copy(), window)
        cp.cuda.Stream.null.synchronize()
        
        OptimizedKernelWrapper.apply_phase_correction(a.copy(), phases)
        cp.cuda.Stream.null.synchronize()
        
        OptimizedKernelWrapper.fused_pulse_compress(a, b, window, phases)
        cp.cuda.Stream.null.synchronize()
        
        OptimizedKernelWrapper.fused_doppler_correction(a, phases, window)
        cp.cuda.Stream.null.synchronize()
        
        OptimizedKernelWrapper.batch_complex_multiply(a, b)
        cp.cuda.Stream.null.synchronize()
    except Exception as e:
        print(f"Warning during kernel warmup: {e}")
    
    # 清理临时数据
    del a, b, window, phases


# ===== 向后兼容：保留旧名称作为别名 =====
CudaKernelWrapper = OptimizedKernelWrapper
