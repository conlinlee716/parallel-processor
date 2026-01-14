# Parallel Processor 并行信号处理库

这是一个基于Python的并行信号处理库，包含多种信号处理算法的并行实现，旨在提高信号处理的效率和性能。

## 项目结构

- **BeamForming-main/**: 波束形成主程序，包含GUI界面和多种场景配置
- **parallelBF/**: 并行波束形成算法实现与性能分析
- **parallelConv/**: 并行卷积算法实现与性能分析
- **parallelConv1/**: 改进的并行卷积算法实现
- **parallelDOA/**: 并行DOA（波达方向）估计算法实现
- **parallelDOA1/**: 改进的并行DOA估计算法实现
- **parallelDOA3/**: 第三个并行DOA估计算法实现
- **parallelFFT/**: 并行FFT（快速傅里叶变换）算法实现
- **parallelGPU/**: GPU加速的并行信号处理实现
- **parallelPulseComp/**: 并行脉冲压缩算法实现

## 主要功能

### 波束形成 (BeamForming)
- 多种波束形成算法实现
- GUI界面支持实时可视化
- 多种应用场景配置（5G、肿瘤消融、超声等）

### 并行卷积 (parallelConv)
- 高性能并行卷积算法
- 支持多种信号类型（LFM、正弦波、QPSK等）
- 性能分析与优化

### DOA估计 (parallelDOA)
- 多种DOA估计算法的并行实现
- 天线阵列配置支持
- 性能与分辨率分析

### 快速傅里叶变换 (parallelFFT)
- 并行FFT算法实现
- 性能与复杂度分析
- 多线程优化

### GPU加速 (parallelGPU)
- CUDA加速的信号处理算法
- GPU与CPU性能对比
- 内存与数据流转优化

### 脉冲压缩 (parallelPulseComp)
- 并行脉冲压缩算法
- 多种波形支持（LFM等）
- 分辨率与效率分析

## 环境要求

- Python 3.7+
- numpy
- scipy
- matplotlib
- （可选）CUDA Toolkit（用于GPU加速）
- （可选）PyQt5（用于GUI界面）

## 安装

1. 克隆仓库：
```bash
git clone <repository-url>
cd parallel-processor
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

### 运行波束形成GUI
```bash
cd BeamForming-main
python main.py
```

### 运行并行波束形成
```bash
cd parallelBF
python parallelBF.py
```

### 运行并行卷积
```bash
cd parallelConv
python parallelConv.py
```

### 运行并行DOA估计
```bash
cd parallelDOA
python parallelDOA.py
```

### 运行并行FFT
```bash
cd parallelFFT
python parallelFFT.py
```

### 运行GPU加速处理
```bash
cd parallelGPU
python dsp_pipeline.py
```

### 运行并行脉冲压缩
```bash
cd parallelPulseComp
python parallelPulseComp.py
```

## 性能分析

每个子项目都包含性能分析脚本和图表，展示了并行算法相对于串行算法的性能提升、复杂度分析、内存使用情况等。

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

本项目采用MIT许可证，详情请查看LICENSE文件。