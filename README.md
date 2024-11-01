# Chitu (赤兔) Attention: A Quantized Attention Library Suite

<p align="center">
  <img src="./resource/chitu.png" alt="Chitu Logo" width="33%">
</p>

ChituAttention is a comprehensive library of quantized Attention implementations. The name "Chitu" (赤兔, meaning "Red Hare") comes from a legendary swift horse in Chinese history, symbolizing speed and efficiency.

This library is designed for Attention computations for long sequences, where Attention calculations typically dominate the computational time compared to subsequent FFN operations. By quantizing the softmax(QK^T)V computation, ChituAttention achieves significant speedups.

We've collected implementations from two leading Quantized Attention repositories, i.e. [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention) and [INT-FlashAttention](https://github.com/INT-FlashAttention2024/INT-FlashAttention), unified their interfaces with FlashAttention. Chitu is designed to integrate seamlessly with the sequence parallel processing in the [feifeibear/Long-Context-Attention](https://github.com/feifeibear/long-context-attention) repository.

## Why a Separate Repository?

We maintain this dedicated repository because both SageAttention and Int8FlashAttention are currently in their demonstration phases, making it challenging to merge our proposed improvements. Additionally, our extensive testing allows us to:
- Rapidly introduce new features
- Address bugs promptly
- Ensure robust performance across a wider range of use cases
- Applied in Sequence Parallel USP


## Performance

FA is FlashAttention, Sage is SageAttention, Int8 is Int8FlashAttention.

### 1-GPU Performance (L40, float16)

| Sequence Length | Method | Max Diff | Min Diff | Latency (sec) |
|----------------|--------|-----------|-----------|---------------|
| 10K            | FA     | 0.00E+00  | 0.00E+00  | 1.57         |
|                | Sage   | 2.20E-03  | 1.64E-04  | 1.08         |
|                | Int8   | 1.95E-02  | 5.02E-04  | 2.74         |
| 100K           | FA     | 0.00E+00  | 0.00E+00  | 154.68       |
|                | Sage   | 6.10E-04  | 5.21E-05  | 111.55       |
|                | Int8   | 7.39E-03  | 2.63E-04  | 262.83       |
| 1M             | FA     | 0.00E+00  | 0.00E+00  | 21055.36     |
|                | Sage   | 1.74E-04  | 1.66E-05  | 12723.77     |
|                | Int8   | 3.97E-03  | 1.34E-04  | 38920.12     |

### Apply in DeepSpeed-Ulysses 8-GPU Performance

Run on 8xL40 PCIe Gen4 GPUs in float16 format. The sequence is local sequence length. The global sequence length is 8 times of the local sequence length.

| Sequence Length | Method | Max Diff | Min Diff | Latency (fp16) |
|----------------|--------|-----------|-----------|----------------|
| 10K            | FA     | 6.10E-05  | 2.56E-06  | 14.79         |
|                | Sage   | 1.45E-03  | 1.64E-04  | 19.72         |
|                | Int8   | 1.04E-02  | 4.99E-04  | 37.01         |
| 100K           | FA     | 0.00E+00  | 0.00E+00  | 126.33        |
|                | Sage   | 5.34E-04  | 5.20E-05  | 117.05        |
|                | Int8   | 8.48E-03  | 2.64E-04  | 158.41        |
| 1M             | FA     | 0.00E+00  | 0.00E+00  | 4726.77       |
|                | Sage   | 1.10E-02  | 1.66E-05  | 3165.62       |
|                | Int8   | 4.03E-03  | 1.34E-04  | 6414.29       |

Based on these results, we can conclude that SageAttention has lower errors than Int8FlashAttention. SageAttention also achieves lower latency than FlashAttention. Int8FlashAttention not only shows noticeable errors but also fails to provide acceleration benefits.

## Citations

**SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration**  
Paper: https://arxiv.org/abs/2410.02367  
Jintao Zhang, Jia Wei, Pengle Zhang, Jun Zhu, Jianfei Chen

**Int-FlashAttention: Enabling Flash Attention for Int8 Quantization**  
Paper: https://arxiv.org/pdf/2409.16997v2  
Shimao Chen, Zirui Liu, Zhiying Wu, Ce Zheng, Peizhuang Cong, Zihan Jiang, Yuhan Wu, Lei Su, Tong Yang
