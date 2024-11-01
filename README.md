# Chitu (赤兔) Attention: A Quantized Attention Library Suite

<p align="center">
  <img src="./resource/chitu.png" alt="Chitu Logo" width="33%">
</p>

ChituAttention is a comprehensive library of quantized Attention implementations. The name "Chitu" (赤兔, meaning "Red Hare") comes from a legendary swift horse in Chinese history, symbolizing speed and efficiency.

This library is designed for Attention computations for long sequences, where Attention calculations typically dominate the computational time compared to subsequent FFN operations. By quantizing the softmax(QK^T)V computation, ChituAttention achieves significant speedups.

We've collected implementations from two leading Quantized Attention repositories, i.e. [SageAttention](https://github.com/OpenBMB/Sage-Attention) and [Int8FlashAttention](https://github.com/OpenBMB/Int8-FlashAttention), unified their interfaces with FlashAttention. Chitu is designed to integrate seamlessly with the sequence parallel processing in the [feifeibear/Long-Context-Attention](https://github.com/feifeibear/long-context-attention) repository.

## Why a Separate Repository?

We maintain this dedicated repository because both SageAttention and Int8FlashAttention are currently in their demonstration phases, making it challenging to merge our proposed improvements. Additionally, our extensive testing allows us to:
- Rapidly introduce new features
- Address bugs promptly
- Ensure robust performance across a wider range of use cases
- Applied in Sequence Parallel USP

## Citations

**SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration**  
Paper: https://arxiv.org/abs/2410.02367  
Jintao Zhang, Jia Wei, Pengle Zhang, Jun Zhu, Jianfei Chen

**Int-FlashAttention: Enabling Flash Attention for Int8 Quantization**  
Paper: https://arxiv.org/pdf/2409.16997v2  
Shimao Chen, Zirui Liu, Zhiying Wu, Ce Zheng, Peizhuang Cong, Zihan Jiang, Yuhan Wu, Lei Su, Tong Yang
