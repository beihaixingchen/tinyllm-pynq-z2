# TinyLLMStory-zh Q6W 量化

## 1. 定义和范围

这里的 **Q6W** 表示 signed-int6 weight-only post-training quantization，不是某个未指定小数位的固定点 `Qm.n` 格式。

- 基线来源是 Experiment 2、step 23,900 的 FP32 checkpoint：`training/checkpoints/masked_v2_step23900.pt`。
- 所有二维 embedding/Linear 权重按输出行做对称量化；每行一个 FP32 scale。
- 每个行的量化范围是 `[-31,31]`，`-32` 编码保留但不会被产生。
- 一行全零时 scale 设为 1；其他行 `scale = max(abs(row)) / 31`。
- `q = clamp(round(weight / scale), -31, 31)`，重建值 `weight_hat = q * scale`。PyTorch round 使用 round-to-nearest ties-to-even。
- RMSNorm scale 参数保持 FP32；激活、KV cache、累加器、RoPE 和 softmax 在参考评估中也保持 FP32。
- 输入 embedding 与输出分类矩阵共享权重，只存储一次。其 output logits 仍使用同一组按词表行存储的 Q6 权重及对应 scale。

因此这个工件将权重存储压缩为 6-bit，但它**不是完整的整数推理网络**。在 FPGA 上把 MAC 全部改为整数，还需要定义 activation quantization、scale 合并、累加器位宽、舍入/饱和、RMSNorm 与非线性实现，再逐层与 FP32 对照。

## 2. 产物

| 文件 | 用途 |
| --- | --- |
| `quantization/artifacts/tinyllamas_zh_story_q6w.bin` | packed Q6 权重、FP32 row scales 和 FP32 RMSNorm 参数的 payload。 |
| `quantization/artifacts/tinyllamas_zh_story_q6w.json` | Header、tensor 名称/shape、字节偏移、量化统计和源 checkpoint SHA-256。 |
| `quantization/quantize_q6w.py` | 可复现导出命令。 |
| `quantization/q6w_format.py` | 打包、解包、导出和解量化参考。 |
| `quantization/evaluate_q6w.py` | 和源 FP32 checkpoint 运行同一提示集/验证 batches 的比较工具。 |
| `quantization/artifacts/tinyllamas_zh_story_q6w_qat_lr1e6.bin` / `.json` | 经 Q6W 感知微调后重新导出的独立实验版本；不覆盖原 Q6W 工件。 |
| `quantization/evaluate_q6w.py` | 生成成对评估 JSONL 和汇总 JSON；这些数据集相关输出不随仓库发布。 |

当前 bin 大小为 **1,130,816 bytes**；源模型唯一参数的 FP32 payload 为 **5,789,184 bytes**，体积缩小约 **5.12x（80.46%）**。其中 1,445,888 个矩阵/embedding 参数以 6-bit 保存，1,408 个 RMSNorm scale 参数以 FP32 保存。工件还有逐 section 的 64-byte 对齐填充，方便软件/PL DMA 和地址生成。

源 checkpoint SHA-256：`8BAEA4008389992AA5D2D3888235A29193CEB8F16540B3A68770025952FDF43F`。Q6 binary SHA-256：`4edc130ef84cbb4450667a3e74ad5f8403608b3f90247256c006ec3604c7a0bf`。

## 3. 二进制格式

所有整数和浮点数均 little-endian。Manifest 是格式的一部分：binary header 不重复存每个 tensor 的名字或地址表，RTL/software loader 应按 JSON 中的 offsets 读取，或把同一固定顺序固化到 ROM/address decoder。

文件开头为 24-byte header，之后每个 payload offset 均对齐到 64 bytes：

```text
struct <8sIIII
magic       = b"TLQ6ROW\0"  # 8 bytes
version     = 1              # uint32
tensor_count= 47             # uint32
weight_bits = 6              # uint32
scale_dtype = 1              # uint32: FP32
```

36 个量化 tensor 按 `tok_embeddings.weight`，随后 layer 0..4 每层按 attention `wq,wk,wv,wo` 和 FFN `w1,w2,w3` 排列。每个 tensor 有两个独立 section：

1. `scale_offset`：按输出行顺序存 `rows` 个 little-endian FP32 scale。
2. `data_offset`：行优先的 packed signed-int6；每行元素数必须被 4 整除，本模型所有量化行满足这一条件。

每连续 4 个 int6 值 `q0..q3` 先按 6-bit two's-complement 解释为无符号位型 `v = q & 0x3f`，低位优先打包：

```text
b0 =  v0        | ((v1 & 0x03) << 6)
b1 = (v1 >> 2)  | ((v2 & 0x0f) << 4)
b2 = (v2 >> 4)  | ( v3        << 2)
```

解包得到 unsigned 0..63 后，若 bit5=1 则减 64 做 sign extension。每 4 个值正好 3 bytes。

其余 11 个 RMSNorm 向量以 FP32 little-endian、连续 row-major section 存储。共享的 `output.weight` 不另存；加载器需要将其 alias 到解包的 `tok_embeddings.weight`。RoPE 表不在此工件中，仍按 [RTL_REFERENCE.md](RTL_REFERENCE.md) 的公式生成或预计算。二进制 section 精确偏移、长度、shape 和 per-tensor 量化误差在 JSON manifest。

## 4. 复现导出

在仓库根目录、已安装 PyTorch 的环境中运行：

```powershell
python quantization/quantize_q6w.py `
  --checkpoint training/checkpoints/masked_v2_step23900.pt `
  --output quantization/artifacts/tinyllamas_zh_story_q6w.bin `
  --manifest quantization/artifacts/tinyllamas_zh_story_q6w.json
```

Exporter 默认拒绝覆盖现有文件。若换了 checkpoint 或量化参数，应选择新的 output/manifest 名称，并重新跑所有评估。

## 5. FP32 对照结果

通过 `evaluate_q6w.py` 将 Q6 权重解量化到 FP32，再用相同模型结构运行。FP32 对照与 Q6 使用同一个 step 23,900 源 checkpoint；验证 loss 用相同 masked objective 和同一组 50 个 validation batches。

| 指标 | FP32 | Q6W 解量化参考 |
| --- | ---: | ---: |
| Masked validation loss | 2.186670 | 2.203846 |
| 平均逐提示 logits cosine similarity | 1.000000 | 0.992569 |
| 平均逐提示 logits RMSE | 0 | 0.387573 |
| 最后提示位置 greedy top-1 一致 | - | 120 / 128（93.75%） |
| 相同采样下 EOS / token cap | 122 / 6 | 121 / 7 |
| 平均输出 token 数 | 75.20 | 73.16 |

Validation loss 增加 0.017177（约 0.79%）。同 seed 随机采样的完整 token 序列在 128/128 个提示上均不完全相同，平均共同前缀 3.57 tokens；小的 logits 扰动会改变采样分支并放大为不同故事，这不等价于 128 个样本都变差。生成语义本来仍有限，少量人工样本不支持“质量不变”的结论。就当前数据看，Q6W 的确定性 logits/validation loss 变化较小，但应在 RTL 量化路径上继续逐层验证。

比较命令（默认 128 prompts、50 validation batches、FP32 reference）：

```powershell
python quantization/evaluate_q6w.py `
  --checkpoint training/checkpoints/masked_v2_step23900.pt `
  --binary quantization/artifacts/tinyllamas_zh_story_q6w.bin `
  --manifest quantization/artifacts/tinyllamas_zh_story_q6w.json `
  --tokenizer tok4096.model `
  --prompts training/data/story_pairs_masked_v2/eval_prompts.jsonl `
  --training-root training `
  --data-dir training/data/story_pairs_masked_v2 `
  --output quantization/artifacts/q6w_eval.jsonl
```

## 6. Q6W 感知微调

可以在 Q6W 的有效前向上继续微调。此实验保留 FP32 master 参数供优化器更新；每次 embedding/Linear 前向都临时按相同 per-row Q6 公式 fake-quantize，反向对 master weights 使用 straight-through estimator (STE)。因此训练看到真实的 Q6W 取整误差，完成后再由相同 exporter 生成 Q6W 文件。它不是对 packed int6 code 直接做梯度下降，也没有量化激活或 KV cache。

- 实现：`training/q6w_fake_quant.py`；`training/train.py` 用 `qat_q6w=True` 启用。全模型 logits 与直接解包后的 Q6W 参考完全相等；STE 和 state-dict 保持测试见 `training/test_q6w_fake_quant.py`。
- 起点：Experiment 2 step23,900 FP32 checkpoint 的独立副本。QAT FP32 master checkpoint 为 `training/checkpoints/q6w_qat_lr1e6_step25000.pt`。训练最多1,200步，CUDA FP16+GradScaler，LR`1e-6`、fresh AdamW、zero weight decay。
- 新 Q6W binary 大小仍为1,130,816 bytes，SHA-256：`ddeaa72c5923f2c445a7543cf53dc13dffc674f53f49f822c0653ef6735c4c90`。

| 相同 masked validation set（50 batches） | 微调前 Q6W | QAT 后 Q6W |
| --- | ---: | ---: |
| Q6W masked loss | 2.203846 | 2.193370 |
| 对应 FP32 master loss | 2.186670 | 2.192481 |
| FP32 master 与 Q6W 的 loss gap | 0.017177 | 0.000889 |
| 同128提示 EOS / cap | 121 / 7 | 122 / 6 |
| 平均生成 token | 73.16 | 73.70 |

QAT Q6W validation loss 降低0.010476（约0.48%），说明它在此评估集上补偿了量化误差。128个采样输出均发生文本变化，平均共同解码文本前缀约8.13字符；人工抽查是混合结果，角色混淆和情节矛盾仍在。因此保留 QAT 文件作实验版本，不保证普遍故事质量改善。基线和 QAT 两个 Q6W 文件都保留，便于 RTL/PYNQ A/B。

复现命令和数据准备见 [`training/README.md`](../training/README.md)。QAT checkpoint 是 FP32 master；部署时请加载 Q6W `.bin` 和配套 manifest，不能把 `.pt` 当作 packed 权重。训练语料不随仓库发布。

## 7. KV cache 精度与片上容量

针对 QAT Q6W 权重，比较了 FP32 KV 存储和 FP16 KV 存储（写 cache 时转 FP16，attention 读取时转 FP32；attention dot/softmax/加权和仍为 FP32）：

| 指标 | FP32 KV | FP16 KV |
| --- | ---: | ---: |
| 512-token K/V cache | 1,310,720 B（1.25 MiB） | 655,360 B（0.625 MiB） |
| Q6W 权重 + KV + 单-token scratch | 2,470,976 B（2.357 MiB） | 1,815,616 B（1.732 MiB） |
| 128 个提示末端 logits top-1 与 FP32 KV 一致 | 基准 | 128 / 128 |
| 平均 logits cosine / RMSE | 基准 | 0.999999868 / 0.001513 |
| 511 个位置逐步 probe：top-1 一致 | 基准 | 510 / 511 |
| 511 个位置整体 logits cosine / RMSE | 基准 | 0.999999943 / 0.001593 |
| 固定采样下完整序列完全一致 | 基准 | 101 / 128 |

因此当前建议 ASIC 首版先使用 **FP16 KV SRAM + FP32 attention 累加**：可节省640 KiB，而测得 logits 扰动很小。生成采样仍会放大小扰动，128条中有27条 token 序列至少一处不同；EOS/cap 为 FP32 `122/6`、FP16 `121/7`。511-token probe 是将 held-out prompt/reference token 串接起来的舍入/长度压力检查，不代表自然故事的全面评测。FP16 乘法/FP16 累加、定点 KV 或 INT8 KV 均未由此验证。

精确基础容量（不含 tokenizer、ECC、UART/FIFO、控制器及测试冗余）：Q6W 文件1,130,816 B + FP16 KV 655,360 B + 当前单-token scratch 29,440 B = 1,815,616 B（1.732 MiB）。实际 SRAM 宏容量需要再留实现余量。

评估脚本：`quantization/compare_kv_cache.py`。成对输出和汇总 JSON 由脚本在本地
生成，不随仓库发布；RTL 规格与注意事项见 [RTL_REFERENCE.md](RTL_REFERENCE.md)。

## 8. PYNQ implementation

对于一行权重和 FP32 输入激活，数学参考是：

```text
acc[o] = sum_i int6_weight[o,i] * activation[i]
output[o] = row_scale[o] * acc[o]
```

若激活也量化为 `activation[i] ~= activation_scale * activation_int[i]`，则整数点积可写成：

```text
output[o] ~= row_scale[o] * activation_scale
             * sum_i int6_weight[o,i] * activation_int[i]
```

Q6W 文件本身只量化权重。当前 PYNQ 自主推理实现会在 host 侧准备阶段把
signed-int6 码值无损展开为 DDR 中的 int8 模型映像；PL 再将每个 activation
vector 量化为 signed-int16。GEMV 使用 int8×int16 乘积，并合并 activation
scale 与每行 FP32 weight scale。通过验证的实现中，attention、softmax、
RMSNorm 和 KV 存储仍使用 FP32 运算；这是混合精度加速器，不是全整数网络。

对于原始 Q6 码值 `[-31,31]`，乘 signed-int8 activation 时单项最大绝对值为
3,968，需要 signed 13-bit 表示。保守估计，长度为32/128/352的点积至少需要
signed 18/20/22-bit accumulator；signed-int16 activation 需要更宽累加器。
更改 datapath 时需保留 scale 应用并重新检查累加器范围。PYNQ overlay、布线
时序、板上吞吐和精确输出校验见[项目 README](../README.md)；构建与部署脚本
位于 `scripts/` 和 `rtl/autonomous/`。

本仓库已经包含 PYNQ-Z2 autonomous PL 实现。相同固定请求下，当前签核 overlay
板测为203.66 generated tok/s，五次运行均与 Q6 PyTorch golden 精确匹配；具体
时钟、资源、时序和比较基线见[项目 README](../README.md)。
