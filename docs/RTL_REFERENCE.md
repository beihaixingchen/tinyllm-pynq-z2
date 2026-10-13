# TinyLLMStory-zh 网络结构与 RTL 参考

本文用于理解本项目的模型结构和 RTL 数值约定。**主参考实现**是
[`../benchmarks/torch_benchmark.py`](../benchmarks/torch_benchmark.py) 中的
`CachedTransformer`；QAT FP32 master 权重位于
`../training/checkpoints/q6w_qat_lr1e6_step25000.pt`，QAT 前基线位于
`../training/checkpoints/masked_v2_step23900.pt`。参考模型将 Q6W 权重解量化到
FP32，执行 batch=1、逐 token 解码和 KV cache。非缓存 Transformer 实现在
`../training/model.py`，可用于交叉检查网络定义。

## 1. 版本与规格

原始仓库 checkpoint 与微调实验 checkpoint 的模型配置、state-dict 键、所有 tensor shape 完全一致。微调改变的是权重值和训练数据/损失策略，不改变网络图。

| 配置 | 值 |
| --- | ---: |
| `dim` | 128 |
| `n_layers` | 5 |
| `n_heads` | 4 |
| `n_kv_heads` | 2 |
| `head_dim = dim / n_heads` | 32 |
| `kv_dim = dim * n_kv_heads / n_heads` | 64 |
| `n_rep = n_heads / n_kv_heads` | 2 |
| `hidden_dim` | 352 |
| `vocab_size` | 4096 |
| `max_seq_len` | 512 |
| RMSNorm epsilon | `1e-5` |
| RoPE theta | 10000 |
| dropout | 0（推理时 eval） |
| 参数量（去除 tied output alias） | 1,447,296 |
| checkpoint 权重类型 | FP32 |

`hidden_dim` 来自 `int(2 * (4 * dim) / 3)` 向上对齐到 `multiple_of=32`：`341 -> 352`。Embedding 与输出分类矩阵共享同一权重。Checkpoint state dict 中二者以两个键出现且值完全相等；计算唯一参数量时只计一次。

参数量核算：Embedding `4096*128=524,288`；每层 `Wq 16,384 + Wk 8,192 + Wv 8,192 + Wo 16,384 + W1 45,056 + W2 45,056 + W3 45,056 + 两个 RMSNorm 256 = 184,576`；5 层为 922,880；末端 RMSNorm 128；合计 `524,288 + 922,880 + 128 = 1,447,296`。

## 2. 拓扑

```text
token_id [scalar]
  -> Embedding E [4096,128]
  -> 5 x PreNorm Transformer block:
       RMSNorm(attention)
       Q: [128->128], K: [128->64], V: [128->64]
       RoPE(Q,K), grouped-query causal attention (4 Q heads / 2 KV heads)
       Wo: [128->128] + residual
       RMSNorm(FFN)
       SwiGLU: W1 [128->352], W3 [128->352], SiLU(W1*x) elementwise* (W3*x)
       W2: [352->128] + residual
  -> final RMSNorm [128]
  -> tied output projection E: [128->4096] (Linear weight [4096,128])
  -> logits [4096]
```

所有线性层无 bias。每个 block 是 pre-norm；残差连接不做缩放。模型前向输出 logits，不在网络内执行最终词表 softmax、top-p 或 token 选择。

## 3. Checkpoint tensor 与矩阵方向

Checkpoint 从 `torch.load(path, map_location="cpu", weights_only=True)["model"]` 取得。线性层采用 PyTorch `nn.Linear` 存储：`weight[out_features, in_features]`；若激活按列向量表示，`y[o] = sum_i weight[o,i] * x[i]`。矩阵在 checkpoint 中是连续行优先的 FP32 tensor。不要转置权重后又按 PyTorch 方向使用。

| State-dict key（每个 layer `l=0..4`） | Shape | 作用 |
| --- | --- | --- |
| `tok_embeddings.weight` | `[4096,128]` | Token embedding `E`，也绑定输出矩阵 |
| `layers.l.attention_norm.weight` | `[128]` | Attention 前 RMSNorm scale |
| `layers.l.attention.wq.weight` | `[128,128]` | Q projection |
| `layers.l.attention.wk.weight` | `[64,128]` | K projection |
| `layers.l.attention.wv.weight` | `[64,128]` | V projection |
| `layers.l.attention.wo.weight` | `[128,128]` | Attention output projection |
| `layers.l.ffn_norm.weight` | `[128]` | FFN 前 RMSNorm scale |
| `layers.l.feed_forward.w1.weight` | `[352,128]` | SwiGLU gate/up branch 1 |
| `layers.l.feed_forward.w3.weight` | `[352,128]` | SwiGLU branch 2 |
| `layers.l.feed_forward.w2.weight` | `[128,352]` | SwiGLU down projection |
| `norm.weight` | `[128]` | Final RMSNorm scale |
| `output.weight` | `[4096,128]` | Tied alias of `tok_embeddings.weight` |

除输入 embedding 与输出矩阵共享权重外，没有 bias、learned position embedding、额外分类头或额外层。RoPE cos/sin 表在 Python 模型中是非持久 buffer，不存入 `ckpt.pt`；按公式生成，或在开始运行前预计算为 `[512,16]` 表。

## 4. 逐 token 前向定义

以下使用列向量记法，所有张量/累加按 FP32 参考。`RMS(x,gamma)` 对最后一维 `D=128` 计算：

```text
ms  = sum(i=0..127, x[i]*x[i]) / 128
inv = 1 / sqrt(ms + 1e-5)
y[i] = x[i] * inv * gamma[i]
```

对当前 token embedding `x`，每层 `l`：

```text
u = RMS(x, attention_norm[l])
q = Wq[l] * u                  # [128] -> 4 heads * 32
k = Wk[l] * u                  # [64]  -> 2 KV heads * 32
v = Wv[l] * u                  # [64]
RoPE q and k at absolute position pos; do not rotate v
Kcache[l,pos] = k_rot          # store rotated key
Vcache[l,pos] = v              # store unrotated value
```

RoPE 对每个 head 的相邻 pair `(2j,2j+1)`, `j=0..15`：

```text
omega[j] = 10000 ** (-2*j / 32)
angle = pos * omega[j]
even' = even*cos(angle) - odd*sin(angle)
odd'  = even*sin(angle) + odd*cos(angle)
```

Query head 到 KV head 的映射为 `kv_head = query_head // 2`：Q heads 0、1 共用 KV head 0；Q heads 2、3 共用 KV head 1。对每个 Q head `h`，仅注意已缓存位置 `t=0..pos`：

```text
score[t] = dot(q[h], Kcache[l,t, h//2]) / sqrt(32)
a[t] = stable_softmax(score[0..pos])  # subtract max before exp
head_out[h] = sum(t=0..pos, a[t] * Vcache[l,t,h//2])
att = Wo[l] * concat(head_out[0..3])
x = x + att
```

接着执行 SwiGLU 和第二个残差：

```text
u = RMS(x, ffn_norm[l])
a = W1[l] * u                    # [352]
b = W3[l] * u                    # [352]
g[i] = (a[i] / (1 + exp(-a[i]))) * b[i]   # SiLU(a) * b
x = x + W2[l] * g                # W2 shape [128,352]
```

5 层之后：

```text
h = RMS(x, norm.weight)
logits[vocab_id] = dot(tok_embeddings.weight[vocab_id], h)  # one dot product for each of 4096 vocab IDs
```

`exp/sqrt/sin/cos/softmax` 的有限精度实现会产生数值误差。Python logits 对照以 CPU FP32 作为首先要匹配的基准；不同乘加树/融合方式可能不保证逐 bit 相同。

## 5. KV cache 与解码时序

主参考 `CachedTransformer` 面向 batch=1，cache 逻辑 shape：

```text
K: [5, 512, 2, 32] FP32
V: [5, 512, 2, 32] FP32
```

总 cache 为 `2 * 5 * 512 * 2 * 32 * 4 = 1,310,720 bytes`。每个新会话清空/重置 cache。对 token 序列长度 `P` 的 prompt，Python 参考依次将 prompt token 以位置 `0..P-1` 喂入 `step()`；最后一个 prompt token 的 logits 用来采样第一个续写 token。采样得到的续写 token 若后面还需要 logits，则再以位置 `P, P+1, ...` 输入 `step()`。每步只计算当前 token 的 Q/K/V 和 MLP，attention 读位置 `0..pos` 的 K/V；不可读取未来位置。

### KV cache 存储精度实验

对当前 QAT Q6W 权重做了 FP32-cache 与 FP16-cache 对照。FP16 方案只在写 cache 时转成 IEEE binary16；attention 读取时转回 FP32，Q/K 点积、softmax、V 加权和仍按 FP32 做。因此它测试的是 **FP16 KV 存储**，不是 FP16 attention datapath。

- 当前上下文512时，FP32 KV 为1,310,720 bytes；FP16 KV 为655,360 bytes，节省640 KiB。Q6W 权重、工作区不变时，模型+KV+工作区从2,470,976 bytes（2.357 MiB）降到1,815,616 bytes（1.732 MiB），不含 tokenizer/协议缓冲。
- 128条固定验证开头：末端 logits top-1 一致128/128，平均 cosine `0.999999868`、RMSE `0.001513`。在固定 sampler 下，101/128条完整 token 序列一致，平均共同前缀67.73 tokens；EOS/cap 从122/6变为121/7。采样在轻微 logits 变化下仍可能分叉，不能用逐字相同率判断算子精度。
- 另外将 held-out prompt/reference 文本串接成511-token上下文，对每个位置比较 logits：所有位置整体 RMSE `0.001593`、平均 position cosine `0.999999943`，top-1 为510/511一致。256-token probe 为255/256一致。此 probe 是上下文长度/舍入压力检查，不是自然长篇语义测试。
- 基于这组结果，FP16 KV storage + FP32 attention accumulation 是当前建议的 ASIC 首版折衷；尚需在 RTL/门级算术路径上复现，并用更多真实长上下文检查。若 attention 乘加本身改成 FP16、定点或 INT8，那是另一种数值格式，不能由本测试直接推断。
- 可复现实验：`quantization/compare_kv_cache.py`；逐提示输出和长上下文汇总：`quantization/artifacts/kv_fp16_q6w_qat_lr1e6_context_probes.jsonl` 与 `.summary.json`。

结束 EOS/BOS token 由采样控制器识别，不应再作为普通输出文本解码。tokenizer 的 ID：BOS=1、EOS=2、PAD=-1。原始模型训练文本没有显式 EOS；较新的故事续写微调样本加 EOS 并训练模型预测它。推荐硬件接口输出 logits，由 ARM 端做 sampler/stopping；若 sampler 做进 PL，必须严格复现所选的 temperature/top-p/重复控制配置。

## 6. tokenizer、prompt 与 sampler 不属于神经网络

- tokenizer 文件：`tok4096.model`，SentencePiece vocab=4096。RTL 核心建议输入 token ID，不要把 UTF-8/SentencePiece 字符串处理误算为网络层。
- 本仓库 tokenizer 的序列化大小：`tok4096.model` 56,837 B，`tokenizer.bin` 52,053 B，`tok4096.vocab` 49,985 B。`tokenizer.bin` 只是文件大小，不等于加载后的 SRAM。按 ESP32 C tokenizer 的 32-bit 指针结构计算，4096 个 vocab 指针16,384 B、FP32 scores16,384 B、排序索引 `TokenIndex[4096]` 32,768 B、piece 字符串含 NUL 23,377 B、Tokenizer 结构/byte table 532 B、临时 BPE 合并字符串 buffer 75 B，合计基础占用约 **89,520 B（87.4 KiB）**。这还不含4096次独立 `malloc` 的分配器元数据/碎片，也不含 prompt/output token buffers；沿用此 C 分配方式建议预留约128–160 KiB。ASIC 可用连续字符串池+offset/ID 表减少指针、allocator 开销和碎片。
- 若 ARM/外部主机负责 SentencePiece，仅向芯片发送 token ID，则 ASIC 无需存完整 tokenizer；512 个 token 的16-bit token buffer最多1 KiB，也可通过 FIFO 流式送入。若芯片必须从 UART 接收 UTF-8 文本并自行编码，则需预留 tokenizer 表和字符串输入缓冲。
- 原始 checkpoint 用 raw prompt：`[BOS] + SentencePiece(user_text)`。
- 结构化微调 checkpoint 的 prompt：先构造 `故事开头：{opening}\n续写：`，再做 SentencePiece 编码并在前面加 BOS。
- 建议硬上限 `prompt_token_count + max_new_tokens <= 512`。实际 EOS 结束时输出 token 数可以更短。
- 试验过的一套可选 sampler 是 temperature=0.65、top-p=0.9、重复 token penalty=1.15、最短32 token、EOS/BOS logit bias +2（生成96 token后启用）、禁止重复 4-gram。以上不参与前向图；进行 RTL 算子校验应绕过采样，直接比较 logits/argmax。

## 7. 权重文件格式

### Python checkpoint

`training/checkpoints/` 中的 `.pt` 是 PyTorch training checkpoint（含模型、
优化器和训练 metadata），不是硬件可直接读取的平面数组。网络只需要
`checkpoint["model"]` 和 `checkpoint["model_args"]`。读取时使用
`weights_only=True`；本项目的 Q6W exporter 按 manifest 生成 packed `.bin`。

### Packed Q6W 部署文件

本项目实际部署文件是 `quantization/artifacts/` 中的 Q6W `.bin`，而不是
`stories1.4M-zh.bin` 这类 llama2.c FP32 平铺文件。Q6W 文件含 signed-int6
矩阵、FP32 行 scale 和 FP32 RMSNorm 参数；结构、tensor 顺序、对齐、偏移和
哈希见[格式说明](Q6_QUANTIZATION.md)。

在 host 侧准备 autonomous PL 包时，Q6 的 signed-int6 码值按 manifest 展开为
DDR 中的 int8 模型映像；不要把 FP32 `.pt`、Q6W `.bin` 和 PL 的运行时映像当成
相同的序列化格式。

## 8. RTL 建议接口与验证顺序

当前工程已有完整的 PYNQ-Z2 autonomous PL 推理；以下接口和分级验证顺序
适用于新算术/数据通路修改，便于定位数值偏差：

建议将 tokenizer、随机 sampler 与神经网络分开，token-step 接口可采用：

```text
输入：start/reset_cache, token_id[11:0], position[8:0], 权重基址/配置
输出：done, logits[4096] FP32（可用流式输出）
内部状态：两组 K/V cache，每层按 position 写入
```

推荐验证顺序：embedding 查表；RMSNorm；独立 Linear；RoPE；单 head attention；GQA 头映射；SwiGLU；单 block；5 层 logits；最后测试多 token cache 序列。每一级对同一 token、权重、位置和 cache 前缀比较 Python 中间值，记录 max absolute error、max relative error、top-1 token 是否一致。先用 FP32 累加建立数值正确的基线，再评估定点/低精度。不要用生成故事的主观好坏替代算子级数值验证。

Python benchmark 的 `verify_reference()` 可将缓存 step 的最后提示 logits 与非缓存
`Transformer.forward()` 对照。使用随仓库提供的训练模型代码可运行：

```powershell
python benchmarks/torch_benchmark.py --checkpoint training/checkpoints/q6w_qat_lr1e6_step25000.pt --tokenizer tok4096.model --reference-repo training --prompt "从前，有一只小猫。" --devices cpu --cpu-threads 1 --new-tokens 1 --runs 1 --temperature 0
```

此前本机 FP32 最大绝对误差 CPU 约 `1.72e-5`、CUDA 约 `9.78e-6`。这可作为两个 Python 实现相互校验的参考，但不代替 RTL 对逐层结果的校验。

本项目在 PYNQ-Z2 上由 ARM 负责文本/Tokenizer、请求准备和结果解码，PL 自主执行
embedding 至 logits/argmax 的完整 autoregressive 推理。已签核 overlay 的资源、
时钟和板测结果见[项目 README](../README.md)。修改算术或数据通路时，仍应按前述
顺序逐级比较 Python 参考输出，再做完整 token 流校验。

现成的 Q6W 权重导出采用逐输出行 scale，布局与解码公式、实验误差和二进制偏移表见 [Q6W 量化说明](Q6_QUANTIZATION.md)。该工件只压缩权重；不能据此推断 activation、MAC 累加器或 softmax 已经量化。
