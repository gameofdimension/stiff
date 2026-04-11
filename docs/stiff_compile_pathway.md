# STIFF Compile 路径深度调查

## 一、项目概述

STIFF 是从 PyTorch Flex Attention（论文 [arXiv:2412.05496](https://arxiv.org/abs/2412.05496)）中提取并扩展的独立研究项目。核心能力是：**让用户用纯 Python 函数自定义注意力模式，通过 `torch.compile` 编译为高性能的融合 GPU 内核**。

### 架构（四层设计）

```
Layer 1: 用户 API          → stiff/attention/flex_attention.py      (1800 行)
Layer 2: 高阶算子          → stiff/_higher_order_ops/flex_attention.py (1371 行)
Layer 3: Inductor 降级     → stiff/_inductor/  (triton / flash / cpu / decoding)
Layer 4: 内核代码生成      → stiff/_inductor/templates/*.jinja + stiff/codegen/
```

| 层级 | 关键组件 | 作用 |
|------|----------|------|
| L1 | `flex_attention()`, `BlockMask`, `create_block_mask()` | 用户调用入口 |
| L2 | `FlexAttentionHOP` | 将 `score_mod`/`mask_mod` 追踪为 FX 子图 |
| L3 | `flex_attention.py` (降级) | 后端选择：Triton / Flash / CPU / Decode |
| L4 | `flex_attention.py.jinja` 等 Jinja2 模板 | 生成 Triton / CuteDSL / C++ 内核代码 |

### Demo 入口

`gym/stiff_demo.py` 是一个性能基准测试 Demo，对比三种注意力实现：

1. `F.scaled_dot_product_attention(is_causal=True)` — PyTorch 原生 FA2
2. `F.scaled_dot_product_attention(attn_mask=mask)` — SDPA + 显式 mask
3. `torch.compile(flex_attention)` — 本项目的 Flex Attention

测试三种场景：`noop`（恒等）、`causal_bias`（score_mod 因果）、`causal_mask`（mask_mod + block_mask 因果）。

---

## 二、Eager 模式工作流

### 2.1 关键发现：`flex_attention()` 始终会内部编译

`flex_attention()` 并非一个纯粹的 eager/compile 二分结构。即使没有显式 `torch.compile()`，函数内部也会自动编译：

```
用户调用 flex_attention(q, k, v, score_mod=...)
        │
        ▼
  ┌─ is_dynamo_compiling()? ─┐
  │ 否                        │ 是
  ▼                           ▼
  内部 torch.compile()       直接调用 HOP
  （即使没显式 compile）      （被外层 compile 包裹时）
```

代码位置 `stiff/attention/flex_attention.py:1747-1799`：

```python
if torch.compiler.is_dynamo_compiling():
    # 被外层 torch.compile 包裹时，直接调用 HOP
    out, lse, max_scores = flex_attention_hop(query, key, value, score_mod, block_mask.as_tuple(), ...)
else:
    # 否则，内部自动 torch.compile
    flex_fn = torch.compile(_flex_attention_hop_wrapper, backend=backend, fullgraph=True)
    out, lse, max_scores = flex_fn(query, key, value, score_mod, block_mask.as_tuple(), ...)
```

### 2.2 唯一真正的 eager 路径

只有设置 `_FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True` 时（debug 用途），才会跳过内部编译，走纯 eager 路径。此路径通过 PyTorch Dispatch 机制进入 `math_attention()`。

### 2.3 PyTorch Dispatch 分发链

```
flex_attention_hop(q, k, v, score_mod, block_mask, ...)
        │
        ▼  PyTorch Dispatch 机制
  ┌─ DispatchKey 选择 ─────────────────────────────────────┐
  │                                                         │
  │  Autograd (requires_grad=True)                          │
  │    → FlexAttentionAutogradOp.apply()                    │
  │      → _AutoDispatchBelowAutograd() 下重分发             │
  │        → CompositeExplicitAutograd                       │
  │          → sdpa_dense() → math_attention()              │
  │                                                         │
  │  CompositeExplicitAutograd (requires_grad=False)        │
  │    → sdpa_dense() → math_attention()                    │
  │                                                         │
  │  ProxyTorchDispatchMode (Dynamo 追踪时)                 │
  │    → trace_flex_attention() ← 这是 compile 路径          │
  └─────────────────────────────────────────────────────────┘
```

### 2.4 `math_attention()` — eager 参考实现

位于 `_higher_order_ops/flex_attention.py:214-283`，纯 Python 参考实现：

```
math_attention(q, k, v, score_mod, block_mask, ...)
│
├─ 1. GQA 处理 (line 243-253)
│     value = repeat_interleave(value, G, dim=1)   # 扩展 KV heads
│     key   = repeat_interleave(key,   G, dim=1)
│
├─ 2. _math_attention_inner() (line 254)
│     │
│     ├─ scores = q_fp32 @ k_fp32.T          # 物化完整 S×S 矩阵 [B,H,S,S]
│     │
│     ├─ b = arange(B), h = arange(H)        # 创建索引张量
│     │   m = arange(S), n = arange(S)
│     │
│     ├─ score_mod = _vmap_for_bhqkv(score_mod)  # 4 层 vmap 向量化
│     │   vmap(in_dims=(None,None,None,0))  → 映射 kv_idx
│     │   vmap(in_dims=(None,None,0,None))  → 映射 q_idx
│     │   vmap(in_dims=(None,0,None,None))  → 映射 head
│     │   vmap(in_dims=(0,None,None,None))  → 映射 batch
│     │
│     ├─ mask_mod = block_mask[-1]            # 从 block_mask 元组提取
│     │   mask_mod = _vmap_for_bhqkv(mask_mod, prefix=())
│     │
│     └─ post_mod_scores = torch.where(       # 应用 mask + score_mod
│           mask_mod(b, h, m, n, ...),         #   True 位置
│           score_mod(scores, b, h, m, n, ...),#  应用 score_mod
│           -inf                              #   False → -inf
│         )
│
├─ 3. Softmax 计算 (line 267-274)
│     logsumexp   = post_mod_scores.logsumexp(dim=-1)
│     max_scores  = torch.max(post_mod_scores, dim=-1)
│     softmax_out = torch._safe_softmax(post_mod_scores, dim=-1)
│
└─ 4. 输出 (line 279-283)
      output = softmax_out @ value
      return (output, logsumexp / log(2), max_scores / log(2))
```

### 2.5 `_vmap_for_bhqkv()` 的作用

用户的 `score_mod` 签名为 `score_mod(score, batch, head, q_idx, kv_idx) -> score`，一次只处理一个标量。`_vmap_for_bhqkv` 对它应用 4 层 `torch.vmap`，使其能一次性处理整个 `[B, H, S, S]` 矩阵：

| vmap 层 | `in_dims` | 映射维度 |
|----------|-----------|----------|
| 1 | `(0, None, None, None, None)` | batch |
| 2 | `(None, 0, None, None, None)` | head |
| 3 | `(None, None, 0, None, None)` | q_idx |
| 4 | `(None, None, None, 0, None)` | kv_idx |

### 2.6 mask_mod 在 eager 模式下起作用

mask_mod 通过 `block_mask[-1]` 提取并在 `_math_attention_inner()` 中通过 `torch.where` 应用。但 public API `flex_attention()` 没有 `mask_mod` 参数，用户必须先通过 `create_block_mask(mask_mod_fn)` 包装成 `BlockMask` 对象，再通过 `block_mask=` 传入。

```python
# mask_mod 是 callable 函数
def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

# 必须先转换成 BlockMask 对象
block_mask = create_block_mask(causal_mask, B, H, S, S)

# 然后传给 flex_attention
output = flex_attention(q, k, v, block_mask=block_mask)
```

### 2.7 Eager 模式的核心限制

1. **物化完整 S×S 矩阵** — S=8192 时，fp32 分数矩阵约 64GB
2. **无内核融合** — QK^T、score_mod、softmax、V 乘法是 4 次独立 kernel launch
3. **vmap 开销** — 4 层 vmap 本身有调度开销
4. 仅用于正确性验证和调试，生产环境必须走 compile 路径

---

## 三、torch.compile 路径

### 3.1 无自定义 score_mod / block_mask 时的完整流程

以 `flex_attn(q, k, v)` 为例。

#### 阶段 1：入口预处理

`flex_attention.py:1541-1799`，默认值填充：

```
score_mod  → _identity(score, b, h, q_idx, kv_idx) → return score
block_mask → _create_empty_block_mask()
              kv_num_blocks = ones([1,1,1])   # 只有 1 个块
              kv_indices    = zeros([1,1,1,1]) # 块索引为 0
              BLOCK_SIZE    = (2^30, 2^30)    # 块大小=整个序列
              mask_mod      = noop_mask       # 始终返回 True
```

空 block_mask 的含义：整个注意力矩阵只有 1 个"稀疏块"，覆盖全部 Q 和 KV，无需跳过任何位置。

#### 阶段 2：Dynamo 追踪

`flex_attention.py:1784-1798` 内部 `torch.compile()` → Dynamo 追踪 `_flex_attention_hop_wrapper` → 触发 `ProxyTorchDispatchMode` → `trace_flex_attention()`（`_higher_order_ops/flex_attention.py:402-468`）：

```
trace_flex_attention()
│
├─ 1. 用 fake tensor 调一次 flex_attention() 获取 example_out
│     （触发 CompositeExplicitAutograd → math_attention → 正确形状输出）
│
├─ 2. reenter_make_fx(score_mod)(*example_vals)
│     将 _identity 追踪为 FX GraphModule：
│       graph():
│           %score = placeholder[target=score]
│           %batch = placeholder[target=batch]  # 未使用
│           ...
│           return %score
│
├─ 3. reenter_make_fx(mask_mod)(*mask_example_vals)
│     将 noop_mask 追踪为 FX GraphModule：
│       graph():
│           %batch = placeholder[target=batch]
│           ...
│           %ones = call_function[new_ones](%batch, size=(), dtype=torch.bool)
│           return %ones
│
├─ 4. 替换 block_mask 中的函数为 FX GraphModule：
│     block_mask = (*block_mask[:-1], mask_graph)
│
└─ 5. 注册两个 GraphModule 到 tracer.root
      qualname "sdpa_score" → score_graph
      qualname "sdpa_mask"  → mask_graph
```

**产出物**：Dynamo 生成的 FX 计算图中，`flex_attention_hop` 节点携带两个 **FX GraphModule 子图**（score_graph 和 mask_graph），以及 block_mask 元组。

#### 阶段 3：Inductor 降级

`stiff/_inductor/flex_attention.py:105-478`，`@register_lowering(torch.ops.higher_order.flex_attention)`：

```
flex_attention(query, key, value, subgraph, block_mask, ...)
│
├─ 1. 解构 block_mask 元组
│     kv_num_blocks, kv_indices, ..., mask_graph = block_mask
│
├─ 2. build_subgraph_buffer()   ← 关键步骤
│     │
│     ├─ 创建 placeholder 输入（score, b, h, m, n 各一个 TensorBox）
│     │
│     ├─ 对 score_graph 调用 PointwiseSubgraphLowering
│     │   FX GraphModule → Inductor IR (ComputedBuffer)
│     │   noop: trivial ComputedBuffer（直接返回输入 score）
│     │
│     └─ 对 mask_graph 调用 PointwiseSubgraphLowering
│         FX GraphModule → Inductor IR (ComputedBuffer)
│         noop: 返回常量 True 的 ComputedBuffer
│
├─ 3. 后端选择 → AUTO 默认走 Triton
│
├─ 4. 准备 kernel_options
│     SM_SCALE = 1/sqrt(d), GQA_SHARED_HEADS = 1
│     HAS_FULL_BLOCKS = False, BLOCK_M/BLOCK_N = autotune 配置
│
├─ 5. autotune_select_algorithm()
│     多个 (BLOCK_M, BLOCK_N, num_warps, num_stages) 配置
│     GPU 上 benchmark 选最优
│
└─ 6. 返回 (out, logsumexp, max_scores) 的 IR 节点
```

**产出物**：Inductor IR 中的 ComputedBuffer 节点，绑定 TritonTemplate 实例及所有配置。

#### 阶段 4：Triton 模板实例化（代码生成）

模板 `flex_attention.py.jinja` + `common.py.jinja` + `utilities.py.jinja` 生成 Triton Python 源码。

内核结构（三层嵌套）：

```python
# 外层：每个 (batch, head, q_block) 启动一个 GPU 线程块
@triton.jit
def kernel(Q, K, V, ..., KV_NUM_BLKS, KV_IDX, ...):
    # 1. 计算 batch/head/q 位置指针偏移
    # 2. 加载当前 Q block
    # 3. 加载稀疏块索引
    # 4. 调用 forward_inner() 迭代 KV blocks

# 中层：迭代 KV blocks
@triton.jit
def forward_inner(...):
    for start_n in range(block_n_start, block_n_end):
        acc, l_i, m_i = forward_block_mn(...)

# 内层：单个 KV block 的计算
@triton.jit
def forward_block_mn(...):
    k = tl.load(...)          # 加载 K block
    qk = tl.dot(q, k)        # QK^T
    qk *= SM_SCALE            # 缩放

    # ★ score_mod 内联位置 ★
    {{ modification(subgraph_number=0, score="qk", b="off_z",
                    h="off_h", m="m", n="n", out="qk") }}

    # ★ mask_mod 内联位置（非 full blocks 时）★
    {{ modification(subgraph_number=1, score="qk", b="off_z",
                    h="off_h", m="m", n="n") }}

    # Online softmax + 累积
    m_ij = tl.maximum(m_i, tl.max(post_mod_scores, 1))
    alpha = tl.math.exp2(m_i - m_ij)
    p = tl.math.exp2(post_mod_scores - m_ij)
    acc = acc * alpha + tl.dot(p, V)

# 返回最终结果
acc = acc / l_i
tl.store(out_ptr, acc)
```

**`modification` 宏**：遍历 `subgraph_buffer` 中的 Inductor IR 节点（ComputedBuffer），将每个 IR 节点翻译成等价的 Triton 代码。对于 `_identity`，翻译结果就是 `qk = qk`。

#### 阶段 5：最终产出物

```
┌───────────────────────────────────────────────────────┐
│  最终产出：一个 Triton GPU kernel（.py → 编译为 PTX）  │
│                                                        │
│  对于 noop 情况，内核等价于标准 Flash Attention：        │
│  - QK^T → scale → softmax → V，全程 fused 在一个内核   │
│  - score_mod 内联 = 空操作                              │
│  - mask_mod 内联 = 始终 True                           │
│  - 稀疏块迭代 = 只有 1 个块（覆盖全部序列）             │
└───────────────────────────────────────────────────────┘
```

### 3.2 自定义 score_mod / block_mask 时的差异

以 `causal_bias`（score_mod）和 `causal_mask`（block_mask）为例。

#### 各阶段对比

| 阶段 | noop | causal_bias (score_mod) | causal_mask (block_mask) |
|------|------|------------------------|------------------------|
| 入口预处理 | score_mod=`_identity`, block_mask=空 | score_mod=`causal_bias`, block_mask=空 | score_mod=`_identity`, block_mask=因果稀疏块 |
| Dynamo: reenter_make_fx | 追踪 `_identity` → trivial graph | 追踪 `causal_bias` → 含 `where/ge` 的 graph | 追踪 `noop_mask`, block_mask 带因果稀疏索引 |
| Inductor: build_subgraph_buffer | ComputedBuffer = 直通 | ComputedBuffer = `where(m>=n, score, -inf)` | 同 noop 的 score_graph + 稀疏块数据 |
| 模板: modification(0) 展开 | `qk = qk` | `qk = tl.where(m >= n, qk, -inf)` | `qk = qk` |
| 模板: modification(1) 展开 | `mask = True` | `mask = True` | `mask = m >= n` |
| 稀疏块迭代 | 1 个块覆盖全序列 | 1 个块覆盖全序列 | ~50% 的块（下三角） |

#### Dynamo 追踪阶段的 FX GraphModule 差异

`causal_bias` 被追踪为（`_higher_order_ops/flex_attention.py:439`）：

```python
# reenter_make_fx(causal_bias)(score_fake, b_fake, h_fake, m_fake, n_fake)
# 产出 FX GraphModule:
graph():
    %score = placeholder[target=score]
    %b     = placeholder[target=b]
    %h     = placeholder[target=h]
    %m     = placeholder[target=m]
    %n     = placeholder[target=n]
    %ge    = call_function[ge](%m, %n)
    %inf   = call_function[full](..., fill_value=-inf)
    %where = call_function[where](%ge, %score, %inf)
    return %where
```

而 `_identity` 的 GraphModule 只有一个 placeholder 和一个 return。

#### Triton 模板展开差异

`modification` 宏根据 ComputedBuffer 的 IR 节点生成对应 Triton 代码：

```python
# noop (_identity) 的 modification 展开：
post_mod_scores = qk    # 直接赋值，无额外计算

# causal_bias 的 modification 展开：
_cond = m >= n
post_mod_scores = tl.where(_cond, qk, float("-inf"))
```

#### block_mask 的稀疏迭代差异

```
noop block_mask:
  kv_num_blocks = [[1]]        # 每行 1 个块
  kv_indices    = [[[0]]]       # 块 0（覆盖全部序列）
  BLOCK_SIZE    = (2^30, 2^30)

causal block_mask (S=8192, BLOCK_SIZE=128):
  kv_num_blocks = [[1, 2, 3, ..., 64]]  # 第 i 行有 ceil((i+1)*128/128) 个块
  kv_indices    = [[[0,1,2,...,63], ...]] # 下三角结构
  BLOCK_SIZE    = (128, 128)
  → 约 50% 的块被跳过（上三角不计算）
```

内核中的稀疏迭代：

```python
# noop: 1 个块，循环执行若干次 BLOCK_N 切片
for start_n in range(0, block_n_end):
    forward_block_mn(...)

# causal: 只迭代下三角的块，通过 kv_indices 做间接寻址
for start_n in range(0, block_n_end):
    forward_block_mn(...)
    offset = get_offset_for_next_block(...)  # 可能跳到非连续位置
    offs_n = offs_n + offset
```

#### 最终产出物对比

```
noop:
┌────────────────────────────────────────────────────────────┐
│  Triton kernel ≈ 标准 scaled dot-product attention          │
│                                                            │
│  modification(0) → (空操作)                                │
│  modification(1) → (空操作)                                │
│  稀疏迭代 → 1 个覆盖全序列的块                              │
│                                                            │
│  本质上等价于一个写法略有不同的 Flash Attention              │
└────────────────────────────────────────────────────────────┘

causal score_mod:
┌────────────────────────────────────────────────────────────┐
│  Triton kernel = SDPA + 内联因果 mask                      │
│                                                            │
│  modification(0) → tl.where(m >= n, qk, -inf)             │
│  modification(1) → (空操作)                                │
│  稀疏迭代 → 1 个覆盖全序列的块（但分数被 -inf 置零）        │
│                                                            │
│  每个元素都计算了，但被 mask 的位置在 softmax 中权重为 0    │
└────────────────────────────────────────────────────────────┘

causal block_mask:
┌────────────────────────────────────────────────────────────┐
│  Triton kernel = blocksparse SDPA + 内联因果 mask          │
│                                                            │
│  modification(0) → (空操作)                                │
│  modification(1) → tl.where(m >= n, True, False)           │
│  稀疏迭代 → ~50% 的块（跳过上三角）                        │
│                                                            │
│  被跳过的块根本不加载 K/V、不做 QK^T，直接节省计算量       │
│  块边界处的部分 mask 由 modification(1) 精确处理            │
└────────────────────────────────────────────────────────────┘
```

---

## 四、核心设计总结

### score_mod 与 mask_mod 的区别

| | score_mod | mask_mod |
|---|---|---|
| **函数签名** | `(score, b, h, q_idx, kv_idx) → score` | `(b, h, q_idx, kv_idx) → bool` |
| **public API 参数** | `flex_attention(score_mod=fn)` 直接传入 | 不存在此参数 |
| **使用方式** | 直接传给 `flex_attention()` | 通过 `create_block_mask(mask_mod_fn)` 包装成 `BlockMask`，再通过 `block_mask=` 传入 |
| **语义** | 修改注意力分数（乘系数、加 bias、置 -inf 等） | 控制哪些位置参与计算（True 保留 / False 置 -inf） |
| **compile 中的作用** | 被追踪为 FX 子图，内联到 Triton 内核 | 同上，且同时用于生成 block sparse 结构跳过计算 |

### 编译产物的差异根源

核心差异在于 **`modification` 宏展开的 Triton 代码** 不同，以及 **稀疏迭代的数据驱动控制** 不同。这两者都源于 Dynamo 阶段 `reenter_make_fx()` 对不同函数生成的 FX GraphModule 不同，最终被 `build_subgraph_buffer()` 降级为不同的 Inductor IR，再被模板系统翻译为不同的 Triton 代码。

### 关键源码文件索引

| 文件 | 行数 | 作用 |
|------|------|------|
| `stiff/attention/flex_attention.py` | 1800 | L1: 用户 API、BlockMask、create_block_mask、_vmap_for_bhqkv |
| `stiff/_higher_order_ops/flex_attention.py` | 1371 | L2: FlexAttentionHOP、math_attention、trace_flex_attention、autograd |
| `stiff/_inductor/flex_attention.py` | 1014 | L3: Inductor 降级、后端选择、autotune |
| `stiff/_inductor/common.py` | 340 | L3: build_subgraph_buffer、子图降级工具 |
| `stiff/_inductor/templates/flex_attention.py.jinja` | 225 | L4: 前向 Triton 内核模板 |
| `stiff/_inductor/templates/common.py.jinja` | 205 | L4: 外层内核 + modification 宏调用点 |
| `stiff/_inductor/templates/utilities.py.jinja` | 60 | L4: 工具函数（load_checked_2d、稀疏块寻址等） |
| `stiff/_inductor/templates/flex_backwards.py.jinja` | 751 | L4: 反向 Triton 内核模板 |
| `stiff/_inductor/flex_flash_attention.py` | 664 | L3: Flash Attention 4 (CuteDSL) 后端 |
| `stiff/_inductor/flex_decoding.py` | 426 | L3: 短序列优化后端 |
| `stiff/_inductor/flex_cpu.py` | 314 | L3: CPU 后端 (C++ template) |
| `stiff/codegen/cutedsl/` | 多文件 | L4: CuteDSL 代码生成基础设施 |
| `stiff/codegen/cpp_flex_attention_template.py` | 1048 | L4: CPU C++ 内核代码生成 |
