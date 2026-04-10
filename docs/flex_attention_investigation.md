# Flex Attention 调查报告：调用入口、数据流、模块结构与最小自包含源码集合

## 一、整体架构 (4 层)

Flex Attention 采用 **四层架构**，从用户 API 到硬件内核层层递进：

```
Layer 1: User API          → torch/nn/attention/flex_attention.py
Layer 2: Higher-Order Op   → torch/_higher_order_ops/flex_attention.py
Layer 3: Inductor Lowering → torch/_inductor/kernel/flex/
Layer 4: Kernel Codegen    → torch/_inductor/kernel/flex/templates/*.jinja
```

**关键设计特点**: Flex Attention 完全不依赖 `aten/src/ATen/native/` 或
`native_functions.yaml`。它是纯 Python + Triton + Jinja2 代码生成的实现，
没有传统的 C++/CUDA dispatch 层。

---

## 二、调用入口

### 主入口函数

`torch.nn.attention.flex_attention.flex_attention()`

```python
def flex_attention(
    query: Tensor,                              # (B, Hq, L, E)
    key: Tensor,                                # (B, Hkv, S, E)
    value: Tensor,                              # (B, Hkv, S, Ev)
    score_mod: _score_mod_signature | None = None,
    block_mask: BlockMask | None = None,
    scale: float | None = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    kernel_options: FlexKernelOptions | None = None,
    *,
    return_aux: AuxRequest | None = None,
) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, AuxOutput]:
```

- 输出: `(B, Hq, L, Ev)` 的 attention 结果
- 两个核心可定制参数:
  - `score_mod`: 自定义注意力分数变换函数 `(score, b, h, q_idx, kv_idx) -> score`
  - `block_mask`: 块稀疏注意力掩码 (通过 `create_block_mask()` 创建)

### 辅助入口

```python
# 创建块稀疏掩码
def create_block_mask(
    mask_mod: _mask_mod_signature,
    B: int | None,
    H: int | None,
    Q_LEN: int,
    KV_LEN: int,
    device: DeviceLikeType | None = None,
    BLOCK_SIZE: int | tuple[int, int] = _DEFAULT_SPARSE_BLOCK_SIZE,
    _compile=False,
) -> BlockMask:

# 创建稠密掩码 (用于调试)
def create_mask(mod_fn, ...)
```

### 公开符号 (`__all__`)

```python
__all__ = [
    "BlockMask",
    "flex_attention",
    "AuxOutput",
    "AuxRequest",
    "FlexKernelOptions",
    "create_block_mask",
    "create_mask",
    "or_masks",
    "and_masks",
    "noop_mask",
]
```

---

## 三、数据流 (从调用到执行)

```
用户调用 flex_attention(q, k, v, score_mod, block_mask)
    │
    ▼
[Layer 1] torch/nn/attention/flex_attention.py
    ├── 输入验证、layout 强制 (BHSD 格式)
    ├── 处理 score_mod / mask_mod
    ├── 通过 _vmap_for_bhqkv() 向量化 mod 函数
    └── 调用 flex_attention HOP (higher-order op)
    │
    ▼
[Layer 2] torch/_higher_order_ops/flex_attention.py
    ├── FlexAttentionHOP.__call__() — forward 算子
    ├── FlexAttentionBackwardHOP.__call__() — backward 算子
    ├── 不同 dispatch mode:
    │   ├── Eager 模式 → math_attention() (Python 参考实现)
    │   ├── FakeTensor 模式 → 形状推断
    │   ├── Autograd 模式 → FlexAttentionAutogradOp (前/反向绑定)
    │   └── torch.compile 模式 → trace_flex_attention()
    └── 在 compile 模式下，score_mod 被 trace 为 FX 子图
    │
    ▼
[Layer 3] torch/_inductor/kernel/flex/
    ├── flex_attention.py: @register_lowering 注册降级
    │   └── @register_lowering(torch.ops.higher_order.flex_attention)
    ├── 后端选择逻辑:
    │   ├── CPU 设备 → flex_cpu.py: lower_cpu() (需 AVX2)
    │   ├── TRITON_DECODE / AUTO+短序列 → flex_decoding.py: create_flex_decoding_kernel()
    │   ├── FLASH / AUTO+简单模式 → flex_flash_attention.py: create_flex_flash_attention_kernel()
    │   └── 默认 → flex_attention.py: Triton 模板实现
    ├── common.py: 共享工具 (子图构建、stride 推断等)
    └── select_algorithm.py: 自动调优参数
    │
    ▼
[Layer 4] templates/*.jinja → 生成 Triton/C++ 内核代码
    ├── flex_attention.py.jinja (前向 Triton 内核, 224 行)
    ├── flex_backwards.py.jinja (反向 Triton 内核, 751 行)
    ├── flex_decode.py.jinja (解码内核, 254 行)
    ├── flash_attention.py.jinja (Flash Attention 内核, 85 行)
    ├── flash_attention_backward.py.jinja (Flash 反向内核, 104 行)
    ├── common.py.jinja (共享工具, 205 行)
    ├── utilities.py.jinja (工具函数, 59 行)
    └── cpp_flex_attention_template.py (CPU C++ 内核, 1089 行)
```

---

## 四、模块结构与关键文件

### 核心文件 (最小自包含集合)

#### Layer 1: 用户 API (2 个文件)

| # | 文件路径 | 行数 | 角色 |
|---|---------|------|------|
| 1 | `torch/nn/attention/flex_attention.py` | 1884 | **用户 API**: flex_attention(), BlockMask, create_block_mask(), AuxRequest, AuxOutput, FlexKernelOptions |
| 2 | `torch/nn/attention/_utils.py` | 61 | 输入验证 (`_validate_sdpa_input`)、scale 计算 (`_calculate_scale`) |

#### Layer 2: Higher-Order Op (3 个文件)

| # | 文件路径 | 行数 | 角色 |
|---|---------|------|------|
| 3 | `torch/_higher_order_ops/flex_attention.py` | 1464 | **HOP 定义**: FlexAttentionHOP, FlexAttentionBackwardHOP, math_attention(), trace_flex_attention(), FlexAttentionAutogradOp |
| 4 | `torch/_higher_order_ops/utils.py` | 1370 | HOP 共享工具: setup_compilation_env(), autograd_not_implemented() 等 |
| 5 | `torch/_higher_order_ops/base_hop.py` | 281 | BaseHOP 子类 (继承自 `torch/_ops.py` 中定义的 HigherOrderOperator) |

#### Layer 3: Inductor Lowering (6 个文件)

| # | 文件路径 | 行数 | 角色 |
|---|---------|------|------|
| 6 | `torch/_inductor/kernel/flex/__init__.py` | 3 | 触发 lowering 注册 (import 副作用) |
| 7 | `torch/_inductor/kernel/flex/flex_attention.py` | 1090 | **Triton lowering**: @register_lowering, 后端选择 (AUTO/TRITON/FLASH/DECODE/CPU)、内核创建 |
| 8 | `torch/_inductor/kernel/flex/flex_decoding.py` | 460 | 解码内核: _use_flex_decoding() 判断、create_flex_decoding_kernel()、split-K 优化 |
| 9 | `torch/_inductor/kernel/flex/flex_flash_attention.py` | 689 | Flash 后端: _use_flex_flash_attention()、is_trivial_score_graph()、FlexFlashConfig 自动调优 |
| 10 | `torch/_inductor/kernel/flex/flex_cpu.py` | 346 | CPU 后端: check_cpu_supported() (AVX2 检测)、lower_cpu() C++ 模板实现 |
| 11 | `torch/_inductor/kernel/flex/common.py` | 364 | 共享工具: build_subgraph_buffer(), create_indices_fake(), freeze_irnodes(), infer_dense_strides() |

#### Layer 4: 内核模板 (7 个文件)

| # | 文件路径 | 行数 | 角色 |
|---|---------|------|------|
| 12 | `torch/_inductor/kernel/flex/templates/flex_attention.py.jinja` | 224 | 前向 Triton 内核模板 |
| 13 | `torch/_inductor/kernel/flex/templates/flex_backwards.py.jinja` | 751 | 反向 Triton 内核模板 |
| 14 | `torch/_inductor/kernel/flex/templates/flex_decode.py.jinja` | 254 | 解码 Triton 内核模板 |
| 15 | `torch/_inductor/kernel/flex/templates/common.py.jinja` | 205 | 共享 Triton 工具模板 |
| 16 | `torch/_inductor/kernel/flex/templates/utilities.py.jinja` | 59 | 工具函数模板 |
| 17 | `torch/_inductor/kernel/flex/templates/flash_attention.py.jinja` | 85 | Flash 前向内核模板 |
| 18 | `torch/_inductor/kernel/flex/templates/flash_attention_backward.py.jinja` | 104 | Flash 反向内核模板 |

#### CPU 代码生成 (1 个文件)

| # | 文件路径 | 行数 | 角色 |
|---|---------|------|------|
| 19 | `torch/_inductor/codegen/cpp_flex_attention_template.py` | 1089 | CPU C++ 代码生成: CppFlexAttentionTemplate 类，包含 softmax/reduction/GEMM 生成 |

### 关键支撑文件 (间接依赖但非 flex 专用)

| 文件 | 角色 |
|------|------|
| `torch/_inductor/select_algorithm.py` | 自动调优配置: get_flex_attention_fwd_configs(), get_flex_attention_bwd_configs() |
| `torch/_inductor/lowering.py` | `register_lowering` 装饰器机制 |
| `torch/_inductor/ir.py` | IR 节点定义 |

### 测试文件

| 文件 | 行数 | 覆盖范围 |
|------|------|---------|
| `test/inductor/test_flex_attention.py` | 8603 | 主测试: 前/反向、GQA、各种 mod 函数、torch.compile 集成 |
| `test/inductor/test_flex_decoding.py` | 2308 | 解码内核测试 |
| `test/inductor/test_flex_flash.py` | 1867 | Flash 后端测试 |

---

## 五、关键数据结构

### BlockMask — 块稀疏掩码的核心表示

```python
class BlockMask:
    def __init__(
        self,
        seq_lengths: tuple[int, int],           # (Q_LEN, KV_LEN)
        kv_num_blocks: Tensor,                  # 每个 Q block 对应的 KV block 数量
        kv_indices: Tensor,                     # KV block 索引
        full_kv_num_blocks: Tensor | None,      # 完全不被掩盖的 KV blocks 数量
        full_kv_indices: Tensor | None,         # 完全不被掩盖的 KV block 索引
        q_num_blocks: Tensor | None,            # 反向: Q block 数量 (反向传播用)
        q_indices: Tensor | None,               # 反向: Q block 索引
        full_q_num_blocks: Tensor | None,       # 反向: 完全 Q blocks
        full_q_indices: Tensor | None,          # 反向: 完全 Q block 索引
        BLOCK_SIZE: tuple[int, int],            # (Q_BLOCK_SIZE, KV_BLOCK_SIZE)
        mask_mod: _mask_mod_signature,          # 掩码函数引用
    ):
```

**关键方法**:
- `from_kv_blocks()` — 从 KV block 数据创建 (classmethod)
- `shape` — 返回 `(B, H, Q_len, KV_len)`
- `sparsity()` — 计算稀疏度百分比
- `to_dense()` — 转换为稠密掩码
- `to_string()` — 可视化掩码模式
- `to(device)` — 设备迁移

### FlexKernelOptions — 内核配置

```python
class FlexKernelOptions(TypedDict, total=False):
    # 性能调优
    BLOCK_M: int                    # Query block 大小
    BLOCK_N: int                    # KV block 大小
    num_warps: int
    num_stages: int
    # 后端选择
    BACKEND: _Backend               # "AUTO" | "TRITON" | "FLASH" | "TRITON_DECODE"
    # 数值行为
    PRESCALE_QK: bool
    # 安全标志
    ROWS_GUARANTEED_SAFE: bool
    BLOCKS_ARE_CONTIGUOUS: bool
    # 硬件特定
    USE_TMA: bool                   # Hopper TMA
    kpack: int                      # ROCm
    matrix_instr_nonkdim: int       # ROCm
```

### AuxRequest / AuxOutput — 辅助输出

```python
class AuxRequest(NamedTuple):
    lse: bool = False               # 请求 log-sum-exp
    max_scores: bool = False        # 请求最大分数

class AuxOutput(NamedTuple):
    lse: Tensor | None = None
    max_scores: Tensor | None = None
```

---

## 六、Layer 2 详细: Higher-Order Op

`torch/_higher_order_ops/flex_attention.py` 中定义了以下关键组件:

### HOP 类定义

> **注意**: `HigherOrderOperator` 基类定义在 `torch/_ops.py:277`，通过 `from torch._ops import HigherOrderOperator` 导入 (flex_attention.py:22)。`base_hop.py` 中的 `BaseHOP` 是其子类，flex_attention 的 HOP 直接继承 `HigherOrderOperator`。

```python
class FlexAttentionHOP(HigherOrderOperator):       # Line 94
    """Forward pass higher-order operator"""

class FlexAttentionBackwardHOP(HigherOrderOperator): # Line 128
    """Backward pass higher-order operator"""

# 全局实例
flex_attention = FlexAttentionHOP()                  # Line 125
```

### Dispatch Mode 实现

| Mode | 函数 | 用途 |
|------|------|------|
| Eager | `math_attention()` (Line 218) | Python 参考实现，用于无编译的直接计算 |
| FakeTensor | 注册的 fake impl | 形状推断 (不执行实际计算) |
| Autograd | `FlexAttentionAutogradOp` (Line 747) | torch.autograd.Function，绑定前/反向 |
| Compile/Trace | `trace_flex_attention()` (Line 406) | score_mod 被 trace 为 FX 子图 |
| Autocast CUDA | `flex_attention_autocast_cuda()` | 混合精度自动转换 |
| Autocast CPU | `flex_attention_autocast_cpu()` | CPU 混合精度 |

### Eager 参考实现

```python
def math_attention(
    query, key, value, score_mod, block_mask,
    scale, kernel_options,
    score_mod_other_buffers=(), mask_mod_other_buffers=(),
) -> tuple[Tensor, Tensor, Tensor]:
    """纯 Python 实现的标准 attention，用于 eager 模式"""
```

---

## 七、Layer 3 详细: 后端选择逻辑

`torch/_inductor/kernel/flex/flex_attention.py` 中的后端选择:

```python
@register_lowering(torch.ops.higher_order.flex_attention, type_promotion_kind=None)
def flex_attention(query, key, value, subgraph, block_mask, scale,
                   kernel_options, score_mod_other_buffers, mask_mod_other_buffers):
    # 1. CPU 设备 → lower_cpu()
    if query.get_device().type == "cpu":
        return lower_cpu(...)

    # 2. 提取后端选项
    sanitized, backend = _sanitize_kernel_options_for_triton(kernel_options)
    # backend 默认 "AUTO"

    # 3. 解码模式: TRITON_DECODE 或 AUTO + 短序列
    use_decode = (backend == "TRITON_DECODE") or (backend == "AUTO" and can_use_decode)
    if use_decode:
        return create_flex_decoding_kernel(...)

    # 4. Flash 模式: FLASH 或 AUTO + 简单 score_mod/mask
    if _use_flex_flash_attention(..., backend=backend):
        return create_flex_flash_attention_kernel(...)

    # 5. 默认: Triton 模板
    return create_triton_kernel(...)  # 使用 flex_attention.py.jinja
```

### 后端判定条件

| 后端 | 选择条件 |
|------|---------|
| CPU | `query.get_device().type == "cpu"` 且 AVX2 可用 |
| TRITON_DECODE | 显式指定 `BACKEND="TRITON_DECODE"` 或 AUTO 模式下序列较短 |
| FLASH | 显式指定 `BACKEND="FLASH"` 或 AUTO 模式下 score_mod 是 trivial (即恒等变换) |
| TRITON | 默认后端，通用性最强 |

---

## 八、关键设计特点

### 1. 无 C++/CUDA 原生内核

Flex Attention 完全不依赖 `aten/src/ATen/native/` 或 `native_functions.yaml`。
它是纯 Python + Triton + Jinja2 代码生成的实现。

### 2. 四种后端策略

- **Triton** (默认): 适用于大多数场景，通过 Jinja2 模板生成
- **Flash Attention** (CuteDSL): 当 score_mod 和 mask 是简单模式时选用，性能最优
- **解码模式** (TRITON_DECODE): 短查询长度 (如推理时) 的 split-K 优化
- **CPU**: AVX2 C++ 模板实现，通过 `cpp_flex_attention_template.py`

### 3. 可组合的 score_mod / mask_mod

用户定义的 Python 函数通过 `torch.compile` trace 为 FX 子图，嵌入到生成的内核中:
- `score_mod(score, b, h, q_idx, kv_idx) -> modified_score`
- `mask_mod(b, h, q_idx, kv_idx) -> bool`
- 可通过 `and_masks()` / `or_masks()` 组合多个掩码

### 4. 块稀疏优化

BlockMask 将稠密掩码转换为块稀疏格式:
- **full blocks**: 完全不被掩盖，跳过 mask_mod 检查
- **partial blocks**: 部分掩盖，需要逐元素检查
- 这种区分允许对不同类型的块使用不同的处理路径以最大化性能

### 5. 编译集成

- 通过 `@torch.compile` 触发完整的编译优化管线
- score_mod/mask_mod 函数被 trace 后嵌入到生成的 Triton 内核中
- 支持自动调优 (autotuning) 选择最优内核参数

---

## 九、最小自包含源码集合总结

理解 Flex Attention 完整实现需要 **22 个核心文件**:

### Layer 1: 用户 API (2 个文件, ~1945 行)
```
torch/nn/attention/flex_attention.py          # 1884 行 — 主 API
torch/nn/attention/_utils.py                  #   61 行 — 输入验证
```

### Layer 2: Higher-Order Op (3 个文件, ~3115 行)
```
torch/_higher_order_ops/flex_attention.py     # 1464 行 — HOP 定义
torch/_higher_order_ops/utils.py              # 1370 行 — 共享工具
torch/_higher_order_ops/base_hop.py           #  281 行 — BaseHOP 子类 (HigherOrderOperator 定义于 torch/_ops.py:277)
```

### Layer 3: Inductor Lowering (6 个文件, ~2953 行)
```
torch/_inductor/kernel/flex/__init__.py       #    3 行 — 注册入口
torch/_inductor/kernel/flex/flex_attention.py # 1090 行 — Triton lowering
torch/_inductor/kernel/flex/flex_decoding.py  #  460 行 — 解码内核
torch/_inductor/kernel/flex/flex_flash_attention.py # 689 行 — Flash 后端
torch/_inductor/kernel/flex/flex_cpu.py       #  346 行 — CPU 后端
torch/_inductor/kernel/flex/common.py         #  364 行 — 共享工具
```

### Layer 4: 内核模板 (7 个文件, ~1682 行)
```
torch/_inductor/kernel/flex/templates/flex_attention.py.jinja           # 224 行
torch/_inductor/kernel/flex/templates/flex_backwards.py.jinja           # 751 行
torch/_inductor/kernel/flex/templates/flex_decode.py.jinja              # 254 行
torch/_inductor/kernel/flex/templates/common.py.jinja                   # 205 行
torch/_inductor/kernel/flex/templates/utilities.py.jinja                #  59 行
torch/_inductor/kernel/flex/templates/flash_attention.py.jinja          #  85 行
torch/_inductor/kernel/flex/templates/flash_attention_backward.py.jinja # 104 行
```

### CPU 代码生成 (1 个文件, 1089 行)
```
torch/_inductor/codegen/cpp_flex_attention_template.py  # 1089 行
```

### 测试 (3 个文件, ~12778 行)
```
test/inductor/test_flex_attention.py          # 8603 行
test/inductor/test_flex_decoding.py           # 2308 行
test/inductor/test_flex_flash.py              # 1867 行
```

### 总计

| 类别 | 文件数 | 代码行数 |
|------|--------|---------|
| 核心实现 (Layer 1-4) | 19 | ~10,784 |
| 测试 | 3 | ~12,778 |
| **总计** | **22** | **~23,562** |

---

## 十、使用示例

```python
import torch
from torch.nn.attention.flex_attention import (
    flex_attention, create_block_mask, BlockMask
)

# 基本使用
B, H, L, S, E = 2, 8, 2048, 2048, 64
q = torch.randn(B, H, L, E, device="cuda")
k = torch.randn(B, H, S, E, device="cuda")
v = torch.randn(B, H, S, E, device="cuda")
output = flex_attention(q, k, v)

# 自定义 score_mod
def relative_bias(score, b, h, q_idx, kv_idx):
    return score + (q_idx - kv_idx).float()

output = flex_attention(q, k, v, score_mod=relative_bias)

# 因果掩码
def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

block_mask = create_block_mask(causal_mask, B=B, H=H, Q_LEN=L, KV_LEN=S)
output = flex_attention(q, k, v, block_mask=block_mask)

# 配合 torch.compile 使用
compiled_fa = torch.compile(flex_attention)
output = compiled_fa(q, k, v, score_mod=relative_bias, block_mask=block_mask)

# 指定内核选项
output = flex_attention(q, k, v,
    kernel_options={"BLOCK_M": 64, "BLOCK_N": 64, "BACKEND": "TRITON"})

# 获取辅助输出
from torch.nn.attention.flex_attention import AuxRequest
output, aux = flex_attention(q, k, v, return_aux=AuxRequest(lse=True))
print(aux.lse.shape)  # (B, H, L)
```