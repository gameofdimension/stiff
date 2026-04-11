# FlexAttentionHOP 分派机制与 Inductor 降级到 Triton 模板的完整链路

> 本文档是对 `stiff_compile_pathway.md` 的补充与纠正，所有结论均通过逐行阅读源码验证。
> 源码版本基于项目 `.venv` 中的 PyTorch（`torch/_ops.py`, `torch/_inductor/select_algorithm.py`）。

---

## 一、HigherOrderOperator 的分派机制

### 1.1 FlexAttentionHOP 的定义

`stiff/_higher_order_ops/flex_attention.py:92-123`:

```python
class FlexAttentionHOP(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("flex_attention", cacheable=True)

    def __call__(self, query, key, value, score_mod, block_mask, scale, kernel_options, ...):
        validate_subgraph_args_types(...)
        return super().__call__(query, key, value, score_mod, block_mask, scale, kernel_options, ...)
```

`__call__` 最终调用 `HigherOrderOperator.__call__`（`torch/_ops.py:525`），这是分派的入口。

### 1.2 分派入口：`HigherOrderOperator.__call__`

`torch/_ops.py:525-533`:

```python
def __call__(self, /, *args, **kwargs):
    flat_args = _to_flat_tuple(args, kwargs)
    if torch.overrides.has_torch_function(flat_args):
        return torch.overrides.handle_torch_function(self, flat_args, *args, **kwargs)

    dispatch_key_set = _compute_keyset(args, kwargs, self.non_fallthrough_keys)
    return self.dispatch(dispatch_key_set.highestPriorityTypeId(), *args, **kwargs)
```

两步：① 计算出当前应该激活的 DispatchKeySet，② 取优先级最高的 key 进行分发。

### 1.3 DispatchKeySet 的计算：`_compute_keyset` → `key_extractor`

`torch/_ops.py:570-589`:

```python
def _compute_keyset(args, kwargs, non_fallthrough_keys):
    tensors = _get_tensors(args, kwargs)
    return key_extractor(tensors, non_fallthrough_keys)

def key_extractor(tensors, key_mask):
    key_set = torch._C._dispatch_tls_local_include_set()    # ① TLS include set
    for tensor in tensors:
        key_set = key_set | torch._C._dispatch_keys(tensor) # ② 合并所有张量的 keys
    key_set = key_set - torch._C._dispatch_tls_local_exclude_set() # ③ 减去 TLS exclude set
    key_set = key_set & key_mask                              # ④ 与 HOP 的 non_fallthrough 取交集
    return key_set
```

四个来源决定最终的 key set：

| 来源 | 含义 | 实现位置 |
|------|------|----------|
| ① TLS include set | 线程局部变量，主动**注入** dispatch key | `torch._C._dispatch_tls_local_include_set()` |
| ② 张量自身的 keys | 每个张量根据自身属性携带的 dispatch keys | `torch._C._dispatch_keys(tensor)` |
| ③ TLS exclude set | 线程局部变量，主动**排除** dispatch key | `torch._C._dispatch_tls_local_exclude_set()` |
| ④ `non_fallthrough_keys` | HOP 自身的 mask，决定哪些 key 它要处理 | `HigherOrderOperator.__init__` 初始化 |

#### `non_fallthrough_keys` 的初始化

`torch/_ops.py:266-301`:

```python
_HIGHER_ORDER_OP_DEFAULT_FALLTHROUGH_DISPATCH_KEYS = [
    DispatchKey.PythonDispatcher,
    DispatchKey.PythonTLSSnapshot,
    DispatchKey.ADInplaceOrView,
    DispatchKey.BackendSelect,
    DispatchKey.AutocastCPU,
    DispatchKey.AutocastCUDA,
    DispatchKey.AutocastXPU,
]

class HigherOrderOperator:
    def __init__(self, name, *, cacheable=False):
        ...
        self.non_fallthrough_keys = torch._C._dispatch_keyset_full()  # 全部 key
        for dispatch_key in _HIGHER_ORDER_OP_DEFAULT_FALLTHROUGH_DISPATCH_KEYS:
            self.fallthrough(dispatch_key)  # 从 mask 中移除这些 key
```

初始化时 `non_fallthrough_keys` 包含所有 key，然后移除 fallthrough 列表中的 key。当通过 `@py_impl(DispatchKey.X)` 注册 handler 时，该 key 会被加回 mask（`torch/_ops.py:311-317`）。

对于 `flex_attention`，注册了以下 handler 的 key 会被加回 mask：
- `DispatchKey.AutocastCUDA`（第 325 行）
- `DispatchKey.AutocastCPU`（第 350 行）
- `DispatchKey.CompositeExplicitAutograd`（第 375 行）
- `ProxyTorchDispatchMode`（第 471 行）
- `DispatchKey.Autograd`（第 856 行）

### 1.4 dispatch 方法的路由逻辑

`torch/_ops.py:373-522`，核心流程：

```
dispatch(dispatch_key, *args, **kwargs)
        │
        ├─ dispatch_key == DispatchKey.Python?
        │       │
        │       ├─ Step 1: 检查 TLS 中当前活跃的 TorchDispatchMode
        │       │   curr_mode = _get_current_dispatch_mode()
        │       │   if type(curr_mode) in self.python_key_table:
        │       │       with _pop_mode_temporarily() as mode:  ← 弹出 mode
        │       │           handler(mode, *args, **kwargs)
        │       │
        │       └─ Step 2: 检查张量子类的 __torch_dispatch__
        │
        └─ 其他 dispatch_key:
                final_key = resolve_key(self, dispatch_key)
                kernel = self.py_kernels[final_key]
                return kernel(*args, **kwargs)
```

**`_pop_mode_temporarily()` 的关键作用**（`torch/_ops.py:407-416`）：当 handler 被调用时，当前 dispatch mode 会被**临时弹出** TLS。这意味着 handler 内部如果再次调用 `flex_attention_hop()`，不会再次路由回同一个 mode handler。

### 1.5 各场景的 DispatchKey 确定过程

#### 场景 A：Dynamo 追踪时（torch.compile 内部）

```
TLS include set = {ProxyTorchDispatchMode}   ← Dynamo 压入
张量 keys       = FakeTensor 相关 keys
结果: ProxyTorchDispatchMode 优先级最高

→ flex_attention_proxy_torch_dispatch_mode() (第 472 行)
  → trace_flex_attention()
```

#### 场景 B：trace_flex_attention 内部获取 example_out

`stiff/_higher_order_ops/flex_attention.py:422`:

```python
example_out = flex_attention(query, key, value, score_mod, block_mask, ...)
```

此时 **不在** `ProxyTorchDispatchMode` 下——因为 dispatcher 调用 handler 时使用了 `with _pop_mode_temporarily()`（`torch/_ops.py:413`），在 `trace_flex_attention` 执行期间，ProxyTorchDispatchMode 已从 TLS 弹出。

```
TLS include set = {}                           ← ProxyTorchDispatchMode 已被弹出
张量 keys       = {CompositeExplicitAutograd}  ← FakeTensor 的 key
结果: CompositeExplicitAutograd

→ sdpa_dense() (第 376 行) → math_attention() (第 214 行)
```

#### 场景 C：Eager 模式（无 torch.compile）

```
TLS include set = {}
张量 keys       = {Autograd, CompositeExplicitAutograd}  ← 普通 CUDA 张量
结果: Autograd 优先级最高（requires_grad=True 时）

→ flex_attention_autograd() (第 857 行)
  → torch.is_grad_enabled() and input_requires_grad?
      是 → FlexAttentionAutogradOp.apply() → 自定义 autograd
      否 → _AutoDispatchBelowAutograd() 排除 Autograd key 再 redispatch
            → CompositeExplicitAutograd → sdpa_dense() → math_attention()
```

#### 场景 D：Autocast 开启时

```
TLS include set = {}
张量 keys       = {AutocastCUDA, Autograd, ...}  ← autocast 给张量加了 key
结果: AutocastCUDA 优先级高于 Autograd

→ flex_attention_autocast_cuda() (第 326 行)
  → _flex_attention_autocast_impl()
    → cast Q/K/V 到 autocast dtype
    → _ExcludeDispatchKeyGuard(AutocastCUDA | AutocastCPU)
    → 重新调用 flex_attention() → 回到场景 B 或 C 的路径
```

---

## 二、math_attention 与 Triton 模板是两条分叉路径

### 2.1 关键结论

`sdpa_dense() → math_attention()` 的 Python 源码**不会被映射到 ninja/Triton 模板**。这两条路径在 Dynamo 追踪阶段完成后就完全分叉了。

### 2.2 math_attention 的唯一作用

在 `trace_flex_attention()` 第 422 行：

```python
example_out = flex_attention(query, key, value, score_mod, block_mask, ...)
```

这行代码走 `CompositeExplicitAutograd → sdpa_dense → math_attention()`，**目的仅仅是拿到输出张量的形状和 dtype**（即 `example_out`），用来告诉 Dynamo "这个算子产出什么形状的 tensor"。

`math_attention()` 内部的所有 Python 代码——`vmap`、`@`、`softmax`、`logsumexp`——其执行结果都被丢弃，只保留 `example_out` 的元信息。

### 2.3 两条路径的分叉

```
flex_attention_hop() 被 Dynamo 追踪
        │
        ▼
trace_flex_attention()
        │
        ├── ① flex_attention() ── CompositeExplicit ──→ sdpa_dense → math_attention()
        │       │                                              │
        │       │                                              ▼
        │       │                                         计算出 example_out
        │       │                                         （形状、dtype）
        │       │                                         执行结果全部丢弃
        │       ▼
        │   example_out = (tensor_of_shape_x, ...)
        │
        ├── ② reenter_make_fx(score_mod) → FX GraphModule
        ├── ③ reenter_make_fx(mask_mod)  → FX GraphModule
        │
        └── ④ 在 FX Graph 中插入 opaque 节点：
                call_function[flex_attention_hop](
                    q, k, v, score_graph, block_mask, ...
                )
                这个节点不展开内部实现

═══════════════════════════════════════════════════
         以下是 Inductor 阶段，与上面完全独立
═══════════════════════════════════════════════════

Dynamo 产出的 FX Graph 进入 Inductor
        │
        ├── 看到 call_function[flex_attention_hop] 节点
        │   查 dispatch table:
        │   @register_lowering(torch.ops.higher_order.flex_attention)
        │       │
        │       ▼
        │   stiff/flex/flex_attention.py:106 的 flex_attention()
        │       │
        │       ├── build_subgraph_buffer(score_graph) → Inductor IR (ComputedBuffer)
        │       ├── build_subgraph_buffer(mask_graph)  → Inductor IR (ComputedBuffer)
        │       └── flex_attention_template.maybe_append_choice(...)
        │               │
        │               ▼
        │           TritonTemplateCaller（持有 Jinja2 模板 + IR 节点）
        │
        └── autotune_select_algorithm() 选最优 config
                │
                ▼
            最终产出：Triton GPU kernel（.py → 编译为 PTX → ninja 管理）
```

---

## 三、Inductor 降级到 Triton 模板的完整链路

### 3.1 TritonTemplate 实例化（模块加载时）

`stiff/flex/flex_attention.py:97-102`:

```python
flex_attention_template = TritonTemplate(
    name="flex_attention",
    grid=flex_attention_grid,
    source=load_flex_template("flex_attention")
         + load_flex_template("utilities")
         + load_flex_template("common"),
    always_freeze_layout=True,
)
```

模块加载时，`flex_attention_template` 就绑定了 Jinja2 模板源码（`flex_attention.py.jinja` + `common.py.jinja` + `utilities.py.jinja`）。这个实例在整个 Inductor 降级过程中是**全局唯一**的。

### 3.2 @register_lowering 注册降级函数

`stiff/flex/flex_attention.py:105-116`:

```python
@register_lowering(torch.ops.higher_order.flex_attention, type_promotion_kind=None)
def flex_attention(query, key, value, subgraph, block_mask, scale, kernel_options, ...):
```

Inductor 看到 FX Graph 中的 `call_function[torch.ops.higher_order.flex_attention]` 节点时，会调用这个函数。注意参数 `subgraph` 就是 Dynamo 阶段产出的 FX GraphModule（score_graph）。

### 3.3 子图降级：FX GraphModule → Inductor IR

`stiff/flex/flex_attention.py:177-200`:

```python
placeholder_inps = [
    create_placeholder(name, dtype, query.get_device())
    for name, dtype in [("score", query.get_dtype()), ("b", torch.int32),
                         ("h", torch.int32), ("m", torch.int32), ("n", torch.int32)]
]
subgraph_buffer = build_subgraph_buffer(placeholder_inps + list(score_mod_other_buffers), subgraph)
freeze_irnodes(subgraph_buffer)

mask_graph_buffer = build_subgraph_buffer(mask_graph_placeholder_inps + list(mask_mod_other_buffers), mask_graph)
freeze_irnodes(mask_graph_buffer)
```

`build_subgraph_buffer`（定义在 `stiff/flex/common.py`）将 FX GraphModule 转换为 Inductor IR 的 ComputedBuffer 节点。对于 `_identity`，产出的 ComputedBuffer 是一个直通（直接返回输入 score）；对于 `causal_bias`，产出的 ComputedBuffer 包含 `where(m >= n, score, -inf)` 的 IR 表示。

这些 ComputedBuffer 节点就是模板中 `{{ modification(...) }}` 宏要遍历的对象——宏将每个 IR 节点翻译成等价的 Triton 指令。

### 3.4 autotune config 遍历 → maybe_append_choice

`stiff/flex/flex_attention.py:354-443`:

```python
choices: list[Any] = []
configs: list[FlexConfig] = V.choices.get_flex_attention_fwd_configs(head_dim, dtype, ...)

for conf in configs:
    cur_kernel_options = original_kernel_options.copy()
    cur_kernel_options.setdefault("BLOCK_M", conf.block_m)
    cur_kernel_options.setdefault("BLOCK_N", conf.block_n)
    cur_kernel_options.setdefault("num_warps", conf.num_warps)
    cur_kernel_options.setdefault("num_stages", conf.num_stages)
    ...

    flex_attention_template.maybe_append_choice(
        choices=choices,
        input_nodes=[query, key, value, logsumexp, max_scores,
                     kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices],
        layout=layout,
        subgraphs=[subgraph_buffer, mask_graph_buffer],  # ★ IR 节点传入模板
        mutated_inputs=[logsumexp, max_scores],
        call_sizes=query.get_size(),
        **cur_kernel_options,
    )
```

每个 config（BLOCK_M, BLOCK_N, num_warps, num_stages 的组合）都会尝试生成一个 `TritonTemplateCaller`，追加到 `choices` 列表。

### 3.5 TritonTemplate.maybe_append_choice → generate → TritonTemplateCaller

`torch/_inductor/select_algorithm.py` 中的调用链：

```
maybe_append_choice(choices, **kwargs)          # 第 1794 行
    │
    └─ self.generate(generate_with_caching=True, **kwargs)  # 第 2026 行
         │
         ├─ self.generate_and_load(...)         # 第 1820 行
         │    │
         │    ├─ kernel = self.kernel_type(     # 创建 TritonTemplateKernel
         │    │       kernel_name, input_nodes, output_node,
         │    │       subgraphs=[subgraph_buffer, mask_graph_buffer],
         │    │       num_stages, num_warps, meta=kwargs, ...)
         │    │
         │    ├─ kernel.render(self.template, kwargs)  # ★ Jinja2 渲染
         │    │   self.template = 初始化时加载的 Jinja2 模板字符串
         │    │   kwargs 包含 BLOCK_M, BLOCK_N, subgraphs, ...
         │    │   模板中 {{ modification(subgraph_number=0, ...) }}
         │    │   遍历 subgraphs[0]（即 subgraph_buffer）的 ComputedBuffer IR 节点
         │    │   将每个节点翻译为 Triton 代码
         │    │   → 产出完整 Triton Python 源码
         │    │
         │    └─ PyCodeCache.load(code)         # 缓存编译
         │
         ├─ make_kernel_render = functools.partial(kernel.render, self.template, kwargs)
         │
         └─ return TritonTemplateCaller(        # 第 2213 行
                  kernel_hash_name,
                  codegen_input_nodes,
                  layout,
                  make_kernel_render,            # 延迟渲染闭包
                  bmreq,                        # benchmark 请求
              )
```

`TritonTemplateCaller`（`torch/_inductor/select_algorithm.py:2342`）持有一个 `make_kernel_render` 闭包。当 Inductor scheduler 最终决定使用这个 choice 时，闭包会将 Triton 代码渲染到实际的 output node 上。

### 3.6 autotune_select_algorithm：从 choices 中选最优

`stiff/flex/flex_attention.py:464-472`:

```python
out, _ = autotune_select_algorithm(
    "flex_attention",
    choices,  # TritonTemplateCaller 列表
    [x for x in inputs_for_autotuning if isinstance(x, torch._inductor.ir.IRNode)],
    layout,
    input_gen_fns=input_gen_fns,
)
```

`autotune_select_algorithm`（`torch/_inductor/select_algorithm.py:4488`）委托给 `AlgorithmSelectorCache.__call__`（第 2874 行）：

- 只有 1 个 choice → 直接返回
- `config.deterministic = True` → 选确定性 choice
- 多个 choices → 在 GPU 上实际 benchmark 每个 kernel，选最快的

### 3.7 最终产出物

```
┌───────────────────────────────────────────────────────────┐
│  Inductor 降级产出：TritonTemplateCaller 绑定到 IR 图     │
│                                                           │
│  scheduler 最终调用 make_kernel_render() 时：              │
│  1. kernel.render(jinja_template, kwargs)                  │
│  2. 模板中 {{ modification(subgraph_number=0, ...) }}     │
│     遍历 subgraph_buffer 的 ComputedBuffer IR 节点        │
│     生成对应的 Triton 代码（tl.where, tl.dot, etc.）      │
│  3. 模板中 {{ modification(subgraph_number=1, ...) }}     │
│     遍历 mask_graph_buffer 的 ComputedBuffer IR 节点      │
│  4. 产出完整 .py 文件 → Triton 编译为 PTX → ninja 管理   │
│                                                           │
│  ★ 整个过程中 math_attention() 的 Python 源码完全不参与   │
└───────────────────────────────────────────────────────────┘
```

---

## 四、关键源码文件索引

| 文件 | 行号 | 作用 |
|------|------|------|
| `torch/_ops.py` | 277 | `HigherOrderOperator` 基类定义 |
| `torch/_ops.py` | 266-274 | HOP 默认 fallthrough dispatch keys |
| `torch/_ops.py` | 311-317 | `py_impl` 注册时将 key 加入 non_fallthrough mask |
| `torch/_ops.py` | 373-522 | `dispatch` 方法：Python key 和普通 key 的路由 |
| `torch/_ops.py` | 413 | `_pop_mode_temporarily()`：临时弹出当前 mode |
| `torch/_ops.py` | 525-533 | `HigherOrderOperator.__call__`：计算 keyset 并分发 |
| `torch/_ops.py` | 570-589 | `_compute_keyset` / `key_extractor`：DispatchKeySet 计算 |
| `stiff/_higher_order_ops/flex_attention.py` | 92-123 | `FlexAttentionHOP` 定义 |
| `stiff/_higher_order_ops/flex_attention.py` | 214-283 | `math_attention`：eager 参考实现 |
| `stiff/_higher_order_ops/flex_attention.py` | 325-372 | Autocast handler（CUDA/CPU） |
| `stiff/_higher_order_ops/flex_attention.py` | 375-399 | `sdpa_dense`：CompositeExplicitAutograd handler |
| `stiff/_higher_order_ops/flex_attention.py` | 402-468 | `trace_flex_attention`：Dynamo 追踪 |
| `stiff/_higher_order_ops/flex_attention.py` | 471-497 | ProxyTorchDispatchMode handler |
| `stiff/_higher_order_ops/flex_attention.py` | 856- | Autograd handler |
| `stiff/flex/flex_attention.py` | 97-102 | `flex_attention_template` TritonTemplate 实例化 |
| `stiff/flex/flex_attention.py` | 105-116 | `@register_lowering` 注册 Inductor 降级 |
| `stiff/flex/flex_attention.py` | 177-200 | `build_subgraph_buffer` 调用 |
| `stiff/flex/flex_attention.py` | 354-443 | autotune config 遍历 + `maybe_append_choice` |
| `stiff/flex/flex_attention.py` | 464-472 | `autotune_select_algorithm` 调用 |
| `torch/_inductor/select_algorithm.py` | 1753 | `TritonTemplate` 类定义 |
| `torch/_inductor/select_algorithm.py` | 1794-1817 | `TritonTemplate.maybe_append_choice` |
| `torch/_inductor/select_algorithm.py` | 1820-2024 | `TritonTemplate.generate_and_load`（Jinja2 渲染） |
| `torch/_inductor/select_algorithm.py` | 2026-2240 | `TritonTemplate.generate`（创建 TritonTemplateCaller） |
| `torch/_inductor/select_algorithm.py` | 2342 | `TritonTemplateCaller` 类定义 |
| `torch/_inductor/select_algorithm.py` | 2803 | `AlgorithmSelectorCache` 类定义 |
| `torch/_inductor/select_algorithm.py` | 2874- | `AlgorithmSelectorCache.__call__`（autotune 选择） |
| `torch/_inductor/select_algorithm.py` | 4488-4500 | `autotune_select_algorithm` 入口函数 |
