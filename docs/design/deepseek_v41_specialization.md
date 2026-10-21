# DeepSeek-V4.1-Flash 专用版

本分支已将模型注册入口收紧到 DeepSeek-V4.1，并实际移除不在保守依赖闭包内的模型 Python 源码。目标部署为 NVIDIA B300，当前交付状态为**待 GPU 验收的开发版**。尚未验证模型权重加载、生成质量、CUDA Graph、分布式运行或性能。

基线是 vLLM 提交 `90ba2f34b5a29d85b8301672e5735676f78fa00c`；参考 DeepInfer 的单模型产品范围，但没有复制其引擎或替换 vLLM 内核。本次保留 vLLM 已有的 V4.1 模型实现及在线服务基础设施。

## 已完成的改造

| 项目 | 专用版行为 |
| --- | --- |
| 对外模型架构 | 仅 `DeepseekV41ForCausalLM` |
| 内部草稿架构 | 仅 `DSparkV41DraftModel`，用于模型自带 DSpark |
| 默认权重 | `deepseek-ai/DeepSeek-V4.1-Flash-1017` |
| 本地权重和量化版本 | 允许相同目标架构的兼容权重目录；兼容性仍须实际加载验证 |
| 不支持的架构 | 在模型类解析或检查时明确报错，不执行通用回退 |
| 架构列表 | 包含任何未支持架构即拒绝；不进行名称后缀自动转换 |
| 替代模型实现 | 拒绝 `model_impl=transformers/terratorch` |
| 模型扩展 | 固定只读注册表；拒绝第三方注册和同名覆盖 |
| 物理裁剪 | 两轮累计删除 402 个模型 Python 文件；另删除 16 个非目标预热和投机解码模块 |

主模型和草稿模型的两个架构名不代表支持两个独立产品。DSpark 是 V4.1 的内部投机解码路径；禁止它会破坏现有配置转换与加载过程。注册表限制的是标准模型加载入口，不验证权重来源或证明任意自定义 Python 插件的行为。

默认模型通过 `ModelConfig` 传播到引擎参数，`vllm serve` 的帮助文本也已同步。Tokenizer 自动选择仅保留 V4.1 分支，草稿继承目标 tokenizer 的行为保留。

## 裁剪规模与保留边界

统计范围仅为 `vllm/model_executor/models/` 和 `vllm/models/` 下的 Git 跟踪 Python 文件。下表采用入口改造完成、第一轮物理删除执行前的源码作为分母，不代表整个仓库或容器镜像。第一轮提交为 `c7623a9bd7`。

| 指标 | 第一轮前 | 第一轮后 | 第二轮后 |
| --- | ---: | ---: | ---: |
| 模型源码文件数 | 630 | 302 | 228 |
| 模型源码字节数 | 12,659,514 | 5,563,708 | 4,116,751 |

累计文件数减少约 63.8%，源码字节减少约 67.5%。第二轮删除的 74 个模型文件合计 1,444,926 字节，另精简了保留的 V4 包入口。注册表由基线的 1,577 行缩减到 548 行，净减少 1,029 行；上表的第一轮前字节已经包含精简后的注册表，不重复计入。

裁剪以其他所有 vLLM Python 模块为根，跟踪普通导入、相对导入、父包初始化、延迟导入及可解析的模块字符串。因此仍然会看见 DeepSeek-V4、Qwen、Llama 等名字。例如：

- V4.1 直接复用 DeepSeek-V4 的 MoE、视觉和算子。V4 父包现仅保留量化配置导出，不再连带初始化其完整模型、MTP 和 DSpark。
- V4.1 DSpark 使用 `qwen3_dspark` 中的共享头部，并通过 `qwen3_dflash` 的辅助逻辑判断非因果注意力；这些文件仍需保留。
- Transformers 配置、processor、tokenizer 存在惰性注册和共享依赖。

这些是保留的实现依赖，标准注册入口不允许将它们作为其他模型提供服务。

本次还保留调度、连续批处理、KV 管理、请求取消、流式输出、视觉输入、工具调用与推理解析、现有并行机制以及 V4.1 内核。通用配置选项、其他平台分支、测试矩阵、文档和 CUDA 构建目标尚未全面缩减；保留源码不等于验证所有功能组合可用。依赖清单及第三方二进制也没有缩减，所以这不是最终的最小镜像。

## 第二轮：清理其他模型的条件分支

第一轮只限制注册入口并裁剪无引用模型文件；第二轮进一步移除了让这些文件继续被引用的非目标运行路径。

| 位置 | 本轮清理 | 保留的 V4.1 行为 |
| --- | --- | --- |
| `config/speculative.py` | 净删约 849 行其他模型 MTP 转换、Qwen3-Omni 验证、Qwen/Gemma/K3 草稿选择及 Medusa/MLP 补丁 | 草稿配置转换、量化继承、TP/长度限制、DSpark 宽度与无模型投机方法 |
| `config/model.py` | 删除 Terratorch、Kimi、V3.2、V4、Inkling 的默认 tokenizer 判断 | 主模型 `auto` 选择 V4.1 tokenizer |
| `model_executor/warmup/` | 删除 Qwen、QwenVL、Mamba、Kimi、Qwen4、MiniMax、ReplaySSM 的 7 个模块及调用 | JIT、CuTeDSL、DeepGEMM、Sparse MLA、router、GEMM-RS/AR、watermark 和硬件预热 |
| `v1/spec_decode/` 与 `v1/worker/gpu/spec_decode/` | 删除 Gemma4、Step3、Medusa、LiLiCorr、DFlash2 专属分派和 9 个模块 | V4.1 DSpark 分派、显式草稿后端优先、目标 KV 后端继承及共享辅助逻辑 |
| `models/deepseek_v4/__init__.py` | 删除完整 V4 模型的平台分派与导出 | 量化类导出、V4.1 对共享子模块的直接导入 |

`ReplaySSM` 是非目标状态空间模型的路径；V4.1 的 decoder replay 文件和控制逻辑没有删除。V4.1 的 ViT 也没有继承 Qwen3-VL 的视觉塔，因此删除 QwenVL 预热不会删掉 V4.1 视觉实现。

一个必须保留的名字是 `deepseek_mtp`：草稿配置先借此计算阶段数，并保持主模型架构通过第一次验证，之后才改回 `deepseek_v41` 并选择 `DSparkV41DraftModel`。不能把这个内部过渡状态与不支持的经典 MTP 混为一谈。

本轮删除了 10 个只测试已删除路径的旧测试文件，将其中 2 个共享 DFlash 辅助测试保留到 `tests/v1/spec_decode/test_dflash_model_utils.py`。同步移除了专属 Gemma4 CI 任务及失效测试引用，保留其余通用测试。

仍可继续分析的区域如下；本轮没有将它们视为可以直接删除的代码：

- `transformers_utils/config.py`、`configs/`、`model_arch_config_convertor.py`：仍有多模型配置注册和转换器。需用真实 HF 配置验证视觉子配置、草稿临时别名和错误处理后，再收窄。
- `qwen3_dspark`、`qwen3_dflash`：可进一步提取 V4.1 所需共享头部与注意力辅助逻辑，避免因导入整文件保留其他模型实现。
- 量化注册及相关分支：必须先确定支持哪些 FP8、MXFP4、NVFP4 权重与布局，不能由“单模型”直接推导“单量化实现”。
- 其他平台和 CUDA 构建目标：需要构建、打包与 B300 实测验证后单独裁剪；仅删除 Python 分支不能证明二进制依赖可移除。

## 可复查的裁剪工具

工具只依赖 Python 标准库与 Git，不导入 Torch 或模型。默认仅审计：

```bash
.venv/bin/python tools/prune_models.py --check
```

若以后需要根据新的改动继续裁剪，应先生成并审阅新计划：

```bash
.venv/bin/python tools/prune_models.py --plan /tmp/v41-prune.json
.venv/bin/python tools/prune_models.py --apply /tmp/v41-prune.json
.venv/bin/python tools/prune_models.py --check
```

计划记录全部参与分析的源码哈希、保留原因、删除清单与无法静态解析的加载点。应用时重新核对完整计划，拒绝过期计划、符号链接及已有本地修改的删除候选。只删除两处模型目录下的 Git 跟踪 Python 文件；未跟踪文件和非 Python 文件保留。未知模型内部动态导入会阻止删除。

首次审计报告了 44 个动态加载点，其中模型目录内仅有由固定字面量映射支撑的注册表加载器；其他是插件等通用运行时加载器。它们需要人工检查，静态分析不保证任意外部插件可运行。`--check` 通过只表示当前源码的这些依赖检查通过，不等同于启动测试。

## 当前验证

本地独立运行路径测试包括注册表 36 项、草稿配置 17 项、预热 9 项、分派/DSpark 后端 16 项和共享包导入 1 项，合计 79 项；裁剪工具另有 14 项测试，全部通过。删除后的闭包检查无缺失依赖，2,187 个运行时 Python 文件的语法解析全部通过。执行环境为 macOS ARM64，没有安装本项目的完整 GPU 运行依赖。

独立测试已接入 `.buildkite/test_areas/deepseek_v41.yaml` 的 CPU 任务，本轮远程 CI 尚未运行。检查使用仓库固定版本的 Ruff、格式及类型检查；此前 `actionlint` 的 Go 依赖下载因 `proxy.golang.org` 网络超时失败，本轮也仅跳过未涉及的 GitHub Actions 检查。

独立测试运行生产模块源码，通过依赖替身隔离 Torch/CUDA、HF 及 Pydantic；验证注册入口、DSpark 配置转换、预热顺序和进程边界、后端继承及导入副作用。它们不替代真实 Pydantic 字段校验或 HF 元数据加载。裁剪测试使用临时 Git 仓库，验证依赖保留、实际删除、用户修改保护、计划失效及动态加载边界。

```bash
.venv/bin/python -m pytest --confcutdir=tests/standalone_tests \
  tests/standalone_tests/test_deepseek_v41_*.py -q
.venv/bin/python -m pytest --confcutdir=tests/tools \
  tests/tools/test_prune_models.py -q
```

这两组测试可以在没有 GPU 的机器上运行。它们不覆盖真正的模型 import、元数据子进程、权重加载或数值正确性。GPU 环境下还需运行已收紧的 `tests/models/test_registry.py` 及 V4.1 的 tokenizer、parser、模型测试。完整上游测试矩阵包含已删除模型，不再适合作为本分支的全量验收矩阵。

## B300 安装与启动验收

以下在 Linux B300 服务器、本分支检出目录中执行。主机上需已有与本基线相容的 NVIDIA 驱动和 CUDA 运行环境。当前只知道 GPU 型号，尚未确认卡数、每机卡数及互联，因此不能给出经过容量验证的 TP/EP 配置。

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements/lint.txt
pre-commit install
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

这是从本分支安装；直接安装发布版 `vllm` 不会包含裁剪结果。预编译扩展必须与本基线兼容；若没有对应产物，应按仓库源码构建说明编译。当前未验证该安装流程在 B300 的结果。

使用已有目标权重目录与适合实际机器的 TP 数；环境变量不提供臆测的容量默认值：

```bash
: "${MODEL_DIR:?请设为 DeepSeek-V4.1-Flash 权重目录}"
: "${TP:?请按实际 GPU 数与容量设置张量并行数}"
.venv/bin/vllm serve "$MODEL_DIR" \
  --tensor-parallel-size "$TP" \
  --max-model-len 8192 \
  --served-model-name deepseek-v41-flash
```

先验收无投机模式：真实类导入、权重加载、预热、CUDA Graph 捕获、文本及视觉各至少一个请求，并覆盖流式输出与请求取消。再单独启用 DSpark：在同一命令增加 `--speculative-config '{"method":"dspark","num_speculative_tokens":5}'`。长上下文、跨节点及其他并行组合需分别验收。

仓库已有 NVFP4、TP4、DSpark 的 GSM8K 评估配置，本分支增加了只包含该配置的列表：

```bash
uv pip install -r requirements/test/cuda.in
.venv/bin/python -m pytest tests/models/test_registry.py -v
.venv/bin/python -m pytest -s -v \
  tests/evals/gsm8k/test_gsm8k_correctness.py \
  --config-list-file=configs/models-deepseek-v41.txt
```

该评估使用 `nvidia/DeepSeek-V4.1-Flash-NVFP4`，与默认原始权重不同；它要求满足 TP4 部署条件。其精度阈值、容差与 DSpark 接受长度要求沿用现有配置，不是本次测量结果。ARM64 测试环境应按仓库要求选用相应依赖。原始权重、无投机及视觉能力还需各自完成评估，不能用 NVFP4 的一个测试替代。

## 性能评估方法

目前没有可报告的 B300 性能提升数值。本轮减少了其他模型的导入、无关预热调用和条件分支，但没有更换 V4.1 CUDA 算子、权重精度、调度算法或通信实现，不能从源码删除比例推导吞吐提升、延迟降低或显存节省。冷启动耗时是否改善也需实测；已明确的收益是代码范围、支持边界和后续维护成本。

在同一 B300 集群分别安装基线和本分支，固定模型 revision、量化、驱动、依赖、GPU 拓扑、并行配置及环境变量；两边显式传入相同权重，避免默认模型变化造成错误比较。分别记录安装目录/镜像大小、启动到健康检查的耗时、CPU RSS、GPU 峰值显存，以及稳态吞吐、TTFT、TPOT、P95/P99 延迟、错误率和取消请求后的资源回收。

服务启动后可用同一负载命令收集原始结果，例如：

```bash
.venv/bin/vllm bench serve \
  --model deepseek-v41-flash \
  --tokenizer "$MODEL_DIR" \
  --dataset-name random \
  --random-input-len 1024 --random-output-len 256 \
  --num-prompts 200 --max-concurrency 16 \
  --request-rate inf \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,95,99 --save-result
```

这是短输入、闭环并发的示例，不是已验证的推荐负载。另测长输入、小并发和真实请求到达率；用真实提示词单独测 DSpark 接受长度与生成质量。两边先预热，每组交替运行多次，报告重复运行的波动及满足延迟目标时的吞吐，不能仅比较峰值 tokens/s。若差异处于运行噪声内，应报告性能持平。

上线前的最低验收是：目标模型文本与视觉结果通过、DSpark 精度及接受率通过、长时间服务稳定、相同负载没有性能回退。当前完成的是源码专有化与裁剪，以上 GPU 验收仍待执行。
