# Flash Depth Mixers

Fused kernels and Hugging Face models for **depth mixers**: methods that replace the residual
connection of a transformer across its layers. The layout follows
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention).

| Model | `model_type` | Paper | Kernels |
|---|---|---|---|
| LDM | `ldm` | [Depth Memory](https://arxiv.org/abs/2610.18809) | `chunk` (CuTe DSL), `fused_recurrent` (Triton), `naive` |
| mHC | `mhc` | [mHC](https://arxiv.org/abs/2512.24880) | `fused` (TileLang), `naive` |
| AttnRes | `attnres` | [Attention Residuals](https://arxiv.org/abs/2603.15031) | `fused` (Triton), `naive` |

## Installation

```bash
uv sync --frozen
source .venv/bin/activate
```

mHC's kernels need `nvcc` on `PATH`. LDM's `chunk` kernels train with per-GPU tuned schedules
(H100, B200, RTX PRO 6000); tune other shapes with `python -m depth_mixer.ops.ldm.chunk.tune`.

## Usage

```python
import torch
import depth_mixer  # registers the models with the transformers Auto classes
from depth_mixer.models import LDMConfig
from transformers import AutoModelForCausalLM

config = LDMConfig(hidden_size=1024, num_hidden_layers=24, num_heads=16, intermediate_size=2736,
                   vocab_size=50280, ldm_mode="chunk")
model = AutoModelForCausalLM.from_config(config).cuda().to(torch.bfloat16)
```

Each fused op has a naive reference with the same signature:

```python
from depth_mixer.ops.ldm import chunk_ldm, fused_recurrent_ldm, naive_recurrent_ldm
from depth_mixer.ops.mhc import fused_mhc_pre, fused_mhc_post, naive_mhc_pre, naive_mhc_post
from depth_mixer.ops.attnres import fused_attnres, naive_attnres
```

Convert a [Depth Memory checkpoint](https://huggingface.co/collections/davidgonmar/depth-memory-6ac5045a1bd8e798d2ed1392)
(native OLMo-core) to a Hugging Face model:

```bash
python utils/convert_from_olmo_core.py --checkpoint <run>/lr1e-3 --output <dir>
```

## Tests, benchmarks, evaluation

```bash
pytest tests
python benchmarks/ops/benchmark_ldm.py
python benchmarks/benchmark_training_throughput.py --model ldm --size 400m
python -m evals.harness --model depth_mixer --model_args pretrained=<dir> --tasks lambada_openai
```

## Layout

```text
depth_mixer/ops/       kernels and naive references
depth_mixer/layers/    the depth mixers and attention as modules
depth_mixer/models/    Hugging Face configs and models
depth_mixer/modules/   shared norms, MLP, rotary embedding and losses (flash-linear-attention)
legacy/                the Depth Memory paper's OLMo-core code
```

Built on [flash-linear-attention](https://github.com/fla-org/flash-linear-attention),
[DeepSeek TileKernels](https://github.com/deepseek-ai/TileKernels) and, in `legacy/`,
[OLMo-core](https://github.com/allenai/OLMo-core).
