<h1 align="center">Depth Mixer</h1>

<p align="center">
  <a href="https://arxiv.org/pdf/2610.18809"><img src="https://img.shields.io/badge/Paper-B31B1B?logo=arxiv&amp;logoColor=white" alt="Paper"></a>
  &nbsp;
  <a href="https://huggingface.co/collections/davidgonmar/depth-memory-6ac5045a1bd8e798d2ed1392"><img src="https://img.shields.io/badge/Checkpoints-FFD21E?logo=huggingface&amp;logoColor=black" alt="Checkpoints"></a>
</p>

Efficient implementations of **depth mixers**, the methods that replace the residual connection of a
transformer across its layers, as Hugging Face models with fused kernels. The layout follows
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention): `ops` holds the kernels
and their naive PyTorch references, `layers` the modules that apply them inside a block, `models` the
`transformers` models, and `modules` the shared building blocks.

This is also the code of the [Depth Memory](https://arxiv.org/pdf/2610.18809) paper. **Linear Depth
Memory (LDM)** augments transformers with a fixed-size associative memory for storing and retrieving
information across layers; the paper's training, evaluation and throughput code is kept under
[Reproducing the paper](#reproducing-the-paper).

<p align="center">
  <img src="assets/figure_one.gif" alt="Illustration of Linear Depth Memory" width="800">
</p>

## Models

| Model | `model_type` | Paper | Kernel modes |
|---|---|---|---|
| LDM | `ldm` | [Depth Memory](https://arxiv.org/abs/2610.18809) | `chunk` (CuTe DSL, the paper's training kernels), `fused_recurrent` (Triton), `naive` |
| mHC | `mhc` | [Manifold-constrained Hyper-Connections](https://arxiv.org/abs/2512.24880) | `fused` (DeepSeek TileKernels, TileLang), `naive` |
| AttnRes | `attnres` | [Attention Residuals](https://arxiv.org/abs/2603.15031) | `fused` (Triton, from flash-linear-attention), `naive` |

Every model is a pre-norm transformer (Llama or Qwen3 shapes: GQA, optional per-head QK norm, tied or
untied embeddings) with the depth mixer on its residual path, and supports KV-cached generation.

Kernel support:

- **LDM `chunk`**: Hopper (H100) and Blackwell (B200). Training launches read per-GPU tuned schedule
  tables (`depth_mixer/ops/ldm/chunk/tuned/`): SM90 and SM100 cover the paper's shapes, SM120 (RTX PRO
  6000) the shapes listed in its table. Tune a new shape with `python -m depth_mixer.ops.ldm.chunk.tune`.
  Launches without gradients run on any of these GPUs.
- **LDM `fused_recurrent`**, **AttnRes `fused`**: any recent CUDA GPU.
- **mHC `fused`**: four BF16 streams of a width divisible by 64; H100, B200 and SM120. It needs `nvcc`
  on `PATH` (TileLang compiles the kernels at first use).

## Installation

Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --frozen
source .venv/bin/activate
```

This installs `depth_mixer` and the paper's `pretrain/OLMo-core` in editable mode. The commands
below assume the environment is active. mHC's kernels need a CUDA toolkit with `nvcc` on `PATH`.

## Usage

Load a model converted from the paper's checkpoints (see [Converting the paper's checkpoints](#converting-the-papers-checkpoints)):

```python
import torch
import depth_mixer  # registers the models with the transformers Auto classes
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "checkpoints/converted/400m-ldm"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda()
inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda")
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=16)[0]))
```

Any config field can be overridden at load time, for instance the kernel mode:
`from_pretrained(path, ldm_mode="fused_recurrent")`.

Initialize a new model from a config:

```python
from depth_mixer.models import LDMConfig, LDMForCausalLM

config = LDMConfig(
    hidden_size=1024, num_hidden_layers=24, num_heads=16, intermediate_size=2736, vocab_size=50280,
    ldm_groups=16, ldm_key_dim=8, ldm_address_hidden_dim=32, ldm_mode="chunk",
)
model = LDMForCausalLM(config).cuda().to(torch.bfloat16)
```

The ops can be used on their own; each fused op has a naive reference with the same signature:

```python
from depth_mixer.ops.attnres import fused_attnres, naive_attnres
from depth_mixer.ops.ldm import chunk_ldm, fused_recurrent_ldm, naive_recurrent_ldm
from depth_mixer.ops.mhc import fused_mhc_pre, fused_mhc_post, naive_mhc_pre, naive_mhc_post
```

An LDM layer threads one `LDMState` through the blocks of a forward: the embedding is written first,
then every block reads before attention, writes the attention output, reads before the MLP and
writes the MLP output, and a final read precedes the LM head (`depth_mixer/models/ldm/modeling_ldm.py`).

## Converting the paper's checkpoints

The released checkpoints are native OLMo-core checkpoints. `utils/convert_from_olmo_core.py` rebuilds
each one as the OLMo-core model it was trained as and writes the Hugging Face model and tokenizer:

```bash
hf download davidgonmar/depth-memory-400m-ldm --include "lr1e-3/*" --local-dir checkpoints/released/400m-ldm
python utils/convert_from_olmo_core.py --checkpoint checkpoints/released/400m-ldm/lr1e-3 \
    --output checkpoints/converted/400m-ldm
```

It supports the LDM (any backend's parameter layout), mHC and AttnRes checkpoints. The converted models
reproduce the OLMo-core ones at BF16 precision: on 4 x 2048 held-out FineWeb-Edu tokens,

| Model | OLMo-core loss | Hugging Face loss | Top-1 agreement |
|---|---|---|---|
| LDM 400M (`chunk` / `fused_recurrent` / `naive`) | 2.5598 | 2.5600 / 2.5604 / 2.5596 | 98.7% |
| AttnRes (Block) 400M | 2.5652 | 2.5655 | 99.1% |
| mHC 400M | 2.5703 | 2.5702 | 99.1% |
| LDM 1.5B (`chunk` / `fused_recurrent`) | 2.2929 | 2.2935 / 2.2932 | 98.8% / 98.6% |
| AttnRes (Block) 1.5B | 2.2995 | 2.2990 | 99.1% |
| mHC 1.5B | 2.2959 | 2.2964 | 99.2% |

The logits differ by about as much as one model's own BF16 and FP32 forwards
(`tests/models/test_olmo_core_parity.py`).

## Evaluation

`evals/harness.py` runs the [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
on a depth-mixer model; the paper's zero-shot suite is

```bash
python -m evals.harness --model depth_mixer \
    --model_args pretrained=checkpoints/converted/400m-ldm,dtype=bfloat16 \
    --tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande,openbookqa,lambada_openai,wikitext \
    --batch_size 32 --device cuda:0
```

## Benchmarks and tests

```bash
python benchmarks/ops/benchmark_ldm.py                         # the three LDM modes, one forward's 2L+1 edges
python benchmarks/benchmark_training_throughput.py --model ldm --size 400m --batch-size 8
pytest tests                                                   # ops against their references, models, checkpoint parity
```

`tests/ops` compares every fused op with its naive reference, forward and backward. `tests/models`
checks forward and backward, the save/load round trip, KV-cached decoding against the full forward,
greedy generation and left padding, and the parity with the paper's checkpoints when they are present.

## Library layout

```text
depth_mixer/
  ops/ldm/                  naive.py, fused_recurrent.py + kernels.py (Triton), chunk/ (CuTe DSL, DESIGN.md)
  ops/mhc/                  naive.py, fused.py, tile_kernels/ (DeepSeek TileKernels, MIT)
  ops/attnres/              naive.py, fused.py (Triton, from flash-linear-attention, MIT)
  layers/                   attn.py, ldm.py, mhc.py, attnres.py
  models/{ldm,mhc,attnres}/ configuration_*.py, modeling_*.py
  modules/                  fused norms, MLP, rotary embedding and losses (flash-linear-attention)
benchmarks/  evals/  tests/  utils/convert_from_olmo_core.py
```

## Reproducing the paper

The paper's runs train and evaluate native OLMo-core models.

### Hardware

- **Fast LDM backends:** NVIDIA Hopper (H100) or Blackwell (B200) GPUs.
- **Other LDM backends:** `flexible` supports recent CUDA GPUs; `reference` supports CPU and CUDA.
- **mHC:** requires a CUDA toolkit with `nvcc` on `PATH`.

### Data

The models are trained on the 100B-token sample of FineWeb-Edu (`sample/100BT` of
[`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)),
tokenized with OLMo's GPT-NeoX Dolma v1.5 tokenizer (vocabulary 50,280).

```bash
# Download the parquet shards of sample/100BT into data/fineweb-edu/100BT, then:
mkdir -p data/fineweb-edu/eval
mv data/fineweb-edu/100BT/013_00008.parquet data/fineweb-edu/eval/

python data_utils/tokenize_from_pretrain_datasets.py \
  --train-parquet-glob "data/fineweb-edu/100BT/*.parquet" \
  --eval-parquet-path data/fineweb-edu/eval/013_00008.parquet \
  --output-dir data/fineweb-edu/tokenized \
  --text-field text \
  --tokenizer-name-or-path pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json \
  --vocab-size 50280 --eos-token-id 50279 --pad-token-id 1 \
  --batch-size 4096 --skip-existing

export DATA_DIR=$PWD/data/fineweb-edu/tokenized
```


### Training setup

Set the data path, Weights & Biases entity, and checkpoint directory:

```bash
export DATA_DIR=... WANDB_ENTITY=... CHECKPOINT_DIR=...
```

See [paper-experiments/README.md](paper-experiments/README.md) for the full training setup.

#### Training and evaluation

Train each model at the learning rate selected in the paper:

```bash
# 400M models, 50B tokens
bash paper-experiments/pretrain/launch.sh 400m preln         0.001
bash paper-experiments/pretrain/launch.sh 400m attnres_full   0.004
bash paper-experiments/pretrain/launch.sh 400m attnres_block  0.002
bash paper-experiments/pretrain/launch.sh 400m mhc           0.002
bash paper-experiments/pretrain/launch.sh 400m ldm_k4        0.001
bash paper-experiments/pretrain/launch.sh 400m ldm           0.001

# 1.5B models, 100B tokens
bash paper-experiments/pretrain/launch.sh 1p5b preln         0.001
bash paper-experiments/pretrain/launch.sh 1p5b attnres_block 0.001
bash paper-experiments/pretrain/launch.sh 1p5b mhc           0.001
bash paper-experiments/pretrain/launch.sh 1p5b ldm           0.001
```

On H100 GPUs, pass `--gpu h100` before the scale, e.g. `bash paper-experiments/pretrain/launch.sh --gpu h100 400m ldm 0.001`.
To reproduce the learning-rate sweeps, repeat at `5e-4`, `1e-3` and `2e-3` (`1e-3`, `2e-3` and `4e-3` for AttnRes at 400M).

Evaluate a checkpoint:

```bash
bash paper-experiments/pretrain/evaluation/run.sh \
  "$CHECKPOINT_DIR/<run>/step47684" my-results/ldm.json
```

#### Ablations

Run 400M ablations for 15B tokens at a learning rate of `1e-3`:

```bash
bash paper-experiments/ablations/launch.sh ldm b200
bash paper-experiments/ablations/launch.sh preln b200
bash paper-experiments/ablations/launch.sh delta_rule b200
```

See [paper-experiments/ablations/README.md](paper-experiments/ablations/README.md)
for all variants and their definitions.

#### Throughput and memory

Benchmark training (8 GPUs), prefill, and decode (1 GPU):

```bash
bash paper-experiments/throughput/rounds.sh train   1p7b b200 3
bash paper-experiments/throughput/rounds.sh prefill 1p7b b200 3
bash paper-experiments/throughput/rounds.sh decode  1p7b b200 3
```

See [paper-experiments/throughput/README.md](paper-experiments/throughput/README.md)
for the benchmark setup.

### Released checkpoints

Native OLMo-core checkpoints are available in our [Hugging Face collection](https://huggingface.co/collections/davidgonmar/depth-memory-6ac5045a1bd8e798d2ed1392). Each model repository contains one folder per learning rate; its model card identifies the run selected in the paper.

| Method | 400M | 1.5B |
|---|---|---|
| PreLN | [Download](https://huggingface.co/davidgonmar/depth-memory-400m-preln) | [Download](https://huggingface.co/davidgonmar/depth-memory-1.5b-preln) |
| AttnRes (Full) | [Download](https://huggingface.co/davidgonmar/depth-memory-400m-attnres-full) | — |
| AttnRes (Block) | [Download](https://huggingface.co/davidgonmar/depth-memory-400m-attnres-block) | [Download](https://huggingface.co/davidgonmar/depth-memory-1.5b-attnres-block) |
| mHC | [Download](https://huggingface.co/davidgonmar/depth-memory-400m-mhc) | [Download](https://huggingface.co/davidgonmar/depth-memory-1.5b-mhc) |
| LDM | [Download](https://huggingface.co/davidgonmar/depth-memory-400m-ldm) | [Download](https://huggingface.co/davidgonmar/depth-memory-1.5b-ldm) |
| LDM, D_k = 4 | [Download](https://huggingface.co/davidgonmar/depth-memory-400m-ldm-dk4) | — |

Download and evaluate a checkpoint:

```bash
hf download davidgonmar/depth-memory-400m-ldm \
  --include "lr1e-3/*" --local-dir ldm-400m

bash paper-experiments/pretrain/evaluation/run.sh ldm-400m/lr1e-3 results.json
```

### Paper code layout

```text
pretrain/OLMo-core/                   OLMo-core (Apache-2.0) with the paper's methods added
  src/olmo_core/nn/ldm/               LDM: config, modules and four backends
    backends/reference.py             readable FP32 reference implementation
    backends/flexible/                Triton kernels; every ablation option (the 400M ablations)
    backends/fast/                    CuTe DSL training kernels with tuned launch tables (main runs)
    backends/fast_inference/          CuTe DSL prefill and decode kernels with tuned launch tables
  src/olmo_core/nn/hyper_connections.py, nn/mhc_ops.py   mHC
  src/olmo_core/kernels/tile_kernels/                    DeepSeek TileKernels, mHC subset (MIT)
  src/olmo_core/kernels/attnres.py                       AttnRes kernels from FLA (MIT)
paper-experiments/
  pretrain/          training recipe (train.py), launcher, model configs, evaluation
  ablations/         the 400M ablation series: generator, configs, launcher
  throughput/        speed and memory benchmark (training, prefill, decode)
config_utils/        model config loader
data_utils/          FineWeb-Edu tokenization
eval/                lm-eval adapter for native checkpoints
```

`backends/BACKENDS.md` compares the four LDM backends, and the fast backends' `DESIGN.md` files
describe their kernels.


## Citation

```bibtex
@misc{gonzalezmartínez2026depthmemory,
      title={Depth Memory}, 
      author={David González-Martínez and Philipp Nazari and T. Konstantin Rusch and Shiwei Liu},
      year={2026},
      eprint={2610.18809},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2610.18809}, 
}
```

## Dependencies

This project builds on:

- [**OLMo-core**](https://github.com/allenai/OLMo-core) for model and training infrastructure.
- [**DeepSeek TileKernels**](https://github.com/deepseek-ai/TileKernels) for mHC kernels.
- [**flash-linear-attention**](https://github.com/fla-org/flash-linear-attention) for the library layout, the shared
  modules and cache, and the Attention Residuals kernels.
