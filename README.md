# A²OPD

**Adaptive Trust-Region On-Policy Distillation** — a heterogeneous on-policy distillation framework built on [verl](https://github.com/volcengine/verl) (HybridFlow).

A²OPD trains a **student** LLM with a **teacher** LLM under a heterogeneous rollout pipeline. When the student fails but the teacher succeeds, the student learns from the teacher via on-policy distillation (reverse KL). A²OPD adds three layers of adaptive trust-region control on top of static-lambda P-OPD:

| Layer | Name | Role |
|-------|------|------|
| **Macro** | Adaptive budget \(B_k\) | Limits policy update magnitude |
| **Meso** | Dual update for \(\lambda\) | Tracks the current budget via Lagrangian dual |
| **Micro** | Token emphasis via \(I_t\) | Importance sampling over student top-k candidates (`opd_top_k > 0`) |

Set `algorithm.a2opd.enable=true` to enable A²OPD; set `false` to fall back to legacy P-OPD.

## Repository Layout

```
A2-OPD/
├── scripts/               # Training, eval, and utility scripts
├── verl/                  # Vendored verl + A²OPD implementation
├── code_eval/             # EvalPlus & LiveCodeBench (code benchmarks)
└── math_eval_server/      # HTTP math verification service for Ray workers
```

## Installation

### Requirements

- Python ≥ 3.10
- CUDA-capable GPUs (training scripts default to 8 GPUs)
- PyTorch + [vLLM](https://github.com/vllm-project/vllm) (rollout engine)
- Ray, Hydra, flash-attn

### Setup

```bash
git clone https://github.com/KnowledgeXLab/A2-OPD.git
cd A2-OPD

pip install -r verl/requirements.txt
pip install -r verl/requirements-cuda.txt   # flash-attn

# Optional: full vLLM/SGLang stack
bash verl/scripts/install_vllm_sglang_mcore.sh
```

verl is **vendored** in this repo. Training scripts set `PYTHONPATH` via `scripts/opd_paths.sh`.

## Quick Start

### 1. Configure paths

```bash
source scripts/opd_paths.sh

export OPD_PUBLIC_MODEL_ROOT=/path/to/models
export OPD_PUBLIC_DATA_ROOT=/path/to/G-OPD-Training-Data
export OPD_A2OPD_MODEL_ROOT=/path/to/checkpoints/a2opd
export WANDB_API_KEY=your_wandb_key
```

### 2. Train

```bash
# Math: Qwen3-8B student + Qwen3-4B teacher expert/base
bash scripts/a2opd_math.sh

# Code
bash scripts/a2opd_code.sh

# Override hyperparameters
opd_top_k=16 total_training_steps=80 bash scripts/a2opd_math.sh

# Resume
bash scripts/a2opd_math.sh --resume_path=/path/to/checkpoint
```

Math training auto-starts the verification HTTP server in `math_eval_server/`.

### 3. Evaluate

```bash
bash scripts/eval/math_eval/run_eval_math.sh
bash scripts/eval/code_eval/run_eval_code.sh

# Offline distribution KL analysis
exp_path=/path/to/run bash scripts/a2opd_distribution_kl_eval.sh
```

Merge FSDP checkpoints before HuggingFace inference:

```bash
bash scripts/merge_checkpoints.sh
```

## Key Configuration

Main Hydra config: `verl/verl/trainer/config/algorithm/a2opd.yaml`

```yaml
+algorithm.train_mode: heterogeneous_distill
+algorithm.hetero_distill.use_opd: true
+algorithm.hetero_distill.opd_top_k: 16
algorithm.a2opd.enable: true
algorithm.a2opd.budget_adaptive_source: target_transfer_kl
algorithm.a2opd.baseline_mode: state
```

## Data

Training data follows the G-OPD-Training-Data layout:

- **Math**: `DeepMath-103K/train_filtered_level6.parquet`, `MathTestTotal/test.parquet`
- **Code**: `Eurus/code_train.parquet`, `Eurus/code_validation.parquet`

Download and set `OPD_PUBLIC_DATA_ROOT` to your local copy.

## Notes

- Do **not** set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` when using vLLM rollout.
- Override any path via `OPD_*` environment variables in `scripts/opd_paths.sh`.

## License

MIT License — see [LICENSE](LICENSE).

## Acknowledgements

Built on [verl / HybridFlow](https://arxiv.org/abs/2409.19256) and the heterogeneous on-policy distillation framework.
