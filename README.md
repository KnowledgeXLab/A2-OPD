<div align="center">

# 🔬 A2-OPD

## Rethinking Relative Policy Differences Transfer of On-Policy Distillation


<p>
  <a href="https://arxiv.org/abs/2610.13205"><img src="https://img.shields.io/badge/arXiv-2610.13205-b31b1b?style=flat-square&logo=arxiv&logoColor=white" height="28" alt="arXiv Paper"/></a>
  &nbsp;
  <a href="https://huggingface.co/papers/2610.13205"><img src="https://img.shields.io/badge/🤗%20Paper-2610.13205-yellow?style=flat-square" height="28" alt="Hugging Face Paper"/></a>
  &nbsp;
  <a href="https://huggingface.co/KnowledgeXLab/P-OPD_Experiments"><img src="https://img.shields.io/badge/Models-HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=yellow" height="28" alt="Hugging Face Models"/></a>
  &nbsp;
  <a href="https://huggingface.co/datasets/KnowledgeXLab/A2-OPD"><img src="https://img.shields.io/badge/🤗%20Dataset-A2--OPD-yellow?style=flat-square" height="28" alt="Hugging Face Dataset"/></a>
  &nbsp;
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue?style=flat-square" height="28" alt="License"/></a>
</p>

**[<img src="https://img.shields.io/badge/-arXiv-B31B1B?logo=arxiv&logoColor=white" height="16" valign="middle"> arXiv](https://arxiv.org/abs/2610.13205)** · **[<img src="https://www.alphaxiv.org/favicon.ico" width="16" height="16" valign="middle"> alphaXiv](https://www.alphaxiv.org/abs/2610.13205)** · **[🤗 Paper](https://huggingface.co/papers/2610.13205)** · **[Models](https://huggingface.co/KnowledgeXLab/P-OPD_Experiments)** · **[Dataset](https://huggingface.co/datasets/KnowledgeXLab/A2-OPD)**

</div>

---

> 💡 On-policy distillation typically needs a teacher stronger than the target. Relative update transfer relaxes this by supervising the target with a **policy difference** $\Delta_\phi = \log(\pi_\phi^+ / \pi_\phi)$ instead of an absolute teacher distribution — but it remains unclear **which differences are informative**, **what roles the expert and reference play**, and **which targets can realize a given difference**. **A2-OPD** formalizes this as implicit-reward KL-regularized optimization and answers these questions through systematic experiments over RL-, SFT-, and scale-induced differences.

## 🔍 Overview

Conventional OPD aligns a target policy to a stronger teacher's absolute distribution, which breaks down when teacher and target have similar capability or mismatched priors. Recent work instead transfers the **relative policy difference** between a proxy expert $\pi_\phi^+$ and a proxy reference $\pi_\phi$:

$$
\Delta_\phi(a \mid s) = \log \frac{\pi_\phi^+(a \mid s)}{\pi_\phi(a \mid s)}
$$

We develop a unified KL-regularized view in which $\Delta_\phi$ acts as an **implicit reward**, inducing an on-policy update for the target while decoupling the proxy pair from the target reference $\pi_{\mathrm{ref}}$. This yields **A2-OPD** (Any-to-Any Policy Transfer): arbitrary proxy pairs can supervise arbitrary targets that differ in scale, training stage, or model family.

Under this formulation, conventional OPD is recovered when the proxy expert is the teacher and the proxy reference equals the target anchor. Several recent methods instantiate the same relative-transfer principle with different proxy pairs: [Direct-OPD](https://arxiv.org/abs/2607.05394) transfers the RL-induced shift between a teacher's post- and pre-RL checkpoints; [OPD²](https://arxiv.org/abs/2607.15161) uses the delta between a reasoning-tuned teacher and its base model; and [P-OPD](https://arxiv.org/abs/2607.11505) extracts reusable update signals from a reward-optimized proxy. Table 1 in the paper unifies these methods under one objective.

The target maximizes:

$$
\mathcal{J}_{\mathrm{A2}}(\theta) = \mathbb{E}_{s \sim d_{\pi_\theta}} \left[ \mathbb{E}_{a \sim \pi_\theta} \left[ \alpha \, \Delta_\phi(s,a) - \beta \log \frac{\pi_\theta(a \mid s)}{\pi_{\mathrm{ref}}(a \mid s)} \right] \right]
$$

where $\alpha/\beta$ controls transfer strength.

## ⚙️ Method at a Glance

| Component | Role |
|-----------|------|
| Proxy expert $\pi_\phi^+$ | Encodes the capability to transfer (e.g., RL-tuned model) |
| Proxy reference $\pi_\phi$ | Defines the baseline improvement direction |
| Target reference $\pi_{\mathrm{ref}}$ | Anchors the target; need not equal $\pi_\phi$ |
| Target $\pi_\theta$ | Trained on its own rollouts with weight $u_\theta = \alpha \Delta_\phi - \beta \ell_\theta$ |

## 📰 Updates

* **`2026-10-09`:** 📄 Paper on [arXiv](https://arxiv.org/abs/2610.13205) · 🎉 Code release for paper experiments and analysis.

## 🛠️ Installation

**Requirements:** Python ≥ 3.10, CUDA ≥ 12.4, 8× GPU recommended for default scripts (Qwen3-8B with TP=2).

```bash
git clone https://github.com/KnowledgeXLab/A2-OPD.git
cd A2-OPD

conda create -n a2opd python=3.10 -y
conda activate a2opd

pip install -r verl/requirements.txt
pip install -r verl/requirements-cuda.txt   # flash-attn
pip install vllm pebble wandb

# Optional: full vLLM / SGLang stack
# bash verl/scripts/install_vllm_sglang_mcore.sh

export WANDB_API_KEY=your_wandb_key
```

## 📦 Model Weights

Paper-trained checkpoints are hosted on **[KnowledgeXLab/P-OPD_Experiments](https://huggingface.co/KnowledgeXLab/P-OPD_Experiments)** (shared with [P-OPD](https://github.com/KnowledgeXLab/P-OPD)). G-OPD public RL checkpoints link to [Keven16](https://huggingface.co/Keven16) below.

### RL checkpoints (Math)

| Checkpoint | Role | Training |
|------------|------|----------|
| [Qwen3-1.7B-Math-GRPO-Steps500](https://huggingface.co/KnowledgeXLab/P-OPD_Experiments/tree/main/Qwen3-1.7B-Math-GRPO-Steps500) | Proxy expert | DeepMath-103K · GRPO · 500 steps |
| [Qwen3-1.7B-Math-GRPO-Steps800](https://huggingface.co/KnowledgeXLab/P-OPD_Experiments/tree/main/Qwen3-1.7B-Math-GRPO-Steps800) | Proxy expert | DeepMath-103K · GRPO · 800 steps |
| [Qwen3-4B-Math-GRPO-Early-RL](https://huggingface.co/KnowledgeXLab/P-OPD_Experiments/tree/main/Qwen3-4B-Math-GRPO-Early-RL) | Proxy expert | Early-stage RL checkpoint |
| [Qwen3-4B-Non-Thinking-RL-Math-Step500](https://huggingface.co/Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500) | Proxy expert | DeepMath · GRPO · 500 steps |
| [Qwen3-4B-Non-Thinking-RL-Math-Step1200](https://huggingface.co/Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step1200) | Proxy expert | DeepMath · GRPO · 1200 steps |


### SFT checkpoints (Math)

| Checkpoint | Role | Training |
|------------|------|----------|
| [Qwen3-4B-Math-SFT](https://huggingface.co/KnowledgeXLab/P-OPD_Experiments/tree/main/Qwen3-4B-Math-SFT) | Proxy expert | Task-directed SFT |

### RL checkpoints (Code)

| Checkpoint | Role | Training |
|------------|------|----------|
| [Qwen3-4B-Non-Thinking-RL-Code-Step300](https://huggingface.co/Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step300) | Proxy expert | Eurus-RL-Code · GRPO · 300 steps |

Download checkpoints under `./models/` (or set `A2OPD_MODEL_ROOT`):

## 📂 Datasets

All training and evaluation data are released in the Hugging Face dataset repo: **[KnowledgeXLab/A2-OPD](https://huggingface.co/datasets/KnowledgeXLab/A2-OPD)**, organized by `math/` and `code/` folders.

```bash
# One-click download (requires huggingface_hub: `hf` or `huggingface-cli`)
hf download KnowledgeXLab/A2-OPD --repo-type dataset --local-dir data
export A2OPD_DATA_ROOT=$PWD/data   # default layout matches training/eval scripts
```

| Path | Split | Used for |
|------|-------|----------|
| `math/train.parquet` | train | DeepMath-103K training |
| `math/test.parquet` | test | Training validation and offline math eval |
| `code/train.parquet` | train | Eurus-RL-Code training |
| `code/validation.parquet` | validation | Training validation during code RL |
| `code/benchmarks/Humaneval_Plus/HumanEvalPlus.jsonl` | test | Offline HumanEval+ |
| `code/benchmarks/Mbpp_Plus/MbppPlus.jsonl` | test | Offline MBPP+ |
| `code/benchmarks/LivecodeBench/test.jsonl` | test | Offline LiveCodeBench |

Training data must be in **verl parquet format** (see [verl data docs](https://github.com/volcengine/verl)). Each row needs:

| Field | Description |
|-------|-------------|
| `data_source` | Dataset name; routes to the correct reward function |
| `prompt` | Chat messages, e.g. `[{"role": "user", "content": "..."}]` |
| `ability` | Task type (`Math` or `Code`) |
| `reward_model.ground_truth` | Reference answer or test cases |
| `extra_info` | Optional metadata (e.g. benchmark source tag) |

### 📐 Math

`math/test.parquet` is a multi-benchmark mix. Benchmarks referenced in training and offline eval: `AIME2024`, `AIME2025`, `AIME2026`, `SMT2025`, `CMIMC2025`, `HMMT2025FEB`, `HMMT2025NOV`, `HMMT2026FEB`. Set `data_source` in each row so the math-verify reward router can identify them.

### 💻 Code

Supported training sources in `code/train.parquet`: `taco`, `apps`, `codecontests`, `codeforces`.

Offline code eval reads benchmarks under `data/code/benchmarks/` (EvalPlus + LiveCodeBench). Override via `CODE_BENCHMARK_ROOT=$A2OPD_DATA_ROOT/code/benchmarks` if needed.


## 🏋️ Training

Run from the **repository root** (default: 8 GPUs).

```bash
bash scripts/a2opd_math.sh   # Math
bash scripts/a2opd_code.sh   # Code
```

Checkpoints are saved under `checkpoints/` (override via `A2OPD_CKPT_ROOT`).

## 📏 Evaluation

```bash
# Math offline benchmarks
bash scripts/eval/math_eval/run_eval_math.sh

# Code — HumanEval+ / MBPP+ / LiveCodeBench
bash scripts/eval/code_eval/run_eval_code.sh

# Offline distribution KL analysis
exp_path=/path/to/run bash scripts/a2opd_distribution_kl_eval.sh
```

## 📁 Repository Structure

```
A2-OPD/
├── scripts/                    # Training, eval, path helpers
│   ├── a2opd_math.sh / a2opd_code.sh
│   ├── merge_checkpoints.sh / opd_paths.sh
│   └── eval/
│       ├── math_eval/          # Math eval entry
│       └── code_eval/          # Code eval entry
├── verl/                       # verl + A2-OPD implementation
└── math_eval_server/           # HTTP math verification for Ray workers
```

## 🙏 Acknowledgements

Our training and evaluation code builds upon:

* **[P-OPD](https://github.com/KnowledgeXLab/P-OPD)**
* **[G-OPD](https://github.com/RUCBM/G-OPD)**
* **[verl](https://github.com/volcengine/verl)**

We thank the authors and contributors of these projects for their open-source work.

## 📬 Contact

For questions or feedback:

* Open an issue in this repository
* Email: wurong1159@zju.edu.cn

## 📜 Citation

If you find this work helpful, please cite:

```bibtex
@misc{wu2026rethinkingrelativepolicydifferences,
      title={Rethinking Relative Policy Differences Transfer of On-Policy Distillation}, 
      author={Rong Wu and Daocheng Fu and Yu Yang and Xuemeng Yang and Siqi Li and Shu Zou and Yao Zhu and Botian Shi and Bo Zhang and Cong Zhang and Pinlong Cai and Jianbiao Mei},
      year={2026},
      eprint={2610.13205},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2610.13205}, 
}
```
