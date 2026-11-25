# MUSE: Multi-Trajectory Skill Evolution

This repository contains the runnable implementation used for the MUSE online
experiments. It includes:

- Base LLM prediction without persistent memory;
- ACE-style online playbook adaptation;
- MUSE multi-trajectory distillation, uncertainty-aware gating, and skill
  lifecycle management;
- matched online evaluation, token accounting, frozen holdout evaluation, and
  paired analysis utilities.

Historical result folders, API credentials, checkpoints, and Python caches are
intentionally excluded. Every run writes to a new output directory.

## 1. Environment

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set the Alibaba Cloud Model Studio
OpenAI-compatible API credentials:

```text
DASHSCOPE_API_KEY=your_api_key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

Never commit `.env` or an API key.

## 2. Data

The compact repository keeps the local files required by the default three
dataset experiment:

```text
gsm-hard/train.json
math/test.jsonl
hotpot_dev_distractor_v1/test.jsonl
```

The driver maps these files to the dataset names `gsm-hard`, `math`, and
`hotpotqa`. Each run samples a deterministic ordered stream without replacement
and records the item IDs and source indices in `experiment_manifest.json`.

## 3. Main online experiment

All methods use the same ordered item stream. For each item, every method
predicts before receiving the label; adaptive methods then update before the next
item. The default is 100 online items, seed 41, temperature 0.3, an 8,000-token
output budget, and sequential updates across items.

Run Base LLM, ACE, and MUSE on all three datasets:

```powershell
python run_main_experiments.py `
  --experiment main `
  --model qwen3.5-35b-a3b `
  --datasets gsm-hard math hotpotqa `
  --methods base_llm ACE MUSE `
  --seed 41 `
  --stream-size 100 `
  --holdout-size 0 `
  --batch-parallel-size 1
```

Run one dataset only by changing `--datasets`, for example:

```powershell
python run_main_experiments.py --experiment main --datasets math --model qwen3.5-35b-a3b
```

`--batch-parallel-size` controls independent predictions for the current item.
The online item order and adaptive state updates remain sequential.

## 4. MUSE controls

The principal controls are exposed as command-line arguments and are recorded
in every run manifest:

| Argument | Default | Meaning |
| --- | ---: | --- |
| `--skill-update-interval` | 100 | Items between skill distillation updates |
| `--trajectory-window-size` | 20 | Recent trajectories supplied to the distiller |
| `--retrieval-pool-size` | 20 | Candidate skills considered by retrieval |
| `--skill-top-k` | 8 | Skills injected into the solver prompt |
| `--gate-threshold` | 0.45 | Minimum activation score |
| `--retain-threshold` | 0.20 | Utility threshold for retaining a skill |
| `--retire-threshold` | 0.05 | Utility threshold for retiring a skill |
| `--max-library-size` | 32 | Maximum active skills |
| `--temperature` | 0.3 | Shared decoding temperature |
| `--max-tokens` | 8000 | Shared generation budget |

For the manuscript's per-item update setting, pass:

```powershell
--skill-update-interval 1 --trajectory-window-size 10 `
--retrieval-pool-size 10 --skill-top-k 3 `
--gate-threshold 0.45 --retain-threshold 0.20 `
--retire-threshold 0.05 --max-library-size 32
```

The solver records reasoning separately from the final answer; only the final
answer field is scored. HotpotQA reports both exact match (EM) and token-level
F1. GSM-Hard and MATH report EM.

## 5. Ablation experiment

The ablation driver uses the same online protocol and compares Base LLM, ACE,
MUSE, single-trajectory MUSE, MUSE without the uncertainty gate, and MUSE
without lifecycle management:

```powershell
python run_main_experiments.py `
  --experiment ablation `
  --model qwen3.5-35b-a3b `
  --datasets math `
  --seed 41 `
  --stream-size 100 `
  --batch-parallel-size 1 `
  --skill-update-interval 1 `
  --trajectory-window-size 10 `
  --retrieval-pool-size 10 `
  --skill-top-k 3 `
  --max-library-size 32
```

Each ablation changes only its named mechanism. Use separate output
directories for repeated seeds.

## 6. Frozen holdout and transfer

To create a disjoint frozen holdout for a completed main run, set
`--holdout-size 100`:

```powershell
python run_main_experiments.py `
  --experiment main `
  --model qwen3.5-35b-a3b `
  --datasets gsm-hard math hotpotqa `
  --seed 41 `
  --stream-size 100 `
  --holdout-size 100
```

The holdout is evaluated without feedback-driven updates. Transfer evaluation
loads the frozen MUSE skill library produced by a completed source run and
requires a different target dataset:

```powershell
python run_main_experiments.py `
  --experiment transfer `
  --model qwen3.5-35b-a3b `
  --seed 41 `
  --source-run results\revision_main_qwen3.5-35b-a3b_41_TIMESTAMP `
  --source-dataset math `
  --target-dataset gsm-hard
```

The source run and transfer run must use the same model, seed, and protocol.
ACE is evaluated in the matched online comparison; its evolving playbook is not
treated as a fixed reusable skill set for the held-out skill-transfer claim.

## 7. Outputs

Each run contains:

- `experiment_manifest.json`: model, seed, parameters, source hashes, and item
  IDs;
- `online/`: per-method predictions, metrics, skill libraries, playbooks, and
  learning curves;
- `online_*_results.md` and CSV summaries;
- token-usage summaries including solver, reflection, distillation, and total
  tokens;
- `holdout/` and `frozen_results.md` when a holdout is requested.

Failed requests, parse failures, and truncated responses are recorded per item.
The client retries failed generations up to three attempts; a sample remains a
failed sample if all attempts fail.

## 8. Checks

Run syntax checks without calling a hosted model:

```powershell
python -m compileall -q ace_framework main.py run_main_experiments.py
python run_main_experiments.py --help
```

For a single configurable run, `main.py` remains available as a compatibility
entry point and forwards command-line arguments to `run_main_experiments.py`.
The recommended driver is `run_main_experiments.py`.
