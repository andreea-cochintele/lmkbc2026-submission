# LM-KBC 2026 Submission — Gemma3-27B

Few-shot solution for the [AKBC Shared Task 2026 / LM-KBC](https://github.com/lm-kbc/dataset2026),
run with `gemma3:27b` through a local Ollama server on Kaggle (T4 GPU).

## Structure

- `src/run_gemma3_27b.py` — self-contained Kaggle script. It clones the official
  `dataset2026` repo itself (for `data/`, `prompt_templates/`, `abstract_model.py`),
  pulls the `gemma3:27b` model via Ollama, and runs the same few-shot prompting setup
  used for the other baselines we compared.
- `src/kernel-metadata.json` — Kaggle kernel config (used with `kaggle kernels push`).
  Lives next to the script on purpose — `kaggle kernels push -p .` requires it in the
  same folder as the code file.
- `requirements.txt` — local-only dependency (the `kaggle` CLI). The actual ML stack
  installs itself inside the Kaggle kernel, not here.
- `results/predictions.jsonl` — model predictions from the original run, in the
  required submission format (`SubjectEntity`, `Relation`, `ObjectEntities`).
- `results/run.log` — full Kaggle run log for that original run.
- `results/results.txt` — output of `evaluate.py` for that original run (macro/micro
  precision, recall, F1 per relation).

Re-running the kernel (see [How to reproduce](#how-to-reproduce)) writes its output
to `results1/` instead, so a fresh run never overwrites the original results above —
compare the two folders directly to see if a change helped or hurt.

## Results

Validation set (`val.jsonl`), macro/micro precision, recall, F1 per relation:

| Relation | macro-p | macro-r | macro-f1 | micro-p | micro-r | micro-f1 | avg. #preds | #empty preds |
|---|---|---|---|---|---|---|---|---|
| awardWonBy | 0.616 | 0.157 | 0.168 | 0.485 | 0.076 | 0.132 | 22.900 | 3 |
| companyTradesAtStockExchange | 0.693 | 0.788 | 0.638 | 0.607 | 0.654 | 0.630 | 0.840 | 25 |
| countryLandBordersCountry | 0.976 | 0.981 | 0.975 | 0.961 | 0.966 | 0.964 | 2.647 | 18 |
| hasArea | 0.570 | 0.570 | 0.570 | 0.570 | 0.570 | 0.570 | 1.000 | 0 |
| hasCapacity | 0.210 | 0.210 | 0.210 | 0.210 | 0.210 | 0.210 | 1.000 | 0 |
| personHasCityOfDeath | 0.480 | 0.690 | 0.440 | 0.366 | 0.492 | 0.420 | 0.820 | 18 |
| **All Relations** | **0.560** | **0.615** | **0.531** | **0.572** | **0.224** | **0.322** | 1.621 | 64 |

Full breakdown in `results/results.txt`. Note: that file is UTF-16 encoded (PowerShell's
`Tee-Object` writes UTF-16 by default) — if you open it and see odd spacing or it looks
"corrupted" in a plain text viewer, that's why; any editor that auto-detects encoding
handles it fine.

Weakest relations right now are `hasCapacity` and `awardWonBy` — worth revisiting the
prompt/parsing for those before a final test-set run.

## Prerequisites

- A Kaggle account with GPU quota available (free tier gives ~30h/week on a T4;
  check under Settings > Accelerator on kaggle.com).
- `enable_internet: true` in the kernel config — the script needs it for `git clone`,
  `pip install`, and `ollama pull`.
- No Hugging Face account/token is strictly required for the Ollama path (the script's
  HF login attempt is leftover from the `transformers` path and fails silently if skipped).


## How to reproduce
 
### 1. Set up a local environment for the Kaggle CLI
 
You only need this to push the script and pull results back — none of the model
dependencies run locally. Run this from inside `my-solution/` specifically — a
`requirements.txt` in a parent folder is a different file and will fail or install
things you don't need here.
 
```bash
cd my-solution
python -m venv venv
```
 
```bash
# Windows (PowerShell)
venv\Scripts\Activate.ps1
# Linux/macOS
source venv/bin/activate
```
 
```bash
pip install -r requirements.txt
```
 
### 2. Set up your Kaggle API token
 
Get a token from your [Kaggle account settings](https://www.kaggle.com/settings)
("Create New Token"). Depending on your account, this either downloads a
`kaggle.json` directly, or shows the key as text to copy — in which case, build
the file yourself:
 
- Windows: `%USERPROFILE%\.kaggle\kaggle.json`
- Linux/macOS: `~/.kaggle/kaggle.json`
```powershell
mkdir "$env:USERPROFILE\.kaggle" -Force
 
@'
{"username":"USERNAME","key":"KEY"}
'@ | Set-Content -Path "$env:USERPROFILE\.kaggle\kaggle.json" -NoNewline
```
 
### 3. Push the kernel to Kaggle
 
`kernel-metadata.json` already lives in `src/`, right next to the script. Stay in
this folder after pushing, and don't delete this file afterwards -- it's not a
temporary copy, it's the only one you have:
 
```bash
cd src
kaggle kernels push -p . --accelerator NvidiaTeslaT4
```
 
`--accelerator NvidiaTeslaT4` is what actually puts the run on a GPU -- without it,
Kaggle falls back to `enable_gpu` in the metadata, which isn't always honored the
same way by every CLI version, so passing it explicitly is safer.
 
### 4. Monitor the run
 
Still from `src/`:
 
```bash
kaggle kernels status andreeacochintele/lmkbc-gemma3-27b
```
 
A full run on `val.jsonl` (100+ rows) takes anywhere from tens of minutes to a
couple of hours on a T4, since Gemma3-27B is still a big model even GGUF-quantized
through Ollama. Re-run this command until it prints `COMPLETE`.
 
### 5. Pull the predictions back
 
Still from `src/`. Downloading into `results1/` (not `results/`) keeps this run's
output separate from the original run already committed:
 
```bash
kaggle kernels output andreeacochintele/lmkbc-gemma3-27b -p ./results1 --file-pattern "predictions\.jsonl$" -o
```
 
This creates `src/results1/predictions.jsonl` -- note the path for the next step.
 
### 6. Evaluate locally
 
Move back up to `my-solution/` first, since `evaluate.py` lives one level up in
`dataset2026/`:
 
```bash
cd ..
python ../dataset2026/evaluate.py -g ../dataset2026/data/val.jsonl -p src/results1/predictions.jsonl
```
 
## Notes
 
- `HFTransformersBaselineModel` is the `transformers`-based path (used when
  `USE_OLLAMA_MODEL = False`); it is kept for local comparisons but was not used
  to produce `results/predictions.jsonl`.
- `OllamaBaselineModel` (subclass) is what actually ran here — it reuses the same
  prompting/few-shot/parsing logic but calls a local Ollama server instead of
  loading a `transformers` model directly, which is how Gemma is run without
  needing an HF token or license click-through.