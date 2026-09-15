# LM-KBC 2026 Submission — Gemma3-27B

Few-shot solution for the [AKBC Shared Task 2026 / LM-KBC](https://github.com/lm-kbc/dataset2026),
run with `gemma3:27b` through a local Ollama server on Kaggle (T4 GPU).

## Structure

Both `my-solution/` and `my-solutionval/` contain the same self-contained Kaggle
script at `src/run_gemma3_27b.py`: it clones the official `dataset2026` repo itself
(for `data/`, `prompt_templates/`, `abstract_model.py`), pulls the `gemma3:27b`
model via Ollama, and runs the same few-shot prompting setup. The two copies differ
only in which data file they load (plus one relation-set difference) — kept as
separate folders because their outputs mean different things:

- `my-solutionval/` — points at `val.jsonl`, which ships with public gold labels,
  so its output can be scored locally with `evaluate.py`. The [Results](#results)
  table below is based on this run.
  - `results/predictions.jsonl`, `results/run.log`, `results/results.txt` — the
    predictions, full Kaggle run log, and `evaluate.py` output from that run.
- `my-solution/` — points at `test.jsonl` instead, whose gold labels are hidden
  (only Codabench can score it). This is what was actually submitted.
  - `src/results1/predictions.jsonl`, `src/results1/lmkbc-gemma3-27b.log` — the
    predictions and run log from the official submission (Codabench test score:
    **0.4733**). Despite the folder name, this *is* the committed result here, not
    a disposable rerun output — see the warning under
    [How to reproduce](#how-to-reproduce) before reproducing this one.

Each folder's `src/kernel-metadata.json` is the Kaggle kernel config (used with
`kaggle kernels push`) — it lives next to the script on purpose, since
`kaggle kernels push -p .` requires it in the same folder as the code file.

`requirements.txt` (repo root) is a local-only dependency (the `kaggle` CLI),
shared by both folders above. The actual ML stack installs itself inside the
Kaggle kernel, not here.

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

Full breakdown in `my-solutionval/results/results.txt`. Note: that file is UTF-16
encoded (PowerShell's `Tee-Object` writes UTF-16 by default) — if you open it and
see odd spacing or it looks "corrupted" in a plain text viewer, that's why; any
editor that auto-detects encoding handles it fine.

`hasCapacity` and `awardWonBy` are the weakest relations here — worth revisiting
the prompt/parsing for those in a future submission round.

**Test set:** submitted using the same setup (`my-solution/`, unchanged prompting),
scoring **0.4733** on Codabench. `test.jsonl`'s gold labels aren't public, so there's
no local per-relation breakdown for it the way there is for validation above.

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
dependencies run locally. `requirements.txt` lives at the repo root (shared by
both `my-solution/` and `my-solutionval/`), so set the venv up from there:
 
```bash
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
 
Pick which run you want to reproduce — `my-solution` (test set, the official
submission) or `my-solutionval` (validation set, locally scoreable) — and `cd`
into its `src/` folder. `kernel-metadata.json` already lives there, right next to
the script; don't delete it afterwards, it's not a temporary copy, it's the only
one you have. Both folders' `kernel-metadata.json` point at the same Kaggle
kernel id (`<your-kaggle-username>/lmkbc-gemma3-27b`), so pushing one after the other
reuses that same kernel slot on Kaggle — fine for reproducing one at a time, just
don't expect both runs to exist there simultaneously:
 
```bash
cd my-solution/src   # or my-solutionval/src
kaggle kernels push -p . --accelerator NvidiaTeslaT4
```
 
`--accelerator NvidiaTeslaT4` is what actually puts the run on a GPU -- without it,
Kaggle falls back to `enable_gpu` in the metadata, which isn't always honored the
same way by every CLI version, so passing it explicitly is safer.
 
### 4. Monitor the run
 
Still from `src/`:
 
```bash
kaggle kernels status <your-kaggle-username>/lmkbc-gemma3-27b
```
 
A full run on either data file (475 rows) takes anywhere from tens of minutes to a
couple of hours on a T4, since Gemma3-27B is still a big model even GGUF-quantized
through Ollama. Re-run this command until it prints `COMPLETE`.
 
### 5. Pull the predictions back
 
Still from `src/`. If you're reproducing `my-solutionval`, download into
`results1/` to keep this run's output separate from the original committed run in
`results/`:
 
```bash
kaggle kernels output <your-kaggle-username>/lmkbc-gemma3-27b -p ./results1 --file-pattern "predictions\.jsonl$" -o
```
 
If you're reproducing `my-solution` instead, `results1/` already holds the
official submission output (predictions + log, Codabench score 0.4733) — use a
different folder name so you don't overwrite it:
 
```bash
kaggle kernels output <your-kaggle-username>/lmkbc-gemma3-27b -p ./results2 --file-pattern "predictions\.jsonl$" -o
```
 
Either way, this creates `src/<that folder>/predictions.jsonl` -- note the path
for the next step.
 
### 6. Evaluate locally
 
Move back up to whichever folder you were reproducing first, since `evaluate.py`
lives one level up in `dataset2026/`. This step only makes sense for
`my-solutionval` — `test.jsonl`'s gold labels are hidden, so there's nothing for
`evaluate.py` to score `my-solution`'s predictions against locally:
 
```bash
cd ..
python ../dataset2026/evaluate.py -g ../dataset2026/data/val.jsonl -p src/results1/predictions.jsonl
```
 
## Notes
 
- `HFTransformersBaselineModel` is the `transformers`-based path (used when
  `USE_OLLAMA_MODEL = False`); it is kept for local comparisons but was not used
  to produce any of the committed predictions above.
- `OllamaBaselineModel` (subclass) is what actually ran here — it reuses the same
  prompting/few-shot/parsing logic but calls a local Ollama server instead of
  loading a `transformers` model directly, which is how Gemma is run without
  needing an HF token or license click-through.