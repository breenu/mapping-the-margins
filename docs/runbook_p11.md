# P11 lab runbook: running the experiment on a university GPU

This is the step-by-step procedure for running the Mapping the Margins experiment (P11) on a lab
computer with an NVIDIA GPU. It is written so you can follow it top to bottom without knowing the
codebase. **Every command in grey boxes is exact: copy it as written.** Where a value changes from
run to run it is shown in angle brackets, for example `<N>`, and the text says where to get it.

Every command in this file is checked by the test suite (`tests/test_p11_orchestration.py`) against
the real scripts, so a flag that does not exist cannot survive here.

**The one rule that overrides everything below:** if something fails, **stop and tell the project
lead**, with the last ~50 lines of terminal output. Never unpin a dependency, never
`pip install` anything, never swap in a different model, and never edit a file tracked by git
to make something work. Each of those would silently change what is being measured. See section F.

---

## What you need before you start

| Need | Why | How to check |
|---|---|---|
| **Linux** machine with an **NVIDIA GPU**, ≥ 16 GB VRAM (24 GB recommended) | The models run at 4-bit NF4 via `bitsandbytes`, which in this project is Linux-only | `nvidia-smi` prints the card |
| NVIDIA **driver** already installed | torch brings its own CUDA runtime, but not the driver, and installing a driver needs admin rights | `nvidia-smi` works |
| ~**100 GB** free disk | Two ~16 GB checkpoints, the scorer models, the outputs | `df -h .` |
| Internet access to github.com, huggingface.co, pypi.org, astral.sh | Code, models, packages | `curl -sI https://huggingface.co` |
| A **Hugging Face account** that has accepted the licence on `meta-llama/Llama-3.1-8B-Instruct` | The Llama model is gated | Visit the model page while logged in |
| A Hugging Face **access token** (read scope) | Downloads the gated model | huggingface.co → Settings → Access Tokens |
| **GitHub access** to `breenu/mapping-the-margins` (a private repository) | To clone the code | Ask the project lead |

The run takes **days of GPU time** in total (section C has the numbers). Every generation step can
be stopped and restarted without losing work, so it fits into separately booked sessions.

---

## A. Setup on a fresh lab machine

### A1. Check the GPU

```bash
nvidia-smi
```

You should see the card's name, its memory, and a "Driver Version". If the command is not found or
errors, **stop**: the driver is missing, and installing it needs whoever administers the machine.

### A2. Install `uv` (no admin rights needed)

`uv` is the project's package manager. It installs into your home directory.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version
```

If `curl` to astral.sh is blocked, **stop** and ask the project lead. Do not install `uv` another
way on your own.

### A3. Get the code

```bash
git clone https://github.com/breenu/mapping-the-margins.git
cd mapping-the-margins
git checkout main
git pull
git log -1 --oneline
```

The repository is private. `git clone` will ask for your GitHub username and a **personal access
token** (not your password). Write down the commit that `git log -1 --oneline` prints; it goes in
your notes for the day.

### A4. Install the environment

```bash
uv sync --locked --extra dev --extra inference
```

This takes a few minutes and downloads torch with CUDA support. `--locked` makes it fail rather than
change any version. **If it fails, stop and report it.** Do not try other versions.

### A5. Set your Hugging Face token and cache location

Do this in **every new terminal**, before any other command:

```bash
export HF_TOKEN=<your Hugging Face token>
export HF_HOME=<a directory on a disk with at least 60 GB free>/hf-cache
```

Never put the token in a file inside the repository.

### A6. Confirm the model pins are reachable

```bash
uv run python scripts/pin_models.py --models llama31_8b_instruct mistral7b_instruct_v03
```

Each line should say `pinned`. `UNREACHABLE` means a token or licence problem (section F).
`DRIFTED` is information, not an error: it means the upstream model moved on, while this project
keeps using the recorded revision. Carry on.

### A7. Prepare the inputs (no GPU needed)

```bash
uv run python scripts/run_p11.py prepare
```

This fetches the pinned word list and CrowS-Pairs dataset, checks their digests, and renders the
prompt sets: E1's 2,970 prompts at `data/prompts/e1_sampled/prompts_v1.jsonl`, and E4's two
432-prompt files at `data/prompts/e4_sampled_t10/` and `data/prompts/e4_greedy/`.

### A8. VERIFY that each model loads at NF4 on this GPU. Do not skip this.

```bash
uv run python scripts/run_p11.py verify-gpu --model llama31_8b_instruct
uv run python scripts/run_p11.py verify-gpu --model mistral7b_instruct_v03
```

Each one downloads its model the first time (~16 GB, 5–30 min), loads it at 4-bit NF4, generates
one batch, and measures the speed. It must end with **`VERIFY OK`**. It stops early, with a
plain-language reason, if:

- the card, driver or `bitsandbytes` cannot run NF4;
- the token is missing, or the licence was not accepted (HTTP 401/403);
- the GPU runs out of memory while loading or generating;
- the model loaded at anything other than NF4.

If either model does not print `VERIFY OK`, **stop here**. Nothing later will work, and a run that
fell back to a different precision would not be the pre-registered measurement.

### A9. Get the time budget for this card

```bash
uv run python scripts/run_p11.py budget
```

This prints how many GPU-hours each step will take **on this card**, using the speeds `verify-gpu`
just measured. Use it to plan your bookings. It also prints the `--max-frames` number that section D
needs.

### Keep long runs alive: use `tmux`

Generation runs for hours. If your SSH connection drops, a plain terminal kills the run (it can be
resumed, but time is lost). Start a `tmux` session first:

```bash
tmux new -s p11
```

Detach with `Ctrl-b` then `d`. Re-attach later with `tmux attach -t p11`. If `tmux` is not
installed, prefix a command with `nohup` and end it with `> run.log 2>&1 &`, then watch it with
`tail -f run.log`.

---

## B. The smoke run: must PASS before anything else

The smoke run pushes a tiny but **multi-rendering** slice of the experiment through the entire
pipeline on this GPU:

- one confirmatory pair's full grid (age × class, 12 cells) × 3 prompts × 3 samples = 108
  generations, plus 40 CrowS-Pairs items;
- the two **E4** arms (temperature 1.0, and greedy) on their first 8 prompts: 32 generations, using
  the exact prompt files and decoding settings the full E4 run will use;
- then scoring → metrics → **FDR correction** + **B3 permutation test** → **claim gate** →
  **one figure**.

It exists because the correction step, and the B3 permutation test beside it, have never run on
real model output. That is where a bug hid through four earlier phases, and finding a bug here
costs about an hour instead of days of GPU time.

```bash
uv run python scripts/run_p11.py smoke --model llama31_8b_instruct
```

It takes roughly **40–75 minutes**: most of that is the 140 generations, plus a few minutes of
scoring. Everything it writes goes under `data/smoke/`; it cannot touch the real results.

### PASS criterion

The run checks this itself and prints `SMOKE PASS` or `SMOKE FAIL`, one line per clause. **It
passes only if all of these hold:**

| # | Clause | What it proves |
|---|---|---|
| 1 | Every stage exited cleanly | The scripts chain on this machine |
| 2 | An **estimable** confirmatory pair (`class x age`) | The data is multi-rendering, so there is something to test |
| 3 | **FDR correction reached**: family terms carry a `q`, a `family_size` and the `fdr_bh` correction | `attach_wald_p → correct` ran on real output |
| 4 | **ClaimGate ran on every corrected term** | The significance gate ran. Its verdict here is `underpowered`, and **that is the expected, correct verdict** at smoke size |
| 5 | The CrowS-Pairs members are **excused as not estimable** | The second paradigm joins the family without crashing it |
| 6 | **B3 permutation p on every corrected term** | The independent permutation test ran on the item-level scores and its p reached every claim row |
| 7 | **E4 arms generated under their own profiles** | Each E4 arm produced the expected count, and every response RECORDS the right decoding profile, temperature (1.0 / 0.0) and seeds |
| 8 | The emergence figure is written as PDF + SVG, with its traceability audit clean | The last link: a figure drawn from corrected output |

**A run that prints `NOTHING TO CORRECT` is a FAIL**, however cleanly it exits. It means the
correction path was never reached, which is exactly what this run exists to rule out.

The result is also saved to `data/smoke/smoke_result.json`. Keep it: it records the stages, their
timings and the verdict.

> **Do not proceed to the full run until this passes.**

If a stage fails, fix nothing yourself. Send the project lead the terminal output and
`data/smoke/smoke_result.json`. Once the cause is fixed, you can restart from the stage that failed
instead of from scratch, for example:

```bash
uv run python scripts/run_p11.py smoke --model llama31_8b_instruct --from-stage scoring
```

The stages are `prompts`, `inference`, `e4`, `scoring`, `metrics`, `stats`, `tables`, `figure`,
`verdict`.

---

## C. The full sweep: E1, CrowS-Pairs, then E4

Run it in this order: **E1** (C2) → **CrowS-Pairs** (C2) → **E4** (C7) → then the red-team pass
(section D).

E1 is the main experiment: all 2,970 prompts (27 prompts × 110 identity cells), each sampled
**4 times** at the pre-registered seeds `20260908 20260909 20260910 20260911`. That is
**11,880 generations per model**, which gives **108 renderings per cell**, exactly the
pre-registered `min_cell_n`. It runs once for each of the two models, plus a short CrowS-Pairs pass
per model.

### C1. Before each session

```bash
cd mapping-the-margins
git status
export HF_TOKEN=<your Hugging Face token>
export HF_HOME=<the same cache directory as before>/hf-cache
```

`git status` should list no modified files. New run records under `results/run_manifests/` are
expected and fine. The orchestrator **refuses to start** if tracked files are modified, or if the
code is older than the pre-registration tag `prereg-v1` (section F).

### C2. Run it

One model at a time, in `tmux`:

```bash
uv run python scripts/run_p11.py e1 --model llama31_8b_instruct
uv run python scripts/run_p11.py e1 --model mistral7b_instruct_v03
uv run python scripts/run_p11.py crows --model llama31_8b_instruct
uv run python scripts/run_p11.py crows --model mistral7b_instruct_v03
```

Each `run_p11.py` step prints, then runs, exactly one underlying command. For the record, these are
the underlying commands:

```bash
uv run python scripts/run_inference.py --model llama31_8b_instruct --prompts data/prompts/e1_sampled/prompts_v1.jsonl --decoding sampled_v1 --seeds 20260908 20260909 20260910 20260911 --batch-size 8 --out-dir data/responses/e1 --run-name e1-llama31_8b_instruct --resume
uv run python scripts/run_inference.py --model mistral7b_instruct_v03 --prompts data/prompts/e1_sampled/prompts_v1.jsonl --decoding sampled_v1 --seeds 20260908 20260909 20260910 20260911 --batch-size 8 --out-dir data/responses/e1 --run-name e1-mistral7b_instruct_v03 --resume
uv run python scripts/run_inference.py --model llama31_8b_instruct --prompts data/prompts/crows_pll_v1.jsonl --decoding pll_v1 --batch-size 8 --out-dir data/responses/crows --run-name crows-llama31_8b_instruct --extractive-set crows_pll_v1 --resume
uv run python scripts/run_inference.py --model mistral7b_instruct_v03 --prompts data/prompts/crows_pll_v1.jsonl --decoding pll_v1 --batch-size 8 --out-dir data/responses/crows --run-name crows-mistral7b_instruct_v03 --extractive-set crows_pll_v1 --resume
```

Use the `run_p11.py` form. It adds the safety checks: a clean tree, the pre-registration tag, and
NF4.

### C3. Resuming after an interruption

**Re-run exactly the same command.** That is the whole procedure.

```bash
uv run python scripts/run_p11.py e1 --model llama31_8b_instruct
```

Every generation step runs with `--resume`, and each arm and model writes to its own folder, so the
command picks up the run file it was writing, skips every response already on disk, repairs a
half-written last line if the session was killed mid-write, and carries on. Nothing is duplicated
and nothing is lost.

Three things must stay the same between sessions. The orchestrator fixes them for you, so only
worry about this if you have been typing the underlying commands by hand:

- the **seeds** (changing them raises an error);
- the **batch size** (a resume at a different batch size is refused, because batch size is part of
  the determinism record);
- the **model**.

### C4. Checking progress

```bash
wc -l data/responses/e1/llama31_8b_instruct/*.jsonl
```

Done means **11,880** lines for E1 on each model. The CrowS pass has one line per CrowS item. Each
step also prints a live `done/total` counter.

### C5. What gets recorded

Before generating anything, each run writes a manifest under `results/run_manifests/`. It records:

- the git commit, and whether the tree was modified;
- every package version;
- the **actual GPU model, NVIDIA driver version, CUDA and cuDNN versions, torch version and
  torch's BLAS**;
- the thread settings and the batch size;
- the quantization that was requested and the one that actually loaded.

To look:

```bash
grep -A20 '"environment"' results/run_manifests/e1-llama31_8b_instruct-*.json
```

Reproduction on different hardware is **metric-level, within a reported tolerance, not
bit-identical**. That is the project's stated determinism scope, and it is why the hardware is
recorded.

### C6. How long it takes

Estimates from the measured Kaggle **T4** pilot (NF4, one prompt at a time). **Your card will
differ. Use the numbers `run_p11.py budget` printed in step A9**, which are measured on your card at
the real batch size of 8.

| Step | Generations | T4 estimate |
|---|---|---|
| E1, Llama-3.1-8B | 11,880 | **~46 GPU-hours** |
| E1, Mistral-7B | 11,880 | **~49 GPU-hours** |
| CrowS-Pairs, per model | ~1,350 scored pairs, no generation | minutes |
| E4 (C7), Llama-3.1-8B | 1,728 | **~6.7 GPU-hours** |
| E4 (C7), Mistral-7B | 1,728 | **~7.1 GPU-hours** |
| Red-team (section D), both models | see D | ~20 GPU-hours (budgeted) |

At T4 speed each model's E1 is about **2 days of continuous GPU**, or about 4 half-day bookings;
E4 adds under 7 hours per model. A modern 24 GB card at batch 8 should be several times faster;
trust the measured `budget` output over this table.

### C7. E4: the sensitivity arms (after E1 and CrowS-Pairs)

E4 asks how much the results move when only the decoding **temperature** changes. It uses one
identity pair, **class × race** (16 cells × 27 prompts = 432 prompts), through **both** models, at
three temperatures:

| Temperature | Decoding profile | Seeds | Generated here? |
|---|---|---|---|
| 0.0 (greedy) | `greedy_v1` | **1** (`20260908`). Greedy output is the same for every seed | yes: 432 per model |
| 0.7 | `sampled_v1` | E1's 4 seeds | **no**: reused from E1, which already covers these prompts |
| 1.0 | `sampled_t10_v1` (identical to `sampled_v1` except the temperature) | 3 (`20260908 20260909 20260910`) | yes: 1,296 per model |

So E4 **generates 1,728 responses per model** and regenerates nothing E1 already made.

```bash
uv run python scripts/run_p11.py e4 --model llama31_8b_instruct
uv run python scripts/run_p11.py e4 --model mistral7b_instruct_v03
```

Each `e4` step runs the temperature-1.0 arm, then the greedy arm. It is resumable exactly like E1
(section C3): re-run the identical command. Done means **1,296** lines in
`data/responses/e4_t10/<model>/` and **432** in `data/responses/e4_greedy/<model>/`.

The underlying commands:

```bash
uv run python scripts/run_inference.py --model llama31_8b_instruct --prompts data/prompts/e4_sampled_t10/prompts_v1.jsonl --decoding sampled_t10_v1 --seeds 20260908 20260909 20260910 --batch-size 8 --out-dir data/responses/e4_t10 --run-name e4-t10-llama31_8b_instruct --resume
uv run python scripts/run_inference.py --model llama31_8b_instruct --prompts data/prompts/e4_greedy/prompts_v1.jsonl --decoding greedy_v1 --seeds 20260908 --batch-size 8 --out-dir data/responses/e4_greedy --run-name e4-greedy-llama31_8b_instruct --resume
uv run python scripts/run_inference.py --model mistral7b_instruct_v03 --prompts data/prompts/e4_sampled_t10/prompts_v1.jsonl --decoding sampled_t10_v1 --seeds 20260908 20260909 20260910 --batch-size 8 --out-dir data/responses/e4_t10 --run-name e4-t10-mistral7b_instruct_v03 --resume
uv run python scripts/run_inference.py --model mistral7b_instruct_v03 --prompts data/prompts/e4_greedy/prompts_v1.jsonl --decoding greedy_v1 --seeds 20260908 --batch-size 8 --out-dir data/responses/e4_greedy --run-name e4-greedy-mistral7b_instruct_v03 --resume
```

The two E4 prompt files were built by `prepare` (step A7).

---

## D. The red-team pass (descriptive only)

The red-team arm uses one language model to write new prompts aimed at specific identity
combinations. By a recorded project decision it is **descriptive**:

- it runs at a small, budgeted size;
- it is `powered=False`;
- its results are reported with confidence intervals and **no significance claim** (no p, no q).

That is a scope choice driven by the GPU budget, not a statement that red-team effects are small.

```bash
uv run python scripts/run_p11.py redteam-elicit
uv run python scripts/run_p11.py budget
uv run python scripts/run_p11.py redteam-prompts --max-frames <N>
uv run python scripts/run_p11.py redteam-audit --model llama31_8b_instruct
uv run python scripts/run_p11.py redteam-audit --model mistral7b_instruct_v03
```

1. **`redteam-elicit`**: Llama writes candidate prompt frames (~1,440 generations, ~5.6 h at T4
   speed). They are then filtered, deduplicated and checked for identity leakage. Resumable: re-run
   the same command.
2. **`budget`**: prints `--max-frames <N>`, the number of frames the red-team budget buys on this
   card. Use that number, not a guess.
3. **`redteam-prompts --max-frames <N>`**: picks N frames spread evenly across the three prompt
   styles and the six identity pairs, and renders each into all 80 target cells.
4. **`redteam-audit`**: runs each model on those prompts, with **one** sample each. Resumable.

The underlying generation commands:

```bash
uv run python scripts/run_inference.py --model llama31_8b_instruct --prompts data/redteam/elicitation_prompts.jsonl --decoding redteam_v1 --seeds 20260921 20260922 20260923 20260924 20260925 20260926 --batch-size 8 --out-dir data/responses/redteam_elicitation --run-name redteam-elicit --resume
uv run python scripts/build_redteam_prompts.py --redteam cell_targeted_v1 --max-frames <N>
uv run python scripts/run_inference.py --model llama31_8b_instruct --prompts data/prompts/cell_targeted_v1.jsonl --decoding sampled_v1 --seeds 20260908 --batch-size 8 --out-dir data/responses/redteam --run-name redteam-llama31_8b_instruct --resume
uv run python scripts/run_inference.py --model mistral7b_instruct_v03 --prompts data/prompts/cell_targeted_v1.jsonl --decoding sampled_v1 --seeds 20260908 --batch-size 8 --out-dir data/responses/redteam --run-name redteam-mistral7b_instruct_v03 --resume
```

Later, the project lead hand-labels a sample of frames (`data/redteam/on_target_sample.jsonl`) for
whether they hit their target. That needs no GPU and is not part of your session.

---

## E. Processing the results (only generation needs the GPU)

Everything after generation runs on an ordinary CPU, so it can run on the lab machine or on a
laptop afterwards.

| Stage | GPU? | Notes |
|---|---|---|
| Generation (sections B–D) | **Yes** | The only GPU step |
| Scoring | No (faster with one) | Two small classifiers score each response. Needs `--extra inference`, including on a laptop. Can take hours on a laptop CPU for ~24,000 responses |
| Metrics, statistics, tables, figures | No | Core environment only |

### E1. On the lab machine

```bash
uv run python scripts/run_p11.py process
```

If the red-team arm has not been run yet, process E1 and CrowS-Pairs alone:

```bash
uv run python scripts/run_p11.py process --no-redteam
```

`process` runs these, in order:

1. scoring of every response file (per arm, per model);
2. metrics (per arm, per model);
3. **one** statistics run that FDR-corrects both models' E1 and CrowS-Pairs terms together as one
   family, and runs the **B3 permutation test** on every family term from the item-level scores
   (CPU work: minutes, not hours);
4. the result tables, with the red-team rows added as descriptive, carrying no p and no q;
5. the five headline figures.

The underlying commands are in the command list at the end of this document.

**`process` does not touch the E4 responses.** They are generated in this session and travel with
`data/responses/`. The E4 sensitivity analysis runs afterwards: it needs no GPU and is not part of
this procedure. Just make sure `data/responses/e4_t10/` and `data/responses/e4_greedy/` are in what
you hand back (they are, if you copy `data/responses`).

If it stops partway, re-run from the stage that failed:

```bash
uv run python scripts/run_p11.py process --from-stage metrics
```

The stages are `scoring`, `metrics`, `stats`, `tables`, `figures`.

### E2. Or on a laptop

Pack the outputs on the lab machine:

```bash
tar czf p11_outputs.tar.gz data/responses data/redteam data/prompts data/p11 results/run_manifests
```

Copy `p11_outputs.tar.gz` to the laptop by any means, but **not into a OneDrive or Dropbox
folder**. Then, in a checkout of the **same commit**:

```bash
tar xzf p11_outputs.tar.gz
uv sync --locked --extra dev --extra inference
uv run python scripts/run_p11.py process
```

### E3. Hand-back

Give the project lead:

- `results/run_manifests/`, `results/tables/`, `results/paper_numbers.json` and `figures/`;
- `data/smoke/smoke_result.json`;
- `data/p11/gpu_check_*.json`;
- your notes: the dates, the commit, and anything that stopped.

The project lead commits them. **Do not commit or push from the lab machine.**

---

## F. If something goes wrong

**The standing rule: never unpin a dependency, never `pip install`, never substitute a different
model, never edit a tracked file.** Stop and send the project lead the command you ran and the last
~50 lines of output. Every item below either has a safe fix you may apply or says to stop.

### The session was interrupted: SSH dropped, the machine rebooted, or the booking ended

Re-run **the identical command** (section C3). The run continues where it stopped. You do not
need to delete or move anything.

### `verify-gpu` says bitsandbytes is not installed, or torch sees no CUDA device

- On **Windows**, NF4 is not available in this project at all. Use a Linux machine (or WSL2 with
  GPU support, if IT can set that up).
- On **Linux**, re-run `uv sync --locked --extra dev --extra inference`. If `nvidia-smi` fails, the
  driver is broken; that needs the machine's administrator.

**Stop** until `verify-gpu` prints `VERIFY OK`.

### The 8B model won't load

- **Out of memory while loading:** the card has too little free VRAM. Check whether someone else is
  using the GPU (`nvidia-smi`). The model needs roughly 7–9 GB at NF4. Stop if the card cannot
  provide that.
- **Any other load error:** stop and report it with the full output.

### HTTP 401 / 403, "gated repo", or `UNREACHABLE` from `pin_models.py`

- `echo $HF_TOKEN` prints nothing: set it (step A5). You must do this in every new terminal.
- The token was revoked or has no read scope: make a new one.
- **403 on `meta-llama/...`**: this account has not accepted the Llama 3.1 licence. Accept it on
  the model's Hugging Face page (approval can take a while), then re-run `verify-gpu`.

**Never** substitute an ungated or different model: the model pair is pre-registered.

### Out of memory **during** generation (`CUDA out of memory`)

- Close anything else using the GPU (`nvidia-smi` lists the processes), then re-run the identical
  command.
- If it still fails, the batch size of 8 is too large for this card. Batch size is fixed in
  `configs/experiment/p11_v1.yaml` and recorded with the run. Changing it is a **decision for the
  project lead**, and it must be made before a model's run starts, never partway through: a resumed
  run at a different batch size is refused.

**Stop and report.**

### `STOP: a CONFIRMATORY model loaded at quant='none' but 'nf4' was requested`

The model fell back from 4-bit NF4 to full precision, almost always because bitsandbytes or CUDA
is not working. The run stopped **before** generating anything, which is intended. Go to the
bitsandbytes entry above. Do not pass `--allow-quant-fallback`: it exists for debugging and makes
the run not the pre-registered measurement.

### `STOP: the git worktree has uncommitted changes`

Run `git status`. Something tracked was modified. Do not commit it and do not discard it: report
it. (New files under `results/run_manifests/` do not count and are expected.)

### `STOP: HEAD does not descend from 'prereg-v1'`, or the tag could not be found

```bash
git fetch --tags
git checkout main
git pull
```

Then re-run. No full run may execute on code older than the pre-registration.

### `RuntimeError: ... does not have a deterministic implementation`

The project switches on torch's deterministic mode, and some GPU operation refused it. **Stop and
report.** Do not turn deterministic mode off.

### `process` says a folder "holds 2 run files"

A generation step was started twice as separate runs, for example with a hand-typed command that
omitted `--resume`. Scoring needs exactly one run file per model per arm. Do not delete either
file: **move** the incomplete one out of `data/` (for example `mkdir -p ~/p11_aside && mv <file>
~/p11_aside/`) and tell the project lead which one you moved and why.

### The disk is full

`df -h .` and `du -sh $HF_HOME data`. Free space outside the repository, or point `HF_HOME` at a
bigger disk and re-run. Never delete files under `data/responses/`.

### The smoke run prints `SMOKE FAIL`

Do not start section C. Send `data/smoke/smoke_result.json` and the terminal output. The failing
clause names what did not happen. `NOTHING TO CORRECT` means the multi-rendering data never reached
the correction step: exactly the failure the smoke run exists to catch before days of GPU time are
spent.

---

## The full command list, in order

Setup (section A):

```bash
nvidia-smi
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
git clone https://github.com/breenu/mapping-the-margins.git
cd mapping-the-margins
git checkout main
git pull
uv sync --locked --extra dev --extra inference
export HF_TOKEN=<your Hugging Face token>
export HF_HOME=<a directory on a disk with at least 60 GB free>/hf-cache
uv run python scripts/pin_models.py --models llama31_8b_instruct mistral7b_instruct_v03
uv run python scripts/run_p11.py prepare
uv run python scripts/run_p11.py verify-gpu --model llama31_8b_instruct
uv run python scripts/run_p11.py verify-gpu --model mistral7b_instruct_v03
uv run python scripts/run_p11.py budget
```

Smoke run (section B). **Must print `SMOKE PASS` before anything below:**

```bash
uv run python scripts/run_p11.py smoke --model llama31_8b_instruct
```

Full E1 and CrowS-Pairs (section C). Each step is resumable by re-running it:

```bash
uv run python scripts/run_p11.py e1 --model llama31_8b_instruct
uv run python scripts/run_p11.py e1 --model mistral7b_instruct_v03
uv run python scripts/run_p11.py crows --model llama31_8b_instruct
uv run python scripts/run_p11.py crows --model mistral7b_instruct_v03
```

E4 sensitivity arms (section C7), after E1 and CrowS-Pairs. Resumable:

```bash
uv run python scripts/run_p11.py e4 --model llama31_8b_instruct
uv run python scripts/run_p11.py e4 --model mistral7b_instruct_v03
```

Red-team, descriptive (section D):

```bash
uv run python scripts/run_p11.py redteam-elicit
uv run python scripts/run_p11.py budget
uv run python scripts/run_p11.py redteam-prompts --max-frames <N>
uv run python scripts/run_p11.py redteam-audit --model llama31_8b_instruct
uv run python scripts/run_p11.py redteam-audit --model mistral7b_instruct_v03
```

Processing (section E). CPU is fine:

```bash
uv run python scripts/run_p11.py process
```

What `prepare` runs underneath:

```bash
uv run python scripts/fetch_lexicons.py --fetch-pinned
uv run python scripts/fetch_crows_pairs.py --fetch-pinned
uv run python scripts/build_crows_items.py --no-manifest --no-paper-numbers
uv run python scripts/build_crows_prompts.py --no-manifest
uv run python scripts/build_prompts.py --prompt-set open_gen_v1 --decoding sampled_v1 --out-dir data/prompts/e1_sampled --run-name build-prompts-e1-sampled
uv run python scripts/build_prompts.py --prompt-set open_gen_v1 --decoding sampled_t10_v1 --pair socioeconomic_class race_ethnicity --out-dir data/prompts/e4_sampled_t10 --run-name build-prompts-e4-t10
uv run python scripts/build_prompts.py --prompt-set open_gen_v1 --decoding greedy_v1 --pair socioeconomic_class race_ethnicity --out-dir data/prompts/e4_greedy --run-name build-prompts-e4-greedy
```

What `redteam-elicit` runs underneath, besides the generation command in section D:

```bash
uv run python scripts/build_redteam_elicitation.py --redteam cell_targeted_v1
uv run python scripts/build_redteam_frames.py --responses <the run file in data/responses/redteam_elicitation/llama31_8b_instruct/> --redteam cell_targeted_v1
```

What `process` runs underneath. `<run file>` is the single `.jsonl` in that folder, which
`run_p11.py` finds for you:

```bash
uv run python scripts/run_scoring.py --responses <run file in data/responses/e1/llama31_8b_instruct/> --prompts data/prompts/e1_sampled/prompts_v1.jsonl --benchmark open_gen_templates --out-dir data/scoring/e1/llama31_8b_instruct --run-name scoring-e1-llama31_8b_instruct
uv run python scripts/run_scoring.py --responses <run file in data/responses/e1/mistral7b_instruct_v03/> --prompts data/prompts/e1_sampled/prompts_v1.jsonl --benchmark open_gen_templates --out-dir data/scoring/e1/mistral7b_instruct_v03 --run-name scoring-e1-mistral7b_instruct_v03
uv run python scripts/run_scoring.py --responses <run file in data/responses/crows/llama31_8b_instruct/> --prompts data/prompts/crows_pll_v1.jsonl --scoring crows_pll_v1 --benchmark crows_pairs --out-dir data/scoring/crows/llama31_8b_instruct --run-name scoring-crows-llama31_8b_instruct
uv run python scripts/run_scoring.py --responses <run file in data/responses/crows/mistral7b_instruct_v03/> --prompts data/prompts/crows_pll_v1.jsonl --scoring crows_pll_v1 --benchmark crows_pairs --out-dir data/scoring/crows/mistral7b_instruct_v03 --run-name scoring-crows-mistral7b_instruct_v03
uv run python scripts/run_scoring.py --responses <run file in data/responses/redteam/llama31_8b_instruct/> --prompts data/prompts/cell_targeted_v1.jsonl --benchmark redteam --out-dir data/scoring/redteam/llama31_8b_instruct --run-name scoring-redteam-llama31_8b_instruct
uv run python scripts/run_scoring.py --responses <run file in data/responses/redteam/mistral7b_instruct_v03/> --prompts data/prompts/cell_targeted_v1.jsonl --benchmark redteam --out-dir data/scoring/redteam/mistral7b_instruct_v03 --run-name scoring-redteam-mistral7b_instruct_v03
uv run python scripts/run_metrics.py --scores data/scoring/e1/llama31_8b_instruct/scores.parquet --drift data/scoring/e1/llama31_8b_instruct/drift_control.parquet --benchmark open_gen_templates --out-dir data/metrics/e1/llama31_8b_instruct --run-name metrics-e1-llama31_8b_instruct
uv run python scripts/run_metrics.py --scores data/scoring/e1/mistral7b_instruct_v03/scores.parquet --drift data/scoring/e1/mistral7b_instruct_v03/drift_control.parquet --benchmark open_gen_templates --out-dir data/metrics/e1/mistral7b_instruct_v03 --run-name metrics-e1-mistral7b_instruct_v03
uv run python scripts/run_metrics.py --scores data/scoring/crows/llama31_8b_instruct/scores.parquet --drift data/scoring/crows/llama31_8b_instruct/drift_control.parquet --benchmark crows_pairs --out-dir data/metrics/crows/llama31_8b_instruct --run-name metrics-crows-llama31_8b_instruct
uv run python scripts/run_metrics.py --scores data/scoring/crows/mistral7b_instruct_v03/scores.parquet --drift data/scoring/crows/mistral7b_instruct_v03/drift_control.parquet --benchmark crows_pairs --out-dir data/metrics/crows/mistral7b_instruct_v03 --run-name metrics-crows-mistral7b_instruct_v03
uv run python scripts/run_metrics.py --scores data/scoring/redteam/llama31_8b_instruct/scores.parquet --drift data/scoring/redteam/llama31_8b_instruct/drift_control.parquet --benchmark redteam --out-dir data/metrics/redteam/llama31_8b_instruct --run-name metrics-redteam-llama31_8b_instruct
uv run python scripts/run_metrics.py --scores data/scoring/redteam/mistral7b_instruct_v03/scores.parquet --drift data/scoring/redteam/mistral7b_instruct_v03/drift_control.parquet --benchmark redteam --out-dir data/metrics/redteam/mistral7b_instruct_v03 --run-name metrics-redteam-mistral7b_instruct_v03
uv run python scripts/run_stats.py --metrics-dir data/metrics/e1/llama31_8b_instruct data/metrics/e1/mistral7b_instruct_v03 data/metrics/crows/llama31_8b_instruct data/metrics/crows/mistral7b_instruct_v03 --scores data/scoring/e1/llama31_8b_instruct/scores.parquet data/scoring/e1/mistral7b_instruct_v03/scores.parquet data/scoring/crows/llama31_8b_instruct/scores.parquet data/scoring/crows/mistral7b_instruct_v03/scores.parquet --out-dir data/stats --paper-numbers results/paper_numbers.json --run-name stats-e1
uv run python scripts/build_result_tables.py --stats-dir data/stats --metrics-dir data/metrics/e1/llama31_8b_instruct data/metrics/e1/mistral7b_instruct_v03 data/metrics/crows/llama31_8b_instruct data/metrics/crows/mistral7b_instruct_v03 --descriptive-metrics-dir data/metrics/redteam/llama31_8b_instruct data/metrics/redteam/mistral7b_instruct_v03 --out-dir results/tables
uv run python scripts/make_figures.py --only essential --table-dir results/tables --paper-numbers results/paper_numbers.json --out-dir figures
```

`run_stats.py` deliberately receives **no** red-team directory. `redteam` is not a declared
family source, so the statistics stage would refuse it, and the arm is descriptive by decision.
