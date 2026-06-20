# Local Arena

Minimal local Mahjong arena scaffolding for the third-party `mahjong_env` backend.

## Environment

Use the isolated conda environment created for arena work:

```bash
conda activate csmj-arena
```

Or run commands without activating:

```bash
conda run -n csmj-arena python eval/arena/check_env.py
```

## Dependency Check

```bash
conda run -n csmj-arena python eval/arena/check_env.py
```

This checks:

- `third_party/botzone-mahjong-environment`
- `numpy`
- `yaml`
- `MahjongGB.MahjongFanCalculator`
- `mahjong_env.core.Mahjong`
- `mahjong_env.base_bot.RandomMahjongBot`

On Windows, `PyMahjongGB` requires Microsoft C++ Build Tools. After installing it, run:

```bash
conda run -n csmj-arena pip install PyMahjongGB
```

## Random Smoke Test

After `check_env.py` passes:

```bash
conda run -n csmj-arena python eval/arena/random_vs_random.py \
  --games 4 \
  --seed-begin 0
```

The runner writes one JSON object per game. By default, result files are written under `eval/results/YYYY-MM-DD/`.

## Packaged Model Minimum Test

This verifies that the packaged Botzone-style agents can start, load their checkpoints, handle `Wind`, `Deal`, and one `Draw` request, and emit keep-running sentinels.

```bash
conda run -n csmj-arena python eval/arena/minimum_model_test.py
```

The finite input intentionally ends after the draw request. The agent may print an `EOFError` traceback on stderr after all expected responses are captured; this does not fail the test unless the expected responses or sentinels are missing.

## One-vs-Three Arena

Run a candidate model against three baseline copies with duplicate seat rotation:

```bash
conda run -n csmj-arena python eval/arena/run_arena.py \
  --candidate-model ResNet_Policy_v1 \
  --baseline-model CNN_Baseline \
  --num-seeds 20 \
  --seed-begin 0
```

`--candidate-model` and `--baseline-model` may be either names under `eval/models/` or explicit model directory paths. The runner expects each model directory to contain:

- `__main__.py`
- `checkpoint/*.pkl`

The runner writes one JSON object per game and a summary JSON next to the output file. By default, it writes to `eval/results/YYYY-MM-DD/<candidate>_vs_<baseline>_<games>games.jsonl`. Use `--out` and `--summary-out` to override the location.

## Parallel Runs

`run_arena.py` supports seed-level parallelism with `--workers`. Each worker runs a batch of seeds and keeps its own reusable agent pool. With the default `--reuse-agents`, each worker keeps one candidate process and three baseline processes alive across games, so the approximate number of model subprocesses is `workers * 4`.

For small debug runs, keep the default single worker:

```bash
conda run -n csmj-arena python eval/arena/run_arena.py \
  --candidate-model ResNet_Policy_v1 \
  --baseline-model CNN_Baseline \
  --num-seeds 4 \
  --seed-begin 0 \
  --workers 1
```

For larger runs, use a small worker count to improve CPU utilization without creating too many model subprocesses:

```bash
conda run -n csmj-arena python eval/arena/run_arena.py \
  --candidate-model ResNet_Policy_v1 \
  --baseline-model CNN_Baseline \
  --num-seeds 256 \
  --seed-begin 0 \
  --workers 2
```

Use `--no-reuse-agents` only when debugging agent startup/reset behavior. It starts fresh model subprocesses for every game and is much slower.

## Reversibility

Remove the isolated conda environment:

```bash
conda env remove -n csmj-arena
```

Remove the third-party backend:

```bash
rmdir /s /q third_party\botzone-mahjong-environment
```
