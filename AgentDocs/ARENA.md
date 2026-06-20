# ARENA.md

## 1. Purpose

This document describes how to build a local arena for evaluating Chinese Official Mahjong / Botzone-style agents.

The goal is to provide a fast, repeatable, statistically meaningful evaluation pipeline for comparing:

- baseline CNN + SL agent
- modified neural agent, e.g. ResNet / Res2Net / Transformer agent
- rule-based agent
- previous best checkpoint
- distilled / reranked / self-play-tuned agent

The local arena is intended to complement, not replace, Botzone online testing.

Recommended usage:

```text
Local arena:
  - fast debugging
  - illegal action detection
  - checkpoint comparison
  - large-scale statistical evaluation

Botzone online:
  - final compatibility smoke test
  - real platform validation
  - final submission testing
```

If Botzone does not provide convenient fully automated batch evaluation, the local arena should be the main evaluation method.

---

## 2. Evaluation Philosophy

Mahjong has very high variance. A single game, or even dozens of games, is not enough to judge model strength.

The arena should therefore use:

```text
fixed seeds
seat rotation
duplicate matches
large number of games
paired score statistics
confidence intervals
illegal / crash / timeout tracking
```

The core comparison should not be simple win rate. Recommended primary metric:

```text
mean_score_diff = candidate_score - baseline_reference_score
```

Secondary metrics:

```text
average rank
rank-1 rate
rank-4 rate
hu rate
deal-in rate
average score
average fan when winning
illegal action rate
crash rate
timeout rate
average decision time
```

A candidate model should only replace the current best model if it is stronger and stable.

---

## 3. Recommended Evaluation Levels

### Level 0: Offline Supervised Learning Metrics

Run after each training epoch or checkpoint.

Metrics:

```text
policy loss
top-1 action accuracy
top-3 action accuracy
top-5 action accuracy
illegal action rate before mask
illegal action rate after mask
value loss
fan-route F1 / accuracy
danger AUC / F1
belief loss, if used
```

These are proxy metrics. They are useful for debugging, but they do not directly prove stronger gameplay.

Minimum requirement:

```text
after-mask illegal action rate = 0
```

---

### Level 1: Local Smoke Test

Purpose:

```text
make sure the agent can play complete games
catch illegal actions
catch crashes
catch severe latency problems
```

Recommended scale:

```text
20 seeds ? 4 seat rotations = 80 games
```

Pass conditions:

```text
illegal = 0
crash = 0
timeout = 0
mean_score_diff is not obviously worse than baseline
average decision time is acceptable
```

---

### Level 2: Development Arena

Purpose:

```text
compare checkpoints
choose between model architectures
compare reranking strategies
compare auxiliary heads
```

Recommended scale:

```text
200 seeds ? 4 seat rotations = 800 games
```

Run this for candidate models that pass smoke test.

---

### Level 3: Release Arena

Purpose:

```text
final model selection
statistical confidence
submission decision
```

Recommended scale:

```text
1000 seeds ? 4 seat rotations = 4000 games or more
```

Use this before Botzone submission.

---

### Level 4: Botzone Online Smoke Test

Purpose:

```text
validate final zip package
validate Botzone JSON I/O
validate actual platform compatibility
```

Do not rely on a small number of online games for statistical model selection.

---

## 4. Recommended Local Arena Backend

Recommended backend:

```text
ccr-cheng/botzone-mahjong-environment
```

Reason:

```text
- Python environment
- close to Botzone interaction style
- suitable for local simulation
- usable for large-scale arena evaluation
- easier to integrate with PyTorch agents than online Botzone
```

The arena should run agents in-process whenever possible.

Recommended:

```text
Python process starts once
  ?
load environment
  ?
load all agents
  ?
load model checkpoint once
  ?
run many games
  ?
write JSONL results
```

Avoid this for large-scale evaluation:

```text
for every decision:
  start python bot.zip
  feed JSON
  wait stdout
  terminate process
```

That mode is useful for Botzone compatibility testing, but too slow for arena evaluation.

---

## 5. Suggested Directory Structure

```text
eval/
  arena/
    match.py
    run_arena.py
    run_round_robin.py
    stats.py
    seed.py
    timeout_runner.py
    report.py

  agents/
    random_agent.py
    rule_agent.py
    baseline_agent.py
    torch_agent.py
    zip_agent.py

  configs/
    smoke.yaml
    dev.yaml
    release.yaml

  seeds/
    smoke_20.jsonl
    dev_200.jsonl
    release_1000.jsonl

  results/
    smoke/
    dev/
    release/

third_party/
  botzone-mahjong-environment/

checkpoints/
  baseline_cnn/
    model.pt
    config.yaml
    action_mapping.json
    obs_config.json

  candidate_res2_transformer/
    model.pt
    config.yaml
    action_mapping.json
    obs_config.json
```

Important:

```text
Each checkpoint must be bundled with:
  - model config
  - observation config
  - action mapping
  - normalization settings, if any
```

Do not evaluate a checkpoint with mismatched action mapping.

---

## 6. Environment Setup

Clone the environment:

```bash
mkdir -p third_party
cd third_party
git clone https://github.com/ccr-cheng/botzone-mahjong-environment.git
```

Add it to `PYTHONPATH`:

```bash
export PYTHONPATH=$PWD/third_party/botzone-mahjong-environment:$PYTHONPATH
```

Install or build the Mahjong fan calculator if required by the environment:

```bash
cd third_party/botzone-mahjong-environment/fan-calculator-usage/Mahjong-GB-Python
python setup.py install
```

Run environment tests:

```bash
cd third_party/botzone-mahjong-environment
python test_mahjong.py
python test_bot.py
```

If these tests fail, fix the environment before integrating neural agents.

---

## 7. Agent Interface

All local arena agents should implement a common interface:

```python
class Agent:
    name: str

    def action(self, obs: dict):
        """Return an environment-compatible Action."""
        ...
```

Recommended agent types:

```text
RandomAgent:
  sanity check only

RuleAgent:
  stable non-neural baseline

BaselineAgent:
  original CNN + SL model

TorchAgent:
  current neural model checkpoint

ZipAgent:
  optional Botzone zip compatibility wrapper
```

The main arena should use `TorchAgent` in-process, not `ZipAgent`, for performance.

---

## 8. Torch Agent Design

A neural agent should perform:

```text
1. receive env observation
2. encode observation
3. build legal action list / action mask
4. run model forward
5. apply action mask
6. optionally run top-k afterstate reranking
7. convert selected action id to environment Action
8. return Action
```

Skeleton:

```python
import torch

class TorchMahjongAgent:
    def __init__(self, name, model, device="cpu"):
        self.name = name
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def action(self, obs: dict):
        legal_actions = build_legal_actions(obs)
        if not legal_actions:
            return fallback_action(obs)

        x = encode_observation(obs).to(self.device)
        logits = self.model(x)
        masked_logits = mask_logits(logits, legal_actions)

        if USE_RERANKER:
            action_id = rerank_topk(obs, masked_logits, legal_actions)
        else:
            action_id = int(torch.argmax(masked_logits, dim=-1).item())

        return action_id_to_env_action(action_id, obs, legal_actions)
```

Strong recommendation:

```text
Reuse the same observation encoder, action mapping, and action mask used during training.
```

Do not implement a separate arena-only action mapping unless absolutely necessary.

---

## 9. Fallback Action Policy

Every agent should have a safe fallback.

Fallback behavior:

```text
if Hu is legal:
  Hu
elif response phase:
  PASS
elif discard phase:
  discard lowest-value legal tile using simple rule heuristic
else:
  PASS
```

Purpose:

```text
- prevent one model crash from killing the whole arena job
- record crash/timeout while allowing the match batch to continue
- detect stability separately from playing strength
```

All fallback uses must be logged.

---

## 10. Match Protocol

### 10.1 One-vs-Three

Most useful for comparing candidate against a fixed baseline.

For each seed:

```text
Game 1: candidate, baseline,  baseline,  baseline
Game 2: baseline,  candidate, baseline,  baseline
Game 3: baseline,  baseline,  candidate, baseline
Game 4: baseline,  baseline,  baseline,  candidate
```

This is called duplicate seat rotation.

Primary statistic per seed:

```text
seed_diff = mean(candidate_score - mean(other_baseline_scores)) over 4 rotated games
```

Then compute mean and confidence interval over seeds.

---

### 10.2 Two-vs-Two

Useful when one-vs-three seems biased by opponent homogeneity.

Example:

```text
candidate, candidate, baseline, baseline
```

Rotate seats across games.

Metric:

```text
mean(candidate_scores) - mean(baseline_scores)
```

---

### 10.3 Round Robin

Useful for final release evaluation.

Example pool:

```text
candidate
previous_best
baseline_cnn
rule_bot
```

Run all meaningful permutations or a fixed balanced schedule.

Metrics:

```text
average score
average rank
rank distribution
pairwise score difference
illegal / crash / timeout rate
```

---

## 11. Match Runner Skeleton

```python
# eval/arena/match.py

import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class GameResult:
    seed: int
    players: List[str]
    scores: Dict[int, float]
    ranks: Dict[int, int]
    illegal: bool
    crash: bool
    timeout: bool
    num_steps: int
    avg_decision_ms: Dict[str, float]
    extra: Dict[str, Any]


def play_game(seed: int, agents: List[Any], agent_names: List[str]) -> GameResult:
    # Replace this import with the actual environment import.
    from mahjong_env.core import Mahjong

    env = Mahjong(random_seed=seed)
    env.init()

    decision_time = {name: 0.0 for name in agent_names}
    decision_count = {name: 0 for name in agent_names}

    crash = False
    timeout = False
    illegal = False
    num_steps = 0

    while not env.done:
        actions = []

        for pid in range(4):
            obs = env.player_obs(pid)
            agent = agents[pid]
            name = agent_names[pid]

            t0 = time.perf_counter()
            try:
                action = agent.action(obs)
            except Exception:
                crash = True
                traceback.print_exc()
                action = fallback_action(obs)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            decision_time[name] += elapsed_ms
            decision_count[name] += 1

            actions.append(action)

        env.step(actions)
        num_steps += 1

        if num_steps > 500:
            crash = True
            break

    # Adapt these fields to the environment's actual result format.
    scores = getattr(env, "rewards", {}) or {}

    avg_decision_ms = {
        name: decision_time[name] / max(1, decision_count[name])
        for name in decision_time
    }

    # Recommended: detect illegal through environment flags if available.
    # If unavailable, approximate by known illegal-score pattern or env logs.
    illegal = detect_illegal(env, scores)

    return GameResult(
        seed=seed,
        players=agent_names,
        scores=scores,
        ranks=compute_ranks(scores),
        illegal=illegal,
        crash=crash,
        timeout=timeout,
        num_steps=num_steps,
        avg_decision_ms=avg_decision_ms,
        extra={
            "fans": getattr(env, "fans", None),
        },
    )
```

---

## 12. Duplicate Seat-Rotation Runner

```python
# eval/arena/run_arena.py

import json
from pathlib import Path

from eval.arena.match import play_game


def run_one_vs_three(candidate_factory, baseline_factory, seeds, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for seed in seeds:
            for candidate_seat in range(4):
                agents = []
                names = []

                for pid in range(4):
                    if pid == candidate_seat:
                        agents.append(candidate_factory())
                        names.append("candidate")
                    else:
                        agents.append(baseline_factory())
                        names.append("baseline")

                result = play_game(seed, agents, names)

                row = {
                    "seed": seed,
                    "candidate_seat": candidate_seat,
                    "players": result.players,
                    "scores": result.scores,
                    "ranks": result.ranks,
                    "illegal": result.illegal,
                    "crash": result.crash,
                    "timeout": result.timeout,
                    "num_steps": result.num_steps,
                    "avg_decision_ms": result.avg_decision_ms,
                    "extra": result.extra,
                }

                f.write(json.dumps(row, ensure_ascii=False) + "\n")
```

---

## 13. Result Format

Save one JSON object per game.

Recommended JSONL schema:

```json
{
  "seed": 12345,
  "candidate_seat": 2,
  "players": ["baseline", "baseline", "candidate", "baseline"],
  "scores": {"0": -8, "1": -8, "2": 24, "3": -8},
  "ranks": {"0": 3, "1": 2, "2": 1, "3": 4},
  "illegal": false,
  "crash": false,
  "timeout": false,
  "num_steps": 71,
  "avg_decision_ms": {
    "candidate": 6.2,
    "baseline": 1.1
  },
  "extra": {
    "fans": [[24, "???"]]
  }
}
```

Do not only print results to stdout. Always save JSONL.

---

## 14. Statistics

### 14.1 Primary Metric

For one-vs-three:

```text
candidate_diff = candidate_score - mean(baseline_scores)
```

Aggregate over the four seat rotations for the same seed, then compute statistics over seeds.

Do not treat all rotated games as fully independent samples.

---

### 14.2 Bootstrap Confidence Interval

```python
import numpy as np


def bootstrap_ci(values, n_boot=10000, alpha=0.05):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    means = []

    for _ in range(n_boot):
        sample = np.random.choice(values, size=n, replace=True)
        means.append(sample.mean())

    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)
```

---

### 14.3 Model Selection Rule

A candidate is considered a strong improvement if:

```text
mean_score_diff > 0
95% CI lower bound > 0
illegal = 0
crash = 0
timeout = 0
rank-4 rate does not increase significantly
deal-in rate does not increase significantly, if tracked
```

A candidate is suspicious if:

```text
mean_score_diff > 0 but 95% CI crosses 0
or rank-4 rate increases
or deal-in rate increases sharply
```

A candidate should be rejected if:

```text
mean_score_diff < 0
or illegal > 0
or crash > 0
or timeout > 0 during release arena
```

---

## 15. Suggested Commands

Smoke evaluation:

```bash
python eval/arena/run_arena.py \
  --mode one_vs_three \
  --candidate checkpoints/candidate/model.pt \
  --baseline checkpoints/baseline_cnn/model.pt \
  --seeds eval/seeds/smoke_20.jsonl \
  --out eval/results/smoke/candidate_vs_baseline.jsonl

python eval/arena/stats.py \
  --input eval/results/smoke/candidate_vs_baseline.jsonl
```

Development evaluation:

```bash
python eval/arena/run_arena.py \
  --mode one_vs_three \
  --candidate checkpoints/candidate/model.pt \
  --baseline checkpoints/baseline_cnn/model.pt \
  --seeds eval/seeds/dev_200.jsonl \
  --out eval/results/dev/candidate_vs_baseline.jsonl

python eval/arena/stats.py \
  --input eval/results/dev/candidate_vs_baseline.jsonl \
  --bootstrap 10000
```

Release evaluation:

```bash
python eval/arena/run_round_robin.py \
  --agents candidate,previous_best,baseline_cnn,rule_bot \
  --seeds eval/seeds/release_1000.jsonl \
  --out eval/results/release/round_robin.jsonl

python eval/arena/report.py \
  --input eval/results/release/round_robin.jsonl \
  --out eval/results/release/report.md
```

---

## 16. Performance Expectations

Local arena performance depends on:

```text
model size
CPU vs GPU inference
afterstate top-k reranking
number of environment workers
whether agents run in-process
whether zip/subprocess mode is used
```

Recommended rule:

```text
local machine:
  debug: 4-20 games
  smoke: 80 games
  small dev: 200-800 games if speed is acceptable

cloud server:
  full dev: 800 games
  release: 4000+ games
  self-play data generation
```

Benchmark before large runs:

```python
import time

t0 = time.perf_counter()
# run 20 games
elapsed = time.perf_counter() - t0

print("sec per game:", elapsed / 20)
print("estimated 800 games minutes:", elapsed / 20 * 800 / 60)
print("estimated 4000 games hours:", elapsed / 20 * 4000 / 3600)
```

Rough interpretation:

```text
< 1 sec/game:
  local development arena is easy

1-3 sec/game:
  800 games is acceptable locally; 4000 games better overnight or cloud

> 5 sec/game:
  optimize agent or run on cloud
```

---

## 17. Local vs Cloud Recommendation

Given the workflow:

```text
local development
cloud training
similar Python environment except torch backend
```

Recommended split:

```text
Local:
  - environment installation
  - agent wrapper debugging
  - 4-20 game debug
  - 80 game smoke test
  - Botzone zip I/O test

Cloud:
  - 800 game dev arena
  - 4000+ game release arena
  - self-play generation
  - teacher model evaluation
  - expensive afterstate reranking evaluation
```

If the local machine has a GPU and model inference is fast, development arena can also run locally.

If the model uses BI-V150 / GPU-specific checkpoints, run large arena on cloud to avoid device mismatch.

---

## 18. Parallelism Strategy

Avoid starting too many workers if each worker loads a large neural model.

Recommended:

```text
CPU-only rule agents:
  4-8 workers depending on local CPU

single GPU neural agent:
  1-4 workers
  or one shared GPU inference service

BI-V150 cloud with 6 CPU cores:
  4-6 environment workers max
```

Do not run:

```text
16+ workers, each loading a full PyTorch model copy
```

This may waste memory and reduce throughput.

---

## 19. Botzone Zip Compatibility Test

This is separate from the main arena.

The final submission should be tested as a zip package:

```bash
python candidate.zip < tests/sample_request_001.json
python candidate.zip < tests/sample_request_hu.json
python candidate.zip < tests/sample_request_chi_peng_gang.json
```

Check:

```text
stdout is valid JSON
stdout contains exactly one response field
no debug text appears in stdout
response is legal
model loads correctly
decision latency is acceptable
```

Debug output should go to stderr, not stdout.

---

## 20. Common Failure Modes

### 20.1 Action Mapping Mismatch

Symptom:

```text
model loads, but plays nonsense or illegal actions
```

Fix:

```text
bind action_mapping.json to every checkpoint
assert action space size at load time
unit test action_id_to_env_action
```

---

### 20.2 Observation Encoding Mismatch

Symptom:

```text
offline validation looks good, arena performance is terrible
```

Fix:

```text
bind obs_config.json to every checkpoint
use the same encoder code for training and arena
add encoder snapshot tests
```

---

### 20.3 Masking Bug

Symptom:

```text
illegal action rate > 0
```

Fix:

```text
unit test legal action generation
compare environment legal actions and model mask
fail fast if selected action is not legal
```

---

### 20.4 Slow Evaluation

Symptom:

```text
arena takes many hours for 800 games
```

Fix:

```text
run agents in-process
load model once
reduce afterstate top-k
batch neural inference if possible
move release arena to cloud
profile environment vs model time
```

---

### 20.5 Botzone Works Locally but Fails Online

Possible causes:

```text
missing dependency in zip
wrong entry point
printing logs to stdout
model file too large
path assumptions
different Python version
slow cold start
```

Fix:

```text
minimize dependencies
put __main__.py at zip root
write logs to stderr
use relative paths inside zip
run python candidate.zip locally before submission
```

---

## 21. Minimal Implementation Checklist

### Phase 1: Basic Environment

```text
[ ] clone botzone-mahjong-environment
[ ] install fan calculator
[ ] run test_mahjong.py
[ ] run test_bot.py
[ ] run random vs random locally
```

### Phase 2: Agent Wrappers

```text
[ ] implement common Agent interface
[ ] implement RandomAgent
[ ] implement RuleAgent or wrapper for baseline rule bot
[ ] implement BaselineAgent
[ ] implement TorchAgent
[ ] implement fallback_action
```

### Phase 3: Match Runner

```text
[ ] implement play_game(seed, agents)
[ ] log per-game JSONL
[ ] record illegal/crash/timeout
[ ] record decision time
[ ] record scores and ranks
```

### Phase 4: Seat Rotation

```text
[ ] implement one-vs-three duplicate match
[ ] implement two-vs-two, optional
[ ] implement round robin, optional
```

### Phase 5: Statistics

```text
[ ] compute mean_score_diff
[ ] aggregate by seed
[ ] compute bootstrap 95% CI
[ ] compute average rank
[ ] compute rank distribution
[ ] compute illegal/crash/timeout counts
```

### Phase 6: Evaluation Pipeline

```text
[ ] smoke: 20 seeds ? 4 rotations
[ ] dev: 200 seeds ? 4 rotations
[ ] release: 1000 seeds ? 4 rotations
[ ] generate markdown report
```

### Phase 7: Botzone Compatibility

```text
[ ] create candidate.zip
[ ] test python candidate.zip < sample_request.json
[ ] verify stdout JSON format
[ ] verify no debug logs in stdout
[ ] verify cold-start latency
```

---

## 22. Recommended Decision Policy

For each new model checkpoint:

```text
1. Run offline validation.
2. If after-mask illegal action rate is not zero, reject.
3. Run 80-game local smoke test.
4. If illegal/crash/timeout occurs, reject or fix.
5. Run 800-game development arena.
6. If mean_score_diff is promising, keep candidate.
7. Run 4000+ game release arena for final candidates.
8. Submit only if release arena is stable and Botzone zip test passes.
```

Recommended replacement rule:

```text
Replace current best only if:
  - release mean_score_diff > 0
  - 95% CI lower bound > 0
  - illegal/crash/timeout = 0
  - rank-4 rate does not increase
  - Botzone zip smoke test passes
```

---

## 23. Summary

The local arena should be the main model-selection tool.

The most practical setup is:

```text
in-process Python arena
+ Botzone-like Mahjong environment
+ fixed seeds
+ seat rotation
+ JSONL logging
+ bootstrap confidence intervals
+ separate zip compatibility test
```

Recommended workload split:

```text
local machine:
  debug and smoke tests

cloud server:
  large development arena
  release arena
  self-play generation
```

This evaluation setup is essential because supervised validation accuracy alone cannot reliably predict competition performance in Chinese Official Mahjong.
