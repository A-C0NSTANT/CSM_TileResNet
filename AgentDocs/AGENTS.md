# AGENTS.md

## 1. Project Purpose

This project modifies and evaluates the baseline Chinese Official Mahjong / ???? intelligent agent provided by the AI Lab assignment. The baseline already includes a CNN + supervised learning implementation and an interactive agent interface. The goal is to improve the baseline under limited and heterogeneous compute resources while keeping the agent compatible with the local evaluation environment and possible Botzone-style interaction.

Primary target:

- Build a stronger Mahjong policy network than the baseline CNN model.
- Preserve 100% legal action output via rule-based action masking.
- Improve decision quality using multi-task learning, opponent/danger modeling, fan-route awareness, and optional afterstate value reranking.
- Keep the final model small and reliable enough for local evaluation and deployment.

Recommended project name:

> Hybrid Mahjong Agent: Supervised Policy Network with Belief-Aware Fan-Route Afterstate Reranking

Short name:

> BAF-AI: Belief-Aware Fan-route Mahjong Agent

---

## 2. High-Level Strategy

Use supervised learning as the main training method. Do not rely on large-scale reinforcement learning or LLM-based decision making as the main solution.

The recommended system is a hybrid model:

```text
Rule engine / legal action generator
  ?
State encoder
  ?
ResNet or Res2Net tile encoder
  ?
History Transformer
  ?
Multi-task heads
  - policy head
  - value head
  - fan-route head
  - danger / opponent modeling head
  - optional belief head
  ?
Top-K legal action candidates
  ?
Afterstate value + rule-based reranking
  ?
Final action
```

The core philosophy is:

- Let the neural network propose strong candidate actions.
- Let the rule engine guarantee legality.
- Let fan-route, danger, shanten, effective tiles, and afterstate value reduce short-sighted mistakes.

---

## 3. Model Architecture

### 3.1 Input Features

The state encoder should use structured Mahjong features rather than treating the state as only an image.

Recommended feature groups:

```text
Self hand:
  - 34 tile counts
  - concealed hand counts
  - open melds
  - current drawn tile if available

Public information:
  - discard piles of all players
  - exposed melds of all players
  - visible tile counts
  - remaining visible-estimated tile counts

Global information:
  - current player
  - dealer / zhuang
  - seat wind
  - prevailing wind
  - round index
  - turn index
  - scores if available
  - action phase: self turn / response to discard / kong response / win response

History information:
  - recent N actions
  - player id
  - action type
  - tile
  - target player if any
  - relative position
```

Keep hidden information out of inference inputs. Hidden hands may be used only as auxiliary training labels if available from replay data.

### 3.2 Public State / Tile Encoder

Recommended choices:

```text
Option A: ResNet encoder
  - 4 to 8 residual blocks for student model
  - 8 to 12 residual blocks for teacher model

Option B: Res2Net-like encoder
  - More distinctive than plain ResNet
  - Useful for multi-scale tile pattern extraction

Option C: Tile-token encoder
  - 34 tile tokens
  - tile embedding + MLP / Transformer
```

For the first working version, use ResNet because it is easier to integrate with the baseline CNN implementation.

### 3.3 History Transformer

Encode recent actions using a compact Transformer encoder.

Recommended student setting:

```text
history length: 80 to 120 actions
hidden dim: 128 or 256
num layers: 2 to 4
num heads: 4 or 8
dropout: 0.1
```

The history encoder should help identify:

- Opponent attacking tendency.
- Possible tenpai state.
- Dangerous suits and tiles.
- Possible fan routes such as ???, ???, ???, ??, ???, etc.

### 3.4 Main Heads

The model should include the following output heads.

#### Policy head

The primary supervised learning target.

```text
Input: fused state representation
Output: logits over all predefined actions
Loss: masked cross entropy against expert action
```

Always apply legal action mask before action selection.

#### Value head

Predict final reward, score, rank, or win/loss outcome depending on available labels.

```text
Loss: MSE or Huber loss
```

This is useful for afterstate reranking and optional offline RL / AWR.

#### Fan-route head

Predict possible fan routes or whether the current state has a viable 8-fan path.

Possible labels:

```text
- multi-label 81 fan types, if available
- simplified fan categories
- estimated current max fan potential
- probability of valid 8-fan win route
```

This is important because Chinese Official Mahjong requires at least 8 fan to win. Pure shanten optimization is not enough.

#### Danger / opponent modeling head

Predict tile danger and opponent state from public information.

Possible outputs:

```text
discard_danger: 34 values
opponent_tenpai_prob: 3 values
opponent_fan_route: 3 ? fan_category_count
opponent_hand_belief: optional 3 ? 34 values
```

Use only public information as input. Hidden information may be used as labels during training but never as inference input.

#### Afterstate value head

For each legal candidate action, simulate the afterstate and estimate its value.

```text
Q(s, a) = AfterstateValue(encode(afterstate), action_embedding)
```

This can be used only for Top-K candidates to control computation.

Recommended setting:

```text
Top-K candidates: 4 to 8 for student
Top-K candidates: 8 to 16 for teacher/offline evaluation
```

---

## 4. Final Decision Logic

Do not simply return the argmax policy action. Use masked Top-K reranking.

Recommended formula:

```text
score(action) =
    alpha * log_policy_prob(action)
  + beta  * afterstate_value(action)
  + gamma * fan_route_score(action)
  + delta * effective_tiles_score(action)
  - eps   * discard_danger(action)
  - zeta  * bad_meld_penalty(action)
```

Suggested initial weights:

```text
alpha = 1.0
beta  = 0.3
gamma = 0.2
delta = 0.1
eps   = 0.2
zeta  = 0.1
```

Tune these weights using validation games and local self-play.

Hard rules:

- If a winning action is legal and valid under 8-fan rules, strongly prefer winning.
- Never output illegal actions.
- During response phase, pass is a valid fallback.
- During discard phase, use a rule-based discard fallback if model inference fails.

---

## 5. Training Environments

### 5.1 BI-V150 resource

Available resource from the provided platform:

```text
GPU: BI-V150 ? 1
Available GPU memory: 32GB
CPU: 6 cores
System disk: 200GB
RAM: 32GB
```

Recommended role:

```text
Primary training card
Long-running experiments
Teacher model training
Self-play data generation
Multi-seed training
Distillation experiments
```

BI-V150 should be used as the main development and training resource if its PyTorch environment is stable. It is suitable for this project because the planned Mahjong models are small to medium scale, typically 10M to 100M parameters.

Expected limitations:

```text
CPU has only 6 cores, so self-play environment simulation may bottleneck.
RAM is 32GB, so keep dataloaders and replay buffers compact.
Disk is 200GB, so do not save massive JSON logs.
```

### 5.2 Ascend 910B resource

Given budget:

```text
Budget: about 800 RMB
Approximate price: 19 RMB/hour
Estimated available time: about 42 hours
```

Recommended role:

```text
Final compatibility validation
Final fine-tuning if needed
NPU-specific inference test
Submission model verification
```

Use Ascend 910B carefully because its software stack differs from GPU-like environments. Avoid custom CUDA-only code or unsupported dynamic operations if the model must later run on Ascend.

### 5.3 Environment portability rules

Prefer common operations:

```text
Embedding
Linear
Conv2d
LayerNorm
BatchNorm
ReLU / GELU
Softmax
Masked fill
Matmul
Standard MultiheadAttention
```

Avoid or minimize:

```text
custom CUDA kernels
complex dynamic shape logic
large scatter/gather tricks
third-party ops without Ascend support
very new PyTorch features
```

Always test a saved checkpoint on the final target environment before trusting training results.

---

## 6. Training Plan

### Stage 0: Baseline reproduction

Goal: reproduce the provided CNN + SL baseline.

Required checks:

```text
train loss decreases
validation loss is reasonable
top-1 / top-k action accuracy can be computed
illegal action rate is zero after masking
agent can complete full games locally
```

Do not start architecture changes before the baseline is measurable.

### Stage 1: ResNet supervised policy

Replace the baseline CNN backbone with ResNet while keeping the rest of the pipeline unchanged.

Compare:

```text
CNN baseline
vs
ResNet policy
```

Metrics:

```text
top-1 accuracy
top-3 / top-5 accuracy
validation loss
illegal action rate
local game score / win rate
```

### Stage 2: Add History Transformer

Add a history encoder for recent action sequences.

Compare:

```text
ResNet only
vs
ResNet + History Transformer
```

### Stage 3: Add multi-task heads

Add value, fan-route, and danger heads.

Suggested loss:

```text
L_total =
    L_policy
  + 0.2 * L_value
  + 0.1 * L_fan_route
  + 0.1 * L_danger
  + 0.1 * L_shanten_or_waits
```

Adjust weights based on validation stability.

### Stage 4: Add afterstate reranking

Use the policy head to select Top-K legal candidates, then rerank them using afterstate value and rule features.

Compare:

```text
policy argmax
vs
top-k reranking
```

Important test cases:

```text
winning decision
chi / peng / gang decision
dangerous discard
high-fan route preservation
late-game defense
```

### Stage 5: Teacher-student distillation

Train a larger teacher model on BI-V150, then distill into a smaller student model.

Teacher suggestion:

```text
parameters: 50M to 100M
hidden dim: 256 or 384
Transformer layers: 4 to 6
ResNet blocks: 8 to 12
all auxiliary heads enabled
```

Student suggestion:

```text
parameters: 10M to 30M
hidden dim: 128 or 256
Transformer layers: 2 to 4
ResNet blocks: 4 to 8
keep policy/value/fan/danger heads
```

Distillation loss:

```text
L_distill =
    CE(student_policy, expert_action)
  + KL(student_policy, teacher_policy)
  + MSE(student_value, teacher_value)
  + BCE(student_fan, teacher_fan)
  + BCE(student_danger, teacher_danger)
```

### Stage 6: Optional self-play and AWR

Only do this after supervised learning is strong.

Recommended approach:

```text
1. Generate self-play data using the SL model.
2. Train value on final rewards.
3. Compute advantage estimates.
4. Reweight behavior cloning samples.
5. Fine-tune policy/value heads only.
```

Avoid full-scale PPO as the main path unless there is much more compute and engineering time.

---

## 7. Expected Timeline

Approximate time using BI-V150 as the main training card.

```text
Minimal runnable version: 1 to 2 days
Full supervised multi-task version: 3 to 5 days
Competition-oriented version with teacher/student and self-play: 1.5 to 3 weeks
```

Recommended detailed schedule:

```text
Day 1-2:
  - reproduce baseline
  - run small ResNet model
  - verify action mask and agent interaction

Day 3-5:
  - add History Transformer
  - add fan/danger/value heads
  - train full supervised model

Day 6-8:
  - implement Top-K reranker
  - add afterstate features
  - run local game evaluation

Day 9-14:
  - train teacher
  - distill student
  - run multi-seed comparison

After Day 15:
  - optional self-play
  - optional AWR/offline RL fine-tuning
  - final deployment optimization
```

---

## 8. Evaluation Protocol

Always evaluate both supervised metrics and game-level metrics.

### 8.1 Supervised metrics

```text
top-1 action accuracy
top-3 / top-5 action accuracy
masked cross entropy
per-action-type accuracy
  - discard
  - chi
  - peng
  - gang
  - hu
  - pass
illegal action rate
```

Top-k accuracy is important because the expert action may not be the only reasonable action.

### 8.2 Auxiliary metrics

```text
value prediction MSE / correlation
fan-route multi-label F1
8-fan viability accuracy
danger prediction AUC / F1
tenpai probability calibration
shanten / wait prediction accuracy if used
```

### 8.3 Game-level metrics

```text
win rate
average score
average rank
deal-in rate
valid win rate
average fan when winning
time per decision
crash / timeout count
```

Run enough games to reduce noise. Mahjong game-level results are high variance.

### 8.4 Ablations

At minimum, compare:

```text
CNN baseline
ResNet only
ResNet + History Transformer
ResNet + Transformer + auxiliary heads
Full model + Top-K reranking
Distilled student if available
```

---

## 9. Recommended Repository Structure

Suggested local structure:

```text
project_root/
  AGENTS.md
  README.md
  configs/
    baseline_cnn.yaml
    resnet_policy.yaml
    baf_student.yaml
    baf_teacher.yaml
  data/
    raw/
    processed/
    selfplay/
  mahjong/
    env/
    rules/
    feature_encoder.py
    action_space.py
    action_mask.py
  models/
    baseline_cnn.py
    resnet_encoder.py
    history_transformer.py
    baf_model.py
    heads.py
    afterstate_value.py
  training/
    train_sl.py
    train_multitask.py
    distill.py
    train_value.py
    awr_finetune.py
  evaluation/
    eval_supervised.py
    eval_selfplay.py
    eval_ablation.py
    debug_cases.py
  agents/
    baseline_agent.py
    baf_agent.py
    fallback_policy.py
  checkpoints/
  logs/
  scripts/
    preprocess.sh
    train_student.sh
    train_teacher.sh
    eval.sh
```

---

## 10. Implementation Guidelines

### 10.1 Action mask

Action mask correctness is mandatory.

Rules:

```text
Never compute final action without applying legal mask.
Use -inf or a very negative value for illegal logits.
Keep a fallback action if all logits become invalid.
Unit-test every action type.
```

### 10.2 Fallback policy

The agent must never crash or output illegal actions.

Recommended fallback:

```text
if legal hu and valid fan:
    return hu
elif response phase:
    return pass if legal
else:
    return heuristic_discard_lowest_value_tile()
```

### 10.3 Logging

Log enough information for debugging but avoid huge files.

Recommended logs:

```text
model config
random seed
training loss
validation loss
top-k accuracy
per-action accuracy
illegal action count
sample failed cases
checkpoint path
```

Avoid saving full uncompressed replay JSON for massive self-play runs.

### 10.4 Reproducibility

Use fixed seeds:

```text
python random
numpy
pytorch
environment shuffle
```

Save:

```text
config file
commit hash if using git
checkpoint
training log
validation split metadata
```

---

## 11. Local Agent Modification Checklist

Before modifying baseline:

```text
[ ] Baseline training runs successfully.
[ ] Baseline agent can complete a full local game.
[ ] Legal action mask is understood.
[ ] Action space mapping is documented.
[ ] Evaluation script exists.
```

After modifying model:

```text
[ ] Forward pass works on one batch.
[ ] Overfit test works on a tiny dataset.
[ ] Masked logits contain no illegal selected action.
[ ] Validation metrics are logged.
[ ] Agent inference works without training-time labels.
[ ] Model checkpoint can be loaded by the agent.
[ ] Fallback policy works when model errors occur.
```

Before final evaluation:

```text
[ ] Compare against CNN baseline.
[ ] Run ablation tests.
[ ] Run local self-play or arena evaluation.
[ ] Measure decision latency.
[ ] Test edge cases: hu, chi, peng, gang, pass, last tile, response phase.
[ ] Test on the final deployment environment if different from training environment.
```

---

## 12. Practical Defaults

Student model defaults:

```text
resnet_blocks: 6
transformer_layers: 2
hidden_dim: 256
num_heads: 4
dropout: 0.1
batch_size: 512 to 2048 depending on memory
mixed_precision: true
optimizer: AdamW
learning_rate: 1e-4 to 3e-4
weight_decay: 1e-4
max_history_len: 100
top_k_rerank: 4 or 8
```

Teacher model defaults:

```text
resnet_blocks: 10
transformer_layers: 4 to 6
hidden_dim: 256 or 384
num_heads: 8
dropout: 0.1
batch_size: tune by memory
mixed_precision: true
optimizer: AdamW
learning_rate: 1e-4
top_k_rerank: 8 or 16
```

Loss weights initial values:

```text
policy: 1.0
value: 0.2
fan_route: 0.1
danger: 0.1
belief: 0.1 if available
shanten_or_waits: 0.1 if available
```

---

## 13. What Not To Do First

Avoid these as first steps:

```text
Do not start with full-scale PPO from scratch.
Do not use an LLM as the core Mahjong decision maker.
Do not build a very large Transformer before validating the data pipeline.
Do not skip baseline reproduction.
Do not trust top-1 accuracy alone.
Do not submit a model without fallback and illegal-action tests.
```

---

## 14. Final Recommended Path

The strongest practical path for this project is:

```text
1. Reproduce CNN + SL baseline.
2. Replace CNN with ResNet / Res2Net.
3. Add History Transformer.
4. Add fan-route, value, and danger auxiliary heads.
5. Add Top-K action reranking with afterstate value and rule features.
6. Train a larger teacher on BI-V150.
7. Distill a smaller student for deployment.
8. Verify the student on Ascend 910B or the final target environment.
9. Optionally add self-play + AWR fine-tuning after supervised learning is strong.
```

This provides a distinctive architecture while remaining realistic under the available compute resources.
