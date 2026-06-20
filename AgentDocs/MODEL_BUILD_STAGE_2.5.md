# Model Build Stage 2.5: RARN Public V2 Summary

????????????????? Stage 2.5 ?????????????????????????/?????arena ???????????

??????? AI agent?????????? History Transformer / public action sequence encoder / auxiliary heads ????

## 1. ??????

Stage 2.5 ????? Stage 2 ? `rarn_public_v1` ???? public state modeling?

???????

```text
rarn_public_v2
```

?????

```text
1. rarn_public_v2 ?? structured public encoder + tile-level mid fusion?
2. public features ????? flat vector ??? MLP ?????? tile grid ??? RARN ?????
3. rarn_public_v2 ??? validation ? arena ?????? rarn_public_v1 ? plain ResNet?
4. ????? value head / rank head / transformer / history encoder?
5. ?????????????????????? history/action transformer?????????? observation?
```

## 2. ?????????

?? observation ??? `feature.py` ?? `FeatureAgent`?

?? `FeatureAgent._obs()` ???

```python
{
    'observation': 6 x 4 x 9,
    'action_mask': 235,
    'public': 442,
}
```

?????

```text
observation: 6 x 4 x 9
action_mask: 235
public:      442
```

?????????

```text
Pass:   0
Hu:     1
Play:   2   - 35    34 actions
Chi:    36  - 98    63 actions
Peng:   99  - 132   34 actions
Gang:   133 - 166   34 actions
AnGang: 167 - 200   34 actions
BuGang: 201 - 234   34 actions
```

?????????? `action_mask`?

?? masking ???

```python
def apply_action_mask(action_logits, action_mask):
    action_mask = action_mask.float()
    return action_logits.masked_fill(action_mask <= 0, torch.finfo(action_logits.dtype).min)
```

## 3. Public Feature Layout

Public feature ????

```text
DEFAULT_PUBLIC_SIZE = 442
```

Stage 2.5 ??????????????

```text
public[0:408]   -> tile-level public features
public[408:435] -> opponent-level public features
public[435:442] -> game-stage public features
```

?????? `model.py`?

```python
PUBLIC_TILE_LEVEL_SIZE = 408
PUBLIC_OPPONENT_SIZE = 27
PUBLIC_STAGE_SIZE = 7
PUBLIC_TILE_CHANNELS = 12
```

### 3.1 Tile-Level Public Features

Tile-level feature ??

```text
12 x 34 = 408
```

? `FeatureAgent.TILE_LIST` ? 34 ???????

```text
W1-W9
T1-T9
B1-B9
F1-F4
J1-J3
```

12 ? tile-level channel ???

```text
visible_count[34]
remaining_count_est[34]
self_hand_count[34]
self_discard_count[34]
opponent_discard_count_sum[34]
meld_visible_count[34]
discard_count_by_relative_player[3, 34]
meld_count_by_relative_player[3, 34]
```

???????? `feature.py::_public_features()`?

```python
for values in (
    self._visible_counts,
    self._remaining_counts,
    self_counts,
    self.discard_counts[0],
    self._opponent_discard_sum,
    self._clipped_meld_visible,
):
    public[offset:offset + NUM_TILE_TYPES] = values
    public[offset:offset + NUM_TILE_TYPES] *= 0.25
    offset += NUM_TILE_TYPES

public[offset:offset + 3 * NUM_TILE_TYPES] = self._clipped_opponent_discard.reshape(-1)
public[offset:offset + 3 * NUM_TILE_TYPES] *= 0.25
offset += 3 * NUM_TILE_TYPES

public[offset:offset + 3 * NUM_TILE_TYPES] = self._clipped_meld_by_player.reshape(-1)
public[offset:offset + 3 * NUM_TILE_TYPES] *= 0.25
offset += 3 * NUM_TILE_TYPES
```

???

```text
visible_count = shown_counts + self_counts
```

??? `meld_visible_count` ?????? `visible_count`?`meld_visible_count` ??? feature?

### 3.2 Opponent-Level Public Features

Opponent-level feature?

```text
3 opponents x 9 = 27
```

?????? 9 ??

```text
meld_count normalized
discard_count normalized
has_open_meld
has_multiple_melds
has_honor_triplet
W suit meld ratio
T suit meld ratio
B suit meld ratio
max suit concentration
```

### 3.3 Game-Stage Public Features

Game-stage feature?

```text
7 dims
```

???

```text
turn_index normalized
wall_remaining normalized
middle bucket
late bucket
self_shanten normalized
self_effective_tiles_count normalized
```

?? shanten / effective tiles ??????????????

```text
PUBLIC_EXACT_SHANTEN = 0
estimate_shanten_fast
estimate_effective_tile_type_count_fast
```

?????????? shanten?

```bash
PUBLIC_EXACT_SHANTEN=1 DATA_DIR=data_public python preprocess.py
```

???????????????? shanten???????????

## 4. ??????

???????

```text
model.py
```

???????

```text
CNNModel
ResNetPolicyModel
RankAwareResNetPolicyModel
RankAwareResNetPublicPolicyModel
RankAwareResNetPublicV2PolicyModel
```

?????

```python
create_model(name)
```

???????

```text
cnn
resnet
rarn
rarn_v2
rarn_public
rarn_public_v2
```

`rarn_public_v2` ???

```text
rarn_public_struct
rarn_public_midfusion
rarn_public_policy_v2
rarn_public_policy_model_v2
rank_aware_resnet_public_v2
rank_aware_resnet_public_policy_model_v2
```

## 5. RARN Public V2 Architecture

????

```text
rarn_public_v2
```

???

```python
RankAwareResNetPublicV2PolicyModel
```

?????

```text
1. ?? RARN_m ??? tile topology ??????
2. ? 442 ? public vector ?? tile / opponent / stage ?? encoder?
3. ? tile-level public features reshape ? 4 x 9 tile grid??? tile ??? RARN ?????
4. ? public state ?????????????? RARN residual blocks ????????
5. ?????????forward(input_dict) -> masked logits 235?
```

?????

```text
observation: B x 6 x 4 x 9
  -> RARN stem
  -> 2 RankAwareResidualBlocks
  -> rarn_tile_feature: B x 128 x 4 x 9

public[0:408]
  -> reshape: B x 12 x 34
  -> public tile grid: B x 12 x 4 x 9
  -> PublicTileEncoder
  -> public_tile_feature: B x 64 x 4 x 9

concat(rarn_tile_feature, public_tile_feature)
  -> 1x1 Conv / BatchNorm / ReLU
  -> fused_tile_feature: B x 128 x 4 x 9

fused_tile_feature
  -> 6 RankAwareResidualBlocks
  -> valid 34 tile flatten
  -> state_feature: B x 512

public[408:435], public[435:442]
  -> GlobalPublicEncoder
  -> global_public_feature: B x 256

concat(state_feature, global_public_feature): B x 768
  -> Linear / LayerNorm / ReLU
  -> policy logits: B x 235
  -> action_mask
```

### 5.1 RARN Backbone

RARN backbone ?????

```text
channels: 128
blocks: 8
pre_fusion_blocks: 2
post_fusion_blocks: 6
state_hidden: 512
```

RARN ????

```text
RankAwareResidualBlock
SuitContextMixer
ValidTileFlatten / flatten_valid_tiles
```

?????

```text
Conv2d(..., kernel_size=(1, 3), padding=(0, 1))
```

?????

```text
Conv2d(..., kernel_size=1)
```

?? tile mask?

```text
4 x 9 grid ? honor row ??????????? mask ? 0?
```

### 5.2 PublicTileEncoder

???

```python
PublicTileEncoder
```

???

```text
B x 12 x 4 x 9
```

???

```text
B x 64 x 4 x 9
```

???

```text
numeric branch:
  Conv2d(12, 64, kernel_size=(1,3), padding=(0,1), bias=False)
  BatchNorm2d(64)
  ReLU
  Conv2d(64, 64, kernel_size=(1,3), padding=(0,1), bias=False)
  BatchNorm2d(64)
  ReLU

honor branch:
  Conv2d(12, 64, kernel_size=1, bias=False)
  BatchNorm2d(64)
  ReLU
  Conv2d(64, 64, kernel_size=1, bias=False)
  BatchNorm2d(64)
  ReLU
```

? encoder ? RARN ? inductive bias ?????

```text
???? rank direction?
???? rank convolution?
honor row ????????? masked?
```

### 5.3 GlobalPublicEncoder

???

```python
GlobalPublicEncoder
```

???

```text
opponent_public: B x 27
stage_public:    B x 7
```

???

```text
global_public_feature: B x 256
```

???

```text
opponent head:
  Linear(27, 128)
  LayerNorm(128)
  ReLU
  Linear(128, 128)
  ReLU

stage head:
  Linear(7, 64)
  LayerNorm(64)
  ReLU
  Linear(64, 64)
  ReLU

fusion:
  Linear(128 + 64, 256)
  LayerNorm(256)
  ReLU
```

?????? FiLM / channel modulation?????? ablation?

```text
global_public_feature -> gamma/beta -> modulate tile_feature
```

? Stage 2.5 ?????????? FiLM?

### 5.4 Tile-Level Mid Fusion

fusion ???

```text
RARN stem + 2 residual blocks ??
```

?????

```python
for block in self._blocks[:self.pre_fusion_blocks]:
    x = block(x, valid_tile_mask)

x = self._tile_fusion(torch.cat([x, public_tile_feature], dim=1)) * valid_tile_mask

for block in self._blocks[self.pre_fusion_blocks:]:
    x = block(x, valid_tile_mask)
```

?????

```text
1. ? 2 ? RARN blocks ??????????
2. public tile feature ?????????????? tile encoding?
3. ? 6 ? RARN blocks ? public-aware tile representation ??????
```

### 5.5 Fusion Head

?? head?

```text
state_feature:         512
global_public_feature: 256
concat:                768

Linear(768, 512)
LayerNorm(512)
ReLU
Linear(512, 235)
action_mask
```

## 6. ???

??????????

| Model | Params | Notes |
|---|---:|---|
| CNN baseline | about 0.73M | baseline CNN |
| ResNet v1 | 2,420,971 | plain ResNet baseline |
| RARN_s / rarn | 2,017,387 | channels 96, blocks 6, hidden 384 |
| RARN_m / rarn_v2 | 3,938,027 | channels 128, blocks 8, hidden 512 |
| RARN_public_v1 | 4,512,491 | RARN_m + flat public MLP |
| RARN_public_v2 | 4,452,651 | structured public encoder + tile-level mid fusion |

???

```text
rarn_public_v2 ?????? rarn_public_v1????? arena ??????
```

## 7. ???????

??????

```text
preprocess.py
```

?????

```text
DATA_DIR/data.txt
```

???????

```text
data/
```

Stage 2 / 2.5 public ???????

```text
data_public/
```

Windows PowerShell ?? public ???

```powershell
$env:DATA_DIR="data_public"
python preprocess.py
```

Linux?

```bash
DATA_DIR=data_public python preprocess.py
```

?? `preprocess.py` ???

```text
1. ???? data.txt?
2. ???? 4 ? FeatureAgent?
3. ???????? request2obs / response2action?
4. ??????????????
5. filterData ?? action_mask.sum() <= 1 ????
6. ?????? .npz?
7. count.json ????????
```

?? `.npz` keys?

```text
obs:     N x 6 x 4 x 9, int8
mask:    N x 235, int8
public:  N x 442, float32
act:     N, int
```

?????

```python
np.stack([x['observation'] for i in range(4) for x in obs[i]])
np.stack([x['action_mask'] for i in range(4) for x in obs[i]])
np.stack([x['public'] for i in range(4) for x in obs[i]])
np.array([x for i in range(4) for x in actions[i]])
```

??????? 0,1,2,3 ?????????????????

???????? value/rank labels ? history labels????? `.npz`??????????????? `data.txt`?

## 8. Dataset ????

Dataset ???

```text
dataset.py
```

???

```python
MahjongGBDataset
```

?????

```text
1. ?? data_dir/count.json?
2. ?? begin/end ? match ???????/????
3. ??? .npz ? obs/mask/act/public ??????
4. __getitem__ ?? obs, mask, act, public?
5. ??????? public key??? 442 ?????
6. Dataset ?? has_public ???
```

?????

```python
return obs, mask, act, public
```

????????????

```python
input_dict = {
    'is_training': is_training,
    'obs': {
        'observation': obs.to(device),
        'action_mask': mask.to(device),
        'public': public.to(device),
    }
}
```

`rarn_public` ? `rarn_public_v2` ??? public ????????? public key???/????????

## 9. ???????

### 9.1 ?? rarn_public_v2

```bash
python supervised.py --model rarn_public_v2 --data-dir data_public --output-dir checkpoints/rarn_public_v2 --epochs 20 --batch-size 2048 --lr 0.0007 --num-workers 2
```

### 9.2 ???? rarn_public_v2

```bash
python evaluation/eval_supervised.py --model rarn_public_v2 --data-dir data_public --checkpoint checkpoints/rarn_public_v2/rarn_public_v2_model_latest.pkl --output-dir evaluation/results/rarn_public_v2 --device auto
```

### 9.3 Arena ??

Arena ???

```text
eval/arena/run_arena.py
```

Stage 2.5 ????????

```text
eval/models/rarn_public_v2/
```

plain ResNet baseline?

```text
eval/models/resnet_policy_v1/
```

6 ? 16 ??????

```text
eval/results/2026-06-16/
```

## 10. ?? Validation ??

????

```text
split_begin: 0.9
split_end:   1.0
num_samples: 589,402
```

Overall metrics?

| Model | Params | Loss | Top1 | Top3 | Top5 | Entropy | Illegal |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN baseline | about 0.73M | 0.7453 | 73.04% | 94.90% | 98.61% | 0.6334 | 0.00% |
| ResNet v1 | 2.42M | 0.7267 | 74.64% | 95.57% | 98.88% | 0.4960 | 0.00% |
| RARN_m / v2 | 3.94M | 0.6972 | 74.97% | 95.47% | 98.70% | 0.5827 | 0.00% |
| RARN_public_v1 | 4.51M | 0.5400 | 82.39% | 97.75% | 99.51% | 0.2727 | 0.00% |
| RARN_public_v2 | 4.45M | 0.3867 | 87.50% | 98.90% | 99.78% | 0.1693 | 0.00% |

rarn_public_v2 ?? rarn_public_v1?

```text
Loss: 0.5400 -> 0.3867   -0.1533
Top1: 82.39% -> 87.50%   +5.11 pp
Top3: 97.75% -> 98.90%   +1.15 pp
Top5: 99.51% -> 99.78%   +0.27 pp
Params: 4.51M -> 4.45M   slightly fewer
Illegal: 0 -> 0
```

Action-type recall?

| Action | RARN_public_v1 Recall | RARN_public_v2 Recall | Change |
|---|---:|---:|---:|
| Pass | 91.40% | 94.52% | +3.12 pp |
| Hu | 100.00% | 100.00% | 0 |
| Play type | 99.95% | 99.97% | +0.02 pp |
| Chi | 81.54% | 85.50% | +3.96 pp |
| Peng | 88.20% | 90.29% | +2.08 pp |
| AnyGang | 86.73% | 88.51% | +1.78 pp |

Play exact top1?

```text
RARN_public_v1: 80.58%
RARN_public_v2: 86.25%
Change: +5.67 pp
```

Legal-count bucket accuracy?

| Bucket | RARN_public_v1 Top1 | RARN_public_v2 Top1 | Change |
|---|---:|---:|---:|
| 2 | 90.01% | 92.98% | +2.96 pp |
| 3-5 | 83.69% | 89.17% | +5.48 pp |
| 6-10 | 78.02% | 86.38% | +8.36 pp |
| >10 | 83.04% | 85.67% | +2.63 pp |

?????? `6-10` ???????? tile-level public fusion ????????????

?? confusion improvements?rarn_public_v2 ?? rarn_public_v1?

```text
Pass -> Chi:      3843 -> 2371   -1472
Pass -> Peng:     2270 -> 1500   -770
Pass -> AnyGang:  168  -> 130    -38
Chi -> Pass:      4440 -> 3483   -957
Peng -> Pass:     2040 -> 1687   -353
AnyGang -> Play:  180  -> 140    -40
Play -> AnyGang:  242  -> 150    -92
```

## 11. Arena ??

Arena baseline?

```text
plain ResNet / resnet_policy_v1
```

6 ? 16 ? rarn_public_v2 vs ResNet?

| Run | Games | Mean Score Diff | 95% CI | Rank1 | Rank4 | Win Rate | Deal-in | Candidate Illegal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| smoke | 128 | +13.44 | [8.21, 18.84] | 42.19% | 17.19% | 39.84% | 14.06% | 0 |
| formal | 1024 | +8.78 | [6.66, 10.98] | 36.33% | 18.95% | 35.16% | 13.28% | 1 |

?? rarn_public_v1 vs ResNet 1024 games?

```text
Mean Score Diff: +3.86
95% CI: [1.84, 5.91]
Rank1: 29.69%
Rank4: 22.85%
Win Rate: 27.54%
Deal-in: 16.21%
```

rarn_public_v2 ?? rarn_public_v1?

```text
Mean Score Diff: +3.86 -> +8.78   +4.92
Rank1: 29.69% -> 36.33%           +6.64 pp
Rank4: 22.85% -> 18.95%           -3.91 pp
Win Rate: 27.54% -> 35.16%        +7.62 pp
Deal-in: 16.21% -> 13.28%         -2.93 pp
```

???

```text
rarn_public_v2 ?????????? arena ???
???? rarn_public_v2 ?? Stage 2.5 ?????
```

## 12. ?????????

?????

```text
__main__.py
```

???????

```python
def obs2response(model, obs):
    model_obs = {
        'observation': torch.from_numpy(np.expand_dims(obs['observation'], 0)),
        'action_mask': torch.from_numpy(np.expand_dims(obs['action_mask'], 0)),
    }
    if 'public' in obs:
        model_obs['public'] = torch.from_numpy(np.expand_dims(obs['public'], 0))
    with torch.no_grad():
        logits = model({'is_training': False, 'obs': model_obs})
    action = logits.detach().numpy().flatten().argmax()
    response = agent.action2response(action)
    return response
```

???????

```text
1. FeatureAgent ?? BotZone / arena request ?????
2. request2obs ?? observation/action_mask/public?
3. ?? forward ?? masked logits?
4. ?? argmax ?? action id?
5. FeatureAgent.action2response(action) ????????
```

?????

```text
Top-K reranking
afterstate value
rule-based danger reranking
temperature sampling
beam search
history transformer
```

### 12.1 BotZone / Arena Request Flow

`__main__.py` ??????

```text
0 seatWind prevalentWind
  -> initialize FeatureAgent
  -> print PASS

1 Deal ...
  -> FeatureAgent.request2obs('Deal ...')
  -> print PASS

2 Draw tile
  -> obs = request2obs('Draw tile')
  -> model action
  -> print HU / PLAY / GANG / BUGANG

3 other player event
  -> update FeatureAgent
  -> if response opportunity, call model
  -> print HU / PASS / GANG / PENG / CHI
```

Chi/Peng ????????

```python
obs = agent.request2obs('Player %d ' % seatWind + response)
response2 = obs2response(model, obs)
print(' '.join([t[0].upper(), *t[1:], response2.split()[-1]]))
agent.request2obs('Player %d Un' % seatWind + response)
```

### 12.2 ??????

`__main__.py::resolve_model_name()` ?????

```text
1. ?????????????????????
2. ?????? MODEL_NAME ?????? MODEL_NAME?
3. ?? checkpoint ??????????
```

????????

```text
rarn_public_v2
rarn_public
rarn_v2 / rarn_m_
rarn
resnet
cnn
```

??????????

```text
MODEL_NAME=rarn_public_v2
```

???? checkpoint ??????

```text
rarn_public_v2
```

????? `/data/model.pkl` ????????????

## 13. ?????????

### 13.1 Entropy ??

rarn_public_v2 ?? entropy ???

```text
avg_entropy: 0.1693
```

???

```text
RARN_m:          0.5827
RARN_public_v1:  0.2727
RARN_public_v2:  0.1693
```

Arena ??????? deal-in ????? deal-in ??????????? Transformer ??????

```text
over-confidence
bad open meld
late-game defense regression
rank4 increase
```

### 13.2 Arena Candidate Illegal

6 ? 16 ? 1024 ? arena?

```text
candidate_illegal_count: 1
candidate_illegal_rate: 0.0009765625
baseline_illegal_count: 7
```

???? illegal ? 0?????????

```text
1. arena / BotZone interaction edge case
2. action2response ??
3. FeatureAgent ????
4. ????????????????
```

???????????????

### 13.3 ?? public features ?????

rarn_public_v2 ????? public summary?????? public action history?

???

```text
recent N actions
action order
discard timing
who called from whom
pass history
late reaction information
opponent tempo
```

????????? History Transformer ??????

## 14. ???? Transformer ??

?? `FeatureAgent` ??????????

```text
self.history
self.packs
self.tileWall
self.shownTiles
self.discard_counts
self.meld_by_player
```

??? `.npz` ???? action sequence?????????????

### 14.1 ?? A???????? history tokens

???? `.npz` key?

```text
history: B x L x D
history_mask: B x L
```

?? token ???

```text
relative_player: 0 self, 1 next, 2 opposite, 3 previous
action_type: draw/play/chi/peng/gang/angang/bugang/hu/pass
tile: 0-33 plus none
source_player: relative id or none
target_player: relative id or none
turn_index bucket
is_response_event
```

?? Transformer student ???

```text
history length: 80-120
hidden dim: 128 or 256
layers: 2-4
heads: 4 or 8
dropout: 0.1
```

???????

```text
history_feature -> concat with state_feature and global_public_feature before final policy head
```

???????? Transformer ???? tile grid??? late fusion ???

### 14.2 ?? B????? `.npz`????? history

?????? `data_public/`????

```text
1. ???? obs/mask/act/public .npz?
2. ? data.txt ?????????
3. ? preprocess.py ??????? history sidecar ???
4. Dataset ?? .npz + sidecar history?
```

sidecar ?????

```text
data_public_history/0_history.npz
  history: N x L x D
  history_mask: N x L
```

???

```text
????? data_public?
?????? history ???
```

???

```text
?????? preprocess ???????????
```

### 14.3 ???????

?????

```text
rarn_public_v2_hist
```

???

```text
observation + public
  -> rarn_public_v2 encoder
  -> state_feature 512
  -> global_public_feature 256

history tokens
  -> History Transformer
  -> history_feature 256

concat 512 + 256 + 256
  -> policy head
  -> logits 235
  -> action_mask
```

?????

```text
1. ??? rarn_public_v2 checkpoint?
2. ??? History Transformer ?? fusion head?
3. ?? freeze RARN/public encoder 1-2 epochs???? history + head?
4. ??? fine-tune?
```

????????

```text
freeze warmup: lr 0.001 for new modules
full fine-tune: lr 0.0003 - 0.0005
```

## 15. ??????

???

```text
? Stage 2.5 ????? RARN_public_v1 ?????????? public state ????????? 442 ??????????? flat MLP?????? policy head ?? RARN state feature ?????? RARN_public_v2 ? public features ????? tile-level?opponent-level ? game-stage ??????? tile-level ? 408 ??????? 12 x 4 x 9 ????????? RARN ????? rank convolution ??? 1x1 branch ????? RARN stem ?? 2 ? residual block ???????? tile-level mid fusion?opponent ? stage ????? MLP ????? public context???? state feature ???? 235 ??? logits??????????? v1 ????????? top-1 ? 82.39% ??? 87.50%?loss ? 0.5400 ?? 0.3867??? 1024 ? arena ??? plain ResNet ?? +8.78 ?????? [6.66, 10.98] ? 95% ????????? v1 ? +3.86???????????????? tile-level ??????????????????????????????????
```

English?

```text
In Stage 2.5, we improved public state modeling on top of RARN_public_v1. The previous model encoded the 442-dimensional public vector with a flat MLP and fused it only at the final policy head. The new RARN_public_v2 explicitly separates public features into tile-level, opponent-level, and game-stage groups. The 408-dimensional tile-level features are reshaped into a 12 x 4 x 9 tile grid, encoded with rank-direction convolutions for suited tiles and 1x1 branches for honor tiles, and fused with the RARN tile representation after the first two residual blocks. Opponent and stage features are encoded as a global public context and fused with the final state feature before producing 235 masked action logits. With slightly fewer parameters than v1, RARN_public_v2 improves validation top-1 accuracy from 82.39% to 87.50% and reduces loss from 0.5400 to 0.3867. In a 1024-game arena against the plain ResNet baseline, it achieves a mean score difference of +8.78 with a 95% confidence interval of [6.66, 10.98], substantially outperforming v1's +3.86. These results show that structured public encoding and tile-level fusion significantly improve medium-complexity decisions, response-phase disambiguation, and discard selection.
```
