# CSM_TileResNet

Supervised-learning agent for Chinese Standard Mahjong. The project starts from a course baseline CNN policy and iteratively improves it with a tile-structure-aware ResNet backbone, public-state features, tile-level public fusion, suit permutation augmentation, and a larger backbone variant.

## Highlights

- Legal action space with 235 actions: `Pass`, `Hu`, `Play`, `Chi`, `Peng`, `Gang`, `AnGang`, `BuGang`.
- Rule-generated action masks are applied in every model forward pass to avoid illegal actions.
- Structured inputs: tile grid `6 x 4 x 9`, public-state vector `442`, and action mask `235`.
- Main model: `rarn_public_v2`, a Rank-Aware ResNet with structured public-state mid-fusion.
- Final experimental model: `rarn_public_v2_large`, a deeper `1.49x` backbone variant with online suit augmentation.
- Includes preprocessing, supervised training, offline evaluation, local arena utilities, and a Botzone-style inference entrypoint.

## Repository Layout

```text
.
|-- __main__.py                  # Botzone-style inference entrypoint
|-- agent.py                     # Agent interface base class
|-- feature.py                   # Mahjong feature encoder and action mapping
|-- dataset.py                   # Dataset loader and suit augmentation
|-- model.py                     # CNN / ResNet / RARN / RARN public v2 models
|-- preprocess.py                # Raw log to .npz preprocessing
|-- supervised.py                # Supervised training script
|-- history_features.py          # Historical-feature helpers kept for experiments
|-- evaluation/
|   `-- eval_supervised.py       # Offline supervised metrics
|-- eval/
|   |-- arena/                   # Local arena evaluation utilities
|   `-- models/rarn_public_v2/   # Deployable model-code snapshot
|-- AgentDocs/                   # Experiment notes and stage summaries
|-- data/                        # Sample data and data-format notes only
|-- checkpoints/                 # Placeholder only; real checkpoints are not tracked
`-- third_party/                 # Botzone Mahjong environment dependency
```

## Environment

```bash
conda env create -f environment.yml
conda activate csmj-arena
```

For GPU training, install a CUDA-enabled PyTorch build matching your machine, then keep the remaining dependencies from `environment.yml`.

## Data

The full course dataset is not tracked in this repository because raw logs and processed `.npz` files are large. The expected raw input is:

```text
data/data.txt
```

Only `data/sample.txt` and format notes are committed. After placing the full raw data file, preprocess it with:

```bash
DATA_DIR=data_public python preprocess.py
```

On Windows PowerShell:

```powershell
$env:DATA_DIR="data_public"
python preprocess.py
```

The generated directory should contain `count.json` and per-match `.npz` files.

## Models

Supported final variants:

| Model | Params | Description |
| --- | ---: | --- |
| `rarn_public_v2` | 4.45M | Rank-aware tile encoder with structured public-state mid-fusion |
| `rarn_public_v2_large` | 6.63M | Deeper 19-block backbone, about 1.49x the base model |

`rarn_public_v2_large` keeps the same channel width as `rarn_public_v2`, so most base checkpoint weights can be loaded directly and only the extra residual blocks are randomly initialized.

## Training

```bash
python supervised.py \
  --model rarn_public_v2_large \
  --data-dir data_public \
  --output-dir checkpoints/rarn_public_v2_large_aug \
  --epochs 30 \
  --batch-size 1024 \
  --num-workers 2 \
  --init-checkpoint checkpoints/rarn_public_v2/rarn_public_v2_model_latest.pkl \
  --suit-augment random \
  --lr 3e-4 \
  --weight-decay 1e-4
```

Suit augmentation modes:

- `none`: no augmentation.
- `random`: online random W/T/B suit permutation per training sample.
- `all6`: six deterministic suit permutations, expanding dataset length by 6x.

Validation data is never augmented.

## Offline Evaluation

```bash
python evaluation/eval_supervised.py \
  --model rarn_public_v2_large \
  --data-dir data_public \
  --checkpoint checkpoints/rarn_public_v2_large_aug/rarn_public_v2_large_model_latest.pkl \
  --output-dir evaluation/results/rarn_public_v2_large_aug \
  --device auto \
  --batch-size 1024 \
  --topk 1,3,5 \
  --num-workers 2
```

## Deployment

`__main__.py` is a Botzone-style interactive entrypoint:

```bash
python __main__.py checkpoints/rarn_public_v2_large_aug/rarn_public_v2_large_model_latest.pkl rarn_public_v2_large
```

## Reported Results

| Model | Validation Loss | Top-1 | Top-3 | Top-5 | Arena Summary |
| --- | ---: | ---: | ---: | ---: | --- |
| `rarn_public_v1` | 0.5400 | 82.39% | 97.75% | 99.51% | Better than CNN / plain ResNet |
| `rarn_public_v2` | 0.3867 | 87.50% | - | - | +8.78 mean score vs plain ResNet, 95% CI [6.66, 10.98] |
| `rarn_public_v2_large + suit_aug` | pending | pending | pending | pending | pending |

Mahjong game-level results have high variance. Use both offline metrics and arena matches for model selection.

## Reproducibility Notes

- Full training data and checkpoints are intentionally excluded from git.
- Put large checkpoints in GitHub Releases or external storage.
- Keep `count.json` and `.npz` generated files out of version control.
- Use the same `feature.py`, `model.py`, and action mapping for training, evaluation, arena, and deployment.

## License

This repository is released under the MIT License. The bundled third-party Botzone Mahjong environment keeps its own license under `third_party/botzone-mahjong-environment/LICENSE`.
