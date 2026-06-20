# CSM_TileResNet

面向国标麻将的监督学习智能体。本项目以课程基线 CNN 策略网络为起点，逐步引入面向麻将牌结构的 ResNet 主干、公开状态特征、牌级公开信息融合、花色置换数据增强，以及更大的主干网络变体，用于提升国标麻将决策策略的离线指标和本地竞技场表现。

## 项目亮点

- 使用 235 维合法动作空间，覆盖 `Pass`、`Hu`、`Play`、`Chi`、`Peng`、`Gang`、`AnGang`、`BuGang`。
- 每次模型前向传播都会应用规则生成的动作掩码，避免模型选择非法动作。
- 输入结构化表示包括牌面网格 `6 x 4 x 9`、公开状态向量 `442` 和动作掩码 `235`。
- 主模型为 `rarn_public_v2`，即带结构化公开状态中层融合的 Rank-Aware ResNet。
- 最终实验模型为 `rarn_public_v2_large`，在基础模型上使用更深的 `1.49x` 主干，并支持在线花色增强。
- 仓库包含数据预处理、监督训练、离线评估、本地竞技场工具，以及 Botzone 风格的推理入口。

## 仓库结构

```text
.
|-- __main__.py                  # Botzone 风格交互式推理入口
|-- agent.py                     # 智能体接口基类
|-- feature.py                   # 麻将特征编码与动作映射
|-- dataset.py                   # 数据集加载与花色增强
|-- model.py                     # CNN / ResNet / RARN / RARN public v2 模型
|-- preprocess.py                # 原始日志到 .npz 的预处理脚本
|-- supervised.py                # 监督学习训练脚本
|-- history_features.py          # 历史特征相关实验辅助代码
|-- evaluation/
|   `-- eval_supervised.py       # 离线监督学习指标评估
|-- eval/
|   |-- arena/                   # 本地竞技场评估工具
|   `-- models/rarn_public_v2/   # 可部署模型代码快照
|-- AgentDocs/                   # 实验记录与阶段总结
|-- data/                        # 样例数据与数据格式说明
|-- checkpoints/                 # 检查点目录；真实权重不纳入版本控制
`-- third_party/                 # Botzone 麻将环境依赖
```

## 环境配置

本地当前提供的是最小竞技场环境文件：

```bash
conda env create -f environment-arena-minimal.yml
conda activate csmj-arena
```

如果需要进行 GPU 训练，请安装与本机 CUDA 版本匹配的 PyTorch，并保留 `environment-arena-minimal.yml` 中的其余依赖。远程仓库 README 中的完整环境文件名为 `environment.yml`；若本地没有该文件，请以当前仓库实际存在的环境文件为准。

## 数据

完整课程数据集不会提交到仓库中，因为原始日志和处理后的 `.npz` 文件体积较大。预期的原始输入文件为：

```text
data/data.txt
```

仓库只应提交 `data/sample.txt` 与数据格式说明。放置完整原始数据后，可执行预处理：

```bash
DATA_DIR=data_public python preprocess.py
```

Windows PowerShell：

```powershell
$env:DATA_DIR="data_public"
python preprocess.py
```

生成的数据目录应包含 `count.json` 和按对局保存的 `.npz` 文件。

## 模型

主要最终模型变体如下：

| 模型 | 参数量 | 说明 |
| --- | ---: | --- |
| `rarn_public_v2` | 4.45M | 带结构化公开状态中层融合的 rank-aware 牌面编码器 |
| `rarn_public_v2_large` | 6.63M | 19 个残差块的更深主干，约为基础模型的 1.49 倍 |

`rarn_public_v2_large` 与 `rarn_public_v2` 保持相同通道宽度，因此多数基础检查点权重可以直接加载，额外残差块会随机初始化。

## 训练

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

花色增强模式：

- `none`：不进行数据增强。
- `random`：训练样本在线随机执行万/条/饼花色置换。
- `all6`：使用 6 种确定性花色置换，将训练集长度扩展为 6 倍。

验证数据不会进行花色增强。

## 离线评估

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

## 部署

`__main__.py` 是 Botzone 风格的交互式入口：

```bash
python __main__.py checkpoints/rarn_public_v2_large_aug/rarn_public_v2_large_model_latest.pkl rarn_public_v2_large
```

## 已报告结果

| 模型 | 验证损失 | Top-1 | Top-3 | Top-5 | 竞技场摘要 |
| --- | ---: | ---: | ---: | ---: | --- |
| `rarn_public_v1` | 0.5400 | 82.39% | 97.75% | 99.51% | 优于 CNN / 普通 ResNet |
| `rarn_public_v2` | 0.3867 | 87.50% | 98.90% | 99.78% | 相比普通 ResNet 平均分 +8.78，95% CI [6.66, 10.98] |
| `rarn_public_v2_large + suit_aug` | pending | pending | pending | pending | pending |

麻将对局级结果方差较高。模型选择应同时参考离线指标与竞技场对局结果。

## 可复现性说明

- 完整训练数据和模型检查点有意排除在 git 之外。
- 大体积检查点建议放入 GitHub Releases 或外部存储。
- `count.json` 与生成的 `.npz` 文件不应纳入版本控制。
- 训练、评估、竞技场和部署应使用一致的 `feature.py`、`model.py` 与动作映射。

## 许可证

本仓库采用 MIT License 发布。随仓库包含的第三方 Botzone 麻将环境保留其自身许可证，详见 `third_party/botzone-mahjong-environment/LICENSE`。
