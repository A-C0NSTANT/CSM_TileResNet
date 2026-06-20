# Model part
import torch
from torch import nn


DEFAULT_OBS_CHANNELS = 6
DEFAULT_ACT_SIZE = 235
DEFAULT_PUBLIC_SIZE = 442
PUBLIC_TILE_LEVEL_SIZE = 408
PUBLIC_OPPONENT_SIZE = 27
PUBLIC_STAGE_SIZE = 7
PUBLIC_TILE_CHANNELS = 12
NUMERIC_SUIT_ROWS = 3
TILE_RANKS = 9
HONOR_VALID_TILES = 7
VALID_TILE_COUNT = NUMERIC_SUIT_ROWS * TILE_RANKS + HONOR_VALID_TILES


def apply_action_mask(action_logits, action_mask):
    action_mask = action_mask.float()
    return action_logits.masked_fill(action_mask <= 0, torch.finfo(action_logits.dtype).min)


def init_policy_modules(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def extract_policy_input(input_dict):
    obs = input_dict["obs"]["observation"].float()
    action_mask = input_dict["obs"]["action_mask"]
    return obs, action_mask


def extract_public_input(input_dict, batch_size, device):
    public = input_dict["obs"].get("public")
    if public is None:
        return torch.zeros(batch_size, DEFAULT_PUBLIC_SIZE, device=device)
    return public.float()


def make_valid_tile_mask():
    mask = torch.ones(1, 1, 4, 9)
    mask[:, :, 3, HONOR_VALID_TILES:] = 0
    return mask


def flatten_valid_tiles(x):
    batch_size, channels = x.size(0), x.size(1)
    numeric = x[:, :, :NUMERIC_SUIT_ROWS, :].reshape(batch_size, channels, -1)
    honors = x[:, :, 3:4, :HONOR_VALID_TILES].reshape(batch_size, channels, -1)
    return torch.cat([numeric, honors], dim=2).reshape(batch_size, channels * VALID_TILE_COUNT)


def pad_honor_tiles(honors):
    if honors.size(-1) == TILE_RANKS:
        return honors
    pad_width = TILE_RANKS - honors.size(-1)
    padding = honors.new_zeros(honors.size(0), honors.size(1), honors.size(2), pad_width)
    return torch.cat([honors, padding], dim=3)


def public_tile_features_to_grid(tile_public):
    batch_size = tile_public.size(0)
    tile_public = tile_public.reshape(batch_size, PUBLIC_TILE_CHANNELS, VALID_TILE_COUNT)
    grid = tile_public.new_zeros(batch_size, PUBLIC_TILE_CHANNELS, 4, TILE_RANKS)
    grid[:, :, :NUMERIC_SUIT_ROWS, :] = tile_public[:, :, :NUMERIC_SUIT_ROWS * TILE_RANKS].reshape(
        batch_size, PUBLIC_TILE_CHANNELS, NUMERIC_SUIT_ROWS, TILE_RANKS
    )
    grid[:, :, 3:4, :HONOR_VALID_TILES] = tile_public[:, :, NUMERIC_SUIT_ROWS * TILE_RANKS:].reshape(
        batch_size, PUBLIC_TILE_CHANNELS, 1, HONOR_VALID_TILES
    )
    return grid


class CNNModel(nn.Module):

    OBS_CHANNELS = DEFAULT_OBS_CHANNELS
    ACT_SIZE = DEFAULT_ACT_SIZE

    def __init__(self):
        nn.Module.__init__(self)
        self._tower = nn.Sequential(
            nn.Conv2d(self.OBS_CHANNELS, 64, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(64 * 4 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, self.ACT_SIZE),
        )
        init_policy_modules(self)

    def forward(self, input_dict):
        obs, action_mask = extract_policy_input(input_dict)
        action_logits = self._tower(obs)
        return apply_action_mask(action_logits, action_mask)


class ResidualBlock(nn.Module):

    def __init__(self, channels):
        nn.Module.__init__(self)
        self._block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self._relu = nn.ReLU(True)

    def forward(self, x):
        return self._relu(x + self._block(x))


class ResNetPolicyModel(nn.Module):

    OBS_CHANNELS = DEFAULT_OBS_CHANNELS
    ACT_SIZE = DEFAULT_ACT_SIZE

    def __init__(self, channels=96, blocks=6, hidden=384):
        nn.Module.__init__(self)
        self.channels = channels
        self.blocks = blocks
        self.hidden = hidden
        self._tower = nn.Sequential(
            nn.Conv2d(self.OBS_CHANNELS, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            *(ResidualBlock(channels) for _ in range(blocks)),
            nn.Flatten(),
            nn.Linear(channels * 4 * 9, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, self.ACT_SIZE),
        )
        init_policy_modules(self)

    def forward(self, input_dict):
        obs, action_mask = extract_policy_input(input_dict)
        action_logits = self._tower(obs)
        return apply_action_mask(action_logits, action_mask)


class SuitContextMixer(nn.Module):

    def __init__(self, channels):
        nn.Module.__init__(self)
        self._mixer = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        suit_mean = x.mean(dim=2, keepdim=True).expand_as(x)
        suit_max = x.max(dim=2, keepdim=True).values.expand_as(x)
        return self._mixer(torch.cat([x, suit_mean, suit_max], dim=1))


class RankAwareResidualBlock(nn.Module):

    def __init__(self, channels):
        nn.Module.__init__(self)
        self._numeric_rank_block = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 3), 1, (0, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, (1, 3), 1, (0, 1), bias=False),
            nn.BatchNorm2d(channels),
        )
        self._honor_block = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels),
        )
        self._suit_mixer = SuitContextMixer(channels)
        self._relu = nn.ReLU(True)

    def forward(self, x, valid_tile_mask):
        numeric = x[:, :, :NUMERIC_SUIT_ROWS, :]
        honors = x[:, :, 3:4, :HONOR_VALID_TILES]
        numeric_update = self._numeric_rank_block(numeric)
        numeric_update = numeric_update + self._suit_mixer(numeric_update)
        honor_update = pad_honor_tiles(self._honor_block(honors))
        update = torch.cat([numeric_update, honor_update], dim=2)
        return self._relu((x + update) * valid_tile_mask)


class RankAwareResNetPolicyModel(nn.Module):

    OBS_CHANNELS = DEFAULT_OBS_CHANNELS
    ACT_SIZE = DEFAULT_ACT_SIZE

    def __init__(self, channels=96, blocks=6, hidden=384):
        nn.Module.__init__(self)
        self.channels = channels
        self.blocks = blocks
        self.hidden = hidden
        self._numeric_stem = nn.Sequential(
            nn.Conv2d(self.OBS_CHANNELS, channels, (1, 3), 1, (0, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
        )
        self._honor_stem = nn.Sequential(
            nn.Conv2d(self.OBS_CHANNELS, channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
        )
        self._blocks = nn.ModuleList(RankAwareResidualBlock(channels) for _ in range(blocks))
        self._head = nn.Sequential(
            nn.Linear(channels * VALID_TILE_COUNT, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, self.ACT_SIZE),
        )
        self.register_buffer('_valid_tile_mask', make_valid_tile_mask())
        init_policy_modules(self)

    def forward(self, input_dict):
        obs, action_mask = extract_policy_input(input_dict)
        valid_tile_mask = self._valid_tile_mask.to(dtype=obs.dtype)
        numeric = self._numeric_stem(obs[:, :, :NUMERIC_SUIT_ROWS, :])
        honors = pad_honor_tiles(self._honor_stem(obs[:, :, 3:4, :HONOR_VALID_TILES]))
        x = torch.cat([numeric, honors], dim=2) * valid_tile_mask
        for block in self._blocks:
            x = block(x, valid_tile_mask)
        action_logits = self._head(flatten_valid_tiles(x))
        return apply_action_mask(action_logits, action_mask)


class RankAwareResNetPublicPolicyModel(nn.Module):

    OBS_CHANNELS = DEFAULT_OBS_CHANNELS
    ACT_SIZE = DEFAULT_ACT_SIZE
    PUBLIC_SIZE = DEFAULT_PUBLIC_SIZE

    def __init__(self, channels=128, blocks=8, state_hidden=512, public_hidden=256, fusion_hidden=512):
        nn.Module.__init__(self)
        self.channels = channels
        self.blocks = blocks
        self.state_hidden = state_hidden
        self.public_hidden = public_hidden
        self.fusion_hidden = fusion_hidden
        self._numeric_stem = nn.Sequential(
            nn.Conv2d(self.OBS_CHANNELS, channels, (1, 3), 1, (0, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
        )
        self._honor_stem = nn.Sequential(
            nn.Conv2d(self.OBS_CHANNELS, channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
        )
        self._blocks = nn.ModuleList(RankAwareResidualBlock(channels) for _ in range(blocks))
        self._state_head = nn.Sequential(
            nn.Linear(channels * VALID_TILE_COUNT, state_hidden),
            nn.ReLU(True),
        )
        self._public_head = nn.Sequential(
            nn.Linear(self.PUBLIC_SIZE, public_hidden),
            nn.LayerNorm(public_hidden),
            nn.ReLU(True),
            nn.Linear(public_hidden, public_hidden),
            nn.ReLU(True),
        )
        self._fusion_head = nn.Sequential(
            nn.Linear(state_hidden + public_hidden, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.ReLU(True),
            nn.Linear(fusion_hidden, self.ACT_SIZE),
        )
        self.register_buffer('_valid_tile_mask', make_valid_tile_mask())
        init_policy_modules(self)

    def encode_state(self, obs):
        valid_tile_mask = self._valid_tile_mask.to(dtype=obs.dtype)
        numeric = self._numeric_stem(obs[:, :, :NUMERIC_SUIT_ROWS, :])
        honors = pad_honor_tiles(self._honor_stem(obs[:, :, 3:4, :HONOR_VALID_TILES]))
        x = torch.cat([numeric, honors], dim=2) * valid_tile_mask
        for block in self._blocks:
            x = block(x, valid_tile_mask)
        return self._state_head(flatten_valid_tiles(x))

    def forward(self, input_dict):
        obs, action_mask = extract_policy_input(input_dict)
        state_feature = self.encode_state(obs)
        public_feature = self._public_head(extract_public_input(input_dict, obs.size(0), obs.device))
        action_logits = self._fusion_head(torch.cat([state_feature, public_feature], dim=1))
        return apply_action_mask(action_logits, action_mask)


class PublicTileEncoder(nn.Module):

    def __init__(self, output_channels=64):
        nn.Module.__init__(self)
        self.output_channels = output_channels
        self._numeric_branch = nn.Sequential(
            nn.Conv2d(PUBLIC_TILE_CHANNELS, output_channels, (1, 3), 1, (0, 1), bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(True),
            nn.Conv2d(output_channels, output_channels, (1, 3), 1, (0, 1), bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(True),
        )
        self._honor_branch = nn.Sequential(
            nn.Conv2d(PUBLIC_TILE_CHANNELS, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(True),
            nn.Conv2d(output_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(True),
        )

    def forward(self, public_grid, valid_tile_mask):
        numeric = self._numeric_branch(public_grid[:, :, :NUMERIC_SUIT_ROWS, :])
        honors = pad_honor_tiles(self._honor_branch(public_grid[:, :, 3:4, :HONOR_VALID_TILES]))
        return torch.cat([numeric, honors], dim=2) * valid_tile_mask


class GlobalPublicEncoder(nn.Module):

    def __init__(self, opponent_hidden=128, stage_hidden=64, output_hidden=256):
        nn.Module.__init__(self)
        self.output_hidden = output_hidden
        self._opponent_head = nn.Sequential(
            nn.Linear(PUBLIC_OPPONENT_SIZE, opponent_hidden),
            nn.LayerNorm(opponent_hidden),
            nn.ReLU(True),
            nn.Linear(opponent_hidden, opponent_hidden),
            nn.ReLU(True),
        )
        self._stage_head = nn.Sequential(
            nn.Linear(PUBLIC_STAGE_SIZE, stage_hidden),
            nn.LayerNorm(stage_hidden),
            nn.ReLU(True),
            nn.Linear(stage_hidden, stage_hidden),
            nn.ReLU(True),
        )
        self._fusion = nn.Sequential(
            nn.Linear(opponent_hidden + stage_hidden, output_hidden),
            nn.LayerNorm(output_hidden),
            nn.ReLU(True),
        )

    def forward(self, opponent_public, stage_public):
        opponent_feature = self._opponent_head(opponent_public)
        stage_feature = self._stage_head(stage_public)
        return self._fusion(torch.cat([opponent_feature, stage_feature], dim=1))


class RankAwareResNetPublicV2PolicyModel(nn.Module):

    OBS_CHANNELS = DEFAULT_OBS_CHANNELS
    ACT_SIZE = DEFAULT_ACT_SIZE
    PUBLIC_SIZE = DEFAULT_PUBLIC_SIZE

    def __init__(
        self,
        channels=128,
        blocks=8,
        pre_fusion_blocks=2,
        state_hidden=512,
        public_tile_channels=64,
        global_public_hidden=256,
        fusion_hidden=512,
    ):
        nn.Module.__init__(self)
        if pre_fusion_blocks < 0 or pre_fusion_blocks > blocks:
            raise ValueError('pre_fusion_blocks must be between 0 and blocks')
        self.channels = channels
        self.blocks = blocks
        self.pre_fusion_blocks = pre_fusion_blocks
        self.state_hidden = state_hidden
        self.public_tile_channels = public_tile_channels
        self.global_public_hidden = global_public_hidden
        self.fusion_hidden = fusion_hidden
        self._numeric_stem = nn.Sequential(
            nn.Conv2d(self.OBS_CHANNELS, channels, (1, 3), 1, (0, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
        )
        self._honor_stem = nn.Sequential(
            nn.Conv2d(self.OBS_CHANNELS, channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
        )
        self._blocks = nn.ModuleList(RankAwareResidualBlock(channels) for _ in range(blocks))
        self._public_tile_encoder = PublicTileEncoder(public_tile_channels)
        self._global_public_encoder = GlobalPublicEncoder(output_hidden=global_public_hidden)
        self._tile_fusion = nn.Sequential(
            nn.Conv2d(channels + public_tile_channels, channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
        )
        self._state_head = nn.Sequential(
            nn.Linear(channels * VALID_TILE_COUNT, state_hidden),
            nn.ReLU(True),
        )
        self._fusion_head = nn.Sequential(
            nn.Linear(state_hidden + global_public_hidden, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.ReLU(True),
            nn.Linear(fusion_hidden, self.ACT_SIZE),
        )
        self.register_buffer('_valid_tile_mask', make_valid_tile_mask())
        init_policy_modules(self)

    def encode_public(self, public, valid_tile_mask):
        tile_public = public[:, :PUBLIC_TILE_LEVEL_SIZE]
        opponent_start = PUBLIC_TILE_LEVEL_SIZE
        stage_start = opponent_start + PUBLIC_OPPONENT_SIZE
        opponent_public = public[:, opponent_start:stage_start]
        stage_public = public[:, stage_start:stage_start + PUBLIC_STAGE_SIZE]
        public_grid = public_tile_features_to_grid(tile_public)
        tile_feature = self._public_tile_encoder(public_grid, valid_tile_mask)
        global_feature = self._global_public_encoder(opponent_public, stage_public)
        return tile_feature, global_feature

    def encode_state(self, obs, public):
        valid_tile_mask = self._valid_tile_mask.to(dtype=obs.dtype)
        numeric = self._numeric_stem(obs[:, :, :NUMERIC_SUIT_ROWS, :])
        honors = pad_honor_tiles(self._honor_stem(obs[:, :, 3:4, :HONOR_VALID_TILES]))
        x = torch.cat([numeric, honors], dim=2) * valid_tile_mask
        public_tile_feature, global_public_feature = self.encode_public(public, valid_tile_mask)

        for block in self._blocks[:self.pre_fusion_blocks]:
            x = block(x, valid_tile_mask)
        x = self._tile_fusion(torch.cat([x, public_tile_feature], dim=1)) * valid_tile_mask
        for block in self._blocks[self.pre_fusion_blocks:]:
            x = block(x, valid_tile_mask)

        state_feature = self._state_head(flatten_valid_tiles(x))
        return state_feature, global_public_feature

    def forward(self, input_dict):
        obs, action_mask = extract_policy_input(input_dict)
        public = extract_public_input(input_dict, obs.size(0), obs.device)
        state_feature, global_public_feature = self.encode_state(obs, public)
        action_logits = self._fusion_head(torch.cat([state_feature, global_public_feature], dim=1))
        return apply_action_mask(action_logits, action_mask)


class RankAwareResNetPublicV2LargePolicyModel(RankAwareResNetPublicV2PolicyModel):

    def __init__(self):
        RankAwareResNetPublicV2PolicyModel.__init__(self, blocks=19)


def model_requires_public(name):
    name = name.lower().replace('-', '_')
    return name in (
        'rarn_public', 'rarn_public_v1', 'rarn_public_policy', 'rarn_public_policy_model',
        'rank_aware_resnet_public', 'rank_aware_resnet_public_policy_model',
        'rarn_public_v2', 'rarn_public_struct', 'rarn_public_midfusion', 'rarn_public_policy_v2',
        'rarn_public_policy_model_v2', 'rank_aware_resnet_public_v2',
        'rank_aware_resnet_public_policy_model_v2',
        'rarn_public_v2_large', 'rarn_public_v2_xl', 'rarn_public_v2_1_5x',
        'rank_aware_resnet_public_v2_large',
    )


def model_requires_history(name):
    return False


def create_model(name='cnn'):
    name = name.lower().replace('-', '_')
    if name in ('cnn', 'cnn_model', 'cnnmodel'):
        return CNNModel()
    if name in ('resnet', 'resnet_policy', 'resnet_policy_model', 'resnetpolicymodel'):
        return ResNetPolicyModel()
    if name in ('rarn_v2', 'rarn_m', 'rarn_model_v2', 'rarn_policy_v2', 'rarn_policy_model_v2', 'rank_aware_resnet_v2', 'rank_aware_resnet_policy_v2', 'rank_aware_resnet_policy_model_v2'):
        return RankAwareResNetPolicyModel(channels=128, blocks=8, hidden=512)
    if name in ('rarn_public_v2_large', 'rarn_public_v2_xl', 'rarn_public_v2_1_5x', 'rank_aware_resnet_public_v2_large'):
        return RankAwareResNetPublicV2LargePolicyModel()
    if name in ('rarn_public_v2', 'rarn_public_struct', 'rarn_public_midfusion', 'rarn_public_policy_v2', 'rarn_public_policy_model_v2', 'rank_aware_resnet_public_v2', 'rank_aware_resnet_public_policy_model_v2'):
        return RankAwareResNetPublicV2PolicyModel()
    if name in ('rarn_public', 'rarn_public_v1', 'rarn_public_policy', 'rarn_public_policy_model', 'rank_aware_resnet_public', 'rank_aware_resnet_public_policy_model'):
        return RankAwareResNetPublicPolicyModel()
    if name in ('rarn', 'rarn_s', 'rarn_model', 'rarn_policy', 'rarn_policy_model', 'rarnpolicymodel', 'rank_aware_resnet', 'rank_aware_resnet_policy', 'rank_aware_resnet_policy_model', 'rankawareresnetpolicymodel'):
        return RankAwareResNetPolicyModel()
    raise ValueError('Unknown model name: %s' % name)
