import logging
from torch.utils.data import Dataset
import numpy as np
from bisect import bisect_right
from pathlib import Path

from history_features import encode_history_from_events

logger = logging.getLogger(__name__)
PUBLIC_FEATURE_SIZE = 442
DEFAULT_HISTORY_MAX_LEN = 64
HISTORY_NUM_FIELDS = 7
SUIT_PERMUTATIONS = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)
NUM_TILE_TYPES = 34
ACTION_SIZE = 235
PUBLIC_TILE_LEVEL_SIZE = 408
PUBLIC_OPPONENT_START = 408
PUBLIC_STAGE_START = 435


def normalize_suit_augment(augment):
    if isinstance(augment, bool):
        return 'random' if augment else 'none'
    if augment is None:
        return 'none'
    augment = str(augment).lower().replace('-', '_')
    if augment in ('0', 'false', 'no', 'none', 'off'):
        return 'none'
    if augment in ('1', 'true', 'yes', 'on', 'random'):
        return 'random'
    if augment in ('all', 'all6', 'full'):
        return 'all6'
    raise ValueError('Unknown suit augment mode: %s' % augment)


def build_tile_permutation(suit_perm):
    tile_perm = np.arange(NUM_TILE_TYPES, dtype=np.int64)
    for old_suit, new_suit in enumerate(suit_perm):
        for rank in range(9):
            tile_perm[old_suit * 9 + rank] = new_suit * 9 + rank
    return tile_perm


def build_action_permutation(tile_perm, suit_perm):
    action_perm = np.arange(ACTION_SIZE, dtype=np.int64)
    for start in (2, 99, 133, 167, 201):
        action_perm[start:start + NUM_TILE_TYPES] = start + tile_perm
    for old_suit, new_suit in enumerate(suit_perm):
        old_start = 36 + old_suit * 21
        new_start = 36 + new_suit * 21
        action_perm[old_start:old_start + 21] = np.arange(new_start, new_start + 21, dtype=np.int64)
    return action_perm


TILE_PERMUTATIONS = tuple(build_tile_permutation(perm) for perm in SUIT_PERMUTATIONS)
ACTION_PERMUTATIONS = tuple(build_action_permutation(tile_perm, perm) for tile_perm, perm in zip(TILE_PERMUTATIONS, SUIT_PERMUTATIONS))
HISTORY_TILE_PERMUTATIONS = []
for tile_perm in TILE_PERMUTATIONS:
    history_tile_perm = np.arange(36, dtype=np.uint8)
    history_tile_perm[:NUM_TILE_TYPES] = tile_perm.astype(np.uint8)
    HISTORY_TILE_PERMUTATIONS.append(history_tile_perm)
HISTORY_TILE_PERMUTATIONS = tuple(HISTORY_TILE_PERMUTATIONS)


def permute_obs(obs, suit_perm):
    augmented = obs.copy()
    augmented[:, suit_perm, :] = obs[:, :3, :]
    return augmented


def permute_action_mask(mask, action_perm):
    augmented = np.zeros_like(mask)
    augmented[action_perm] = mask
    return augmented


def permute_public(public, tile_perm, suit_perm):
    augmented = public.copy()
    tile_level = public[:PUBLIC_TILE_LEVEL_SIZE].reshape(12, NUM_TILE_TYPES)
    augmented_tile_level = augmented[:PUBLIC_TILE_LEVEL_SIZE].reshape(12, NUM_TILE_TYPES)
    augmented_tile_level[:, tile_perm] = tile_level

    for player in range(3):
        base = PUBLIC_OPPONENT_START + player * 9
        augmented[base + 5 + np.asarray(suit_perm, dtype=np.int64)] = public[base + 5:base + 8]
    return augmented


def permute_history(history, history_tile_perm):
    augmented = history.copy()
    tile_ids = augmented[:, 2]
    in_vocab = tile_ids < len(history_tile_perm)
    tile_ids[in_vocab] = history_tile_perm[tile_ids[in_vocab]]
    return augmented


def apply_suit_permutation(obs, mask, act, public, history, perm_index):
    suit_perm = SUIT_PERMUTATIONS[perm_index]
    tile_perm = TILE_PERMUTATIONS[perm_index]
    action_perm = ACTION_PERMUTATIONS[perm_index]
    history_tile_perm = HISTORY_TILE_PERMUTATIONS[perm_index]

    obs = permute_obs(obs, suit_perm)
    mask = permute_action_mask(mask, action_perm)
    act = int(action_perm[int(act)])
    public = permute_public(public, tile_perm, suit_perm)
    if history is not None:
        history = permute_history(history, history_tile_perm)
    return obs, mask, act, public, history

class MahjongGBDataset(Dataset):
    
    def __init__(self, begin = 0, end = 1, augment = False, data_dir = 'data', split_name = None, include_history = False, history_max_len = None):
        import json
        self.data_dir = Path(data_dir)
        self.split_name = split_name or 'dataset'
        self.include_history = include_history
        self.history_max_len = history_max_len
        with open(self.data_dir / 'count.json') as f:
            self.match_samples = json.load(f)
        self.total_matches = len(self.match_samples)
        self.total_samples = sum(self.match_samples)
        self.begin = int(begin * self.total_matches)
        self.end = int(end * self.total_matches)
        self.match_samples = self.match_samples[self.begin : self.end]
        self.matches = len(self.match_samples)
        self.samples = sum(self.match_samples)
        self.suit_augment = normalize_suit_augment(augment)
        t = 0
        for i in range(self.matches):
            a = self.match_samples[i]
            self.match_samples[i] = t
            t += a
        self.cache = {'obs': [], 'mask': [], 'act': [], 'public': []}
        if self.include_history:
            self.cache['history'] = []
            self.cache['history_length'] = []
            self.cache['history_events'] = []
            self.cache['history_event_index'] = []
            self.cache['history_viewpoint'] = []
            self.cache['history_max_len'] = []
        self.has_public = True
        self.has_history = True
        for i in range(self.matches):
            if i % 128 == 0:
                logger.info('[Data:%s] loading_match=%d/%d', self.split_name, i, self.matches)
            d = np.load(self.data_dir / ('%d.npz' % (i + self.begin)))
            for k in ('obs', 'mask', 'act'):
                self.cache[k].append(d[k])
            if 'public' not in d:
                self.has_public = False
                sample_count = d['obs'].shape[0]
                self.cache['public'].append(np.zeros((sample_count, PUBLIC_FEATURE_SIZE), dtype=np.float32))
            else:
                self.cache['public'].append(d['public'])

            if self.include_history:
                sample_count = d['obs'].shape[0]
                match_history_max_len = self._resolve_match_history_max_len(d)
                self.cache['history_max_len'].append(match_history_max_len)
                if 'history_events' in d and 'history_event_index' in d and 'history_viewpoint' in d:
                    history_length = d['history_length'].astype(np.int64, copy=False)
                    history_length = np.minimum(history_length, match_history_max_len).astype(np.int64, copy=False)
                    self.cache['history'].append(None)
                    self.cache['history_events'].append(d['history_events'])
                    self.cache['history_event_index'].append(d['history_event_index'].astype(np.int64, copy=False))
                    self.cache['history_viewpoint'].append(d['history_viewpoint'].astype(np.int64, copy=False))
                    self.cache['history_length'].append(history_length)
                elif 'history' in d and 'history_length' in d:
                    history = d['history']
                    history_length = d['history_length'].astype(np.int64, copy=False)
                    history, history_length = self._normalize_history(history, history_length, match_history_max_len)
                    self.cache['history'].append(history)
                    self.cache['history_events'].append(None)
                    self.cache['history_event_index'].append(None)
                    self.cache['history_viewpoint'].append(None)
                    self.cache['history_length'].append(history_length)
                else:
                    self.has_history = False
                    self.cache['history'].append(np.zeros((sample_count, match_history_max_len, HISTORY_NUM_FIELDS), dtype=np.uint8))
                    self.cache['history_events'].append(None)
                    self.cache['history_event_index'].append(None)
                    self.cache['history_viewpoint'].append(None)
                    self.cache['history_length'].append(np.zeros(sample_count, dtype=np.int64))

    def _resolve_match_history_max_len(self, d):
        if self.history_max_len is not None:
            return int(self.history_max_len)
        if 'history_max_len' in d:
            return int(np.asarray(d['history_max_len']).item())
        if 'history' in d:
            return int(d['history'].shape[1])
        return DEFAULT_HISTORY_MAX_LEN

    def _normalize_history(self, history, history_length, target_len):
        if history.shape[1] == target_len:
            return history, history_length
        target_len = int(target_len)
        if target_len <= 0:
            raise ValueError('history_max_len must be positive')
        sample_count, source_len, num_fields = history.shape
        normalized = np.zeros((sample_count, target_len, num_fields), dtype=history.dtype)
        normalized_length = np.minimum(history_length, target_len).astype(np.int64, copy=False)
        for i, length in enumerate(history_length):
            length = int(length)
            if length <= 0:
                continue
            copy_len = min(length, target_len)
            source_start = max(0, length - copy_len)
            normalized[i, :copy_len] = history[i, source_start:source_start + copy_len]
        return normalized, normalized_length
    
    def __len__(self):
        if self.suit_augment == 'all6':
            return self.samples * len(SUIT_PERMUTATIONS)
        return self.samples
    
    def __getitem__(self, index):
        perm_index = 0
        if self.suit_augment == 'all6':
            perm_index = index % len(SUIT_PERMUTATIONS)
            index = index // len(SUIT_PERMUTATIONS)
        elif self.suit_augment == 'random':
            perm_index = np.random.randint(len(SUIT_PERMUTATIONS))
        match_id = bisect_right(self.match_samples, index, 0, self.matches) - 1
        sample_id = index - self.match_samples[match_id]
        obs = self.cache['obs'][match_id][sample_id]
        mask = self.cache['mask'][match_id][sample_id]
        act = self.cache['act'][match_id][sample_id]
        public = self.cache['public'][match_id][sample_id]
        history = None
        history_length = None
        history_padding_mask = None
        if not self.include_history:
            if perm_index:
                obs, mask, act, public, history = apply_suit_permutation(obs, mask, act, public, history, perm_index)
            return obs, mask, act, public
        cached_history = self.cache['history'][match_id]
        if cached_history is None:
            history, history_length = encode_history_from_events(
                self.cache['history_events'][match_id],
                self.cache['history_event_index'][match_id][sample_id],
                self.cache['history_viewpoint'][match_id][sample_id],
                self.cache['history_max_len'][match_id],
            )
        else:
            history = cached_history[sample_id]
            history_length = self.cache['history_length'][match_id][sample_id]
        if perm_index:
            obs, mask, act, public, history = apply_suit_permutation(obs, mask, act, public, history, perm_index)
        history_padding_mask = np.arange(history.shape[0]) >= int(history_length)
        return (
            obs,
            mask,
            act,
            public,
            history,
            history_length,
            history_padding_mask,
        )
