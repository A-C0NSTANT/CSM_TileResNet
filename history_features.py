from dataclasses import dataclass
from enum import IntEnum

import numpy as np


DEFAULT_HISTORY_MAX_LEN = 64
HISTORY_NUM_FIELDS = 7

PLAYER_NONE = 4
TILE_NONE = 34
TILE_UNKNOWN = 35

FLAG_PRIVATE_TILE = 1


class HistoryActionType(IntEnum):
    PAD = 0
    INIT = 1
    DRAW = 2
    PLAY = 3
    PASS = 4
    CHI = 5
    PENG = 6
    GANG = 7
    BUGANG = 8
    ANGANG = 9
    HU = 10


class HistoryPhase(IntEnum):
    NORMAL = 0
    DISCARD_RESPONSE = 1
    SELF_DRAW_DECISION = 2
    POST_MELD_DISCARD = 3


class WallBucket(IntEnum):
    UNKNOWN = 0
    EARLY = 1
    MIDDLE = 2
    LATE = 3
    VERY_LATE = 4


HISTORY_FIELD_NAMES = (
    'relative_player',
    'action_type',
    'tile_id',
    'target_player',
    'wall_bucket',
    'phase',
    'flags',
)
HISTORY_EVENT_FIELD_NAMES = (
    'player',
    'action_type',
    'tile_id',
    'target_player',
    'wall_bucket',
    'phase',
    'flags',
)

TILE_LIST = [
    *('W%d' % (i + 1) for i in range(9)),
    *('T%d' % (i + 1) for i in range(9)),
    *('B%d' % (i + 1) for i in range(9)),
    *('F%d' % (i + 1) for i in range(4)),
    *('J%d' % (i + 1) for i in range(3)),
]
OFFSET_TILE = {tile: i for i, tile in enumerate(TILE_LIST)}


@dataclass(frozen=True)
class HistoryEvent:
    player: int
    action_type: int
    tile_id: int = TILE_NONE
    target_player: int = PLAYER_NONE
    wall_bucket: int = WallBucket.UNKNOWN
    phase: int = HistoryPhase.NORMAL
    flags: int = 0
    private_tile: bool = False


def tile_to_id(tile):
    if tile is None:
        return TILE_NONE
    return OFFSET_TILE.get(tile, TILE_UNKNOWN)


def wall_bucket_from_remaining(wall_remaining):
    if wall_remaining is None:
        return WallBucket.UNKNOWN
    if wall_remaining > 56:
        return WallBucket.EARLY
    if wall_remaining > 28:
        return WallBucket.MIDDLE
    if wall_remaining > 14:
        return WallBucket.LATE
    return WallBucket.VERY_LATE


def relative_player(player, viewpoint_player):
    if player == PLAYER_NONE or player is None:
        return PLAYER_NONE
    return (int(player) + 4 - int(viewpoint_player)) % 4


def encode_event(event, viewpoint_player):
    tile_id = int(event.tile_id)
    flags = int(event.flags)
    if event.private_tile:
        flags |= FLAG_PRIVATE_TILE
        if int(event.player) != int(viewpoint_player):
            tile_id = TILE_UNKNOWN

    return np.array(
        [
            relative_player(event.player, viewpoint_player),
            int(event.action_type),
            tile_id,
            relative_player(event.target_player, viewpoint_player),
            int(event.wall_bucket),
            int(event.phase),
            flags,
        ],
        dtype=np.uint8,
    )


def event_to_row(event):
    flags = int(event.flags)
    if event.private_tile:
        flags |= FLAG_PRIVATE_TILE
    return np.array(
        [
            int(event.player),
            int(event.action_type),
            int(event.tile_id),
            int(event.target_player),
            int(event.wall_bucket),
            int(event.phase),
            flags,
        ],
        dtype=np.uint8,
    )


def encode_event_row(row, viewpoint_player):
    player = int(row[0])
    tile_id = int(row[2])
    flags = int(row[6])
    if flags & FLAG_PRIVATE_TILE and player != int(viewpoint_player):
        tile_id = TILE_UNKNOWN

    return np.array(
        [
            relative_player(player, viewpoint_player),
            int(row[1]),
            tile_id,
            relative_player(int(row[3]), viewpoint_player),
            int(row[4]),
            int(row[5]),
            flags,
        ],
        dtype=np.uint8,
    )


def encode_history_from_events(event_rows, end_index, viewpoint_player, max_len=DEFAULT_HISTORY_MAX_LEN):
    max_len = int(max_len)
    history = np.zeros((max_len, HISTORY_NUM_FIELDS), dtype=np.uint8)
    end_index = int(end_index)
    start = max(0, end_index - max_len)
    length = max(0, end_index - start)
    if length == 0:
        return history, 0
    rows = event_rows[start:end_index]
    viewpoint_player = int(viewpoint_player)
    players = rows[:, 0].astype(np.int16, copy=False)
    targets = rows[:, 3].astype(np.int16, copy=False)
    tiles = rows[:, 2].copy()
    flags = rows[:, 6]

    private_mask = (flags & FLAG_PRIVATE_TILE) != 0
    tiles[private_mask & (players != viewpoint_player)] = TILE_UNKNOWN

    encoded = np.empty((length, HISTORY_NUM_FIELDS), dtype=np.uint8)
    encoded[:, 0] = np.where(players == PLAYER_NONE, PLAYER_NONE, (players + 4 - viewpoint_player) % 4)
    encoded[:, 1] = rows[:, 1]
    encoded[:, 2] = tiles
    encoded[:, 3] = np.where(targets == PLAYER_NONE, PLAYER_NONE, (targets + 4 - viewpoint_player) % 4)
    encoded[:, 4] = rows[:, 4]
    encoded[:, 5] = rows[:, 5]
    encoded[:, 6] = flags
    history[:length] = encoded
    return history, length


class HistoryFeatureBuilder:
    def __init__(self, max_len=DEFAULT_HISTORY_MAX_LEN, track_inline=True):
        if max_len <= 0:
            raise ValueError('history max_len must be positive')
        self.max_len = int(max_len)
        self.track_inline = track_inline
        self.buffers = None
        self.lengths = None
        if self.track_inline:
            self.buffers = np.zeros((4, self.max_len, HISTORY_NUM_FIELDS), dtype=np.uint8)
            self.lengths = np.zeros(4, dtype=np.uint16)
        self.events = []

    def append(self, event):
        self.events.append(event_to_row(event))
        if not self.track_inline:
            return
        for viewpoint_player in range(4):
            row = encode_event(event, viewpoint_player)
            length = int(self.lengths[viewpoint_player])
            if length < self.max_len:
                self.buffers[viewpoint_player, length] = row
                self.lengths[viewpoint_player] = length + 1
            else:
                self.buffers[viewpoint_player, :-1] = self.buffers[viewpoint_player, 1:].copy()
                self.buffers[viewpoint_player, -1] = row

    def snapshot(self, viewpoint_player):
        if not self.track_inline:
            raise RuntimeError('HistoryFeatureBuilder snapshot requires track_inline=True')
        viewpoint_player = int(viewpoint_player)
        return self.buffers[viewpoint_player].copy(), int(self.lengths[viewpoint_player])

    def event_count(self):
        return len(self.events)

    def events_array(self):
        if not self.events:
            return np.zeros((0, HISTORY_NUM_FIELDS), dtype=np.uint8)
        return np.stack(self.events).astype(np.uint8, copy=False)
