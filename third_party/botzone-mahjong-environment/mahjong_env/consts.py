from enum import Enum

NUM_PLAYERS = 4
NUM_HAND_TILES = 13


# ??
class RoundWind(Enum):
    EAST = 0
    SOUTH = 1
    WEST = 2
    NORTH = 3


class TileSuit(Enum):
    CHARACTERS = 1  # ?
    BAMBOO = 2  # ?
    DOTS = 3  # ?
    HONORS = 4  # ?


class ActionType(Enum):
    PASS = 0  # ???
    DRAW = 1  # ??
    PLAY = 2  # ??
    CHOW = 3  # ??
    PUNG = 4  # ??
    KONG = 5  # ??
    MELD_KONG = 6  # ??
    HU = 7  # ??


class ClaimingType:
    CHOW = "CHI"
    PUNG = "PENG"
    KONG = "GANG"


TILE_SET = (
    'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9',  # ?
    'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9',  # ?
    'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9',  # ?
    'F1', 'F2', 'F3', 'F4', 'J1', 'J2', 'J3',  # ??
)
