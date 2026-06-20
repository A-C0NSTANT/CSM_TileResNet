from agent import MahjongGBAgent
from collections import defaultdict
import numpy as np
import os


NUM_TILE_TYPES = 34
PUBLIC_FEATURE_SIZE = 442
SHANTEN_CACHE = {}
USE_EXACT_PUBLIC_SHANTEN = os.environ.get('PUBLIC_EXACT_SHANTEN', '0').lower() in ('1', 'true', 'yes', 'on')

try:
    from MahjongGB import MahjongFanCalculator
except:
    print('MahjongGB library required! Please visit https://github.com/ailab-pku/PyMahjongGB for more information.')
    raise


def _can_sequence(index):
    return index < 27 and index % 9 <= 6


def _can_adjacent(index):
    return index < 27 and index % 9 <= 7


def _can_gap(index):
    return index < 27 and index % 9 <= 6


def estimate_shanten_from_counts(counts, open_melds):
    key = (tuple(int(x) for x in counts), int(open_melds))
    if key in SHANTEN_CACHE:
        return SHANTEN_CACHE[key]

    work = list(key[0])
    max_closed_melds = max(0, 4 - open_melds)
    best = [8]

    def update_best(melds, taatsu, pair):
        melds = min(melds, max_closed_melds)
        total_melds = open_melds + melds
        taatsu = min(taatsu, max(0, 4 - total_melds))
        best[0] = min(best[0], 8 - 2 * total_melds - taatsu - pair)

    def dfs(start, melds, taatsu, pair):
        while start < NUM_TILE_TYPES and work[start] == 0:
            start += 1
        if start >= NUM_TILE_TYPES:
            update_best(melds, taatsu, pair)
            return

        work[start] -= 1
        dfs(start, melds, taatsu, pair)
        work[start] += 1

        if melds < max_closed_melds and work[start] >= 3:
            work[start] -= 3
            dfs(start, melds + 1, taatsu, pair)
            work[start] += 3

        if melds < max_closed_melds and _can_sequence(start) and work[start + 1] > 0 and work[start + 2] > 0:
            work[start] -= 1
            work[start + 1] -= 1
            work[start + 2] -= 1
            dfs(start, melds + 1, taatsu, pair)
            work[start] += 1
            work[start + 1] += 1
            work[start + 2] += 1

        if pair == 0 and work[start] >= 2:
            work[start] -= 2
            dfs(start, melds, taatsu, 1)
            work[start] += 2

        if melds + taatsu < max_closed_melds and work[start] >= 2:
            work[start] -= 2
            dfs(start, melds, taatsu + 1, pair)
            work[start] += 2

        if melds + taatsu < max_closed_melds and _can_adjacent(start) and work[start + 1] > 0:
            work[start] -= 1
            work[start + 1] -= 1
            dfs(start, melds, taatsu + 1, pair)
            work[start] += 1
            work[start + 1] += 1

        if melds + taatsu < max_closed_melds and _can_gap(start) and work[start + 2] > 0:
            work[start] -= 1
            work[start + 2] -= 1
            dfs(start, melds, taatsu + 1, pair)
            work[start] += 1
            work[start + 2] += 1

    dfs(0, 0, 0, 0)
    if open_melds == 0:
        pairs = sum(1 for count in counts if count >= 2)
        unique = sum(1 for count in counts if count > 0)
        best[0] = min(best[0], 6 - pairs + max(0, 7 - unique))

    value = max(-1, best[0])
    SHANTEN_CACHE[key] = value
    return value


def estimate_shanten_fast(counts, open_melds):
    counts = [int(x) for x in counts]
    work = counts[:]
    max_closed_melds = max(0, 4 - open_melds)
    melds = 0
    taatsu = 0
    pair = 0

    for i in range(NUM_TILE_TYPES):
        while melds < max_closed_melds and work[i] >= 3:
            work[i] -= 3
            melds += 1

    for start in (0, 9, 18):
        for i in range(start, start + 7):
            while melds < max_closed_melds and work[i] > 0 and work[i + 1] > 0 and work[i + 2] > 0:
                work[i] -= 1
                work[i + 1] -= 1
                work[i + 2] -= 1
                melds += 1

    for i in range(NUM_TILE_TYPES):
        if pair == 0 and work[i] >= 2:
            pair = 1
            work[i] -= 2
            break

    for i in range(NUM_TILE_TYPES):
        if melds + taatsu >= max_closed_melds:
            break
        if work[i] >= 2:
            taatsu += 1
            work[i] -= 2

    for start in (0, 9, 18):
        for i in range(start, start + 8):
            while melds + taatsu < max_closed_melds and work[i] > 0 and work[i + 1] > 0:
                work[i] -= 1
                work[i + 1] -= 1
                taatsu += 1
        for i in range(start, start + 7):
            while melds + taatsu < max_closed_melds and work[i] > 0 and work[i + 2] > 0:
                work[i] -= 1
                work[i + 2] -= 1
                taatsu += 1

    total_melds = open_melds + melds
    taatsu = min(taatsu, max(0, 4 - total_melds))
    normal = 8 - 2 * total_melds - taatsu - pair
    if open_melds == 0:
        pairs = sum(1 for count in counts if count >= 2)
        unique = sum(1 for count in counts if count > 0)
        normal = min(normal, 6 - pairs + max(0, 7 - unique))
    return max(-1, normal)


def estimate_effective_tile_type_count_fast(counts, remaining_counts, shanten):
    effective = 0
    for tile_id in range(NUM_TILE_TYPES):
        if remaining_counts[tile_id] <= 0:
            continue
        count = counts[tile_id]
        if count >= 2:
            effective += 1
            continue
        if tile_id < 27:
            rank = tile_id % 9
            suit_start = tile_id - rank
            near = False
            for delta in (-2, -1, 1, 2):
                other_rank = rank + delta
                if 0 <= other_rank < 9 and counts[suit_start + other_rank] > 0:
                    near = True
                    break
            if near:
                effective += 1
        elif count == 1 and shanten <= 2:
            effective += 1
    return effective

class FeatureAgent(MahjongGBAgent):
    
    '''
    observation: 6*4*9
        (men+quan+hand4)*4*9
    action_mask: 235
        pass1+hu1+discard34+chi63(3*7*3)+peng34+gang34+angang34+bugang34
    '''
    
    OBS_SIZE = 6
    ACT_SIZE = 235
    PUBLIC_FEATURE_SIZE = PUBLIC_FEATURE_SIZE
    
    OFFSET_OBS = {
        'SEAT_WIND' : 0,
        'PREVALENT_WIND' : 1,
        'HAND' : 2
    }
    OFFSET_ACT = {
        'Pass' : 0,
        'Hu' : 1,
        'Play' : 2,
        'Chi' : 36,
        'Peng' : 99,
        'Gang' : 133,
        'AnGang' : 167,
        'BuGang' : 201
    }
    TILE_LIST = [
        *('W%d'%(i+1) for i in range(9)),
        *('T%d'%(i+1) for i in range(9)),
        *('B%d'%(i+1) for i in range(9)),
        *('F%d'%(i+1) for i in range(4)),
        *('J%d'%(i+1) for i in range(3))
    ]
    OFFSET_TILE = {c : i for i, c in enumerate(TILE_LIST)}
    
    def __init__(self, seatWind, public_for_single_action = True):
        self.seatWind = seatWind
        self.public_for_single_action = public_for_single_action
        self.packs = [[] for i in range(4)]
        self.history = [[] for i in range(4)]
        self.tileWall = [21] * 4
        self.shownTiles = defaultdict(int)
        self.wallLast = False
        self.isAboutKong = False
        self.obs = np.zeros((self.OBS_SIZE, 36), dtype=np.int8)
        self.obs[self.OFFSET_OBS['SEAT_WIND']][self.OFFSET_TILE['F%d' % (self.seatWind + 1)]] = 1
        self.hand_counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        self.discard_counts = np.zeros((4, NUM_TILE_TYPES), dtype=np.float32)
        self.shown_counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        self.meld_visible_counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        self.meld_by_player = np.zeros((3, NUM_TILE_TYPES), dtype=np.float32)
        self._public_buffer = np.empty(self.PUBLIC_FEATURE_SIZE, dtype=np.float32)
        self._visible_counts = np.empty(NUM_TILE_TYPES, dtype=np.float32)
        self._remaining_counts = np.empty(NUM_TILE_TYPES, dtype=np.float32)
        self._opponent_discard_sum = np.empty(NUM_TILE_TYPES, dtype=np.float32)
        self._clipped_opponent_discard = np.empty((3, NUM_TILE_TYPES), dtype=np.float32)
        self._clipped_meld_visible = np.empty(NUM_TILE_TYPES, dtype=np.float32)
        self._clipped_meld_by_player = np.empty((3, NUM_TILE_TYPES), dtype=np.float32)
        self._opponent_feature_buffer = np.empty(27, dtype=np.float32)
        self._game_stage_buffer = np.empty(7, dtype=np.float32)
    
    '''
    Wind 0..3
    Deal XX XX ...
    Player N Draw
    Player N Gang
    Player N(me) AnGang XX
    Player N(me) Play XX
    Player N(me) BuGang XX
    Player N(not me) Peng
    Player N(not me) Chi XX
    Player N(not me) AnGang
    
    Player N Hu
    Huang
    Player N Invalid
    Draw XX
    Player N(not me) Play XX
    Player N(not me) BuGang XX
    Player N(me) Peng
    Player N(me) Chi XX
    '''
    def request2obs(self, request):
        t = request.split()
        if t[0] == 'Wind':
            self.prevalentWind = int(t[1])
            self.obs[self.OFFSET_OBS['PREVALENT_WIND']][self.OFFSET_TILE['F%d' % (self.prevalentWind + 1)]] = 1
            return
        if t[0] == 'Deal':
            self.hand = t[1:]
            self._hand_embedding_update()
            return
        if t[0] == 'Huang':
            self.valid = []
            return self._obs()
        if t[0] == 'Draw':
            # Available: Hu, Play, AnGang, BuGang
            self.tileWall[0] -= 1
            self.wallLast = self.tileWall[1] == 0
            tile = t[1]
            self.valid = []
            if self._check_mahjong(tile, isSelfDrawn = True, isAboutKong = self.isAboutKong):
                self.valid.append(self.OFFSET_ACT['Hu'])
            self.isAboutKong = False
            self.hand.append(tile)
            self._hand_embedding_update()
            for tile in set(self.hand):
                self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                if self.hand.count(tile) == 4 and not self.wallLast and self.tileWall[0] > 0:
                    self.valid.append(self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[tile])
            if not self.wallLast and self.tileWall[0] > 0:
                for packType, tile, offer in self.packs[0]:
                    if packType == 'PENG' and tile in self.hand:
                        self.valid.append(self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[tile])
            return self._obs()
        # Player N Invalid/Hu/Draw/Play/Chi/Peng/Gang/AnGang/BuGang XX
        p = (int(t[1]) + 4 - self.seatWind) % 4
        if t[2] == 'Draw':
            self.tileWall[p] -= 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            return
        if t[2] == 'Invalid':
            self.valid = []
            return self._obs()
        if t[2] == 'Hu':
            self.valid = []
            return self._obs()
        if t[2] == 'Play':
            self.tileFrom = p
            self.curTile = t[3]
            self.history[p].append(self.curTile)
            self._add_discard_tile(p, self.curTile)
            if p == 0:
                self.hand.remove(self.curTile)
                self._hand_embedding_update()
                return
            else:
                # Available: Hu/Gang/Peng/Chi/Pass
                self.valid = []
                if self._check_mahjong(self.curTile):
                    self.valid.append(self.OFFSET_ACT['Hu'])
                if not self.wallLast:
                    if self.hand.count(self.curTile) >= 2:
                        self.valid.append(self.OFFSET_ACT['Peng'] + self.OFFSET_TILE[self.curTile])
                        if self.hand.count(self.curTile) == 3 and self.tileWall[0]:
                            self.valid.append(self.OFFSET_ACT['Gang'] + self.OFFSET_TILE[self.curTile])
                    color = self.curTile[0]
                    if p == 3 and color in 'WTB':
                        num = int(self.curTile[1])
                        tmp = []
                        for i in range(-2, 3): tmp.append(color + str(num + i))
                        if tmp[0] in self.hand and tmp[1] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 3) * 3 + 2)
                        if tmp[1] in self.hand and tmp[3] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 2) * 3 + 1)
                        if tmp[3] in self.hand and tmp[4] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 1) * 3)
                self.valid.append(self.OFFSET_ACT['Pass'])
                return self._obs()
        if t[2] == 'Chi':
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            self.packs[p].append(('CHI', tile, int(self.curTile[1]) - num + 2))
            self._add_meld_counts(p, 'CHI', tile)
            self._add_shown_tile(self.curTile, -1)
            for i in range(-1, 2):
                self._add_shown_tile(color + str(num + i))
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            if p == 0:
                # Available: Play
                self.valid = []
                self.hand.append(self.curTile)
                for i in range(-1, 2):
                    self.hand.remove(color + str(num + i))
                self._hand_embedding_update()
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                return self._obs()
            else:
                return
        if t[2] == 'UnChi':
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            packType, packTile, _ = self.packs[p].pop()
            self._add_meld_counts(p, packType, packTile, -1)
            self._add_shown_tile(self.curTile)
            for i in range(-1, 2):
                self._add_shown_tile(color + str(num + i), -1)
            if p == 0:
                for i in range(-1, 2):
                    self.hand.append(color + str(num + i))
                self.hand.remove(self.curTile)
                self._hand_embedding_update()
            return
        if t[2] == 'Peng':
            self.packs[p].append(('PENG', self.curTile, (4 + p - self.tileFrom) % 4))
            self._add_meld_counts(p, 'PENG', self.curTile)
            self._add_shown_tile(self.curTile, 2)
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            if p == 0:
                # Available: Play
                self.valid = []
                for i in range(2):
                    self.hand.remove(self.curTile)
                self._hand_embedding_update()
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                return self._obs()
            else:
                return
        if t[2] == 'UnPeng':
            packType, packTile, _ = self.packs[p].pop()
            self._add_meld_counts(p, packType, packTile, -1)
            self._add_shown_tile(self.curTile, -2)
            if p == 0:
                for i in range(2):
                    self.hand.append(self.curTile)
                self._hand_embedding_update()
            return
        if t[2] == 'Gang':
            self.packs[p].append(('GANG', self.curTile, (4 + p - self.tileFrom) % 4))
            self._add_meld_counts(p, 'GANG', self.curTile)
            self._add_shown_tile(self.curTile, 3)
            if p == 0:
                for i in range(3):
                    self.hand.remove(self.curTile)
                self._hand_embedding_update()
                self.isAboutKong = True
            return
        if t[2] == 'AnGang':
            tile = 'CONCEALED' if p else t[3]
            self.packs[p].append(('GANG', tile, 0))
            self._add_meld_counts(p, 'GANG', tile)
            if p == 0:
                self.isAboutKong = True
                for i in range(4):
                    self.hand.remove(tile)
            else:
                self.isAboutKong = False
            return
        if t[2] == 'BuGang':
            tile = t[3]
            for i in range(len(self.packs[p])):
                if tile == self.packs[p][i][1]:
                    oldPackType, oldTile, _ = self.packs[p][i]
                    self._add_meld_counts(p, oldPackType, oldTile, -1)
                    self.packs[p][i] = ('GANG', tile, self.packs[p][i][2])
                    self._add_meld_counts(p, 'GANG', tile)
                    break
            self._add_shown_tile(tile)
            if p == 0:
                self.hand.remove(tile)
                self._hand_embedding_update()
                self.isAboutKong = True
                return
            else:
                # Available: Hu/Pass
                self.valid = []
                if self._check_mahjong(tile, isSelfDrawn = False, isAboutKong = True):
                    self.valid.append(self.OFFSET_ACT['Hu'])
                self.valid.append(self.OFFSET_ACT['Pass'])
                return self._obs()
        raise NotImplementedError('Unknown request %s!' % request)
    
    '''
    Pass
    Hu
    Play XX
    Chi XX
    Peng9
    Gang
    (An)Gang XX
    BuGang XX
    '''
    def action2response(self, action):
        if action < self.OFFSET_ACT['Hu']:
            return 'Pass'
        if action < self.OFFSET_ACT['Play']:
            return 'Hu'
        if action < self.OFFSET_ACT['Chi']:
            return 'Play ' + self.TILE_LIST[action - self.OFFSET_ACT['Play']]
        if action < self.OFFSET_ACT['Peng']:
            t = (action - self.OFFSET_ACT['Chi']) // 3
            return 'Chi ' + 'WTB'[t // 7] + str(t % 7 + 2)
        if action < self.OFFSET_ACT['Gang']:
            return 'Peng'
        if action < self.OFFSET_ACT['AnGang']:
            return 'Gang'
        if action < self.OFFSET_ACT['BuGang']:
            return 'Gang ' + self.TILE_LIST[action - self.OFFSET_ACT['AnGang']]
        return 'BuGang ' + self.TILE_LIST[action - self.OFFSET_ACT['BuGang']]
    
    '''
    Pass
    Hu
    Play XX
    Chi XX
    Peng
    Gang
    (An)Gang XX
    BuGang XX
    '''
    def response2action(self, response):
        t = response.split()
        if t[0] == 'Pass': return self.OFFSET_ACT['Pass']
        if t[0] == 'Hu': return self.OFFSET_ACT['Hu']
        if t[0] == 'Play': return self.OFFSET_ACT['Play'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'Chi': return self.OFFSET_ACT['Chi'] + 'WTB'.index(t[1][0]) * 7 * 3 + (int(t[2][1]) - 2) * 3 + int(t[1][1]) - int(t[2][1]) + 1
        if t[0] == 'Peng': return self.OFFSET_ACT['Peng'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'Gang': return self.OFFSET_ACT['Gang'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'AnGang': return self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'BuGang': return self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[t[1]]
        return self.OFFSET_ACT['Pass']
    
    def _obs(self):
        mask = np.zeros(self.ACT_SIZE, dtype=np.int8)
        for a in self.valid:
            mask[a] = 1
        obs = {
            'observation': self.obs.reshape((self.OBS_SIZE, 4, 9)).copy(),
            'action_mask': mask
        }
        if self.public_for_single_action or mask.sum() > 1:
            obs['public'] = self._public_features()
        return obs

    def _count_tiles(self, tiles):
        counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        for tile in tiles:
            if tile in self.OFFSET_TILE:
                counts[self.OFFSET_TILE[tile]] += 1
        return counts

    def _tile_index(self, tile):
        return self.OFFSET_TILE[tile]

    def _add_hand_tile(self, tile):
        self.hand_counts[self._tile_index(tile)] += 1

    def _remove_hand_tile(self, tile):
        self.hand_counts[self._tile_index(tile)] -= 1

    def _add_shown_tile(self, tile, value=1):
        self.shown_counts[self._tile_index(tile)] += value
        self.shownTiles[tile] += value

    def _add_discard_tile(self, player, tile):
        self.discard_counts[player][self._tile_index(tile)] += 1
        self._add_shown_tile(tile)

    def _meld_tile_indices(self, packType, tile):
        if tile == 'CONCEALED' or tile not in self.OFFSET_TILE:
            return []
        if packType == 'CHI':
            color = tile[0]
            if color not in 'WTB':
                return []
            num = int(tile[1])
            return [self.OFFSET_TILE[color + str(num + offset)] for offset in (-1, 0, 1)]
        count = 4 if packType == 'GANG' else 3
        return [self.OFFSET_TILE[tile]] * count

    def _add_meld_counts(self, player, packType, tile, value=1):
        for tile_id in self._meld_tile_indices(packType, tile):
            delta = float(value)
            self.meld_visible_counts[tile_id] += delta
            if player != 0:
                self.meld_by_player[player - 1][tile_id] += delta

    def _add_meld_tiles(self, counts, packType, tile, value=1):
        if tile == 'CONCEALED' or tile not in self.OFFSET_TILE:
            return
        if packType == 'CHI':
            color = tile[0]
            if color not in 'WTB':
                return
            num = int(tile[1])
            for offset in (-1, 0, 1):
                meld_tile = color + str(num + offset)
                if meld_tile in self.OFFSET_TILE:
                    counts[self.OFFSET_TILE[meld_tile]] += value
        else:
            counts[self.OFFSET_TILE[tile]] += value * (4 if packType == 'GANG' else 3)

    def _meld_visible_counts(self):
        total = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        by_player = np.zeros((3, NUM_TILE_TYPES), dtype=np.float32)
        for player in range(4):
            for packType, tile, _ in self.packs[player]:
                if player == 0:
                    self._add_meld_tiles(total, packType, tile)
                else:
                    before = total.copy()
                    self._add_meld_tiles(total, packType, tile)
                    by_player[player - 1] += total - before
        return total, by_player

    def _opponent_features(self):
        features = self._opponent_feature_buffer
        offset = 0
        for player in range(1, 4):
            packs = self.packs[player]
            meld_count = len(packs)
            discard_count = len(self.history[player])
            has_honor_triplet = 0.0
            suit_melds = [0.0, 0.0, 0.0]
            for packType, tile, _ in packs:
                if tile == 'CONCEALED' or tile not in self.OFFSET_TILE:
                    continue
                if tile[0] in 'FJ' and packType in ('PENG', 'GANG'):
                    has_honor_triplet = 1.0
                if tile[0] in 'WTB':
                    suit_melds['WTB'.index(tile[0])] += 1
            denom = max(1.0, float(meld_count))
            suit_ratio0 = suit_melds[0] / denom
            suit_ratio1 = suit_melds[1] / denom
            suit_ratio2 = suit_melds[2] / denom
            features[offset] = min(meld_count / 4.0, 1.0)
            features[offset + 1] = min(discard_count / 24.0, 1.0)
            features[offset + 2] = 1.0 if meld_count > 0 else 0.0
            features[offset + 3] = 1.0 if meld_count >= 2 else 0.0
            features[offset + 4] = has_honor_triplet
            features[offset + 5] = suit_ratio0
            features[offset + 6] = suit_ratio1
            features[offset + 7] = suit_ratio2
            features[offset + 8] = max(suit_ratio0, suit_ratio1, suit_ratio2) if meld_count else 0.0
            offset += 9
        return features

    def _effective_tile_type_count(self, self_counts, remaining_counts, shanten):
        effective = 0
        for tile_id in range(NUM_TILE_TYPES):
            if remaining_counts[tile_id] <= 0:
                continue
            test_counts = self_counts.astype(np.int16).copy()
            test_counts[tile_id] += 1
            if estimate_shanten_from_counts(test_counts.tolist(), len(self.packs[0])) < shanten:
                effective += 1
        return effective

    def _game_stage_features(self, self_counts, remaining_counts):
        turn_index = sum(len(x) for x in self.history)
        wall_remaining = sum(self.tileWall)
        counts = self_counts.astype(np.int16).tolist()
        if USE_EXACT_PUBLIC_SHANTEN:
            shanten = estimate_shanten_from_counts(counts, len(self.packs[0]))
            effective = self._effective_tile_type_count(self_counts, remaining_counts, shanten)
        else:
            shanten = estimate_shanten_fast(counts, len(self.packs[0]))
            effective = estimate_effective_tile_type_count_fast(counts, remaining_counts, shanten)
        features = self._game_stage_buffer
        features[0] = min(turn_index / 80.0, 1.0)
        features[1] = min(wall_remaining / 84.0, 1.0)
        features[2:5] = 0.0
        if wall_remaining > 56:
            features[2] = 1.0
        elif wall_remaining > 28:
            features[3] = 1.0
        else:
            features[4] = 1.0
        features[5] = min((shanten + 1) / 8.0, 1.0)
        features[6] = min(effective / 34.0, 1.0)
        return features

    def _public_features(self):
        self_counts = self.hand_counts
        opponent_discard_by_player = self.discard_counts[1:4]
        np.add(opponent_discard_by_player[0], opponent_discard_by_player[1], out=self._opponent_discard_sum)
        np.add(self._opponent_discard_sum, opponent_discard_by_player[2], out=self._opponent_discard_sum)
        np.add(self.shown_counts, self_counts, out=self._visible_counts)
        np.clip(self._visible_counts, 0, 4, out=self._visible_counts)
        np.subtract(4, self._visible_counts, out=self._remaining_counts)
        np.clip(self._remaining_counts, 0, 4, out=self._remaining_counts)
        np.clip(self.meld_visible_counts, 0, 4, out=self._clipped_meld_visible)
        np.clip(opponent_discard_by_player, 0, 4, out=self._clipped_opponent_discard)
        np.clip(self.meld_by_player, 0, 4, out=self._clipped_meld_by_player)

        public = self._public_buffer
        offset = 0
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
        public[offset:offset + 27] = self._opponent_features()
        offset += 27
        public[offset:offset + 7] = self._game_stage_features(self_counts, self._remaining_counts)
        offset += 7
        assert offset == self.PUBLIC_FEATURE_SIZE
        return public.copy()
    
    def _hand_embedding_update(self):
        self.obs[self.OFFSET_OBS['HAND'] : ] = 0
        self.hand_counts.fill(0)
        d = defaultdict(int)
        for tile in self.hand:
            d[tile] += 1
            self.hand_counts[self.OFFSET_TILE[tile]] += 1
        for tile in d:
            self.obs[self.OFFSET_OBS['HAND'] : self.OFFSET_OBS['HAND'] + d[tile], self.OFFSET_TILE[tile]] = 1
    
    def _check_mahjong(self, winTile, isSelfDrawn = False, isAboutKong = False):
        try:
            fans = MahjongFanCalculator(
                pack = tuple(self.packs[0]),
                hand = tuple(self.hand),
                winTile = winTile,
                flowerCount = 0,
                isSelfDrawn = isSelfDrawn,
                is4thTile = self.shownTiles[winTile] == 4,
                isAboutKong = isAboutKong,
                isWallLast = self.wallLast,
                seatWind = self.seatWind,
                prevalentWind = self.prevalentWind,
                verbose = True
            )
            fanCnt = 0
            for fanPoint, cnt, fanName, fanNameEn in fans:
                fanCnt += fanPoint * cnt
            if fanCnt < 8: raise Exception('Not Enough Fans')
        except:
            return False
        return True
