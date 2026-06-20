from feature import FeatureAgent
from history_features import (
    DEFAULT_HISTORY_MAX_LEN,
    HISTORY_EVENT_FIELD_NAMES,
    HISTORY_FIELD_NAMES,
    HistoryActionType,
    HistoryEvent,
    HistoryFeatureBuilder,
    HistoryPhase,
    PLAYER_NONE,
    tile_to_id,
    wall_bucket_from_remaining,
)
import numpy as np
import json
from pathlib import Path
import os
import time


DATA_DIR = Path(os.environ.get('DATA_DIR', 'data'))
DATA_FILE = DATA_DIR / 'data.txt'
HISTORY_MAX_LEN = int(os.environ.get('HISTORY_MAX_LEN', DEFAULT_HISTORY_MAX_LEN))
INCLUDE_HISTORY = os.environ.get('INCLUDE_HISTORY', '1').lower() not in ('0', 'false', 'no', 'off')
HISTORY_STORAGE = os.environ.get('HISTORY_STORAGE', 'indexed').lower()
if HISTORY_STORAGE not in ('indexed', 'inline'):
    raise ValueError('HISTORY_STORAGE must be indexed or inline')

obs = [[] for i in range(4)]
actions = [[] for i in range(4)]
histories = [[] for i in range(4)]
history_lengths = [[] for i in range(4)]
history_event_indexes = [[] for i in range(4)]
history_viewpoints = [[] for i in range(4)]
matchid = -1
history_builder = None
curTile = None
curPlayer = PLAYER_NONE

l = []
start_time = time.perf_counter()
saved_samples = 0

def filterData():
    global obs
    global actions
    global histories
    global history_lengths
    global history_event_indexes
    global history_viewpoints
    newobs = [[] for i in range(4)]
    newactions = [[] for i in range(4)]
    newhistories = [[] for i in range(4)] if INCLUDE_HISTORY else histories
    newhistory_lengths = [[] for i in range(4)] if INCLUDE_HISTORY else history_lengths
    newhistory_event_indexes = [[] for i in range(4)] if INCLUDE_HISTORY else history_event_indexes
    newhistory_viewpoints = [[] for i in range(4)] if INCLUDE_HISTORY else history_viewpoints
    for i in range(4):
        for j, o in enumerate(obs[i]):
            if o['action_mask'].sum() > 1: # ignore states with single valid action (Pass)
                newobs[i].append(o)
                newactions[i].append(actions[i][j])
                if INCLUDE_HISTORY:
                    if HISTORY_STORAGE == 'inline':
                        newhistories[i].append(histories[i][j])
                    else:
                        newhistory_event_indexes[i].append(history_event_indexes[i][j])
                        newhistory_viewpoints[i].append(history_viewpoints[i][j])
                    newhistory_lengths[i].append(history_lengths[i][j])
    obs = newobs
    actions = newactions
    histories = newhistories
    history_lengths = newhistory_lengths
    history_event_indexes = newhistory_event_indexes
    history_viewpoints = newhistory_viewpoints

def _current_wall_bucket():
    if 'agents' not in globals():
        return wall_bucket_from_remaining(None)
    return wall_bucket_from_remaining(sum(agents[0].tileWall))

def appendHistoryEvent(player, action_type, tile=None, target_player=PLAYER_NONE, phase=HistoryPhase.NORMAL, private_tile=False):
    if not INCLUDE_HISTORY or history_builder is None:
        return
    history_builder.append(HistoryEvent(
        player=player,
        action_type=action_type,
        tile_id=tile_to_id(tile),
        target_player=target_player,
        wall_bucket=_current_wall_bucket(),
        phase=phase,
        private_tile=private_tile,
    ))

def appendSample(player, sample_obs, action=0):
    obs[player].append(sample_obs)
    actions[player].append(action)
    if INCLUDE_HISTORY:
        if HISTORY_STORAGE == 'inline':
            history, history_length = history_builder.snapshot(player)
            histories[player].append(history)
            history_lengths[player].append(history_length)
        else:
            event_index = history_builder.event_count()
            history_event_indexes[player].append(event_index)
            history_viewpoints[player].append(player)
            history_lengths[player].append(min(event_index, HISTORY_MAX_LEN))

def saveData():
    global saved_samples
    assert [len(x) for x in obs] == [len(x) for x in actions], 'obs actions not matching!'
    if INCLUDE_HISTORY:
        if HISTORY_STORAGE == 'inline':
            assert [len(x) for x in obs] == [len(x) for x in histories], 'obs histories not matching!'
        else:
            assert [len(x) for x in obs] == [len(x) for x in history_event_indexes], 'obs history_event_indexes not matching!'
            assert [len(x) for x in obs] == [len(x) for x in history_viewpoints], 'obs history_viewpoints not matching!'
        assert [len(x) for x in obs] == [len(x) for x in history_lengths], 'obs history_lengths not matching!'
    sample_count = sum([len(x) for x in obs])
    l.append(sample_count)
    saved_samples += sample_count
    arrays = {
        'obs': np.stack([x['observation'] for i in range(4) for x in obs[i]]).astype(np.int8),
        'mask': np.stack([x['action_mask'] for i in range(4) for x in obs[i]]).astype(np.int8),
        'public': np.stack([x['public'] for i in range(4) for x in obs[i]]).astype(np.float32),
        'act': np.array([x for i in range(4) for x in actions[i]]),
    }
    if INCLUDE_HISTORY:
        arrays['history_length'] = np.array([x for i in range(4) for x in history_lengths[i]], dtype=np.uint16)
        arrays['history_field_names'] = np.array(HISTORY_FIELD_NAMES)
        arrays['history_storage'] = np.array(HISTORY_STORAGE)
        arrays['history_max_len'] = np.array(HISTORY_MAX_LEN, dtype=np.uint16)
        if HISTORY_STORAGE == 'inline':
            arrays['history'] = np.stack([x for i in range(4) for x in histories[i]]).astype(np.uint8)
        else:
            arrays['history_events'] = history_builder.events_array()
            arrays['history_event_index'] = np.array([x for i in range(4) for x in history_event_indexes[i]], dtype=np.uint16)
            arrays['history_viewpoint'] = np.array([x for i in range(4) for x in history_viewpoints[i]], dtype=np.uint8)
            arrays['history_event_field_names'] = np.array(HISTORY_EVENT_FIELD_NAMES)
    np.savez(DATA_DIR / ('%d.npz' % matchid), **arrays)
    for x in obs: x.clear()
    for x in actions: x.clear()
    for x in histories: x.clear()
    for x in history_lengths: x.clear()
    for x in history_event_indexes: x.clear()
    for x in history_viewpoints: x.clear()

DATA_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA_FILE, encoding='UTF-8') as f:
    line = f.readline()
    while line:
        t = line.split()
        if len(t) == 0:
            line = f.readline()
            continue
        if t[0] == 'Match':
            agents = [FeatureAgent(i, public_for_single_action=False) for i in range(4)]
            history_builder = HistoryFeatureBuilder(HISTORY_MAX_LEN, track_inline=HISTORY_STORAGE == 'inline')
            curTile = None
            curPlayer = PLAYER_NONE
            matchid += 1
            if matchid % 128 == 0:
                elapsed = time.perf_counter() - start_time
                speed = saved_samples / max(elapsed, 1e-6)
                print('Processing match %d %s... samples=%d samples/sec=%.2f elapsed=%.1fs' % (matchid, t[1], saved_samples, speed, elapsed))
        elif t[0] == 'Wind':
            for agent in agents:
                agent.request2obs(line)
        elif t[0] == 'Player':
            p = int(t[1])
            if t[2] == 'Deal':
                agents[p].request2obs(' '.join(t[2:]))
            elif t[2] == 'Draw':
                draw_obs = None
                for i in range(4):
                    if i == p:
                        draw_obs = agents[p].request2obs(' '.join(t[2:]))
                    else:
                        agents[i].request2obs(' '.join(t[:3]))
                appendHistoryEvent(p, HistoryActionType.DRAW, t[3], phase=HistoryPhase.SELF_DRAW_DECISION, private_tile=True)
                appendSample(p, draw_obs)
            elif t[2] == 'Play':
                actions[p].pop()
                actions[p].append(agents[p].response2action(' '.join(t[2:])))
                appendHistoryEvent(p, HistoryActionType.PLAY, t[3])
                for i in range(4):
                    if i == p:
                        agents[p].request2obs(line)
                    else:
                        appendSample(i, agents[i].request2obs(line))
                curTile = t[3]
                curPlayer = p
            elif t[2] == 'Chi':
                actions[p].pop()
                actions[p].append(agents[p].response2action('Chi %s %s' % (curTile, t[3])))
                appendHistoryEvent(p, HistoryActionType.CHI, t[3], target_player=curPlayer, phase=HistoryPhase.POST_MELD_DISCARD)
                for i in range(4):
                    if i == p:
                        appendSample(p, agents[p].request2obs('Player %d Chi %s' % (p, t[3])))
                    else:
                        agents[i].request2obs('Player %d Chi %s' % (p, t[3]))
            elif t[2] == 'Peng':
                actions[p].pop()
                actions[p].append(agents[p].response2action('Peng %s' % t[3]))
                appendHistoryEvent(p, HistoryActionType.PENG, t[3], target_player=curPlayer, phase=HistoryPhase.POST_MELD_DISCARD)
                for i in range(4):
                    if i == p:
                        appendSample(p, agents[p].request2obs('Player %d Peng %s' % (p, t[3])))
                    else:
                        agents[i].request2obs('Player %d Peng %s' % (p, t[3]))
            elif t[2] == 'Gang':
                actions[p].pop()
                actions[p].append(agents[p].response2action('Gang %s' % t[3]))
                appendHistoryEvent(p, HistoryActionType.GANG, t[3], target_player=curPlayer)
                for i in range(4):
                    agents[i].request2obs('Player %d Gang %s' % (p, t[3]))
            elif t[2] == 'AnGang':
                actions[p].pop()
                actions[p].append(agents[p].response2action('AnGang %s' % t[3]))
                appendHistoryEvent(p, HistoryActionType.ANGANG, t[3], private_tile=True)
                for i in range(4):
                    if i == p:
                        agents[p].request2obs('Player %d AnGang %s' % (p, t[3]))
                    else:
                        agents[i].request2obs('Player %d AnGang' % p)
            elif t[2] == 'BuGang':
                actions[p].pop()
                actions[p].append(agents[p].response2action('BuGang %s' % t[3]))
                appendHistoryEvent(p, HistoryActionType.BUGANG, t[3])
                curTile = t[3]
                curPlayer = p
                for i in range(4):
                    if i == p:
                        agents[p].request2obs('Player %d BuGang %s' % (p, t[3]))
                    else:
                        appendSample(i, agents[i].request2obs('Player %d BuGang %s' % (p, t[3])))
            elif t[2] == 'Hu':
                actions[p].pop()
                actions[p].append(agents[p].response2action('Hu'))
                appendHistoryEvent(p, HistoryActionType.HU, t[3] if len(t) > 3 else None, target_player=curPlayer if len(t) > 3 else PLAYER_NONE)
            # Deal with Ignore clause
            if t[2] in ['Peng', 'Gang', 'Hu']:
                for k in range(5, 15, 5):
                    if len(t) > k:
                        p = int(t[k + 1])
                        if t[k + 2] == 'Chi':
                            actions[p].pop()
                            actions[p].append(agents[p].response2action('Chi %s %s' % (curTile, t[k + 3])))
                        elif t[k + 2] == 'Peng':
                            actions[p].pop()
                            actions[p].append(agents[p].response2action('Peng %s' % t[k + 3]))
                        elif t[k + 2] == 'Gang':
                            actions[p].pop()
                            actions[p].append(agents[p].response2action('Gang %s' % t[k + 3]))
                        elif t[k + 2] == 'Hu':
                            actions[p].pop()
                            actions[p].append(agents[p].response2action('Hu'))
                    else: break
        elif t[0] == 'Score':
            filterData()
            saveData()
        line = f.readline()
with open(DATA_DIR / 'count.json', 'w') as f:
    json.dump(l, f)
