import argparse
import concurrent.futures
import contextlib
import io
import json
import queue
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY_ENV = ROOT / 'third_party' / 'botzone-mahjong-environment'
SENTINEL = '>>>BOTZONE_REQUEST_KEEP_RUNNING<<<'


def add_third_party_path():
    path = str(THIRD_PARTY_ENV)
    if path not in sys.path:
        sys.path.insert(0, path)


add_third_party_path()

from mahjong_env.core import Mahjong
from mahjong_env.consts import ActionType
from mahjong_env.player_data import Action
from mahjong_env.utils import request2str, response2act


def resolve_model_dir(value):
    path = Path(value)
    if path.exists():
        return path.resolve()
    path = ROOT / 'eval' / 'models' / value
    if path.exists():
        return path.resolve()
    raise FileNotFoundError('model directory not found: %s' % value)


def resolve_checkpoint(model_dir, checkpoint_arg):
    if checkpoint_arg:
        path = Path(checkpoint_arg)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise FileNotFoundError('checkpoint not found: %s' % path)
        return path.resolve()

    checkpoint_dir = model_dir / 'checkpoint'
    candidates = sorted(checkpoint_dir.glob('*latest*.pkl')) + sorted(checkpoint_dir.glob('*.pkl'))
    if not candidates:
        raise FileNotFoundError('no checkpoint *.pkl found under %s' % checkpoint_dir)
    return candidates[0].resolve()


def dated_result_dir():
    return ROOT / 'eval' / 'results' / datetime.now().strftime('%Y-%m-%d')


def default_arena_out_path(candidate_name, baseline_name, num_games):
    filename = '%s_vs_%s_%dgames.jsonl' % (candidate_name, baseline_name, num_games)
    return dated_result_dir() / filename


def compute_ranks(scores):
    items = sorted(((int(player), float(score)) for player, score in scores.items()), key=lambda x: (-x[1], x[0]))
    return {str(player): rank + 1 for rank, (player, _) in enumerate(items)}


def detect_invalid(scores):
    values = sorted(int(v) for v in scores.values())
    return values == [-30, 10, 10, 10]


def invalid_player(scores):
    if not detect_invalid(scores):
        return None
    for player, score in scores.items():
        if int(score) == -30:
            return str(player)
    return None


def total_fans(fans):
    if not fans:
        return 0
    return int(sum(int(item[0]) for item in fans))


def infer_win_method(scores, winner, fan_total):
    if winner is None or fan_total <= 0:
        return 'unknown'

    winner_score = int(scores[str(winner)])
    if winner_score == 3 * (8 + fan_total):
        return 'self_draw'
    if winner_score == 3 * 8 + fan_total:
        return 'discard_win'
    return 'unknown'


def infer_discarder(scores, fan_total):
    if fan_total <= 0:
        return None
    discarder_scores = [
        str(player)
        for player, score in scores.items()
        if int(score) == -(8 + fan_total)
    ]
    return discarder_scores[0] if len(discarder_scores) == 1 else None


def build_outcome(env, scores, roles, candidate_seat, invalid, invalid_pid, invalid_role, crash, timeout):
    fan_total = total_fans(env.fans)
    winner = None
    positive_scores = [(int(player), int(score)) for player, score in scores.items() if int(score) > 0]
    if not invalid and positive_scores:
        winner = max(positive_scores, key=lambda item: (item[1], -item[0]))[0]

    if invalid:
        outcome_type = 'illegal'
    elif winner is not None and fan_total > 0:
        outcome_type = 'win'
    elif all(int(score) == 0 for score in scores.values()):
        outcome_type = 'draw'
    elif crash or timeout:
        outcome_type = 'incomplete'
    else:
        outcome_type = 'unknown'

    win_method = infer_win_method(scores, winner, fan_total) if outcome_type == 'win' else None
    discarder = infer_discarder(scores, fan_total) if win_method == 'discard_win' else None
    winner_gain = int(scores[str(winner)]) if winner is not None and str(winner) in scores else None
    loser_losses = {
        str(player): int(score)
        for player, score in scores.items()
        if winner is not None and int(player) != winner and int(score) < 0
    }
    candidate_key = str(candidate_seat)
    candidate_score = int(scores[candidate_key])

    return {
        'type': outcome_type,
        'winner': str(winner) if winner is not None else None,
        'winner_role': roles[winner] if winner is not None else None,
        'win_method': win_method,
        'discarder': discarder,
        'discarder_role': roles[int(discarder)] if discarder is not None else None,
        'fan_total': fan_total if outcome_type == 'win' else None,
        'fans': env.fans,
        'winner_gain': winner_gain if outcome_type == 'win' else None,
        'loser_losses': loser_losses if outcome_type == 'win' else {},
        'invalid_player': invalid_pid,
        'invalid_role': invalid_role,
        'candidate_won': outcome_type == 'win' and winner == candidate_seat,
        'candidate_self_draw': outcome_type == 'win' and winner == candidate_seat and win_method == 'self_draw',
        'candidate_discard_win': outcome_type == 'win' and winner == candidate_seat and win_method == 'discard_win',
        'candidate_deal_in': outcome_type == 'win' and discarder == candidate_key,
        'candidate_win_gain': candidate_score if outcome_type == 'win' and winner == candidate_seat else None,
        'candidate_loss_when_others_win': candidate_score
        if outcome_type == 'win' and winner != candidate_seat and candidate_score < 0 else None,
    }


def fallback_action(obs, player_id):
    if obs.get('last_operation') == ActionType.DRAW and obs.get('last_player') == player_id and obs.get('last_tile'):
        return Action(player_id, ActionType.PLAY, obs['last_tile'])
    return Action(player_id, ActionType.PASS, None)


class BotzoneModelAgent:
    def __init__(self, role, model_dir, checkpoint, timeout_sec):
        self.role = role
        self.model_dir = model_dir
        self.checkpoint = checkpoint
        self.timeout_sec = timeout_sec
        self.proc = None
        self.stdout_queue = queue.Queue()
        self.stderr = ''
        self.failed = False
        self.crash = False
        self.timeout = False
        self.errors = []
        self.decision_ms = 0.0
        self.decision_count = 0

    def _reset_game_metrics(self):
        self.failed = False
        self.crash = False
        self.timeout = False
        self.errors = []
        self.decision_ms = 0.0
        self.decision_count = 0

    def _spawn(self):
        script = self.model_dir / '__main__.py'
        if not script.exists():
            raise FileNotFoundError('missing agent entrypoint: %s' % script)

        self.stdout_queue = queue.Queue()
        self.stderr = ''
        self.proc = subprocess.Popen(
            [sys.executable, str(script), str(self.checkpoint)],
            cwd=str(self.model_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        thread = threading.Thread(target=self._read_stdout, daemon=True)
        thread.start()

        self._send_raw('1')

    def start(self, player_id, round_wind, init_tiles):
        needs_restart = (
            self.proc is None or
            self.proc.poll() is not None or
            self.failed or
            self.crash or
            self.timeout
        )
        if needs_restart:
            self.close()
            self._spawn()

        self._reset_game_metrics()
        try:
            self._exchange('0 %d %d' % (player_id, round_wind))
            self._exchange('1 0 0 0 0 %s' % ' '.join(init_tiles))
        except Exception as exc:
            self.failed = True
            self.crash = self.crash or not isinstance(exc, TimeoutError)
            self.errors.append('%s: %s' % (type(exc).__name__, exc))
            raise

    def _read_stdout(self):
        try:
            for line in self.proc.stdout:
                line = line.strip()
                if line:
                    self.stdout_queue.put(line)
        except Exception:
            pass

    def _send_raw(self, line):
        if self.proc.poll() is not None:
            raise RuntimeError('agent process exited with code %s' % self.proc.returncode)
        self.proc.stdin.write(line + '\n')
        self.proc.stdin.flush()

    def _read_stdout_line(self):
        try:
            return self.stdout_queue.get(timeout=self.timeout_sec)
        except queue.Empty as exc:
            if self.proc.poll() is not None:
                raise RuntimeError('agent process exited before response, code=%s' % self.proc.returncode) from exc
            self.timeout = True
            raise TimeoutError('agent response timeout after %.3fs' % self.timeout_sec) from exc

    def _exchange(self, request_line):
        self._send_raw(request_line)
        response = self._read_stdout_line()
        sentinel = self._read_stdout_line()
        if sentinel != SENTINEL:
            raise RuntimeError('missing sentinel after response=%s sentinel=%s' % (response, sentinel))
        return response

    def action(self, player_id, request_line, obs):
        if self.failed:
            return fallback_action(obs, player_id)

        t0 = time.perf_counter()
        try:
            response = self._exchange(request_line)
            action = response2act(response, player_id)
            return action
        except Exception as exc:
            self.failed = True
            self.crash = True
            self.errors.append('%s: %s' % (type(exc).__name__, exc))
            return fallback_action(obs, player_id)
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.decision_ms += elapsed_ms
            self.decision_count += 1

    def close(self):
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
            _, stderr = self.proc.communicate(timeout=3)
            self.stderr = stderr.strip()
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        finally:
            self.proc = None

    def metrics(self):
        return {
            'role': self.role,
            'crash': self.crash,
            'timeout': self.timeout,
            'errors': self.errors,
            'avg_decision_ms': self.decision_ms / max(1, self.decision_count),
        }


def make_agents(candidate_seat, candidate_spec, baseline_spec, timeout_sec):
    agents = []
    roles = []
    for seat in range(4):
        if seat == candidate_seat:
            roles.append('candidate')
            agents.append(BotzoneModelAgent('candidate', candidate_spec['model_dir'], candidate_spec['checkpoint'], timeout_sec))
        else:
            roles.append('baseline')
            agents.append(BotzoneModelAgent('baseline', baseline_spec['model_dir'], baseline_spec['checkpoint'], timeout_sec))
    return agents, roles


class ReusableAgentPool:
    def __init__(self, candidate_spec, baseline_spec, timeout_sec):
        self.candidate_agent = BotzoneModelAgent(
            'candidate', candidate_spec['model_dir'], candidate_spec['checkpoint'], timeout_sec
        )
        self.baseline_agents = [
            BotzoneModelAgent('baseline', baseline_spec['model_dir'], baseline_spec['checkpoint'], timeout_sec)
            for _ in range(3)
        ]

    def agents_for_game(self, candidate_seat):
        agents = []
        roles = []
        baseline_index = 0
        for seat in range(4):
            if seat == candidate_seat:
                role = 'candidate'
                agent = self.candidate_agent
            else:
                role = 'baseline'
                agent = self.baseline_agents[baseline_index]
                baseline_index += 1
            agents.append(agent)
            roles.append(role)
        return agents, roles

    def start_game(self, candidate_seat, round_wind, init_tiles):
        agents, roles = self.agents_for_game(candidate_seat)
        for seat, agent in enumerate(agents):
            agent.start(seat, round_wind, init_tiles[seat])
        return agents, roles

    def close(self):
        self.candidate_agent.close()
        for agent in self.baseline_agents:
            agent.close()


def play_game(seed, candidate_seat, candidate_spec, baseline_spec, timeout_sec, max_steps, quiet_env, agent_pool=None):
    output_context = contextlib.redirect_stdout(io.StringIO()) if quiet_env else contextlib.nullcontext()
    agents = []
    roles = []
    crash = False
    timeout = False
    num_steps = 0
    errors = []

    with output_context:
        env = Mahjong(random_seed=seed)
        env.init()

    try:
        if agent_pool is None:
            agents, roles = make_agents(candidate_seat, candidate_spec, baseline_spec, timeout_sec)
            for seat, agent in enumerate(agents):
                agent.start(seat, env.round_wind, env.init_tiles[seat])
        else:
            agents, roles = agent_pool.agents_for_game(candidate_seat)
            for seat, agent in enumerate(agents):
                agent.start(seat, env.round_wind, env.init_tiles[seat])

        while not env.done:
            request_action = env.request_hist[-1]
            actions = []
            for player_id, agent in enumerate(agents):
                request_line = request2str(request_action, player_id)
                obs = env.player_obs(player_id)
                actions.append(agent.action(player_id, request_line, obs))

            with output_context:
                env.step(actions)
            num_steps += 1
            if num_steps > max_steps:
                crash = True
                errors.append('max_steps exceeded')
                break
    except Exception as exc:
        crash = True
        errors.append('%s: %s' % (type(exc).__name__, exc))
    finally:
        if agent_pool is None:
            for agent in agents:
                agent.close()

    agent_metrics = [agent.metrics() for agent in agents]
    crash = crash or any(item['crash'] for item in agent_metrics)
    timeout = timeout or any(item['timeout'] for item in agent_metrics)
    for item in agent_metrics:
        errors.extend(item['errors'])

    scores = env.rewards or {i: 0 for i in range(4)}
    scores = {str(k): int(v) for k, v in scores.items()}
    ranks = compute_ranks(scores)
    invalid = detect_invalid(scores)
    invalid_pid = invalid_player(scores)
    invalid_role = roles[int(invalid_pid)] if invalid_pid is not None else None
    outcome = build_outcome(env, scores, roles, candidate_seat, invalid, invalid_pid, invalid_role, crash, timeout)
    candidate_score = scores[str(candidate_seat)]
    baseline_scores = [scores[str(i)] for i in range(4) if i != candidate_seat]
    candidate_diff = candidate_score - float(np.mean(baseline_scores))
    candidate_agent_metrics = agent_metrics[candidate_seat] if len(agent_metrics) > candidate_seat else {}

    return {
        'seed': seed,
        'candidate_seat': candidate_seat,
        'players': roles,
        'candidate_model': candidate_spec['name'],
        'baseline_model': baseline_spec['name'],
        'scores': scores,
        'ranks': ranks,
        'candidate_score': candidate_score,
        'baseline_mean_score': float(np.mean(baseline_scores)),
        'candidate_diff': candidate_diff,
        'candidate_rank': ranks[str(candidate_seat)],
        'outcome': outcome,
        'illegal': invalid,
        'invalid_player': invalid_pid,
        'invalid_role': invalid_role,
        'candidate_illegal': invalid_role == 'candidate',
        'crash': crash,
        'timeout': timeout,
        'candidate_crash': bool(candidate_agent_metrics.get('crash', False)),
        'candidate_timeout': bool(candidate_agent_metrics.get('timeout', False)),
        'num_steps': num_steps,
        'agent_metrics': agent_metrics,
        'extra': {
            'fans': env.fans,
            'errors': errors,
        },
    }


def bootstrap_ci(values, n_boot=10000, alpha=0.05):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), float(values[0])
    means = []
    for _ in range(n_boot):
        sample = np.random.choice(values, size=len(values), replace=True)
        means.append(sample.mean())
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def summarize(rows, bootstrap):
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[row['seed']].append(row['candidate_diff'])
    seed_diffs = [float(np.mean(values)) for _, values in sorted(by_seed.items())]
    ci_lo, ci_hi = bootstrap_ci(seed_diffs, bootstrap) if seed_diffs else (0.0, 0.0)
    candidate_scores = [row['candidate_score'] for row in rows]
    candidate_ranks = [row['candidate_rank'] for row in rows]
    outcomes = [row.get('outcome', {}) for row in rows]
    win_outcomes = [outcome for outcome in outcomes if outcome.get('type') == 'win']
    candidate_win_outcomes = [outcome for outcome in win_outcomes if outcome.get('candidate_won')]

    def mean_or_zero(values):
        return float(np.mean(values)) if values else 0.0

    def rate(count):
        return count / max(1, len(rows))

    fan_totals = [outcome.get('fan_total') for outcome in win_outcomes if outcome.get('fan_total') is not None]
    candidate_win_fans = [
        outcome.get('fan_total')
        for outcome in candidate_win_outcomes
        if outcome.get('fan_total') is not None
    ]
    win_gains = [outcome.get('winner_gain') for outcome in win_outcomes if outcome.get('winner_gain') is not None]
    candidate_win_gains = [
        outcome.get('candidate_win_gain')
        for outcome in candidate_win_outcomes
        if outcome.get('candidate_win_gain') is not None
    ]
    candidate_losses_when_others_win = [
        outcome.get('candidate_loss_when_others_win')
        for outcome in win_outcomes
        if outcome.get('candidate_loss_when_others_win') is not None
    ]

    illegal_count = sum(1 for row in rows if row['illegal'])
    crash_count = sum(1 for row in rows if row['crash'])
    timeout_count = sum(1 for row in rows if row['timeout'])
    candidate_illegal_count = sum(1 for row in rows if row.get('candidate_illegal') or row.get('invalid_role') == 'candidate')
    candidate_crash_count = sum(1 for row in rows if row.get('candidate_crash'))
    candidate_timeout_count = sum(1 for row in rows if row.get('candidate_timeout'))
    return {
        'num_games': len(rows),
        'num_seeds': len(seed_diffs),
        'mean_score_diff': float(np.mean(seed_diffs)) if seed_diffs else 0.0,
        'score_diff_ci95': [ci_lo, ci_hi],
        'avg_candidate_score': mean_or_zero(candidate_scores),
        'avg_candidate_rank': mean_or_zero(candidate_ranks),
        'candidate_rank1_rate': sum(1 for rank in candidate_ranks if rank == 1) / max(1, len(candidate_ranks)),
        'candidate_rank4_rate': sum(1 for rank in candidate_ranks if rank == 4) / max(1, len(candidate_ranks)),
        'win_count': len(win_outcomes),
        'draw_count': sum(1 for outcome in outcomes if outcome.get('type') == 'draw'),
        'incomplete_count': sum(1 for outcome in outcomes if outcome.get('type') == 'incomplete'),
        'unknown_count': sum(1 for outcome in outcomes if outcome.get('type') == 'unknown'),
        'candidate_win_count': len(candidate_win_outcomes),
        'candidate_win_rate': rate(len(candidate_win_outcomes)),
        'candidate_self_draw_count': sum(1 for outcome in win_outcomes if outcome.get('candidate_self_draw')),
        'candidate_self_draw_rate': rate(sum(1 for outcome in win_outcomes if outcome.get('candidate_self_draw'))),
        'candidate_discard_win_count': sum(1 for outcome in win_outcomes if outcome.get('candidate_discard_win')),
        'candidate_discard_win_rate': rate(sum(1 for outcome in win_outcomes if outcome.get('candidate_discard_win'))),
        'candidate_deal_in_count': sum(1 for outcome in win_outcomes if outcome.get('candidate_deal_in')),
        'candidate_deal_in_rate': rate(sum(1 for outcome in win_outcomes if outcome.get('candidate_deal_in'))),
        'avg_fan': mean_or_zero(fan_totals),
        'avg_candidate_win_fan': mean_or_zero(candidate_win_fans),
        'avg_win_gain': mean_or_zero(win_gains),
        'avg_candidate_win_gain': mean_or_zero(candidate_win_gains),
        'avg_candidate_loss_when_others_win': mean_or_zero(candidate_losses_when_others_win),
        'illegal_count': illegal_count,
        'illegal_rate': rate(illegal_count),
        'crash_count': crash_count,
        'crash_rate': rate(crash_count),
        'timeout_count': timeout_count,
        'timeout_rate': rate(timeout_count),
        'candidate_illegal_count': candidate_illegal_count,
        'candidate_illegal_rate': rate(candidate_illegal_count),
        'candidate_crash_count': candidate_crash_count,
        'candidate_crash_rate': rate(candidate_crash_count),
        'candidate_timeout_count': candidate_timeout_count,
        'candidate_timeout_rate': rate(candidate_timeout_count),
        'baseline_illegal_count': sum(1 for row in rows if row.get('invalid_role') == 'baseline'),
    }


def split_batches(items, num_batches):
    batches = [[] for _ in range(num_batches)]
    for index, item in enumerate(items):
        batches[index % num_batches].append(item)
    return [batch for batch in batches if batch]


def run_seed_batch(worker_id, seeds, candidate_spec, baseline_spec, timeout_sec, max_steps, quiet_env, reuse_agents):
    rows = []
    start = time.perf_counter()
    agent_pool = ReusableAgentPool(candidate_spec, baseline_spec, timeout_sec) if reuse_agents else None
    try:
        for seed in seeds:
            for candidate_seat in range(4):
                rows.append(play_game(
                    seed,
                    candidate_seat,
                    candidate_spec,
                    baseline_spec,
                    timeout_sec,
                    max_steps,
                    quiet_env=quiet_env,
                    agent_pool=agent_pool,
                ))
    finally:
        if agent_pool is not None:
            agent_pool.close()
    return {
        'worker_id': worker_id,
        'seeds': seeds,
        'rows': rows,
        'elapsed_sec': time.perf_counter() - start,
    }


def main():
    parser = argparse.ArgumentParser(description='Run a local Mahjong one-vs-three arena.')
    parser.add_argument('--mode', default='one_vs_three', choices=['one_vs_three'])
    parser.add_argument('--candidate-model', required=True, help='Model name under eval/models or explicit model dir path.')
    parser.add_argument('--baseline-model', required=True, help='Model name under eval/models or explicit model dir path.')
    parser.add_argument('--candidate-checkpoint', default=None)
    parser.add_argument('--baseline-checkpoint', default=None)
    parser.add_argument('--num-seeds', type=int, default=20)
    parser.add_argument('--seed-begin', type=int, default=0)
    parser.add_argument('--out', default=None, help='Output JSONL path. Defaults to eval/results/YYYY-MM-DD/<candidate>_vs_<baseline>_<games>games.jsonl.')
    parser.add_argument('--summary-out', default=None)
    parser.add_argument('--timeout-sec', type=float, default=30.0)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--bootstrap', type=int, default=10000)
    parser.add_argument('--reuse-agents', dest='reuse_agents', action='store_true', help='Reuse model subprocesses across games.')
    parser.add_argument('--no-reuse-agents', dest='reuse_agents', action='store_false', help='Start fresh model subprocesses each game.')
    parser.add_argument('--workers', type=int, default=1, help='Number of seed batches to run in parallel.')
    parser.add_argument('--show-env-output', action='store_true')
    parser.set_defaults(reuse_agents=True)
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError('--workers must be >= 1')

    candidate_dir = resolve_model_dir(args.candidate_model)
    baseline_dir = resolve_model_dir(args.baseline_model)
    candidate_spec = {
        'name': candidate_dir.name,
        'model_dir': candidate_dir,
        'checkpoint': resolve_checkpoint(candidate_dir, args.candidate_checkpoint),
    }
    baseline_spec = {
        'name': baseline_dir.name,
        'model_dir': baseline_dir,
        'checkpoint': resolve_checkpoint(baseline_dir, args.baseline_checkpoint),
    }

    num_games = args.num_seeds * 4
    out_path = Path(args.out) if args.out else default_arena_out_path(
        candidate_spec['name'], baseline_spec['name'], num_games
    )
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_out) if args.summary_out else out_path.with_suffix('.summary.json')
    if not summary_path.is_absolute():
        summary_path = ROOT / summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.seed_begin, args.seed_begin + args.num_seeds))
    worker_count = min(args.workers, len(seeds)) if seeds else 1
    worker_results = []
    start = time.perf_counter()
    if worker_count == 1:
        worker_results.append(run_seed_batch(
            0,
            seeds,
            candidate_spec,
            baseline_spec,
            args.timeout_sec,
            args.max_steps,
            quiet_env=not args.show_env_output,
            reuse_agents=args.reuse_agents,
        ))
    else:
        batches = split_batches(seeds, worker_count)
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    run_seed_batch,
                    worker_id,
                    batch,
                    candidate_spec,
                    baseline_spec,
                    args.timeout_sec,
                    args.max_steps,
                    not args.show_env_output,
                    args.reuse_agents,
                )
                for worker_id, batch in enumerate(batches)
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                worker_results.append(result)
                print('completed worker=%d seeds=%s games=%d elapsed_sec=%.3f' % (
                    result['worker_id'],
                    ','.join(str(seed) for seed in result['seeds']),
                    len(result['rows']),
                    result['elapsed_sec'],
                ))

    rows = []
    for result in worker_results:
        rows.extend(result['rows'])
    rows.sort(key=lambda row: (row['seed'], row['candidate_seat']))
    with out_path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    summary = summarize(rows, args.bootstrap)
    summary.update({
        'candidate_model': candidate_spec['name'],
        'baseline_model': baseline_spec['name'],
        'candidate_checkpoint': str(candidate_spec['checkpoint']),
        'baseline_checkpoint': str(baseline_spec['checkpoint']),
        'reuse_agents': args.reuse_agents,
        'workers': worker_count,
        'worker_elapsed_sec': [result['elapsed_sec'] for result in sorted(worker_results, key=lambda item: item['worker_id'])],
        'elapsed_sec': time.perf_counter() - start,
        'out': str(out_path),
    })
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print('summary=%s' % summary_path)
    return 0 if summary['illegal_count'] == 0 and summary['crash_count'] == 0 and summary['timeout_count'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
