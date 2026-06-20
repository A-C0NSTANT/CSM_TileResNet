import argparse
import contextlib
import io
import json
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY_ENV = ROOT / 'third_party' / 'botzone-mahjong-environment'


def add_third_party_path():
    if THIRD_PARTY_ENV.exists():
        path = str(THIRD_PARTY_ENV)
        if path not in sys.path:
            sys.path.insert(0, path)


def import_environment():
    add_third_party_path()
    try:
        from mahjong_env.core import Mahjong
        from mahjong_env.base_bot import RandomMahjongBot
    except Exception as exc:
        message = [
            'Failed to import mahjong_env. Run dependency check first:',
            '  conda run -n csmj-arena python eval/arena/check_env.py',
            'Original error: %s: %s' % (type(exc).__name__, exc),
        ]
        raise RuntimeError('\n'.join(message)) from exc
    return Mahjong, RandomMahjongBot


def default_out_path(filename):
    return ROOT / 'eval' / 'results' / datetime.now().strftime('%Y-%m-%d') / filename


def compute_ranks(scores):
    items = sorted(((int(player), float(score)) for player, score in scores.items()), key=lambda x: (-x[1], x[0]))
    return {str(player): rank + 1 for rank, (player, _) in enumerate(items)}


def detect_invalid(scores):
    values = sorted(int(v) for v in scores.values())
    return values == [-30, 10, 10, 10]


def play_game(seed, max_steps, quiet_env):
    Mahjong, RandomMahjongBot = import_environment()
    random.seed(seed)
    env = Mahjong(random_seed=seed)
    agents = [RandomMahjongBot() for _ in range(4)]
    agent_names = ['random_%d' % i for i in range(4)]
    decision_ms = {name: 0.0 for name in agent_names}
    decision_count = {name: 0 for name in agent_names}
    num_steps = 0
    crash = False
    error = None

    output_context = contextlib.redirect_stdout(io.StringIO()) if quiet_env else contextlib.nullcontext()
    with output_context:
        try:
            env.init()
            while not env.done:
                actions = []
                for player_id, agent in enumerate(agents):
                    obs = env.player_obs(player_id)
                    t0 = time.perf_counter()
                    action = agent.action(obs)
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    name = agent_names[player_id]
                    decision_ms[name] += elapsed_ms
                    decision_count[name] += 1
                    actions.append(action)

                env.step(actions)
                num_steps += 1
                if num_steps > max_steps:
                    crash = True
                    error = 'max_steps exceeded'
                    break
        except Exception:
            crash = True
            error = traceback.format_exc()

    scores = env.rewards or {str(i): 0 for i in range(4)}
    scores = {str(k): int(v) for k, v in scores.items()}
    avg_decision_ms = {
        name: decision_ms[name] / max(1, decision_count[name])
        for name in agent_names
    }

    return {
        'seed': seed,
        'players': agent_names,
        'scores': scores,
        'ranks': compute_ranks(scores),
        'illegal': detect_invalid(scores),
        'crash': crash,
        'timeout': False,
        'num_steps': num_steps,
        'avg_decision_ms': avg_decision_ms,
        'extra': {
            'fans': env.fans,
            'error': error,
        },
    }


def main():
    parser = argparse.ArgumentParser(description='Run a minimal random-vs-random local Mahjong arena smoke test.')
    parser.add_argument('--games', type=int, default=4, help='Number of games to run.')
    parser.add_argument('--seed-begin', type=int, default=0, help='First random seed.')
    parser.add_argument('--out', default=None, help='Output JSONL path. Defaults to eval/results/YYYY-MM-DD/random_vs_random.jsonl.')
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--show-env-output', action='store_true', help='Do not suppress mahjong_env stdout.')
    args = parser.parse_args()

    # Fail before creating result files if the third-party environment is not usable.
    import_environment()

    out_path = Path(args.out) if args.out else default_out_path('random_vs_random.jsonl')
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    rows = []
    with out_path.open('w', encoding='utf-8') as f:
        for i in range(args.games):
            seed = args.seed_begin + i
            row = play_game(seed, args.max_steps, quiet_env=not args.show_env_output)
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    elapsed = time.perf_counter() - start
    illegal = sum(1 for row in rows if row['illegal'])
    crash = sum(1 for row in rows if row['crash'])
    print('games=%d illegal=%d crash=%d elapsed_sec=%.3f sec_per_game=%.3f out=%s' % (
        len(rows),
        illegal,
        crash,
        elapsed,
        elapsed / max(1, len(rows)),
        out_path,
    ))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
