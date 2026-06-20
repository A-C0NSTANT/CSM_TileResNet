import argparse
import importlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY_ENV = ROOT / 'third_party' / 'botzone-mahjong-environment'


def add_third_party_path():
    if THIRD_PARTY_ENV.exists():
        path = str(THIRD_PARTY_ENV)
        if path not in sys.path:
            sys.path.insert(0, path)


def check_import(module_name, attr_name=None):
    try:
        module = importlib.import_module(module_name)
        if attr_name is not None:
            getattr(module, attr_name)
        return {'ok': True, 'error': None}
    except Exception as exc:
        return {'ok': False, 'error': '%s: %s' % (type(exc).__name__, exc)}


def run_checks():
    add_third_party_path()
    checks = {
        'third_party_env_path': {
            'ok': THIRD_PARTY_ENV.exists(),
            'path': str(THIRD_PARTY_ENV),
            'error': None if THIRD_PARTY_ENV.exists() else 'third_party/botzone-mahjong-environment is missing',
        },
        'numpy': check_import('numpy'),
        'yaml': check_import('yaml'),
        'MahjongGB.MahjongFanCalculator': check_import('MahjongGB', 'MahjongFanCalculator'),
        'mahjong_env.core.Mahjong': check_import('mahjong_env.core', 'Mahjong'),
        'mahjong_env.base_bot.RandomMahjongBot': check_import('mahjong_env.base_bot', 'RandomMahjongBot'),
    }
    return checks


def print_text(checks):
    for name, result in checks.items():
        status = 'OK' if result['ok'] else 'FAIL'
        print('%-42s %s' % (name, status))
        if result.get('path'):
            print('  path: %s' % result['path'])
        if result.get('error'):
            print('  error: %s' % result['error'])


def main():
    parser = argparse.ArgumentParser(description='Check local Mahjong arena dependencies.')
    parser.add_argument('--json', action='store_true', help='Print machine-readable JSON.')
    args = parser.parse_args()

    checks = run_checks()
    if args.json:
        print(json.dumps(checks, indent=2, ensure_ascii=False))
    else:
        print_text(checks)

    if not all(item['ok'] for item in checks.values()):
        print('\nRequired fix on Windows: install Microsoft C++ Build Tools, then run:')
        print('  conda run -n csmj-arena pip install PyMahjongGB')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
