import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SENTINEL = '>>>BOTZONE_REQUEST_KEEP_RUNNING<<<'
TILE_RE = r'(?:[WTB][1-9]|F[1-4]|J[1-3])'
VALID_DRAW_RESPONSES = [
    re.compile(r'^HU$'),
    re.compile(r'^PLAY %s$' % TILE_RE),
    re.compile(r'^GANG %s$' % TILE_RE),
    re.compile(r'^BUGANG %s$' % TILE_RE),
]


MODELS = [
    {
        'name': 'CNN_Baseline',
        'workdir': ROOT / 'eval' / 'models' / 'CNN_Baseline',
        'checkpoint': ROOT / 'eval' / 'models' / 'CNN_Baseline' / 'checkpoint' / 'cnn_model_latest.pkl',
    },
    {
        'name': 'ResNet_Policy_v1',
        'workdir': ROOT / 'eval' / 'models' / 'ResNet_Policy_v1',
        'checkpoint': ROOT / 'eval' / 'models' / 'ResNet_Policy_v1' / 'checkpoint' / 'resnet_model_latest.pkl',
    },
]


def default_out_path(filename):
    return ROOT / 'eval' / 'results' / datetime.now().strftime('%Y-%m-%d') / filename


def valid_draw_response(response):
    return any(pattern.match(response or '') for pattern in VALID_DRAW_RESPONSES)


def test_model(model_info, timeout_sec):
    script = model_info['workdir'] / '__main__.py'
    checkpoint = model_info['checkpoint']
    result = {
        'name': model_info['name'],
        'script': str(script),
        'checkpoint': str(checkpoint),
        'ok': False,
        'responses': [],
        'error': None,
        'stderr': '',
    }

    if not script.exists():
        result['error'] = 'missing __main__.py'
        return result
    if not checkpoint.exists():
        result['error'] = 'missing checkpoint'
        return result

    try:
        requests = [
            ('wind', '0 0 0'),
            ('deal', '1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T2 T3 T4'),
            ('draw', '2 T5'),
        ]
        input_text = '1\n' + ''.join(request + '\n' for _, request in requests)
        completed = subprocess.run(
            [sys.executable, str(script), str(checkpoint)],
            cwd=str(model_info['workdir']),
            input=input_text,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
        )
        result['stderr'] = completed.stderr.strip()

        stdout_lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        expected_lines = len(requests) * 2
        if len(stdout_lines) < expected_lines:
            result['error'] = 'too few stdout lines: expected at least %d, got %d' % (expected_lines, len(stdout_lines))
            return result

        for label, request in requests:
            response = stdout_lines[len(result['responses']) * 2]
            sentinel = stdout_lines[len(result['responses']) * 2 + 1]
            item = {
                'label': label,
                'request': request,
                'response': response,
                'sentinel': sentinel,
            }
            result['responses'].append(item)
            if sentinel != SENTINEL:
                result['error'] = 'missing keep-running sentinel after %s' % label
                return result

        if result['responses'][0]['response'] != 'PASS':
            result['error'] = 'wind response is not PASS'
            return result
        if result['responses'][1]['response'] != 'PASS':
            result['error'] = 'deal response is not PASS'
            return result
        if not valid_draw_response(result['responses'][2]['response']):
            result['error'] = 'draw response has invalid format: %s' % result['responses'][2]['response']
            return result

        result['ok'] = True
        return result
    except Exception as exc:
        result['error'] = '%s: %s' % (type(exc).__name__, exc)
        return result


def main():
    parser = argparse.ArgumentParser(description='Minimum Botzone-agent smoke test for packaged models.')
    parser.add_argument('--timeout-sec', type=float, default=15.0)
    parser.add_argument('--out', default=None, help='Output JSON path. Defaults to eval/results/YYYY-MM-DD/minimum_model_test.json.')
    args = parser.parse_args()

    results = [test_model(model_info, args.timeout_sec) for model_info in MODELS]
    summary = {
        'ok': all(item['ok'] for item in results),
        'results': results,
    }

    out_path = Path(args.out) if args.out else default_out_path('minimum_model_test.json')
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    for item in results:
        status = 'OK' if item['ok'] else 'FAIL'
        draw_response = None
        if len(item['responses']) >= 3:
            draw_response = item['responses'][2]['response']
        print('%s %-20s draw_response=%s' % (status, item['name'], draw_response))
        if item['error']:
            print('  error: %s' % item['error'])
    print('out=%s' % out_path)

    return 0 if summary['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
