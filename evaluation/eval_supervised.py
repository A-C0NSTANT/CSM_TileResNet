import argparse
import csv
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


logger = logging.getLogger(__name__)

ACT_SIZE = 235
HU_ACTION = 1
DETAILED_TYPES = ['Pass', 'Hu', 'Play', 'Chi', 'Peng', 'Gang', 'AnGang', 'BuGang', 'AnyGang']
CONFUSION_TYPES = ['Pass', 'Hu', 'Play', 'Chi', 'Peng', 'AnyGang']
BUCKETS = ['2', '3-5', '6-10', '>10']
FAILURE_CATEGORIES = [
    'label_hu_pred_not_hu',
    'hu_legal_pred_not_hu',
    'label_peng_pred_pass',
    'label_chi_pred_pass',
    'label_gang_pred_wrong',
    'confident_wrong',
    'label_not_top5',
    'large_legal_wrong',
]


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a supervised Mahjong policy model.')
    parser.add_argument('--model', default='cnn', choices=['cnn', 'resnet', 'rarn', 'rarn_v2', 'rarn_public', 'rarn_public_v2', 'rarn_public_v2_large', 'rarn_public_v2_1_5x', 'rarn_public_v2_hist', 'rarn_public_v2_hist_gated', 'rarn_public_v2_hist_delta_gang', 'rarn_public_danger'], help='Model architecture to evaluate.')
    parser.add_argument('--data-dir', default='data', help='Directory containing count.json and *.npz files.')
    parser.add_argument('--checkpoint', default='checkpoints/cnn_model_latest.pkl', help='Checkpoint to evaluate.')
    parser.add_argument('--split-begin', type=float, default=0.9, help='Validation split begin ratio.')
    parser.add_argument('--split-end', type=float, default=1.0, help='Validation split end ratio.')
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--device', default='auto', help='cpu, cuda, or auto.')
    parser.add_argument('--output-dir', default='evaluation/results/cnn_baseline_v0')
    parser.add_argument('--max-samples', type=int, default=None, help='Optional smoke-eval sample cap.')
    parser.add_argument('--save-failures', action='store_true', help='Save selected failed cases to failures.jsonl.')
    parser.add_argument('--topk', default='1,3,5', help='Comma-separated top-k values, for example 1,3,5.')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--log-interval', type=int, default=128)
    parser.add_argument('--max-cases-per-type', type=int, default=100)
    parser.add_argument('--history-max-len', type=int, default=None, help='Override history length loaded from preprocessed data.')
    return parser.parse_args()


def setup_logging(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / 'eval.log'

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    return log_file


def set_seed(seed):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_topk(topk_text):
    topk = []
    for part in topk_text.split(','):
        value = int(part.strip())
        if value <= 0:
            raise ValueError('top-k values must be positive')
        if value not in topk:
            topk.append(value)
    return sorted(topk)


def resolve_device(device_arg):
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_arg)


def make_input(obs, mask, public, device, history=None, history_padding_mask=None):
    input_dict = {
        'is_training': False,
        'obs': {
            'observation': obs.to(device, non_blocking=True),
            'action_mask': mask.to(device, non_blocking=True),
        },
    }
    if public is not None:
        input_dict['obs']['public'] = public.to(device, non_blocking=True)
    if history is not None:
        input_dict['obs']['history'] = history.to(device, non_blocking=True)
    if history_padding_mask is not None:
        input_dict['obs']['history_padding_mask'] = history_padding_mask.to(device, non_blocking=True)
    return input_dict


def action_type(action):
    action = int(action)
    if action == 0:
        return 'Pass'
    if action == 1:
        return 'Hu'
    if 2 <= action < 36:
        return 'Play'
    if 36 <= action < 99:
        return 'Chi'
    if 99 <= action < 133:
        return 'Peng'
    if 133 <= action < 167:
        return 'Gang'
    if 167 <= action < 201:
        return 'AnGang'
    if 201 <= action < 235:
        return 'BuGang'
    return 'Invalid'


def merged_action_type(action):
    typ = action_type(action)
    if typ in ('Gang', 'AnGang', 'BuGang'):
        return 'AnyGang'
    return typ


def legal_count_bucket(count):
    count = int(count)
    if count == 2:
        return '2'
    if 3 <= count <= 5:
        return '3-5'
    if 6 <= count <= 10:
        return '6-10'
    return '>10'


def new_action_stats():
    return {
        'support': 0,
        'pred_count': 0,
        'type_correct': 0,
        'exact_top1': 0,
        'topk_correct': defaultdict(int),
    }


def new_bucket_stats():
    return {
        'support': 0,
        'loss_sum': 0.0,
        'topk_correct': defaultdict(int),
    }


def update_action_metrics(stats, label, pred, topk_indices, topk_values):
    label_typ = action_type(label)
    pred_typ = action_type(pred)

    label_keys = [label_typ]
    if label_typ in ('Gang', 'AnGang', 'BuGang'):
        label_keys.append('AnyGang')

    pred_keys = [pred_typ]
    if pred_typ in ('Gang', 'AnGang', 'BuGang'):
        pred_keys.append('AnyGang')

    for key in label_keys:
        if key not in stats:
            stats[key] = new_action_stats()
        stats[key]['support'] += 1
        if pred == label:
            stats[key]['exact_top1'] += 1
        for k in topk_values:
            if label in topk_indices[:k]:
                stats[key]['topk_correct'][k] += 1

    for key in pred_keys:
        if key not in stats:
            stats[key] = new_action_stats()
        stats[key]['pred_count'] += 1

    if label_typ == pred_typ:
        stats[label_typ]['type_correct'] += 1
    if label_typ in ('Gang', 'AnGang', 'BuGang') and pred_typ in ('Gang', 'AnGang', 'BuGang'):
        stats['AnyGang']['type_correct'] += 1


def failure_categories(label, pred, top5, legal_count, mask_row, pred_prob):
    categories = []
    label_typ = action_type(label)
    pred_typ = action_type(pred)
    hu_legal = bool(mask_row[HU_ACTION] > 0)

    if label == HU_ACTION and pred != HU_ACTION:
        categories.append('label_hu_pred_not_hu')
    if hu_legal and pred != HU_ACTION:
        categories.append('hu_legal_pred_not_hu')
    if label_typ == 'Peng' and pred_typ == 'Pass':
        categories.append('label_peng_pred_pass')
    if label_typ == 'Chi' and pred_typ == 'Pass':
        categories.append('label_chi_pred_pass')
    if label_typ in ('Gang', 'AnGang', 'BuGang') and pred != label:
        categories.append('label_gang_pred_wrong')
    if pred != label and pred_prob > 0.8:
        categories.append('confident_wrong')
    if label not in top5:
        categories.append('label_not_top5')
    if legal_count > 10 and pred != label:
        categories.append('large_legal_wrong')
    return categories


def maybe_add_failures(failures, failure_counts, max_cases_per_type, sample_index, label, pred, top5, legal_count, probs, mask_row):
    pred_prob = float(probs[pred])
    categories = failure_categories(label, pred, top5, legal_count, mask_row, pred_prob)
    for category in categories:
        if failure_counts[category] >= max_cases_per_type:
            continue
        failure_counts[category] += 1
        label_prob = float(probs[label])
        failures.append({
            'category': category,
            'sample_index': int(sample_index),
            'label': int(label),
            'label_type': action_type(label),
            'pred': int(pred),
            'pred_type': action_type(pred),
            'legal_count': int(legal_count),
            'top5': [int(x) for x in top5],
            'top5_types': [action_type(x) for x in top5],
            'label_prob': label_prob,
            'pred_prob': pred_prob,
        })


def write_results(output_dir, results, action_stats, confusion, bucket_stats, topk_values, failures):
    with open(output_dir / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(output_dir / 'action_type_metrics.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['action_type', 'support', 'pred_count', 'precision', 'recall'] + ['top%d' % k for k in topk_values])
        for typ in DETAILED_TYPES:
            stat = action_stats.get(typ, new_action_stats())
            support = stat['support']
            pred_count = stat['pred_count']
            precision = stat['type_correct'] / pred_count if pred_count else 0.0
            recall = stat['type_correct'] / support if support else 0.0
            row = [typ, support, pred_count, precision, recall]
            row.extend(stat['topk_correct'][k] / support if support else 0.0 for k in topk_values)
            writer.writerow(row)

    confusion_types = CONFUSION_TYPES[:]
    if any(confusion.get('Invalid', {}).values()) or any(row.get('Invalid', 0) for row in confusion.values()):
        confusion_types.append('Invalid')
    with open(output_dir / 'confusion_matrix.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['label_type'] + confusion_types)
        for label_typ in confusion_types:
            writer.writerow([label_typ] + [confusion[label_typ][pred_typ] for pred_typ in confusion_types])

    with open(output_dir / 'legal_count_buckets.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['bucket', 'support', 'loss'] + ['top%d' % k for k in topk_values])
        for bucket in BUCKETS:
            stat = bucket_stats.get(bucket, new_bucket_stats())
            support = stat['support']
            loss = stat['loss_sum'] / support if support else 0.0
            row = [bucket, support, loss]
            row.extend(stat['topk_correct'][k] / support if support else 0.0 for k in topk_values)
            writer.writerow(row)

    if failures:
        with open(output_dir / 'failures.jsonl', 'w', encoding='utf-8') as f:
            for item in failures:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


@torch.no_grad()
def evaluate(model, loader, device, topk_values, max_samples, save_failures, max_cases_per_type, log_interval):
    model.eval()
    max_topk = min(max(topk_values + [5]), ACT_SIZE)
    start_time = time.perf_counter()

    total_samples = 0
    total_loss = 0.0
    total_topk = defaultdict(int)
    total_illegal = 0
    total_label_illegal = 0
    total_legal_actions = 0.0
    total_entropy = 0.0

    hu_label_count = 0
    hu_label_top1 = 0
    hu_label_top3 = 0
    hu_available_count = 0
    hu_available_pred_hu = 0
    hu_available_top3_hu = 0

    action_stats = {typ: new_action_stats() for typ in DETAILED_TYPES}
    confusion = defaultdict(lambda: defaultdict(int))
    bucket_stats = {bucket: new_bucket_stats() for bucket in BUCKETS}
    failures = []
    failure_counts = defaultdict(int)

    for batch_index, batch in enumerate(loader):
        obs, mask, act = batch[:3]
        public = batch[3] if len(batch) > 3 else None
        history = batch[4] if len(batch) > 4 else None
        history_padding_mask = batch[6] if len(batch) > 6 else None
        if max_samples is not None and total_samples >= max_samples:
            break
        if max_samples is not None and total_samples + act.size(0) > max_samples:
            keep = max_samples - total_samples
            obs = obs[:keep]
            mask = mask[:keep]
            act = act[:keep]
            if public is not None:
                public = public[:keep]
            if history is not None:
                history = history[:keep]
            if history_padding_mask is not None:
                history_padding_mask = history_padding_mask[:keep]

        target = act.to(device, non_blocking=True).long()
        mask_device = mask.to(device, non_blocking=True)
        logits = model(make_input(obs, mask, public, device, history, history_padding_mask))
        losses = F.cross_entropy(logits, target, reduction='none')
        probs = torch.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)

        batch_size = target.size(0)
        pred = logits.argmax(dim=1)
        topk_indices = logits.topk(max_topk, dim=1).indices
        legal_counts = mask_device.float().sum(dim=1)

        total_samples += batch_size
        total_loss += losses.sum().item()
        total_legal_actions += legal_counts.sum().item()
        total_entropy += entropy.sum().item()
        total_illegal += (mask_device[torch.arange(batch_size, device=device), pred] <= 0).sum().item()
        total_label_illegal += (mask_device[torch.arange(batch_size, device=device), target] <= 0).sum().item()

        for k in topk_values:
            total_topk[k] += topk_indices[:, :k].eq(target.view(-1, 1)).any(dim=1).sum().item()

        hu_label_mask = target.eq(HU_ACTION)
        if hu_label_mask.any():
            hu_label_count += hu_label_mask.sum().item()
            hu_label_top1 += pred[hu_label_mask].eq(HU_ACTION).sum().item()
            hu_label_top3 += topk_indices[hu_label_mask, :min(3, max_topk)].eq(HU_ACTION).any(dim=1).sum().item()

        hu_available_mask = mask_device[:, HU_ACTION] > 0
        if hu_available_mask.any():
            hu_available_count += hu_available_mask.sum().item()
            hu_available_pred_hu += pred[hu_available_mask].eq(HU_ACTION).sum().item()
            hu_available_top3_hu += topk_indices[hu_available_mask, :min(3, max_topk)].eq(HU_ACTION).any(dim=1).sum().item()

        losses_cpu = losses.detach().cpu().tolist()
        target_cpu = target.detach().cpu().tolist()
        pred_cpu = pred.detach().cpu().tolist()
        topk_cpu = topk_indices.detach().cpu().tolist()
        legal_counts_cpu = legal_counts.detach().cpu().tolist()
        probs_cpu = probs.detach().cpu().tolist()
        mask_cpu = mask.detach().cpu().tolist()

        for i in range(batch_size):
            label_i = int(target_cpu[i])
            pred_i = int(pred_cpu[i])
            topk_i = [int(x) for x in topk_cpu[i]]
            legal_count_i = int(legal_counts_cpu[i])
            label_merged = merged_action_type(label_i)
            pred_merged = merged_action_type(pred_i)
            sample_index = total_samples - batch_size + i

            update_action_metrics(action_stats, label_i, pred_i, topk_i, topk_values)
            confusion[label_merged][pred_merged] += 1

            bucket = legal_count_bucket(legal_count_i)
            bucket_stats[bucket]['support'] += 1
            bucket_stats[bucket]['loss_sum'] += float(losses_cpu[i])
            for k in topk_values:
                if label_i in topk_i[:k]:
                    bucket_stats[bucket]['topk_correct'][k] += 1

            if save_failures:
                maybe_add_failures(
                    failures,
                    failure_counts,
                    max_cases_per_type,
                    sample_index,
                    label_i,
                    pred_i,
                    topk_i[:5],
                    legal_count_i,
                    probs_cpu[i],
                    mask_cpu[i],
                )

        if batch_index % log_interval == 0:
            elapsed = time.perf_counter() - start_time
            logger.info(
                '[Eval] iter=%d/%d samples=%d loss=%.6f top1=%.4f samples_per_sec=%.2f',
                batch_index,
                len(loader),
                total_samples,
                total_loss / max(total_samples, 1),
                total_topk.get(1, 0) / max(total_samples, 1),
                total_samples / max(elapsed, 1e-6),
            )

    elapsed = time.perf_counter() - start_time
    results = {
        'model': 'CNNModel',
        'num_samples': total_samples,
        'loss': total_loss / total_samples if total_samples else 0.0,
        'illegal_action_rate': total_illegal / total_samples if total_samples else 0.0,
        'label_illegal_rate': total_label_illegal / total_samples if total_samples else 0.0,
        'avg_legal_actions': total_legal_actions / total_samples if total_samples else 0.0,
        'avg_entropy': total_entropy / total_samples if total_samples else 0.0,
        'hu_label_count': hu_label_count,
        'hu_label_top1': hu_label_top1 / hu_label_count if hu_label_count else 0.0,
        'hu_label_top3': hu_label_top3 / hu_label_count if hu_label_count else 0.0,
        'hu_available_count': hu_available_count,
        'hu_available_pred_hu_rate': hu_available_pred_hu / hu_available_count if hu_available_count else 0.0,
        'hu_available_top3_hu_rate': hu_available_top3_hu / hu_available_count if hu_available_count else 0.0,
        'elapsed_sec': elapsed,
        'samples_per_sec': total_samples / max(elapsed, 1e-6),
    }
    for k in topk_values:
        results['top%d_accuracy' % k] = total_topk[k] / total_samples if total_samples else 0.0

    return results, action_stats, confusion, bucket_stats, failures


def main():
    args = parse_args()
    from dataset import MahjongGBDataset
    from model import create_model, model_requires_history, model_requires_public

    output_dir = Path(args.output_dir)
    log_file = setup_logging(output_dir)
    set_seed(args.seed)
    device = resolve_device(args.device)
    topk_values = parse_topk(args.topk)

    logger.info('[Log] file=%s', log_file)
    logger.info(
        '[Config] model=%s data_dir=%s checkpoint=%s split_begin=%.4f split_end=%.4f batch_size=%d device=%s topk=%s max_samples=%s output_dir=%s',
        args.model,
        args.data_dir,
        args.checkpoint,
        args.split_begin,
        args.split_end,
        args.batch_size,
        device,
        ','.join(str(k) for k in topk_values),
        args.max_samples,
        output_dir,
    )

    logger.info('[Data] loading begin=%.4f end=%.4f', args.split_begin, args.split_end)
    include_history = model_requires_history(args.model)
    dataset = MahjongGBDataset(args.split_begin, args.split_end, False, data_dir=args.data_dir, split_name='eval', include_history=include_history, history_max_len=args.history_max_len)
    if model_requires_public(args.model) and not dataset.has_public:
        raise RuntimeError(
            '%s requires data with public features. '
            'Regenerate data with preprocess.py using DATA_DIR=%s' % (args.model, args.data_dir)
        )
    if include_history and not dataset.has_history:
        raise RuntimeError(
            '%s requires data with history features. '
            'Regenerate data with INCLUDE_HISTORY=1 DATA_DIR=%s python preprocess.py' % (args.model, args.data_dir)
        )
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )
    logger.info('[Data] samples=%d batches=%d', len(dataset), len(loader))

    logger.info('[Model] loading checkpoint=%s', args.checkpoint)
    model = create_model(args.model).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info('[Model] name=%s params=%d trainable_params=%d', args.model, param_count, sum(p.numel() for p in model.parameters() if p.requires_grad))
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    results, action_stats, confusion, bucket_stats, failures = evaluate(
        model,
        loader,
        device,
        topk_values,
        args.max_samples,
        args.save_failures,
        args.max_cases_per_type,
        args.log_interval,
    )
    results.update({
        'checkpoint': args.checkpoint,
        'model': args.model,
        'param_count': param_count,
        'data_dir': args.data_dir,
        'split_begin': args.split_begin,
        'split_end': args.split_end,
        'batch_size': args.batch_size,
        'device': str(device),
        'topk': topk_values,
        'failures_saved': len(failures),
    })

    write_results(output_dir, results, action_stats, confusion, bucket_stats, topk_values, failures)

    logger.info(
        '[Summary] samples=%d loss=%.6f top1=%.4f top3=%.4f top5=%.4f illegal=%.6f label_illegal=%.6f entropy=%.4f',
        results['num_samples'],
        results['loss'],
        results.get('top1_accuracy', 0.0),
        results.get('top3_accuracy', 0.0),
        results.get('top5_accuracy', 0.0),
        results['illegal_action_rate'],
        results['label_illegal_rate'],
        results['avg_entropy'],
    )
    logger.info('[Output] results_dir=%s', output_dir)


if __name__ == '__main__':
    main()
