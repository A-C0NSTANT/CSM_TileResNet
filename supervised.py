import argparse
import logging
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import create_model, model_requires_history, model_requires_public

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Mahjong policy model.')
    parser.add_argument('--model', default='cnn', choices=['cnn', 'resnet', 'rarn', 'rarn_v2', 'rarn_public', 'rarn_public_v2', 'rarn_public_v2_large', 'rarn_public_v2_1_5x', 'rarn_public_v2_hist', 'rarn_public_v2_hist_gated', 'rarn_public_v2_hist_delta_gang'], help='Model architecture to train.')
    parser.add_argument('--data-dir', default='data', help='Directory containing count.json and *.npz files.')
    parser.add_argument('--output-dir', default='checkpoints', help='Directory for saved model checkpoints.')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--backbone-lr', type=float, default=None, help='Learning rate for backbone parameters. Defaults to --lr.')
    parser.add_argument('--history-lr', type=float, default=None, help='Learning rate for history/Transformer parameters. Defaults to --lr.')
    parser.add_argument('--fusion-lr', type=float, default=None, help='Learning rate for final fusion head parameters. Defaults to --lr.')
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--freeze-backbone-epochs', type=int, default=0, help='Freeze backbone during the first N epochs; history and fusion remain trainable.')
    parser.add_argument('--split-ratio', type=float, default=0.9)
    parser.add_argument('--suit-augment', default='none', choices=['none', 'random', 'all6'], help='Suit permutation augmentation for training data only.')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--log-interval', type=int, default=128)
    parser.add_argument('--log-file', default=None, help='Path to the training log file. Defaults to output-dir/train_YYYYmmdd_HHMMSS.log.')
    parser.add_argument('--init-checkpoint', default=None, help='Optional checkpoint used to initialize matching parameters before training.')
    parser.add_argument('--history-max-len', type=int, default=None, help='Override history length loaded from preprocessed data.')
    parser.add_argument('--kl-to-base-weight', type=float, default=0.0, help='KL regularization weight from model logits to frozen base logits when available.')
    parser.add_argument('--delta-l2-weight', type=float, default=0.0, help='L2 penalty weight for history delta logits when available.')
    parser.add_argument('--gang-margin-weight', type=float, default=0.0, help='Margin loss weight for AnyGang vs non-gang logits when available.')
    parser.add_argument('--gang-margin', type=float, default=0.2, help='Desired logit margin for AnyGang auxiliary loss.')
    return parser.parse_args()


def setup_logging(output_dir, log_file=None):
    os.makedirs(output_dir, exist_ok=True)
    if log_file is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(output_dir, 'train_%s.log' % timestamp)
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

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


def make_input(obs, mask, public, device, is_training, history=None, history_padding_mask=None):
    input_dict = {
        'is_training': is_training,
        'obs': {
            'observation': obs.to(device, non_blocking=True),
            'action_mask': mask.to(device, non_blocking=True),
        }
    }
    if public is not None:
        input_dict['obs']['public'] = public.to(device, non_blocking=True)
    if history is not None:
        input_dict['obs']['history'] = history.to(device, non_blocking=True)
    if history_padding_mask is not None:
        input_dict['obs']['history_padding_mask'] = history_padding_mask.to(device, non_blocking=True)
    return input_dict


def topk_correct(logits, target, k):
    k = min(k, logits.size(1))
    pred = logits.topk(k, dim=1).indices
    return pred.eq(target.view(-1, 1)).any(dim=1).sum().item()


ANYGANG_START = 133
ANYGANG_END = 235


def auxiliary_loss(model, logits, target, action_mask, args):
    aux = getattr(model, 'last_aux', None)
    if not aux:
        return logits.new_zeros(()), {}

    total = logits.new_zeros(())
    logs = {}
    base_logits = aux.get('base_logits')
    delta_logits = aux.get('delta_logits')
    if args.kl_to_base_weight > 0 and base_logits is not None:
        base_masked = base_logits.masked_fill(action_mask <= 0, torch.finfo(base_logits.dtype).min)
        kl = F.kl_div(
            F.log_softmax(logits, dim=1),
            F.softmax(base_masked.detach(), dim=1),
            reduction='batchmean',
        )
        total = total + args.kl_to_base_weight * kl
        logs['kl'] = kl.detach().item()
    if args.delta_l2_weight > 0 and delta_logits is not None:
        legal_delta = delta_logits.masked_fill(action_mask <= 0, 0.0)
        delta_l2 = legal_delta.square().mean()
        total = total + args.delta_l2_weight * delta_l2
        logs['delta_l2'] = delta_l2.detach().item()
    if args.gang_margin_weight > 0:
        gang_mask = action_mask[:, ANYGANG_START:ANYGANG_END] > 0
        has_gang = gang_mask.any(dim=1)
        is_gang_label = (target >= ANYGANG_START) & (target < ANYGANG_END)
        if has_gang.any():
            gang_logits = logits[:, ANYGANG_START:ANYGANG_END].masked_fill(~gang_mask, torch.finfo(logits.dtype).min)
            max_gang = gang_logits.max(dim=1).values
            non_gang_mask = (action_mask > 0).clone()
            non_gang_mask[:, ANYGANG_START:ANYGANG_END] = False
            max_non_gang = logits.masked_fill(~non_gang_mask, torch.finfo(logits.dtype).min).max(dim=1).values
            label_logits = logits.gather(1, target.view(-1, 1)).squeeze(1)
            gang_pos = is_gang_label & has_gang
            gang_neg = (~is_gang_label) & has_gang
            losses = []
            if gang_pos.any():
                losses.append(F.relu(max_non_gang[gang_pos] + args.gang_margin - label_logits[gang_pos]))
            if gang_neg.any():
                losses.append(F.relu(max_gang[gang_neg] + args.gang_margin - label_logits[gang_neg]))
            if losses:
                gang_margin = torch.cat(losses).mean()
                total = total + args.gang_margin_weight * gang_margin
                logs['gang_margin'] = gang_margin.detach().item()
    return total, logs


def train_one_epoch(model, loader, optimizer, device, epoch, log_interval, args, freeze_backbone=False):
    model.train()
    if freeze_backbone:
        set_backbone_modules_eval(model)
    start_time = time.perf_counter()
    total_loss = 0.0
    total_samples = 0
    total_top1 = 0

    for i, batch in enumerate(loader):
        obs, mask, act = batch[:3]
        public = batch[3] if len(batch) > 3 else None
        history = batch[4] if len(batch) > 4 else None
        history_padding_mask = batch[6] if len(batch) > 6 else None
        act = act.to(device, non_blocking=True).long()
        logits = model(make_input(obs, mask, public, device, True, history, history_padding_mask))
        mask_device = mask.to(device, non_blocking=True)
        ce_loss = F.cross_entropy(logits, act)
        aux_loss, aux_logs = auxiliary_loss(model, logits, act, mask_device, args)
        loss = ce_loss + aux_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = act.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        total_top1 += logits.argmax(dim=1).eq(act).sum().item()

        if i % log_interval == 0:
            elapsed = time.perf_counter() - start_time
            logger.info(
                '[Train] epoch=%d iter=%d/%d samples=%d loss_avg=%.6f top1_avg=%.4f ce=%.6f aux=%.6f kl=%s delta_l2=%s gang_margin=%s samples_per_sec=%.2f elapsed_sec=%.1f',
                epoch,
                i,
                len(loader),
                total_samples,
                total_loss / total_samples,
                total_top1 / total_samples,
                ce_loss.item(),
                aux_loss.item(),
                '%.6f' % aux_logs['kl'] if 'kl' in aux_logs else '-',
                '%.6f' % aux_logs['delta_l2'] if 'delta_l2' in aux_logs else '-',
                '%.6f' % aux_logs['gang_margin'] if 'gang_margin' in aux_logs else '-',
                total_samples / max(elapsed, 1e-6),
                elapsed,
            )

    elapsed = time.perf_counter() - start_time
    return total_loss / total_samples, total_top1 / total_samples, elapsed


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    start_time = time.perf_counter()
    total_loss = 0.0
    total_samples = 0
    total_top1 = 0
    total_top3 = 0
    total_top5 = 0

    for batch in loader:
        obs, mask, act = batch[:3]
        public = batch[3] if len(batch) > 3 else None
        history = batch[4] if len(batch) > 4 else None
        history_padding_mask = batch[6] if len(batch) > 6 else None
        act = act.to(device, non_blocking=True).long()
        logits = model(make_input(obs, mask, public, device, False, history, history_padding_mask))
        loss = F.cross_entropy(logits, act)

        batch_size = act.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        total_top1 += logits.argmax(dim=1).eq(act).sum().item()
        total_top3 += topk_correct(logits, act, 3)
        total_top5 += topk_correct(logits, act, 5)

    return {
        'loss': total_loss / total_samples,
        'top1': total_top1 / total_samples,
        'top3': total_top3 / total_samples,
        'top5': total_top5 / total_samples,
        'elapsed': time.perf_counter() - start_time,
    }


def checkpoint_prefix(model_name):
    return 'cnn_model' if model_name == 'cnn' else '%s_model' % model_name


def save_checkpoint(model, output_dir, epoch, model_name):
    os.makedirs(output_dir, exist_ok=True)
    prefix = checkpoint_prefix(model_name)
    epoch_path = os.path.join(output_dir, '%s_epoch_%d.pkl' % (prefix, epoch))
    latest_path = os.path.join(output_dir, '%s_latest.pkl' % prefix)
    state_dict = model.state_dict()
    torch.save(state_dict, epoch_path)
    torch.save(state_dict, latest_path)
    return epoch_path, latest_path


def load_init_checkpoint(model, checkpoint, device):
    state_dict = torch.load(checkpoint, map_location=device)
    current = model.state_dict()
    matched = {}
    skipped = []
    for key, value in state_dict.items():
        if key in current and current[key].shape == value.shape:
            matched[key] = value
        else:
            skipped.append(key)
    missing = [key for key in current if key not in matched]
    current.update(matched)
    model.load_state_dict(current)
    return len(matched), skipped, missing


PARAMETER_GROUPS = ('backbone', 'history', 'fusion')


def parameter_group_name(name, model=None):
    if model is not None and hasattr(model, 'parameter_group_name'):
        return model.parameter_group_name(name)
    if name == '_history_encoder' or name.startswith('_history_encoder.'):
        return 'history'
    if name == '_fusion_head' or name.startswith('_fusion_head.'):
        return 'fusion'
    return 'backbone'


def optimizer_group_lrs(args):
    return {
        'backbone': args.backbone_lr if args.backbone_lr is not None else args.lr,
        'history': args.history_lr if args.history_lr is not None else args.lr,
        'fusion': args.fusion_lr if args.fusion_lr is not None else args.lr,
    }


def create_optimizer(model, args):
    group_lrs = optimizer_group_lrs(args)
    params_by_group = {group: [] for group in PARAMETER_GROUPS}
    counts_by_group = {group: 0 for group in PARAMETER_GROUPS}
    for name, param in model.named_parameters():
        group = parameter_group_name(name, model)
        params_by_group[group].append(param)
        counts_by_group[group] += param.numel()

    param_groups = []
    for group in PARAMETER_GROUPS:
        if params_by_group[group]:
            param_groups.append({
                'params': params_by_group[group],
                'lr': group_lrs[group],
                'name': group,
            })
    return torch.optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay), counts_by_group, group_lrs


def set_backbone_frozen(model, frozen):
    for name, param in model.named_parameters():
        param.requires_grad = not frozen or parameter_group_name(name, model) != 'backbone'


def set_backbone_modules_eval(model):
    for name, module in model.named_modules():
        if name and parameter_group_name(name, model) == 'backbone':
            module.eval()


def count_trainable_by_group(model):
    counts_by_group = {group: 0 for group in PARAMETER_GROUPS}
    for name, param in model.named_parameters():
        if param.requires_grad:
            counts_by_group[parameter_group_name(name, model)] += param.numel()
    return counts_by_group


def main():
    args = parse_args()
    from dataset import MahjongGBDataset

    if args.freeze_backbone_epochs < 0:
        raise ValueError('--freeze-backbone-epochs must be non-negative')

    log_file = setup_logging(args.output_dir, args.log_file)
    set_seed(args.seed)
    device = torch.device(args.device)

    logger.info('[Log] file=%s', log_file)
    logger.info(
        '[Config] model=%s data_dir=%s output_dir=%s epochs=%d batch_size=%d lr=%.6g backbone_lr=%s history_lr=%s fusion_lr=%s weight_decay=%.6g freeze_backbone_epochs=%d kl_to_base_weight=%.6g delta_l2_weight=%.6g gang_margin_weight=%.6g gang_margin=%.6g split_ratio=%.4f suit_augment=%s num_workers=%d device=%s seed=%d log_interval=%d',
        args.model,
        args.data_dir,
        args.output_dir,
        args.epochs,
        args.batch_size,
        args.lr,
        'default' if args.backbone_lr is None else '%.6g' % args.backbone_lr,
        'default' if args.history_lr is None else '%.6g' % args.history_lr,
        'default' if args.fusion_lr is None else '%.6g' % args.fusion_lr,
        args.weight_decay,
        args.freeze_backbone_epochs,
        args.kl_to_base_weight,
        args.delta_l2_weight,
        args.gang_margin_weight,
        args.gang_margin,
        args.split_ratio,
        args.suit_augment,
        args.num_workers,
        device,
        args.seed,
        args.log_interval,
    )

    logger.info('[Data:train] begin=0.0000 end=%.4f', args.split_ratio)
    include_history = model_requires_history(args.model)
    train_dataset = MahjongGBDataset(0, args.split_ratio, args.suit_augment, data_dir=args.data_dir, split_name='train', include_history=include_history, history_max_len=args.history_max_len)
    logger.info('[Data:valid] begin=%.4f end=1.0000', args.split_ratio)
    validate_dataset = MahjongGBDataset(args.split_ratio, 1, False, data_dir=args.data_dir, split_name='valid', include_history=include_history, history_max_len=args.history_max_len)
    if model_requires_public(args.model):
        if not train_dataset.has_public or not validate_dataset.has_public:
            raise RuntimeError(
                '%s requires data with public features. '
                'Regenerate data with preprocess.py using DATA_DIR=%s' % (args.model, args.data_dir)
            )
    if include_history and (not train_dataset.has_history or not validate_dataset.has_history):
        raise RuntimeError(
            '%s requires data with history features. '
            'Regenerate data with INCLUDE_HISTORY=1 DATA_DIR=%s python preprocess.py' % (args.model, args.data_dir)
        )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )
    validate_loader = DataLoader(
        dataset=validate_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )

    model = create_model(args.model).to(device)
    if args.init_checkpoint:
        matched, skipped, missing = load_init_checkpoint(model, args.init_checkpoint, device)
        logger.info('[Init] checkpoint=%s matched=%d skipped=%d missing=%d', args.init_checkpoint, matched, len(skipped), len(missing))
        if skipped:
            logger.info('[Init] skipped_keys_sample=%s', ','.join(skipped[:12]))
        if missing:
            logger.info('[Init] missing_keys_sample=%s', ','.join(missing[:12]))
    param_count = sum(p.numel() for p in model.parameters())
    optimizer, group_counts, group_lrs = create_optimizer(model, args)

    logger.info('[Data] train_samples=%d valid_samples=%d', len(train_dataset), len(validate_dataset))
    logger.info('[Model] name=%s params=%d trainable_params=%d', args.model, param_count, sum(p.numel() for p in model.parameters() if p.requires_grad))
    logger.info(
        '[Optimizer] backbone_params=%d backbone_lr=%.6g history_params=%d history_lr=%.6g fusion_params=%d fusion_lr=%.6g weight_decay=%.6g',
        group_counts['backbone'],
        group_lrs['backbone'],
        group_counts['history'],
        group_lrs['history'],
        group_counts['fusion'],
        group_lrs['fusion'],
        args.weight_decay,
    )
    logger.info('[Train] start device=%s', device)

    for epoch in range(1, args.epochs + 1):
        freeze_backbone = epoch <= args.freeze_backbone_epochs
        set_backbone_frozen(model, freeze_backbone)
        trainable_counts = count_trainable_by_group(model)
        trainable_count = sum(trainable_counts.values())
        if trainable_count == 0:
            raise RuntimeError('No trainable parameters remain for this stage. Disable --freeze-backbone-epochs or use a model with history/fusion parameters.')
        logger.info(
            '[Stage] epoch=%d/%d freeze_backbone=%s trainable_params=%d trainable_backbone=%d trainable_history=%d trainable_fusion=%d',
            epoch,
            args.epochs,
            freeze_backbone,
            trainable_count,
            trainable_counts['backbone'],
            trainable_counts['history'],
            trainable_counts['fusion'],
        )
        logger.info('[Epoch] start epoch=%d/%d', epoch, args.epochs)
        train_loss, train_top1, train_elapsed = train_one_epoch(
            model, train_loader, optimizer, device, epoch, args.log_interval, args, freeze_backbone
        )
        metrics = validate(model, validate_loader, device)
        epoch_path, latest_path = save_checkpoint(model, args.output_dir, epoch, args.model)
        logger.info(
            '[Epoch] done epoch=%d/%d train_loss=%.6f train_top1=%.4f valid_loss=%.6f valid_top1=%.4f valid_top3=%.4f valid_top5=%.4f train_sec=%.1f valid_sec=%.1f',
            epoch,
            args.epochs,
            train_loss,
            train_top1,
            metrics['loss'],
            metrics['top1'],
            metrics['top3'],
            metrics['top5'],
            train_elapsed,
            metrics['elapsed'],
        )
        logger.info(
            '[Checkpoint] epoch=%s latest=%s',
            epoch_path,
            latest_path,
        )

    logger.info('[Train] finished log_file=%s', log_file)


if __name__ == '__main__':
    main()
