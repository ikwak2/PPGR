"""Equal-weight EER reporting: 3 enrollments x 4 Top-M x 4 folds, then 3 seeds."""
from pathlib import Path
import csv
import fcntl
import functools
import json
import os
import re
import numpy as np

SECONDS = (10, 20, 30)
TOP_M = (1, 3, 5, 10)
OFFSETS = (0, 5000, 10000)
AGGREGATION = 'macro12_v1'


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'.pid{os.getpid()}.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def write_csv(path, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_cells(cells):
    if set(cells) != {str(s) for s in SECONDS}:
        raise ValueError('Macro-12 requires exactly enrollment 10/20/30')
    for length in cells.values():
        if set(length) != {str(m) for m in TOP_M}:
            raise ValueError('Macro-12 requires separate Top-1/3/5/10 EER cells; legacy averaged-score results are invalid')
        for cell in length.values():
            eer = cell['overall']['eer']
            if not np.isfinite(eer) or not 0 <= eer <= 1:
                raise ValueError(f'invalid EER: {eer}')


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return {'eer_mean': float(a.mean()), 'eer_std': float(a.std(ddof=0)), 'per_fold': a.tolist()}


def aggregate_macro12(output, variant, protocol_id, offset, protocol, model):
    output = Path(output)
    folds = []
    for index in range(1, 5):
        fold = json.loads((output / f'oob_fold{index}.json').read_text())
        if (fold.get('status'), fold.get('variant'), fold.get('protocol_id'), fold.get('fold_id'), fold.get('seed_offset'), fold.get('evaluation_aggregation')) != ('complete', variant, protocol_id, index, offset, AGGREGATION):
            raise ValueError(f'invalid fold {index} metadata')
        validate_cells(fold['cells'])
        expected = np.mean([c['overall']['eer'] for length in fold['cells'].values() for c in length.values()])
        if not np.isclose(expected, fold['macro_eer_across_12_cells'], atol=1e-12, rtol=0):
            raise ValueError(f'inconsistent fold {index} macro')
        folds.append(fold)
    cross_fold, rows = {}, []
    for seconds in SECONDS:
        cross_fold[str(seconds)] = {}
        for top_m in TOP_M:
            cells = [fold['cells'][str(seconds)][str(top_m)] for fold in folds]
            overall = stats([cell['overall']['eer'] for cell in cells])
            activities = {name: stats([cell['activity'][name]['eer'] for cell in cells]) for name in cells[0]['activity']}
            cross_fold[str(seconds)][str(top_m)] = {'overall': overall, 'activity': activities}
            rows.append({'seed_offset': offset, 'enrollment_seconds': seconds, 'top_m': top_m, 'eer_mean': overall['eer_mean'], 'eer_std': overall['eer_std'], **{f'fold{i}_eer': value for i, value in enumerate(overall['per_fold'], 1)}})
    macro = float(np.mean([row['eer_mean'] for row in rows]))
    payload = {'status': 'complete', 'variant': variant, 'protocol_id': protocol_id, 'seed_offset': offset, 'evaluation_aggregation': AGGREGATION, 'macro12_eer': macro, 'cross_fold': cross_fold, 'protocol': protocol, 'model': model}
    write_json(output / 'oob_cross_fold.json', payload)
    write_csv(output / 'oob_cross_fold_summary.csv', rows)
    print(f'{variant} offset={offset} macro12={100*macro:.3f}%', flush=True)
    return payload


def summarize_macro12(base_root, result_subdir, variant, protocol_id):
    base_root = Path(base_root)
    protocol_id = re.sub(r"_seed_offset_\d+$", "", protocol_id)
    runs = [json.loads((base_root / f'offset_{o}' / result_subdir / 'oob_cross_fold.json').read_text()) for o in OFFSETS]
    for offset, run in zip(OFFSETS, runs):
        expected_protocol = protocol_id if offset == 0 else f'{protocol_id}_seed_offset_{offset}'
        if (run.get('status'), run.get('variant'), run.get('protocol_id'), run.get('seed_offset'), run.get('evaluation_aggregation')) != ('complete', variant, expected_protocol, offset, AGGREGATION):
            raise ValueError(f'invalid cross-fold metadata at offset {offset}')
        validate_cells({s: {m: {'overall': {'eer': c['overall']['eer_mean']}} for m, c in length.items()} for s, length in run['cross_fold'].items()})
    def seed_row(values, **labels):
        a = np.asarray(values)
        return {**labels, 'eer_mean': float(a.mean()), 'eer_seed_sd_ddof1': float(a.std(ddof=1)), **{f'offset_{o}_eer': float(v) for o, v in zip(OFFSETS, a)}}
    cells = [seed_row([r['cross_fold'][str(s)][str(m)]['overall']['eer_mean'] for r in runs], enrollment_seconds=s, top_m=m) for s in SECONDS for m in TOP_M]
    enrollment = [seed_row([np.mean([r['cross_fold'][str(s)][str(m)]['overall']['eer_mean'] for m in TOP_M]) for r in runs], enrollment_seconds=s) for s in SECONDS]
    top_m_rows = [seed_row([np.mean([r['cross_fold'][str(s)][str(m)]['overall']['eer_mean'] for s in SECONDS]) for r in runs], top_m=m) for m in TOP_M]
    macro = np.asarray([r['macro12_eer'] for r in runs])
    for run, value in zip(runs, macro):
        expected = np.mean([c['overall']['eer_mean'] for length in run['cross_fold'].values() for c in length.values()])
        if not np.isclose(expected, value, atol=1e-12, rtol=0):
            raise ValueError('inconsistent seed macro')
    payload = {'status': 'complete', 'variant': variant, 'protocol_id': protocol_id, 'evaluation_aggregation': AGGREGATION, 'seed_offsets': list(OFFSETS), 'macro12_eer_mean': float(macro.mean()), 'macro12_eer_seed_sd_ddof1': float(macro.std(ddof=1)), 'macro12_eer_by_seed': {str(o): float(v) for o, v in zip(OFFSETS, macro)}, 'cells': cells, 'enrollment_cells': enrollment, 'top_m_cells': top_m_rows}
    out = base_root / 'three_seed_summary'
    write_json(out / 'three_seed_summary.json', payload)
    for filename, rows in [('three_seed_cells.csv', cells), ('three_seed_enrollment.csv', enrollment), ('three_seed_top_m.csv', top_m_rows)]:
        write_csv(out / filename, rows)
    print(f'{variant} three-seed macro12={100*macro.mean():.3f} +/- {100*macro.std(ddof=1):.3f}%', flush=True)
    return payload


def evaluation_lock(function):
    """Serialize reevaluation of the same result-root/fold across launchers."""
    @functools.wraps(function)
    def locked(fold_id, data_root, result_root, device):
        root = Path(result_root)
        root.mkdir(parents=True, exist_ok=True)
        with (root / f'.macro12_fold{fold_id}.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            return function(fold_id, data_root, result_root, device)
    return locked
