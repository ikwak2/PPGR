"""Cache of deterministic OOB inputs, never model features or fitted statistics."""
import hashlib
import inspect
import json
import os
import re
from pathlib import Path
import torch


def build_cached(builder, data_root, user_ids, protocol_id):
    cache_root = os.environ.get('FINAL_OOB_INPUT_CACHE', str(Path(__file__).resolve().parent / 'results/macro12_migration_260908/input_cache'))
    if not cache_root:
        return builder(data_root, user_ids)
    root = Path(data_root).resolve()
    # Bind cache to source implementation and every input CSV's size and mtime.
    files = [(str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(root.rglob('*.csv'))]
    source = Path(inspect.getsourcefile(builder))
    identity = {'data_root': str(root), 'files': files, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'protocol_id': re.sub(r'_seed_offset_\d+$', '', protocol_id), 'user_ids': list(user_ids)}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path(cache_root) / f'{builder.__module__}_{key}.pt'
    if not path.is_file():
        # Earlier cache entries included the seed suffix, although input construction
        # is deterministic and seed-independent. Accept their matching identities.
        for offset in (5000, 10000):
            legacy_identity = dict(identity, protocol_id=f"{identity['protocol_id']}_seed_offset_{offset}")
            legacy_key = hashlib.sha256(json.dumps(legacy_identity, sort_keys=True).encode()).hexdigest()
            legacy_path = path.parent / f'{builder.__module__}_{legacy_key}.pt'
            if legacy_path.is_file():
                path = legacy_path
                break
    if path.is_file():
        payload = torch.load(path, map_location='cpu', weights_only=False)
        stored_identity = dict(payload['identity'])
        stored_identity['protocol_id'] = re.sub(r'_seed_offset_\d+$', '', stored_identity['protocol_id'])
        if stored_identity != identity:
            raise RuntimeError('OOB input cache identity mismatch')
        print(f'OOB input cache hit users={list(user_ids)} path={path.name}', flush=True)
        return payload['users']
    users = builder(data_root, user_ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'.pid{os.getpid()}.tmp')
    torch.save({'identity': identity, 'users': users}, temporary)
    temporary.replace(path)
    print(f'OOB input cache saved users={list(user_ids)} path={path.name}', flush=True)
    return users
