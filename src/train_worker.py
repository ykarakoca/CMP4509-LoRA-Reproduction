import ast
import os
import sys
import json
import hashlib
import random
import gc
import time
import math
import uuid
import atexit
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
from importlib.metadata import version
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, RobertaForSequenceClassification, get_linear_schedule_with_warmup

REPO = Path(__file__).resolve().parents[1]
stage5 = Path(os.environ['LORA_RUN_ROOT']).resolve()
stage5.mkdir(parents=True, exist_ok=True)
code_dir = REPO / 'src'
protocol_path = REPO / 'configs/protocol.json'
LOCKED_HASH = '5433dadcd18642db88a4046e3d5a28e5a229c1b0fb74a96260a6eaa104a7eddf'

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def write_once(path, content):
    if path.exists():
        assert path.read_text(encoding='utf-8') == content, str(path)
    else:
        with path.open('x', encoding='utf-8') as handle:
            handle.write(content)

assert file_sha256(protocol_path) == LOCKED_HASH, 'Protocol hash mismatch'
protocol = json.loads(protocol_path.read_text())
method, seed = sys.argv[1], int(sys.argv[2])
assert method in protocol['methods'] and seed in protocol['seeds']
assert protocol['status'] == 'locked'
actual_packages = {k: version(k) for k in protocol['packages']}
if os.environ.get('LORA_STRICT_ENV') == '1':
    assert actual_packages == protocol['packages'], (actual_packages, protocol['packages'])
elif actual_packages != protocol['packages']:
    print('Environment differs from recorded Stage 5 environment:', actual_packages, flush=True)
for name, digest in protocol['source_sha256'].items():
    assert file_sha256(code_dir / name) == digest, name
runtime_path = code_dir / 'stage5_runtime.py'
assert file_sha256(runtime_path) == protocol['development_runner_sha256']

sys.path.insert(0, str(code_dir))
from lora_linear import LoRALinear
from lora_injection import inject_roberta_lora

# Reuse the exact runtime actually used for Stage 5, without regenerating its AST.
official_runtime_path = code_dir / 'stage5_official_runtime.py'
official_source = official_runtime_path.read_text(encoding='utf-8')
OFFICIAL_RUNTIME_SHA256 = file_sha256(official_runtime_path)
assert OFFICIAL_RUNTIME_SHA256 == '4874667d273f9d543993df352b6346c9f2dee4b77a9b9f85ef02204b169f5a3b'
WORKER_SHA256 = file_sha256(Path(__file__))
write_once(stage5 / 'public_run_metadata.json', json.dumps({
    'protocol_sha256': LOCKED_HASH, 'worker_sha256': WORKER_SHA256,
    'runtime_sha256': OFFICIAL_RUNTIME_SHA256,
    'scope': 'Independent repetition; not original Stage 5 evidence',
    'mode': os.environ.get('LORA_MODE', 'full'),
}, indent=2))
phase = 'smoke' if os.environ.get('LORA_MODE') == 'smoke' else 'official'
for result_path in sorted((stage5 / phase).glob('*/result.json')):
    old = json.loads(result_path.read_text())
    if old['config']['method'] == method and old['config']['seed'] == seed:
        assert old['protocol_sha256'] == LOCKED_HASH
        assert old['official_worker_sha256'] == WORKER_SHA256
        assert old['official_runtime_sha256'] == OFFICIAL_RUNTIME_SHA256
        if old['status'] == 'completed' and old.get('reload', {}).get('verified'):
            cp = old['best']['checkpoint']
            assert file_sha256(cp['path']) == cp['sha256']
            print('Already completed; skipped:', result_path, flush=True)
            sys.exit(0)
        raise RuntimeError(f'Existing incomplete run; inspect before retry: {result_path}')

(stage5 / phase).mkdir(exist_ok=True)
lock_path = stage5 / phase / f'{method}_seed{seed}.lock'
with lock_path.open('x') as handle:
    handle.write(str(Path(__file__)))
atexit.register(lambda: lock_path.unlink(missing_ok=True))

assert torch.cuda.is_available()
device = torch.device('cuda')
if os.environ.get('LORA_STRICT_ENV') == '1':
    assert torch.cuda.get_device_name(device) == protocol['measurement']['gpu']
torch.backends.cuda.matmul.allow_tf32 = protocol['measurement']['cuda_matmul_allow_tf32']
torch.backends.cudnn.allow_tf32 = protocol['measurement']['cudnn_allow_tf32']
assert torch.cuda.memory_allocated() == 0, 'Fresh-process baseline is not zero'
current_env = {
    'python': sys.version,
    'packages': {k: version(k) for k in protocol['packages']},
    'gpu_name': torch.cuda.get_device_name(device),
    'gpu_total_memory_bytes': torch.cuda.get_device_properties(device).total_memory,
    'cuda_available': True,
    'pytorch_cuda_build': torch.version.cuda,
    'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
    'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
    'source_sha256': protocol['source_sha256'],
    'verified_base_revision': protocol['model_revision'],
    'planned_precision': 'float32_no_amp',
}
MODEL_NAME = protocol['model_name']
MODEL_REVISION = protocol['model_revision']
MAX_LENGTH = protocol['training']['max_length']
EXPECTED_COUNTS = protocol['expected_trainable_parameters']
manifest_copy = REPO / protocol['data']['manifest_path']
assert file_sha256(manifest_copy) == protocol['data']['manifest_file_sha256']
manifest = json.loads(manifest_copy.read_text())
raw_train = load_dataset('nyu-mll/glue', 'sst2', split='train')
id_map = {sample_id: pos for pos, sample_id in enumerate(raw_train['idx'])}
assert len(id_map) == len(raw_train)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)

def tokenize_batch(batch):
    encoded = tokenizer(batch['sentence'], truncation=True, max_length=MAX_LENGTH, padding=False)
    encoded['labels'] = batch['label']
    return encoded

def restore_tokens(name):
    info = manifest['splits'][name]
    assert info['source_split'] == 'train'
    data = raw_train.select([id_map[i] for i in info['ids_in_order']])
    rows = [{k: row[k] for k in ('idx', 'sentence', 'label')} for row in data]
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    assert len(data) == info['size'] == protocol['data']['splits'][name]['size']
    assert list(data['idx']) == info['ids_in_order']
    assert hashlib.sha256(payload).hexdigest() == info['sha256'] == protocol['data']['splits'][name]['sha256']
    print(name, 'content hash: PASS', flush=True)
    return data.map(tokenize_batch, batched=True, remove_columns=data.column_names)

train_tokens = restore_tokens('training')
tune_tokens = restore_tokens('tuning')
collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')
exec(compile(official_source, str(official_runtime_path), 'exec'), globals())
print(f'OFFICIAL START: {method} / seed {seed} / protocol {LOCKED_HASH}', flush=True)
run_dir = run_experiment(method, seed, protocol['learning_rates'][method],
    1 if phase == 'smoke' else protocol['training']['epochs'], phase=phase,
    train_size=64 if phase == 'smoke' else None, tune_size=32 if phase == 'smoke' else None)
result = json.loads((Path(run_dir) / 'result.json').read_text())
assert result['status'] == 'completed' and result['reload']['verified']
assert result['baseline_allocated_bytes'] == 0
assert result['planned_optimizer_steps'] == (2 if phase == 'smoke' else 1565)
assert result['warmup_steps'] == (1 if phase == 'smoke' else 94)
print('OFFICIAL RUN COMPLETE; final evaluation not performed.', flush=True)
