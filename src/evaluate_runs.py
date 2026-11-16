"""Evaluate the 12 fixed Stage 5 checkpoints; never train or select models.

Run in a fresh Colab subprocess: python -u stage5_final_evaluation.py
Completed results are reused only after provenance and prediction checks.
An interrupted checkpoint is evaluated again; completed checkpoints are skipped.
"""
import ast
import os
import csv
import gc
import hashlib
import json
import math
import random
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ['LORA_RUN_ROOT']).resolve()
PROTOCOL_HASH = '5433dadcd18642db88a4046e3d5a28e5a229c1b0fb74a96260a6eaa104a7eddf'
# The new index is fixed by the orchestrator only after all 12 repetitions finish.
RUNTIME_HASH = '4874667d273f9d543993df352b6346c9f2dee4b77a9b9f85ef02204b169f5a3b'


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                  allow_nan=False), encoding='utf-8')
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def prediction_metrics(rows):
    """Independently derive accuracy/confusion from saved ordered predictions."""
    cm = [[0, 0], [0, 0]]
    for row in rows:
        label, pred, logits = row['label'], row['prediction'], row['logits']
        assert label in (0, 1) and pred in (0, 1)
        assert len(logits) == 2 and all(math.isfinite(x) for x in logits)
        assert pred == (0 if logits[0] >= logits[1] else 1)
        cm[label][pred] += 1
    assert rows
    return {'samples': len(rows), 'accuracy': (cm[0][0] + cm[1][1]) / len(rows),
            'confusion_matrix': cm,
            'prediction_counts': {str(j): sum(cm[k][j] for k in range(2))
                                  for j in range(2)}}


def main():
    global np, torch, RobertaForSequenceClassification, LoRALinear
    global inject_roberta_lora, MODEL_NAME, MODEL_REVISION, EXPECTED_COUNTS
    global manifest_copy, current_env
    assert file_sha256(REPO / 'configs/protocol.json') == PROTOCOL_HASH
    INDEX_HASH = file_sha256(ROOT / 'official_selected_checkpoints.json')
    assert file_sha256(ROOT / 'official_selected_checkpoints.json') == INDEX_HASH
    protocol = read_json(REPO / 'configs/protocol.json')
    index = read_json(ROOT / 'official_selected_checkpoints.json')
    assert protocol['status'] == 'locked'
    assert index['protocol_sha256'] == PROTOCOL_HASH
    entries = index['checkpoints']
    assert index['checkpoint_count'] == len(entries) == 12
    assert {(e['method'], e['seed']) for e in entries} == {
        (m, s) for m in protocol['methods'] for s in protocol['seeds']}
    code = REPO / 'src'
    assert file_sha256(code / 'stage5_official_runtime.py') == RUNTIME_HASH
    assert file_sha256(code / 'stage5_runtime.py') == protocol['development_runner_sha256']
    for name, digest in protocol['source_sha256'].items():
        assert file_sha256(code / name) == digest, name
    manifest_copy = REPO / protocol['data']['manifest_path']
    assert file_sha256(manifest_copy) == protocol['data']['manifest_file_sha256']
    # Check ALL selected weights and immutable training logs before reading final labels.
    for e in entries:
        checkpoint, result_path = Path(e['checkpoint_path']), Path(e['result_path'])
        assert result_path.parent.parent.resolve() == (ROOT / 'official').resolve()
        assert checkpoint.resolve() == (result_path.parent / 'best_model.pt').resolve()
        assert file_sha256(result_path) == e['result_sha256']
        r = read_json(result_path)
        assert r['phase'] == 'official' and r['status'] == 'completed'
        assert (r['config']['method'], r['config']['seed']) == (e['method'], e['seed'])
        assert r['protocol_sha256'] == PROTOCOL_HASH
        assert r['official_runtime_sha256'] == RUNTIME_HASH
        assert r['reload']['verified']
        best = min(r['epochs'], key=lambda x: (-x['tuning']['accuracy'],
                                              x['tuning']['loss'], x['epoch']))
        assert best['epoch'] == r['best']['epoch'] == e['selected_epoch']
        assert r['best']['checkpoint']['sha256'] == e['checkpoint_sha256']
        assert checkpoint.stat().st_size == e['checkpoint_size_bytes']
        assert file_sha256(checkpoint) == e['checkpoint_sha256']
    print('Fixed checkpoint index and all 12 file hashes: PASS', flush=True)

    import numpy as np
    import torch
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, DataCollatorWithPadding, RobertaForSequenceClassification
    actual_packages = {k: version(k) for k in protocol['packages']}
    if os.environ.get('LORA_STRICT_ENV') == '1':
        assert actual_packages == protocol['packages']
    assert torch.cuda.is_available(), 'Select Tesla T4 GPU in Colab.'
    if os.environ.get('LORA_STRICT_ENV') == '1':
        assert torch.cuda.get_device_name(0) == protocol['measurement']['gpu']
    torch.backends.cuda.matmul.allow_tf32 = protocol['measurement']['cuda_matmul_allow_tf32']
    torch.backends.cudnn.allow_tf32 = protocol['measurement']['cudnn_allow_tf32']
    sys.path.insert(0, str(code))
    from lora_linear import LoRALinear
    from lora_injection import inject_roberta_lora
    MODEL_NAME, MODEL_REVISION = protocol['model_name'], protocol['model_revision']
    EXPECTED_COUNTS = protocol['expected_trainable_parameters']
    current_env = {'source_sha256': protocol['source_sha256']}
    # Reuse exactly the audited restoration functions; no training function is loaded.
    tree = ast.parse((code / 'stage5_official_runtime.py').read_text())
    wanted = {'seed_everything', 'build_model', 'checkpoint_keys', 'restore_model_checkpoint'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in nodes} == wanted
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<audited restoration>', 'exec'), globals())

    out = ROOT / 'final_evaluation'
    out.mkdir(exist_ok=True)
    lock = out / '.evaluation.lock'
    with lock.open('x') as f:
        f.write(datetime.now(timezone.utc).isoformat())
    try:
        provenance = {'protocol_sha256': PROTOCOL_HASH, 'checkpoint_index_sha256': INDEX_HASH,
                      'runtime_sha256': RUNTIME_HASH, 'evaluation_script_sha256': file_sha256(__file__),
                      'final_split_sha256': protocol['data']['splits']['final_evaluation']['sha256']}
        provenance_path = out / 'provenance.json'
        if provenance_path.exists():
            assert read_json(provenance_path) == provenance, 'Existing evaluation provenance differs.'
        else:
            atomic_json(provenance_path, provenance)

        info = read_json(manifest_copy)['splits']['final_evaluation']
        assert info['source_split'] == 'validation'
        raw = load_dataset('nyu-mll/glue', 'sst2', split='validation')
        positions = {idx: pos for pos, idx in enumerate(raw['idx'])}
        assert len(positions) == len(raw)
        data = raw.select([positions[i] for i in info['ids_in_order']])
        rows = [{k: row[k] for k in ('idx', 'sentence', 'label')} for row in data]
        content = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
        assert len(rows) == info['size'] == 872
        assert hashlib.sha256(content).hexdigest() == info['sha256'] == provenance['final_split_sha256']
        print('Final evaluation: 872 samples, content and order hash PASS', flush=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
        def tokenize(batch):
            encoded = tokenizer(batch['sentence'], truncation=True,
                                max_length=protocol['training']['max_length'], padding=False)
            encoded['labels'] = batch['label']
            return encoded
        tokens = data.map(tokenize, batched=True, remove_columns=data.column_names)
        collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')
        environment = {'python': sys.version, 'packages': actual_packages,
                       'gpu_name': torch.cuda.get_device_name(0), 'cuda_build': torch.version.cuda,
                       'precision': 'float32_no_amp'}
        completed = []
        for e in entries:
            tag = f"{e['method']}_seed{e['seed']}"
            result_file, predictions_file = out / f'{tag}.json', out / f'{tag}_predictions.json'
            if result_file.exists():
                result = read_json(result_file)
                assert result['status'] == 'completed' and result['provenance'] == provenance
                assert result['checkpoint'] == e
                assert file_sha256(predictions_file) == result['predictions_sha256']
                saved_rows = read_json(predictions_file)
                assert [(r['idx'], r['label']) for r in saved_rows] == [(r['idx'], r['label']) for r in rows]
                assert all(result['metrics'][k] == v for k, v in prediction_metrics(saved_rows).items())
                assert math.isfinite(result['metrics']['loss'])
                print(tag, 'existing verified result: SKIP', flush=True)
                completed.append(result)
                continue
            print('FINAL EVALUATION:', tag, flush=True)
            model, metadata = restore_model_checkpoint(e['checkpoint_path'], e['checkpoint_sha256'])
            assert (metadata['method'], metadata['seed'], metadata['epoch']) == (
                e['method'], e['seed'], e['selected_epoch'])
            model = model.to('cuda').eval()
            loader = DataLoader(tokens, batch_size=16, shuffle=False, drop_last=False,
                                generator=torch.Generator().manual_seed(e['seed']), collate_fn=collator)
            predictions, total_loss, offset = [], 0.0, 0
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                for cpu_batch in loader:
                    batch = {k: v.to('cuda') for k, v in cpu_batch.items()}
                    output = model(**batch)
                    assert torch.isfinite(output.loss).item() and torch.isfinite(output.logits).all().item()
                    size = batch['labels'].numel()
                    total_loss += output.loss.item() * size
                    logits = output.logits.cpu().tolist()
                    labels = batch['labels'].cpu().tolist()
                    preds = output.logits.argmax(-1).cpu().tolist()
                    for j in range(size):
                        assert labels[j] == rows[offset + j]['label']
                        predictions.append({'idx': rows[offset + j]['idx'], 'label': labels[j],
                                            'prediction': preds[j], 'logits': logits[j]})
                    offset += size
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            assert offset == 872
            metrics = prediction_metrics(predictions)
            metrics.update(loss=total_loss / offset, evaluation_seconds=elapsed)
            atomic_json(predictions_file, predictions)
            result = {'status': 'completed', 'checkpoint': e, 'provenance': provenance,
                      'environment': environment, 'metrics': metrics,
                      'predictions_sha256': file_sha256(predictions_file)}
            atomic_json(result_file, result)
            completed.append(result)
            print(f"{tag} | epoch={e['selected_epoch']} | final acc={metrics['accuracy']:.6f} | loss={metrics['loss']:.6f}", flush=True)
            del model, output, batch, cpu_batch, loader
            gc.collect()
            torch.cuda.empty_cache()

        summary = []
        for method in protocol['methods']:
            group = [r for r in completed if r['checkpoint']['method'] == method]
            assert len(group) == 3
            item = {'method': method, 'n_seeds': 3}
            for metric in ('accuracy', 'loss'):
                values = [r['metrics'][metric] for r in group]
                item[metric + '_mean'] = statistics.mean(values)
                item[metric + '_sample_std_ddof1'] = statistics.stdev(values)
            summary.append(item)
        atomic_json(out / 'summary.json', {'status': 'completed', 'provenance': provenance,
                    'runs': completed, 'aggregate': summary})
        with (out / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        for item in summary:
            print(f"{item['method']:5s} | final accuracy mean={item['accuracy_mean']:.6f} | sample std={item['accuracy_sample_std_ddof1']:.6f}", flush=True)
        print('FINAL EVALUATION COMPLETE: 12/12; no training performed.', flush=True)
        print('Results:', out, flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
