def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_model(method, seed):
    assert method in EXPECTED_COUNTS
    seed_everything(seed)
    fresh = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, revision=MODEL_REVISION, num_labels=2)
    if method == 'full':
        fresh.requires_grad_(True)
    else:
        fresh.requires_grad_(False)
        fresh.classifier.requires_grad_(True)
        if method.startswith('lora'):
            rank = 2 if method == 'lora2' else 8
            assert inject_roberta_lora(fresh, rank=rank, alpha=8) == 24
    adapters = [m for m in fresh.modules() if isinstance(m, LoRALinear)]
    expected_ids = {id(p) for p in fresh.parameters()} if method == 'full' else {id(p) for p in fresh.classifier.parameters()} | {id(p) for m in adapters for p in (m.A, m.B)}
    assert {id(p) for p in fresh.parameters() if p.requires_grad} == expected_ids
    assert sum((p.numel() for p in fresh.parameters() if p.requires_grad)) == EXPECTED_COUNTS[method]
    assert len(adapters) == (24 if method.startswith('lora') else 0)
    return fresh

@torch.inference_mode()
def evaluate(model, loader):
    was_training = model.training
    model.eval()
    model_device = next(model.parameters()).device
    total_loss, total = (0.0, 0)
    confusion = torch.zeros(2, 2, dtype=torch.long)
    torch.cuda.synchronize(model_device)
    started = time.perf_counter()
    try:
        for cpu_batch in loader:
            batch = {k: v.to(model_device) for k, v in cpu_batch.items()}
            output = model(**batch)
            assert torch.isfinite(output.loss).item()
            assert torch.isfinite(output.logits).all().item()
            labels = batch['labels']
            predictions = output.logits.argmax(dim=-1)
            size = labels.numel()
            total_loss += output.loss.item() * size
            total += size
            confusion += torch.bincount((2 * labels + predictions).cpu(), minlength=4).reshape(2, 2)
        torch.cuda.synchronize(model_device)
        elapsed = time.perf_counter() - started
    finally:
        model.train(was_training)
    assert total > 0
    return {'samples': total, 'loss': total_loss / total, 'accuracy': confusion.diag().sum().item() / total, 'prediction_counts': {str(i): confusion[:, i].sum().item() for i in (0, 1)}, 'confusion_matrix': confusion.tolist(), 'evaluation_seconds': elapsed}

def train_one_epoch(model, loader, optimizer, scheduler, accumulation=2):
    assert isinstance(accumulation, int) and accumulation > 0
    assert len(loader) > 0
    model_device = next(model.parameters()).device
    trainable = [p for p in model.parameters() if p.requires_grad]
    frozen = [p for p in model.parameters() if not p.requires_grad]
    optimizer_params = [p for group in optimizer.param_groups for p in group['params']]
    assert len(optimizer_params) == len({id(p) for p in optimizer_params})
    assert {id(p) for p in optimizer_params} == {id(p) for p in trainable}
    model.train()
    model.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_correct = total_samples = group_samples = 0
    optimizer_steps = positive_lr_steps = 0
    torch.cuda.synchronize(model_device)
    started = time.perf_counter()
    for index, cpu_batch in enumerate(loader):
        batch = {k: v.to(model_device) for k, v in cpu_batch.items()}
        output = model(**batch)
        assert torch.isfinite(output.loss).item()
        assert torch.isfinite(output.logits).all().item()
        size = batch['labels'].numel()
        (output.loss * size).backward()
        group_samples += size
        total_loss += output.loss.item() * size
        total_correct += (output.logits.detach().argmax(dim=-1) == batch['labels']).sum().item()
        total_samples += size
        if (index + 1) % accumulation == 0 or index + 1 == len(loader):
            assert all((p.grad is None for p in frozen))
            for p in trainable:
                assert p.grad is not None
                p.grad.div_(group_samples)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=float('inf'), error_if_nonfinite=True)
            positive_lr_steps += int(any((group['lr'] > 0 for group in optimizer.param_groups)))
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            group_samples = 0
    torch.cuda.synchronize(model_device)
    elapsed = time.perf_counter() - started
    assert group_samples == 0
    assert optimizer_steps == math.ceil(len(loader) / accumulation)
    return {'samples': total_samples, 'online_training_loss': total_loss / total_samples, 'online_training_accuracy': total_correct / total_samples, 'optimizer_steps': optimizer_steps, 'steps_with_positive_lr': positive_lr_steps, 'last_gradient_norm': grad_norm.item(), 'training_seconds': elapsed}

def is_better(candidate, best):
    return best is None or (candidate['accuracy'] > best['accuracy'] or (candidate['accuracy'] == best['accuracy'] and candidate['loss'] < best['loss']))

def checkpoint_keys(model, method):
    state = model.state_dict()
    if method == 'full':
        return set(state)
    return {name for name, p in model.named_parameters() if p.requires_grad} | {name for name, _ in model.named_buffers()} & set(state)

def save_model_checkpoint(model, method, seed, epoch, tuning, config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    keys = checkpoint_keys(model, method)
    payload = {'schema_version': 1, 'model_name': MODEL_NAME, 'model_revision': MODEL_REVISION, 'method': method, 'seed': seed, 'epoch': epoch, 'tuning': dict(tuning), 'config': dict(config), 'split_manifest_sha256': file_sha256(manifest_copy), 'source_sha256': dict(current_env['source_sha256']), 'state': {name: state[name].detach().cpu().clone() for name in sorted(keys)}}
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {'path': str(path), 'size_bytes': path.stat().st_size, 'sha256': file_sha256(path), 'saved_state_tensors': len(keys)}

def restore_model_checkpoint(path, expected_sha256):
    assert file_sha256(path) == expected_sha256
    payload = torch.load(path, map_location='cpu', weights_only=True)
    assert payload['schema_version'] == 1
    assert payload['model_name'] == MODEL_NAME
    assert payload['model_revision'] == MODEL_REVISION
    assert payload['split_manifest_sha256'] == file_sha256(manifest_copy)
    assert payload['source_sha256'] == current_env['source_sha256']
    restored = build_model(payload['method'], payload['seed'])
    keys = checkpoint_keys(restored, payload['method'])
    assert set(payload['state']) == keys
    result = restored.load_state_dict(payload['state'], strict=payload['method'] == 'full')
    assert not result.unexpected_keys
    assert set(result.missing_keys) == set(restored.state_dict()) - keys
    state = restored.state_dict()
    for name, tensor in payload['state'].items():
        assert torch.equal(state[name], tensor), name
    metadata = {k: v for k, v in payload.items() if k != 'state'}
    return (restored, metadata)

def run_experiment(method, seed, learning_rate, epochs, phase='smoke', train_size=None, tune_size=None):
    assert phase in ('smoke', 'tuning', 'official')
    assert method in EXPECTED_COUNTS
    assert epochs >= 1 and learning_rate > 0
    if phase in ('tuning', 'official'):
        assert train_size is None and tune_size is None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print(f'GPU başlangıcı: allocated={baseline / 1024 ** 2:.2f} MiB | reserved={reserved / 1024 ** 2:.2f} MiB')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    run_dir = stage5 / phase / f'{method}_seed{seed}_{run_id}'
    run_dir.mkdir(parents=True, exist_ok=False)
    training_tokens = train_tokens if train_size is None else train_tokens.select(range(train_size))
    tuning_tokens = tune_tokens if tune_size is None else tune_tokens.select(range(tune_size))
    config = {'method': method, 'seed': seed, 'epochs': epochs, 'learning_rate': learning_rate, 'batch_size': 16, 'gradient_accumulation': 2, 'weight_decay': 0.01, 'warmup_ratio': 0.06, 'precision': 'float32_no_amp', 'max_length': MAX_LENGTH, 'alpha': 8 if method.startswith('lora') else None, 'rank': {'lora2': 2, 'lora8': 8}.get(method), 'train_samples': len(training_tokens), 'tune_samples': len(tuning_tokens), 'subset_policy': 'full_manifest_splits' if phase in ('tuning', 'official') else 'first_N_for_engineering_smoke_only', 'selection': 'max_tuning_accuracy_then_min_loss_then_earliest', 'gradient_clipping': None}
    log = {'phase': phase, 'status': 'running', 'config': config, 'model_revision': MODEL_REVISION, 'split_manifest_sha256': file_sha256(manifest_copy), 'environment': current_env, 'runtime_source_sha256': file_sha256(code_dir / 'stage5_runtime.py'), 'baseline_allocated_bytes': baseline, 'baseline_reserved_bytes': reserved, 'memory_measurement_scope': 'Fresh-process training; PyTorch allocator measurements, not total physical VRAM', 'epochs': [], 'best': None, 'protocol_sha256': LOCKED_HASH, 'official_runtime_sha256': OFFICIAL_RUNTIME_SHA256, 'official_worker_sha256': WORKER_SHA256}

    def write_log():
        temporary = run_dir / 'result.json.tmp'
        temporary.write_text(json.dumps(log, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(run_dir / 'result.json')
    write_log()
    model = optimizer = scheduler = restored = None
    ref_inputs = reference_logits = reload_logits = None
    started = time.perf_counter()
    try:
        model = build_model(method, seed).to(device)
        log['trainable_parameters'] = sum((p.numel() for p in model.parameters() if p.requires_grad))
        log['total_parameters'] = sum((p.numel() for p in model.parameters()))
        train_loader = DataLoader(training_tokens, batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(seed), collate_fn=collator, drop_last=False)
        tune_loader = DataLoader(tuning_tokens, batch_size=16, shuffle=False, generator=torch.Generator().manual_seed(seed), collate_fn=collator, drop_last=False)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate, weight_decay=0.01)
        total_steps = math.ceil(len(train_loader) / 2) * epochs
        warmup_steps = math.ceil(total_steps * 0.06)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        log['planned_optimizer_steps'] = total_steps
        log['warmup_steps'] = warmup_steps
        reference_batch = collator([tuning_tokens[i] for i in range(min(8, len(tuning_tokens)))])
        ref_inputs = {k: v.to(device) for k, v in reference_batch.items() if k != 'labels'}
        print(f'Training: {len(training_tokens)} | Tuning: {len(tuning_tokens)} | Toplam optimizer step: {total_steps}')
        seed_everything(seed)
        for epoch in range(1, epochs + 1):
            torch.cuda.reset_peak_memory_stats()
            training = train_one_epoch(model, train_loader, optimizer, scheduler, accumulation=2)
            training['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            training['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            tuning = evaluate(model, tune_loader)
            previous = None if log['best'] is None else log['best']['tuning']
            if is_better(tuning, previous):
                checkpoint = save_model_checkpoint(model, method, seed, epoch, tuning, config, run_dir / 'best_model.pt')
                log['best'] = {'epoch': epoch, 'tuning': tuning, 'checkpoint': checkpoint}
                model.eval()
                with torch.inference_mode():
                    reference_logits = model(**ref_inputs).logits.cpu().clone()
                model.train()
            log['epochs'].append({'epoch': epoch, 'training': training, 'tuning': tuning})
            write_log()
            print(f"{method} | epoch {epoch}/{epochs} | train loss={training['online_training_loss']:.4f} | tune loss={tuning['loss']:.4f} | tune acc={tuning['accuracy']:.4f} | pred={tuning['prediction_counts']} | best epoch={log['best']['epoch']}", flush=True)
        assert sum((row['training']['optimizer_steps'] for row in log['epochs'])) == total_steps
        model = optimizer = scheduler = None
        gc.collect()
        torch.cuda.empty_cache()
        selected = log['best']['checkpoint']
        restored, metadata = restore_model_checkpoint(selected['path'], selected['sha256'])
        assert metadata['epoch'] == log['best']['epoch']
        restored = restored.to(device).eval()
        with torch.inference_mode():
            reload_logits = restored(**ref_inputs).logits.cpu()
        torch.testing.assert_close(reload_logits, reference_logits, rtol=1e-05, atol=1e-06)
        reloaded = evaluate(restored, tune_loader)
        original = log['best']['tuning']
        for key in ('samples', 'accuracy', 'prediction_counts', 'confusion_matrix'):
            assert reloaded[key] == original[key], key
        assert math.isclose(reloaded['loss'], original['loss'], rel_tol=1e-05, abs_tol=1e-06)
        log['reload'] = {'verified': True, 'reference_samples': len(reference_logits), 'max_abs_logit_difference': (reload_logits - reference_logits).abs().max().item(), 'tuning': reloaded}
        log['status'] = 'completed'
        log['wall_seconds_including_setup_save_reload'] = time.perf_counter() - started
        write_log()
        print('Best-checkpoint fresh-model reload: PASS')
        print('Run klasörü:', run_dir)
        return str(run_dir)
    except Exception as error:
        log['status'] = 'failed'
        log['error'] = f'{type(error).__name__}: {error}'
        write_log()
        raise
    finally:
        model = optimizer = scheduler = restored = None
        ref_inputs = reference_logits = reload_logits = None
        gc.collect()
        torch.cuda.empty_cache()
