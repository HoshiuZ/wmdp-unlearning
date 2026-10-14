"""Download the pinned official test file; all data and row mappings stay ignored/local."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(__file__).with_name('config.json')


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def build(rows):
    labels = {'harmful', 'unharmful'}
    groups = defaultdict(list)
    outputs, conflicts = [], []
    for index, row in enumerate(rows):
        for field in ('prompt_harm_label', 'response_harm_label'):
            if row[field] is not None and row[field] not in labels:
                raise ValueError(f'Unexpected label at row {index}, field {field}')
        prompt = row['prompt']
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        groups[prompt].append(index)
        response = row['response']
        if isinstance(response, str) and response.strip() and row['response_harm_label'] in labels:
            outputs.append({'task': 'output', 'request': prompt, 'response': response,
                            'gold': row['response_harm_label'], 'rows': [index],
                            'adversarial': row['adversarial'], 'content_sha256': digest([prompt, response])})
    inputs = []
    for prompt, indices in groups.items():
        known = {rows[i]['prompt_harm_label'] for i in indices} & labels
        item = {'task': 'input', 'request': prompt, 'rows': indices,
                'content_sha256': digest([prompt]),
                'adversarial_values': sorted({rows[i]['adversarial'] for i in indices}, key=str)}
        if len(known) > 1:
            conflicts.append(item)
        elif known:
            inputs.append(dict(item, gold=next(iter(known))))
    summary = {'raw_rows': len(rows), 'unique_nonempty_requests': len(groups),
               'repeated_request_extra_rows': sum(len(v)-1 for v in groups.values()),
               'input_conflicting_requests': len(conflicts),
               'input_without_known_label': sum(not ({rows[i]['prompt_harm_label'] for i in v} & labels) for v in groups.values()),
               'eligible_input': len(inputs), 'eligible_output': len(outputs),
               'input_labels': dict(Counter(x['gold'] for x in inputs)),
               'output_labels': dict(Counter(x['gold'] for x in outputs)),
               'missing_labels': {f: sum(r[f] is None for r in rows) for f in
                    ('prompt_harm_label', 'response_harm_label', 'response_refusal_label')},
               'output_unique_pairs': len({x['content_sha256'] for x in outputs})}
    return inputs, outputs, conflicts, summary


def main():
    cfg = json.loads(CONFIG.read_text())
    local = ROOT / cfg['local_directory'] / cfg['revision']
    local.mkdir(parents=True, exist_ok=True)
    # No fallback mirror, train split, automatic gate acceptance, or label substitution.
    path = Path(hf_hub_download(cfg['dataset_id'], cfg['file'], repo_type='dataset',
                              revision=cfg['revision'], cache_dir=str(ROOT / 'cache/huggingface/hub')))
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    required = {'prompt', 'response', 'adversarial', 'prompt_harm_label', 'response_harm_label', 'response_refusal_label'}
    if not required.issubset(table.column_names):
        raise ValueError('Official schema changed: missing required fields')
    inputs, outputs, conflicts, summary = build(table.to_pylist())
    manifest = []
    for name, items in [('input', inputs), ('output', outputs), ('conflicts', conflicts)]:
        target = local / f'{name}.jsonl'
        target.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in items))
        manifest.append({'file': name, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'rows': len(items)})
    summary.update(dataset=cfg, columns=table.column_names, data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), manifests=manifest)
    (local / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
