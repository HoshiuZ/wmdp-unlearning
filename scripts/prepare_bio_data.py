"""Conservative, reproducible Bio corpus preparation. Raw/processed corpora stay local."""
import collections
import hashlib
import json
import math
import random
import re
import unicodedata
from pathlib import Path
from datetime import datetime, timezone, timedelta
import importlib.metadata
import ahocorasick
import numpy as np
import pyarrow.parquet as pq
from datasketch import MinHash, MinHashLSH
from rapidfuzz.fuzz import ratio

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/processed/bio_v1'
SEED = 42
VERSION = 'bio-clean-v1'
FORGET_REV = '5a786ed1041e56fef2d4ed26c1239f12b73a68eb'
RETAIN_REV = '030ad12feed583a291df21471a153ff70bebf002f27ee4eb19c9c41b6ccf4bdf'


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def norm(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def lexical(text):
    return ' '.join(re.findall(r'\w+', norm(text)))


def raw_records():
    for path in sorted((ROOT / 'data/raw/wmdp-bio-forget/data').glob('*.parquet')):
        row_index = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            for row in batch.to_pylist():
                yield 'forget', path, row_index, row, FORGET_REV
                row_index += 1
    path = ROOT / 'open-unlearning/data/wmdp/wmdp-corpora/bio-retain-corpus.jsonl'
    with path.open(encoding='utf-8') as stream:
        for row_index, line in enumerate(stream):
            text = json.loads(line)
            if not isinstance(text, str):
                raise ValueError(f'Expected JSON string at row {row_index}')
            yield 'retain', path, row_index, {'text': text}, RETAIN_REV


def record_id(role, path, index, rev):
    return f'wmdp-bio-{role}-{rev[:12]}-{path.stem}-{index:07d}'


def doi_key(doi):
    return re.sub(r'^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)', '', (doi or '').strip().lower()).strip()


def eval_questions():
    for split in ('test', 'dev'):
        path = ROOT / f'data/evaluation/mmlu/source/all/{split}-00000-of-00001.parquet'
        rows = pq.read_table(path).to_pylist()
        for index, row in enumerate(rows):
            yield f'mmlu-{split}-{index:05d}', row['question'], row['choices']
    path = ROOT / 'open-unlearning/saves/eval/qwen3_4b_base_bio_full_v1/LMEval_EVAL.json'
    rows = json.loads(path.read_text())['wmdp_bio']['wmdp_bio']
    for index, row in enumerate(rows):
        doc = row['doc']
        yield f'wmdp-bio-test-{index:05d}', doc['question'], doc['choices']


def main():
    if OUT.exists():
        raise RuntimeError('Output exists; preserve it and use a new processing version rather than overwriting.')
    OUT.mkdir(parents=True)
    questions = {}
    patterns = collections.defaultdict(list)
    for key, question, choices in eval_questions():
        text = lexical(question)
        words = text.split()
        questions[key] = {'text': text, 'choices': [lexical(c) for c in choices], 'words': len(words)}
        if len(words) >= 8:
            patterns[text].append((key, 'exact'))
        if len(words) >= 16:
            for offset in sorted(set((0, (len(words)-8)//2, len(words)-8))):
                patterns[' '.join(words[offset:offset+8])].append((key, 'anchor'))
    matcher = ahocorasick.Automaton()
    for text, values in patterns.items():
        matcher.add_word(' ' + text + ' ', values)
    matcher.make_automaton()
    rows, texts, hashes, dois = {}, {}, {}, {}
    parent = {}
    lsh = MinHashLSH(threshold=0.8, num_perm=64)
    raw_counts = collections.Counter()
    near_pairs = []
    overlaps = []
    candidate_counts = collections.Counter()
    max_candidate_count = 0

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for role, path, index, row, rev in raw_records():
        raw_counts[role] += 1
        sid = record_id(role, path, index, rev)
        text = row.get('text')
        meta = {'sample_id': sid, 'role': role, 'source_file': str(path.relative_to(ROOT)),
                'source_row_index': index, 'source_version': rev, 'cleaning_pipeline_version': VERSION,
                'language': None, 'language_note': 'Not automatically classified',
                'risk_domain': 'biosecurity' if role == 'forget' else None,
                'status': 'retained', 'exclusion_reason': None, 'split': None,
                'used_for_training': False}
        rows[sid] = meta
        parent[sid] = sid
        if not isinstance(text, str) or not text.strip():
            meta.update(status='excluded', exclusion_reason='empty_or_invalid_text')
            continue
        cleaned = text.replace('\r\n', '\n').replace('\r', '\n').strip()
        normalized = norm(cleaned)
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        meta['normalized_text_sha256'] = digest
        if digest in hashes:
            other = hashes[digest]
            meta.update(status='excluded', exclusion_reason='exact_duplicate', duplicate_of=other)
            if rows[other]['role'] != role:
                rows[other].update(status='excluded', exclusion_reason='cross_role_exact_conflict')
                meta['exclusion_reason'] = 'cross_role_exact_conflict'
            continue
        hashes[digest] = sid
        token_text = lexical(cleaned)
        hits = collections.defaultdict(lambda: {'exact': False, 'anchors': 0})
        for _, values in matcher.iter(' '+token_text+' '):
            for qid, kind in values:
                if kind == 'exact': hits[qid]['exact'] = True
                else: hits[qid]['anchors'] += 1
        confirmed = []
        for qid, hit in hits.items():
            question = questions[qid]
            if hit['exact']:
                # Only exclude substantial verbatim questions; short/generic prompts are candidates.
                if question['words'] >= 20 and len(question['text']) >= 120:
                    confirmed.append(qid)
                    overlaps.append({'sample_id': sid, 'evaluation_sample_id': qid, 'kind': 'long_verbatim_question'})
                else:
                    candidate_counts['short_exact_question_hits'] += 1
                    overlaps.append({'sample_id': sid, 'evaluation_sample_id': qid, 'kind': 'short_exact_candidate'})
            elif hit['anchors'] >= 2:
                candidate_counts['anchor_match_candidates'] += 1
                overlaps.append({'sample_id': sid, 'evaluation_sample_id': qid, 'kind': 'anchor_candidate'})
        if confirmed:
            meta.update(status='excluded', exclusion_reason='evaluation_long_verbatim_question')
            continue
        words = token_text.split()
        count = max(1, len(words)-4)
        positions = np.unique(np.linspace(0, count-1, min(128, count), dtype=int))
        shingles = [' '.join(words[int(i):int(i)+5]).encode() for i in positions]
        mh = MinHash(num_perm=64, seed=SEED)
        mh.update_batch(shingles)
        candidates = sorted(lsh.query(mh))
        max_candidate_count = max(max_candidate_count, len(candidates))
        for other in candidates:
            score = ratio(normalized, texts[other], score_cutoff=90)
            if score >= 90:
                union(sid, other)
                near_pairs.append({'sample_id': sid, 'other_sample_id': other, 'normalized_character_similarity': score/100,
                                   'action': 'group_for_split_only_keep_both'})
        texts[sid] = normalized
        lsh.insert(sid, mh)
        doi = doi_key(row.get('doi'))
        if doi:
            if doi in dois: union(sid, dois[doi])
            else: dois[doi] = sid
        if sum(raw_counts.values()) % 5000 == 0:
            print('Scanned',dict(raw_counts),'near_pairs',len(near_pairs),flush=True)

    groups = collections.defaultdict(list)
    for sid, meta in rows.items():
        if meta['status'] == 'retained': groups[find(sid)].append(sid)
    # One shared group assignment prevents overlap between the two corpora as well.
    keys = sorted(groups)
    random.Random(SEED).shuffle(keys)
    validation_keys = set(keys[:max(1, round(len(keys)*0.05))])
    for key, ids in groups.items():
        split = 'validation' if key in validation_keys else 'train'
        for sid in ids: rows[sid].update(split=split, article_group_id=key)
    paths = {}
    for role in ('forget', 'retain'):
        for split in ('train', 'validation'):
            path = OUT / role / f'{split}.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            paths[role, split] = path
    handles = {key: path.open('w', encoding='utf-8') for key, path in paths.items()}
    counts = collections.Counter()
    try:
        for role, path, index, row, rev in raw_records():
            sid = record_id(role, path, index, rev)
            meta = rows[sid]
            if meta['status'] != 'retained': continue
            split = meta['split']
            output = dict(meta)
            output.update(text=row['text'].replace('\r\n','\n').replace('\r','\n').strip(),
                          source_dataset='cais/wmdp-bio-forget-corpus' if role=='forget' else 'wmdp-s3-bio-retain',
                          included_in_training_split=split=='train')
            if role == 'forget':
                output.update(title=row.get('title'), abstract=row.get('abstract'), doi=row.get('doi'))
            handles[role, split].write(json.dumps(output, ensure_ascii=False, allow_nan=False)+'\n')
            counts[role, split] += 1
    finally:
        for handle in handles.values(): handle.close()
    with (OUT/'record_audit.jsonl').open('w') as f:
        for sid in sorted(rows): f.write(json.dumps(rows[sid],ensure_ascii=False)+'\n')
    dump(OUT/'near_duplicate_pairs.json', near_pairs)
    dump(OUT/'evaluation_overlap_candidates.json', overlaps)
    # Verify saved rows, IDs, text hashes, groups and counts independently of the writer.
    ids = set(); split_hashes = collections.defaultdict(set); split_groups = collections.defaultdict(set)
    output_files = []
    for (role, split), path in paths.items():
        actual = 0
        with path.open() as f:
            for line in f:
                row = json.loads(line); sid = row['sample_id']; assert sid not in ids; ids.add(sid)
                assert isinstance(row['text'],str) and row['text'].strip()
                assert hashlib.sha256(norm(row['text']).encode()).hexdigest()==row['normalized_text_sha256']
                split_hashes[split].add(row['normalized_text_sha256']);split_groups[split].add(row['article_group_id']);actual+=1
        assert actual==counts[role,split]
        h=hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        output_files.append({'path':str(path.relative_to(ROOT)),'role':role,'split':split,'records':actual,
                             'size_bytes':path.stat().st_size,'sha256':h.hexdigest()})
    assert not split_hashes['train'] & split_hashes['validation']
    assert not split_groups['train'] & split_groups['validation']
    exclusions={role:dict(collections.Counter(m['exclusion_reason'] for m in rows.values() if m['role']==role and m['status']=='excluded')) for role in ('forget','retain')}
    for role in ('forget','retain'):
        assert raw_counts[role]==sum(counts[role,s] for s in ('train','validation'))+sum(exclusions[role].values())
    report={'processing_version':VERSION,'completed_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
            'seed':SEED,'validation_group_fraction':0.05,'split_method':'Sort shared article groups, shuffle with Random(42), assign round(5% of groups) to validation. Record fractions can differ.',
            'raw_counts':dict(raw_counts),'excluded_counts':exclusions,'output_files':output_files,
            'deduplication':'NFKC + casefold + whitespace normalization for exact SHA256 comparison; original casing and text preserved in output. Same DOI and verified near duplicates share a split; near duplicates are not deleted.',
            'near_duplicate_screen':'128 evenly spaced five-word shingles; 64-permutation MinHash LSH threshold 0.8; verify candidates with full normalized-character similarity >=0.90. Heuristic, not exhaustive.',
            'verified_near_duplicate_pairs':len(near_pairs),'max_lsh_candidates_per_record':max_candidate_count,
            'evaluation_questions_checked':len(questions),'evaluation_check':'Aho-Corasick normalized question substring matching, plus three 8-word anchors for questions >=16 words. Exclude verbatim questions >=20 words and >=120 normalized chars; shorter exact/anchor matches are unreviewed candidates, not confirmed contamination.',
            'evaluation_confirmed_matches':sum(x['kind']=='long_verbatim_question' for x in overlaps),
            'evaluation_candidates':dict(candidate_counts),'candidate_review_complete':not any(candidate_counts.values()),
            'checks':{'ids_unique':True,'train_validation_exact_overlap':0,'train_validation_group_overlap':0,'counts_reconciled':True,'saved_text_hashes_verified':True},
            'training_started':False,'actual_language_distribution':None,'language_note':'No language classifier run; no inferred distribution fabricated.',
            'redistribution_authorization':None,'distribution_note':'Processed Bio data stays local under source restrictions; receiving-party authorization remains to be confirmed.',
            'dependencies':{n:importlib.metadata.version(n) for n in ['pyahocorasick','datasketch','rapidfuzz','pyarrow','numpy']},
            'limitations':['Approximate duplicate candidate generation is sampled and cannot prove absence of semantic duplicates.', 'Near evaluation matches require review; paraphrases and answer-only overlap may not be detected.', 'Training loader/tokenization compatibility will be verified before RMU short run.']}
    dump(ROOT/'logs/bio_data_preparation_report.json',report)
    dump(OUT/'processing_manifest.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__ == '__main__': main()
