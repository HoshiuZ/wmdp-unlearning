"""Review candidate match structure without exposing corpus contents in public reports."""
import collections
import json
from pathlib import Path
from rapidfuzz.fuzz import partial_ratio
from prepare_bio_data import ROOT, OUT, eval_questions, lexical, dump

questions = {key: (lexical(q), [lexical(c) for c in choices]) for key, q, choices in eval_questions()}
candidates = json.loads((OUT/'evaluation_overlap_candidates.json').read_text())
needed = {x['sample_id'] for x in candidates}
texts = {}
for path in sorted(OUT.glob('*/*.jsonl')):
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if row['sample_id'] in needed:
                texts[row['sample_id']] = lexical(row['text'])
review = []
for candidate in candidates:
    row = dict(candidate)
    q, choices = questions[row['evaluation_sample_id']]
    text = texts[row['sample_id']]
    words = q.split()
    anchors = {' '.join(words[i:i+8]) for i in {0, max(0,(len(words)-8)//2),max(0,len(words)-8)}}
    distinct = sum(' '+a+' ' in ' '+text+' ' for a in anchors)
    score = partial_ratio(q, text)
    row.update(distinct_anchor_count=distinct, question_word_count=len(words),
               normalized_question_partial_similarity=score/100)
    # These are screening outcomes, not a claim of semantic contamination absence.
    if candidate['kind']=='anchor_candidate' and distinct < 2:
        row['review_outcome']='single_anchor_repeated_not_confirmed_near_question'
    elif candidate['kind']=='short_exact_candidate':
        row['review_outcome']='short_exact_question_requires_context_review'
    elif score >= 90:
        row['review_outcome']='high_similarity_requires_context_review'
    else:
        row['review_outcome']='multiple_shared_phrases_requires_context_review'
    review.append(row)
dump(OUT/'evaluation_overlap_candidate_review.json',review)
summary={'candidate_pairs':len(review),'corpus_records':len(needed),
         'outcomes':dict(collections.Counter(x['review_outcome'] for x in review)),
         'confirmed_contamination':None,'automatic_exclusions_from_this_review':0,
         'review_note':'Distinct anchors and full normalized-question partial similarity were checked. Context-dependent candidates remain unconfirmed; no automatic content-based deletion.',
         'local_detail_path':'data/processed/bio_v1/evaluation_overlap_candidate_review.json'}
dump(ROOT/'logs/bio_overlap_candidate_review_summary.json',summary)
print(json.dumps(summary,indent=2))
