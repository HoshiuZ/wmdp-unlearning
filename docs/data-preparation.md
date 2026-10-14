# Bio data preparation v1

Completed on 2026-10-13 (Asia/Shanghai). This prepares article records, not tokenized RMU batches. No training has started.

## Sources and output

Source identities, revisions, file sizes and checksums are archived in `logs/wmdp_download_manifest.json` and `logs/wmdp_bio_forget_download_manifest.json`. Original files were not modified. Official Bio forget Parquet contributes 24,453 records; S3 Bio retain contributes 60,887 JSON-string records. The latter is decoded with JSON parsing before training text is written.

Processed files are local under `data/processed/bio_v1/{forget,retain}/{train,validation}.jsonl`. They inherit source restrictions and are not published. The official Bio forget access terms prohibit distribution beyond the collaborator list without explicit permission. Platform recipients must be covered by authorization; retain-corpus redistribution permission has not been established. Public Git stores code and aggregate provenance/validation records.

| Role | Raw | Excluded | Train | Validation |
| --- | ---: | ---: | ---: | ---: |
| Forget | 24,453 | 16 | 23,220 | 1,217 |
| Retain | 60,887 | 370 | 57,488 | 3,029 |

Forget exclusions: 12 empty text records, 2 normalized exact duplicates, 2 cross-role exact-conflict records. Retain exclusions: 368 normalized exact duplicates and 2 cross-role exact-conflict records. Two identical texts appeared in both roles; both copies were excluded to avoid conflicting objectives. The 12 empty full-text records have title/abstract/DOI, but abstracts were not silently substituted.

## Cleaning and grouping

Output body text only changes CRLF/CR to LF and removes surrounding whitespace. Exact duplicate comparison separately uses NFKC, case folding and whitespace normalization before SHA256 hashing. Sample IDs encode role, fixed source version, original filename and zero-based row index; they are independent of cleaning and split results. Source fields and per-record exclusion decisions are preserved in local audit files. Language was not classified and is null; `used_for_training` is false because training has not started.

Near-duplicate screening uses 128 evenly spaced five-word shingles, 64-permutation MinHash with seed42, LSH threshold0.8, then full normalized-character similarity at least0.90. It found 326 verified pairs. These records are kept and grouped for splitting, not deleted. Same-DOI records are also grouped. This heuristic cannot prove absence of all semantic duplicates.

Groups across both roles are sorted and shuffled with `random.Random(42)`. Approximately 5% of groups go to validation; all members of a group share its split. This shared assignment prioritizes preventing article leakage across the two roles, so individual role record fractions need not equal exactly5%. No text chunking occurs before splitting.

## Evaluation overlap review

15,600 question records were checked: MMLU test14,042 and dev285 plus WMDP-Bio test1,273. Normalized long question substrings were searched with Aho-Corasick; questions of at least20 words and120 normalized characters would trigger exclusion. No such match was found. Three eight-word anchors were also used for candidate screening.

The initial screen yielded 78 candidate pairs covering74 corpus records. Distinct-anchor review rejected72 repeated-single-phrase candidates; the remaining six pairs were reviewed with local surrounding context. They show knowledge statements/source-paper text or generic question stems without corresponding complete test-item copies. All candidate records were retained. This does not mean that target knowledge is absent, or establish absence of all contamination; paraphrases, answer-only overlap and unknown sources remain limitations. Only metadata and aggregate decisions are public, not corpus excerpts.

`review_bio_overlap_candidates.py` reproduces the automated triage; the six context-review decisions and their reasons are preserved in the local reviewed JSON. Re-running triage would require reapplying/rechecking those decisions and must not silently replace the reviewed artifact.

## Evidence and validation

- `scripts/prepare_bio_data.py`: source parsing, audit, cleaning, grouping, split, output verification. Refuses to overwrite an existing processed directory; use a new processing version for changed rules.
- `scripts/review_bio_overlap_candidates.py`: automated overlap candidate triage.
- `logs/bio_data_preparation_report.json`: complete counts, output hashes, dependencies, checks and limitations.
- `logs/bio_overlap_candidate_review_summary.json`: reviewed candidate outcomes.
- `logs/environment-data-processing.txt` and processing console logs: dependency and execution records.
- Local `record_audit.jsonl`, `near_duplicate_pairs.json`, `evaluation_overlap_candidates.json`, `evaluation_overlap_candidate_review.json`, `processing_manifest.json`: per-record/group decisions and restricted metadata.

Saved files were parsed again; sample IDs are unique, text hashes match, input/output counts reconcile, and train/validation have zero normalized exact-text or article-group overlap. The split was independently reconstructed from the saved audit with seed42. This checks the declared grouping rule, not the completeness of near-duplicate detection.


当前实验仅使用纯RMU，配置及结果入口以根目录README为准。
