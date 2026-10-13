# WMDP corpus access and format review

Reviewed 2026-10-13 (Asia/Shanghai). No corpus transformation or training performed.

## Official sources

- https://github.com/centerforaisafety/wmdp : the official README lists Bio forget separately at https://huggingface.co/datasets/cais/wmdp-bio-forget-corpus .
- https://huggingface.co/datasets/cais/wmdp-bio-forget-corpus : login and access application required. The page states that requesting access entails agreement not to distribute the dataset beyond the collaborator list without explicit permission. Approval status for the user's account has not been checked. Do not bypass the gate or publish restricted contents to GitHub.
- https://huggingface.co/datasets/cais/wmdp-cyber-forget-corpus : gated files require acceptance of conditions/contact information sharing; visible card is empty. This does not establish redistribution permission for the downloaded S3 copy.

The S3 archive contains Bio retain, Cyber retain and Cyber forget, but no Bio forget. This is confirmed by the archive directory, not a failed extraction. No general corpus license was established from the reviewed pages. The framework's MIT license cannot be assumed to cover these external corpora. Record corpus license/redistribution authorization as unresolved; obtain authorization applicable to collaborators/platform delivery before redistributing.

## Local format

All three files consist of JSON string values, not JSON objects. Bio retain: 60,887 records; Cyber forget: 1,000; Cyber retain: 4,473. All lines parse successfully; see wmdp_jsonl_inspection.json for decoded-string statistics and emptiness checks.

The current text dataset loader would read serialized strings including JSON quotation marks/escapes. Merely changing the loader to JSON while assuming a text field is insufficient. A later processing step should json.loads each line and write objects containing text plus stable IDs and provenance, preserving original files and recording the transformation. No loader/configuration change has been made yet.

## Next step

The user should apply for access to the official Bio forget repository and confirm data-use/redistribution terms. Do not substitute WMDP test questions or silently switch to Cyber. Independent work such as MMLU protocol/baseline can proceed while approval is pending if authorized. Large raw files have not been uploaded; Git LFS/storage planning remains pending, and storage setup does not grant redistribution permission.
