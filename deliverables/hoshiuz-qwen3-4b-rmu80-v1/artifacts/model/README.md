
# RMU80 model artifact status

The actual selected checkpoint is at the project-local path open-unlearning/saves/unlearn/rmu_step_comparison_v1/h12_a2_s32_seed42_continuous120/checkpoint-80. It is a BF16 uint16 XOR+gzip lossless delta against the pinned local base, not LoRA and not a standalone full model. The three compressed payloads total 74,022,192 bytes and are present in this server's Git LFS object cache with matching SHA-256. The checked-out filenames are Git LFS pointers (133 bytes each), not payload files. Remote upload/retrieval is not verified. This delivery directory contains metadata only; no model bytes are copied.

Reconstruction uses scripts/rmu_adaptation_utils.py::load_delta from the project root after obtaining the exact base model and local LFS payloads. Existing checkpoint-80/rebuild_verification.json verifies compressed payload, base tensor and reconstructed tensor hashes. Delivery preparation reused this record and did not load/rebuild the model again. Four-group runtime model fingerprint: 06fc83a6bd5dec105e2de6fde7aa94ab75af0dee562b3a7b9de4a034c7503ea3.
