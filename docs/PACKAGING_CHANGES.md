# Stage 6 packaging change record

The four numerical/custom source modules are copied byte-for-byte, with SHA-256 recorded in provenance.json. The original Stage 5 archive is retained separately and was not modified.

Changes to the worker: repository-relative source/config paths; user-selected output directory; direct loading of the exact recorded official runtime instead of regenerating it using Python AST unparse; hash validation; explicit strict/portable environment checks; four-method smoke routing; run metadata. Training, optimizer, scheduler, checkpoint selection and restoration functions remain unchanged.

Changes to evaluation: repository-relative config/source paths; selected checkpoint index created from all completed independent runs; actual environment recording; strict/portable environment switch. Original final inference loop, data checks, metrics and restoration functions are retained. New index contents are write-once and all weights/logs are checked before final labels are loaded.

New orchestration runs every method/seed in a fresh subprocess and keeps generated output outside the repository. A new CPU analysis script audits the path-free public records, recomputes statistics, and regenerates clearly labelled figures. This script is separate from the private Stage 5 audit whose hashes target the unsanitized original archive.

Public metadata removes private path fields. It retains original numerical measurements and source-hash fields for provenance; those original fields must not be mistaken for hashes of sanitized metadata. PUBLIC_SHA256SUMS.json is the public integrity inventory.

The retrieved Stage 4 clean notebook contains 28 code cells with execution counts and outputs cleared. Its introductory historical execution statement is not a fresh execution record. It was inspected for cleaning/split methodology; it is not relabelled as a tested Stage 5 or public notebook.

Validation boundary: no model training, dataset download, checkpoint reload, dependency installation or GPU notebook execution occurred during this packaging pass. A fresh GPU smoke and complete notebook execution remain necessary before claiming the course's top-to-bottom reproducibility requirement is met. Original results must remain frozen regardless of the new run's outcomes.
