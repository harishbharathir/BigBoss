# BigBoss: Updated Code-Aware PS05 Documentation Pack
**Prepared:** 2026-10-09 | **Target:** https://github.com/harishbharathir/BigBoss | **Branch inspected:** public `main` | **Verification:** public-source inspection ONLY; no local clone, model execution, or measured benchmarks.

## Start here
This pack is a **change specification**, not evidence the changes have been implemented. Read `docs/00_REPOSITORY_AUDIT.md`, `docs/01_PRD.md`, `docs/02_REQUIREMENTS_ACCEPTANCE.md`, and `docs/13_IMPLEMENTATION_PLAN.md`. Give Antigravity `AGENTS.md` and `ANTIGRAVITY_START_PROMPT.md`. It must open the actual local Git checkout, record the HEAD commit SHA, resolve all discrepancies and run a smoke test **before editing code**.

## Key facts found on 2026-10-09
- Streamlit dashboard `app/ui/dashboard.py` already offers chatbot, multi-camera Re-ID, single-video search, standing alerts, privacy mode, and an ablation page. Do not recreate these from scratch.
- `app/retrieval/conversational_engine.py` uses hand-built query parsing and `tracks_by_camera`; `app/reid/cross_camera.py` primarily indexes vehicle tracks. Person-related query support therefore needs end-to-end verification; do **not** label it complete.
- `app/eval/ablation.py` returns **static numbers**, not executed experiments. The dashboard displays these numbers. No quantitative score from this module may be used in a competition claim.
- `app/reid/cross_camera.py` does proximity matching even though `config.yaml` specifies `tracking.tracker: bytetrack`. A configured option is not proof it is wired.
- `app/retrieval/visual_search.py` selects a capped set of frames; `search_frames` ranks SigLIP frame embeddings with smoothing.
- Conversation matches currently expose cropped still images; the query API does not identify playable source clips. Output original camera/video IDs, robust source timestamps, and seekable segments.
- `ClarifySession` does write a JSON file, but known terms and aliases are limited and require restart/overwrite/concurrency tests.
- `PrivacyFilter.redact_crop` blurs an assumed lower portion of the crop. It is not face/plate detection. Alert rules require enforcement of rule condition and deduplication.

## Truth labels
- **Observed in public source:** linked code exists and its flow is visible.
- **Unverified:** runtime behavior, installation, FPS, memory, accuracy, precision, datasets, persistence after power loss, live-stream reliability.
- **Proposed:** every new algorithm, index, test, experimental gain, UI flow, or performance target.

## Files and recommended order
1. `docs/00_REPOSITORY_AUDIT.md` — evidence-based current-state audit.
2. `docs/01_PRD.md`, `docs/02_REQUIREMENTS_ACCEPTANCE.md` — what to build and done definition.
3. `docs/03_UX_UI.md` ... `docs/12_TECHNICAL_DECISIONS.md` — engineering contracts.
4. `docs/13_IMPLEMENTATION_PLAN.md`, `docs/14_ENGINEERING_TICKETS.md` — execution.
5. `docs/15_TESTING_STRATEGY.md`, `docs/16_BENCHMARK_ABLATION.md`, `docs/24_EXPERIMENT_LOG_TEMPLATE.md` — proof.
6. `docs/17_SECURITY_PRIVACY.md` ... `docs/26_RELEASE_GATE.md` — delivery and jury defense.

**Absolute rule:** no fabricated test results, citations, metric tables, novelty claims, or "100% accurate" phrasing. Add genuine benchmark tables only after reproducible scripts have produced them.

## Setup assumptions
Windows 11, Python 3.11, NVIDIA CUDA optional, Streamlit, Ultralytics YOLO-World, Google SigLIP, FFmpeg; refer to the repo's own README for supported commands. The deployed CUDA/PyTorch versions, model licenses, and GPU configuration must be checked on the build machine.

## Public source anchors
- https://github.com/harishbharathir/BigBoss
- https://raw.githubusercontent.com/harishbharathir/BigBoss/main/app/ui/dashboard.py
- https://raw.githubusercontent.com/harishbharathir/BigBoss/main/app/retrieval/conversational_engine.py
- https://raw.githubusercontent.com/harishbharathir/BigBoss/main/app/reid/cross_camera.py
- https://raw.githubusercontent.com/harishbharathir/BigBoss/main/app/eval/ablation.py

**Source freeze:** Replace floating `main` references with checked-out commit SHA before implementation. These documents describe the code publicly accessible on the date above; if the local repo diverges, the checkout takes precedence and this audit must be amended.
