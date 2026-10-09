# Agent instructions — BigBoss PS05
You are working **inside the user's existing repository**; preserve working functionality. First run `git status`, `git rev-parse HEAD`, inventory the tree, inspect tests and README. Do not claim you inspected a file you could not open. Do not overwrite local changes or delete user data. Read `README_FIRST.md` and the entire requirements and acceptance matrix before changes.

## Hard constraints
1. Baseline + held-out evaluation **before** optimizing. No cherry-picking; no tuning on held-out videos.
2. Never render hardcoded "mAP", "grounding accuracy", "speedup", "verified", or accuracy percentages as measured. Replace current hard-coded ablation path or visibly label simulated fixture data; competition/demo mode must only show validated runs.
3. Do not convert raw SigLIP cosine similarity or heuristic boosted scores to "probability" or "accuracy". Show similarity/ranking score, not fake certainty.
4. Ground each positive answer in registered source video, camera_id, original timebase or PTS, event interval, and a working seekable clip. An unmatched query must abstain.
5. Vehicle-only tracking is not person retrieval. Verify person and carrying-bag tasks end-to-end or mark unsupported.
6. Claimed entrance/departure requires demonstrated direction of movement across an annotated zone boundary. Never infer it merely because a track starts/ends in the middle of video.
7. Config keys are not implementations. Audit wiring for `ByteTrack`, `idle_sample_fps`, `active_sample_fps`, `verification_enabled`, `verify_top_n`, and `result_threshold`.
8. No unbounded RTSP ingestion. Streaming paths require a bounded session, controlled stop, deadlines and backpressure.
9. Do not claim face/plate protection unless a detector identifies areas and tests verify coverage. Prevent unredacted evidence from bypassing redaction when enabled.
10. Treat uploaded video and queries as untrusted data; validate, sanitize, bound resources and keep secrets out of logs. Do not add cloud calls without explicit approval.
11. Keep Windows/Python 3.11 compatibility and low-VRAM fallback; no massive architecture rewrite unless justified by measurements.
12. Optional GPT4All must never invent timestamps/cameras. Data source remains deterministic retrieval; LLM only paraphrases verified structured facts with a validator/fallback.
13. Add tests for every bug and record terminal commands, exit codes, configuration, random seeds, datasets and hashes.
14. Speak to the user plainly: implementation evidence, not confidence theater. Mark every unexecuted step `NOT RUN`.
15. Ask permission before major dependency installs, downloading >1GB models, connecting external services, or replacing project data.

## Agent workflow
Audit -> baseline -> classify failures -> implement ONE scoped change -> unit/integration tests -> development-set tuning -> frozen held-out test -> ablation -> keep/revert -> report. Use `docs/13_IMPLEMENTATION_PLAN.md` and `docs/14_ENGINEERING_TICKETS.md`.

## Required response after each milestone
| Field | Report |
|---|---|
| Current commit / branch | SHA and branch |
| Files inspected / changed | Paths |
| Bug and technical hypothesis | Specific |
| Commands executed | Exact |
| Tests and benchmark | Actual results or NOT RUN |
| Regression / runtime cost | Actual or unknown |
| Remaining risks | Concrete |
| Next gate | Pass/fail |
