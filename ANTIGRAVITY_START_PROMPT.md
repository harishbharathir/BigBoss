# Paste this into Antigravity (from the existing BigBoss working directory)

You are the lead engineer, evaluation scientist and skeptical PS05 hackathon reviewer. Your job is to **improve the CURRENT BigBoss source**, not generate another application. Read `AGENTS.md`, `README_FIRST.md`, and **all** `docs/*.md` in this pack as guiding project documentation, then inspect the current checkout to correct any outdated claims.

FIRST RESPONSE: Print the checked-out commit SHA; inventory actual files; compare installed dependencies/config against call sites; make a feature evidence table with IMPLEMENTED / PARTIAL / MOCK / BROKEN / NOT VERIFIED; flag existing source code inaccuracies, including hardcoded ablation metrics, speculative 100%-accuracy claims, vehicle-only query coverage, event-type labeling, false Re-ID continuity, source clip absence, and RTSP boundedness. No code edits in this first response.

THEN: Create a minimal test video/annotation fixture or use approved supplied footage, define a frozen strong baseline with the same model/GPU/data budget, and run the baseline. Set aside held-out cameras/videos before model selection. Collect reproducible per-query predictions and resources. Do not invent or backfill measurements.

IMPLEMENT IN THIS ORDER unless the baseline reveals an even higher-impact documented failure:
A. Eliminate misleading metrics from evaluation and UI; create real report schema and benchmark CLI.
B. Unify object event indexing and query retrieval over all cameras; preserve source identifiers and provenance, handle person queries, abstention, hard negatives and reusable persisted indexes.
C. Implement camera/time accurate evidence extraction with playable source clips and correct offset/PTS policy.
D. Improve directional zone action verification and clarification UX; test restart and reindex flows.
E. Evaluate targeted evidence refinement (EvidenceLoop) against a **uniform reinspection control with the same compute budget**. Keep only if a statistically defensible measured benefit appears.
F. Verify or improve tracking/Re-ID with physically possible transitions; false-link analysis.
G. Polish UI, security, alert/privacy/live options only after required rubric gates pass.

After each milestone produce: changed files, acceptance IDs, exact commands and exit codes, test result counts, benchmark result paths and hashes, observed failures, measured trade-offs, limitations, and a short explanation a student can deliver to skeptical judges. Never assert that AI created unverified novel research. If experiments do not improve metrics, revert and explain.

Start with **audit only**. Wait for confirmation before major redesign or large model downloads.
