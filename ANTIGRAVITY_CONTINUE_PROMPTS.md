# Antigravity stage prompts

## Following audit
Use `docs/00_REPOSITORY_AUDIT.md` as an initial hypothesis only. Show your own real code audit against this checkout and list mismatches with references; propose a minimal migration plan with no user-data loss. Do not modify code until you report audit findings.

## Baseline phase
Build and execute the benchmark described by `docs/16_BENCHMARK_ABLATION.md`. Do not generate an evaluation summary from literal constants. Obtain explicit approval before using any dataset requiring licensing or download.

## One engineering improvement
Take the highest-impact reproducible failure case. Open `docs/14_ENGINEERING_TICKETS.md`. Implement one acceptance-tested change behind a feature flag, create regression tests, and report exact commands, diffs and outcomes.

## Jury preparation
Use actual execution artifacts (run manifest, prediction CSV/JSONL, ground truth, metric calculation, ablation, clips). Prepare a 3-minute evidence-first demo and 12 skeptical Q&A cards. Any claim not backed by an artifact must be described as pending.

## Release gate
Check `docs/26_RELEASE_GATE.md` item by item. Label PASS/FAIL/NOT RUN with evidence paths. Do not treat a clean UI as proof.
