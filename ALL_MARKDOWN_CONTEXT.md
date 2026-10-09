# All Markdown files — Antigravity context references

Use the **4-file starting set** first; the master prompt directs the agent to consult the rest. `@` references may require editor support and extracted files.

## Minimal starting context
```text
@AGENTS.md
@ANTIGRAVITY_START_PROMPT.md
@docs/00_REPOSITORY_AUDIT.md
@docs/01_PRD.md
```

## Full context list
```text
@AGENTS.md
@ANTIGRAVITY_CONTINUE_PROMPTS.md
@ANTIGRAVITY_START_PROMPT.md
@README_FIRST.md
@docs/00_REPOSITORY_AUDIT.md
@docs/01_PRD.md
@docs/02_REQUIREMENTS_ACCEPTANCE.md
@docs/03_UX_UI.md
@docs/04_ARCHITECTURE.md
@docs/05_QUERY_GROUNDING.md
@docs/06_EVENT_INDEX_TIMESTAMPS_CLIPS.md
@docs/07_EVIDENCELOOP_EXPERIMENT.md
@docs/08_TRACKING_REID.md
@docs/09_CLARIFY_MEMORY.md
@docs/10_DATA_MODEL.md
@docs/11_API_CONTRACTS.md
@docs/12_TECHNICAL_DECISIONS.md
@docs/13_IMPLEMENTATION_PLAN.md
@docs/14_ENGINEERING_TICKETS.md
@docs/15_TESTING_STRATEGY.md
@docs/16_BENCHMARK_ABLATION.md
@docs/17_SECURITY_PRIVACY.md
@docs/18_DEPLOYMENT_RUNBOOK.md
@docs/19_HACKATHON_JURY_SUBMISSION.md
@docs/20_PERFORMANCE_AND_RESOURCES.md
@docs/21_RISK_REGISTER.md
@docs/22_RESEARCH_POSITIONING.md
@docs/23_ALERTS_PRIVACY_LIVE.md
@docs/24_EXPERIMENT_LOG_TEMPLATE.md
@docs/25_PS05_STATUS_MATRIX.md
@docs/26_RELEASE_GATE.md
@templates/ANNOTATION_GUIDELINES.md
@templates/JURY_90_SECOND_EXPLANATION.md
@templates/RUN_REPORT_TEMPLATE.md
```

Do not feed every file simultaneously if the editor truncates context. Start with the minimal set, then have the agent read the docs by milestone.
