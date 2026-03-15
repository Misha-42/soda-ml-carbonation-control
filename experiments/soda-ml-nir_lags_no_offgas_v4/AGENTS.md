# AGENTS.md

## Purpose
Practical rules for Codex in this repository: keep a **minimal baseline ML** workflow for stage-1 NIR on soda process analysis.

## Working style (mandatory)
- Always show a **short plan first** before making changes.
- Keep solutions simple, readable, and easy to run locally/Colab.
- Change only what is required for the current task.
- After edits, briefly explain what each changed file does.

## Baseline-first architecture
- Prefer minimal pipeline: `data -> features -> baseline models -> metrics -> report`.
- Avoid premature architecture expansion.
- Do **not** add by default: MLflow, model registry, drift monitoring, complex config systems, large test suites, microservices.

## Time-series and leakage rules
- If time column exists, use **time-based split** only (no random split).
- Never use future information in features for past predictions.
- Fit preprocessing on train data, then apply to test.
- Keep target strictly separated from feature generation.

## Delivery format in every final response
At the end of each answer, include:
1) what I am learning as an ML developer,
2) the next best step,
3) what should **not** be overcomplicated yet.
