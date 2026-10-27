# Abstract task feature space

A reproducible human-norm feature-space pilot and two-task compositionality study using configurable hosted frontier-model APIs.

The pipeline provides isolated feature discovery, conservative candidate consolidation, strict binary applicability judgments, resumable execution, cumulative API budget controls, and auditable reports. The case studies cover creative writing and synthetic financial reconciliation.

## Run the offline demonstration

Requires Python 3.11. No API credentials or GPU are needed.

```bash
python -m pip install -r requirements-feature-space-pilot.txt
python -m feature_space_pilot.run --config configs/feature_space_pilot.json --stage all --demo
python -m feature_space_pilot.report --config configs/feature_space_pilot.json
python -m pytest -q tests/feature_space_pilot
```

The full synthetic demonstration runs 2,250 generation requests and produces a complete binary matrix, composition audits, cost accounting, and a hashed manifest under `artifacts/feature_space_pilot/`. Synthetic results validate the engineering workflow; they are not human benchmark findings.

## Documentation

- [Setup, hosted API configuration, source gates, and live-run instructions](FEATURE_SPACE_PILOT.md)
- [Artifact and data dictionary](DATA_DICTIONARY.md)
- [Product requirements](PRD_Feature_Space_Human_Norm_Pilot.md)
- [Incremental implementation and test record](scratchpad_feature_space_pilot.md)
- [Human reference source registry](data/feature_space_pilot/source_registry.json)

Generator and judge model IDs remain configurable. Paid execution requires verified model pricing/capabilities, a successful dry run, its matching preflight hash, and explicit `--execute-live`. The configured cumulative API cap is at most $500.

The human benchmark remains blocked pending acquisition/verification of the abstract-concept supplement and the exact Vinson participant protocol. Thirty-six Vinson labels and all fifteen task targets are frozen; missing human stimuli are not fabricated. No paid model calls were made during implementation.

## Validation and reference data

This is a standalone pilot project. Execution uses hosted APIs, and the offline workflow needs only Python, NumPy, SciPy, and pytest.

The pilot's **42 offline tests** cover schemas, source gates, prompts, independent jobs, budget controls, binary judgments, metrics, compositionality, checkpoint recovery, and full CLI runs. CI runs without credentials.

Generated run directories are excluded from Git. Reproduce the sample report with the offline command above. Pilot fixtures and references are committed directly with their byte hashes preserved. Downloaded human norm files retain their source-specific terms recorded in the source registry; this project does not grant additional rights to those data.
