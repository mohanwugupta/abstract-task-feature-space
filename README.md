# Abstract task feature space

A reproducible human-norm feature-space pilot and two-task compositionality study using configurable hosted frontier-model APIs.

The pipeline provides isolated feature discovery, conservative candidate consolidation, strict binary applicability judgments, resumable execution, cumulative API budget controls, and auditable reports. The case studies cover creative writing and synthetic financial reconciliation.

## Complete shared vocabulary

The current applicability stage uses [concurrent immediate execution](IMMEDIATE_APPLICABILITY.md) at the user's request for faster completion, reusing Batch results under the same$100cap.

The expanded workflow uses every accepted human and task feature across all **75 items**, with conservative semantic consolidation, complete provenance, and a shared binary applicability matrix. The original 150-feature runs remain preserved baselines.

```bash
.venv/bin/python -m feature_space_pilot.full.run --dry-run
.venv/bin/python -m feature_space_pilot.full.run --prepare
```

Preparation is offline and exports all available features, figures, actual task propositions, and every blocked cell with its cost. Paid stages require completed source generations, measured costs and fresh whole-stage preflight. The unchanged **$100 cumulative cap** covers previous spending and outstanding reservations. First fund every primary cell, then preflight all known invalid retries; no arbitrary feature subset is substituted.

The current overnight amendment freezes all valid generations already available, records missing samples, and uses one shared $100 limit without separate stage partitions. Late original outputs remain preserved for a subsequent vocabulary version.

See [the full workflow and execution guide](FULL_FEATURE_SPACE_PILOT.md), [starting-state audit](FULL_PILOT_AUDIT.md), and [incremental TDD record](scratchpad_feature_space_pilot.md). The commands below describe the preserved original pilot.

## Run the offline demonstration

Requires Python 3.11. No API credentials or GPU are needed.

```bash
python -m pip install -r requirements-feature-space-viz.txt
python -m feature_space_pilot.scientific --config configs/feature_space_pilot.json --prepare
python -m feature_space_pilot.run --config configs/feature_space_pilot.json --stage all --demo
python -m feature_space_pilot.report --config configs/feature_space_pilot.json
python -m pytest -q tests/feature_space_pilot
```

The full synthetic demonstration runs 675 generation requests (three repetitions per target and prompt family) and produces a complete binary matrix, composition audits, cost accounting, and a hashed manifest under `artifacts/feature_space_pilot/`. Synthetic results validate the engineering workflow; they are not human benchmark findings. The three-repetition default follows the [user sampling amendment](data/feature_space_pilot/sampling_amendment.json), replacing the original PRD’s ten-repetition plan.

## Offline visualization explorer

Install the pinned explorer dependencies, then point it at **one existing frozen run**:

```bash
python -m pip install -r requirements-feature-space-viz.txt
streamlit run feature_space_pilot/viz/app.py -- --run-dir artifacts/feature_space_pilot/YOUR_RUN_ID
```

The default Task Atlas compares creative writing and reconciliation with full proposition text. Other pages cover all six compositions, a four-state matrix, pooled feature discovery, gated human/embedding comparisons, and exact provenance/evidence inspection. Filters on discovery origin never redefine applicability. No API key, network access, model calls, or study execution is needed after dependencies are installed.

Use **Export publication figures** in the app to create the gallery and download its ZIP bundle. For a gallery and paper/slide figures without starting a server:

```bash
python -m feature_space_pilot.viz.cli --run-dir artifacts/feature_space_pilot/YOUR_RUN_ID --inventory
python -m feature_space_pilot.viz.cli --run-dir artifacts/feature_space_pilot/YOUR_RUN_ID
```

Open [the example offline gallery](artifacts/feature_space_visualizations/demo_all_cebc8c4d0142b011/v1/index.html) after rendering that example run, or `artifacts/feature_space_visualizations/YOUR_RUN_ID/v1/index.html` for your selected run. Generated PNG (220 DPI), SVG and PDF files, tidy data, a feature evidence inspector and `figure_manifest.json` stay outside the read-only source run. Synthetic filenames/banner distinguish engineering fixtures from empirical results. Human/embedding panels require verified aligned inputs; unavailable evidence is displayed explicitly.

See [visualization usage and interpretation](VISUALIZATION_EXPLORER.md) and the [visualization TDD/QA record](scratchpad_feature_space_visualizations.md).

## Human CSV inputs

The default configuration now reads both supplied human CSVs: **60 stimuli**, cross-checked against the **24-item abstract category summary**. Original reference labels, German source text, translation status and existing splits are preserved. Model-facing Vinson words use the published noun/verb syntax markers; abstract targets use English, as requested. The abstract summary supplies rounded category proportions rather than feature strings.

```bash
python -m feature_space_pilot.human_inputs --config configs/feature_space_pilot.json --output-dir artifacts/human_input_audit/v1
python -m feature_space_pilot.run --config configs/feature_space_pilot.json --stage human --dry-run
```

The audit is offline; the human stage plans **540 generation jobs**. Scientific generation preflight and the combined $100 budget now pass. Pending abstract translation/coding reviews still gate quantitative cross-language comparisons. See [scientific checks, budget, and execution commands](PREFLIGHT_WORKFLOW.md) and [CSV mappings](HUMAN_INPUTS.md).

## Documentation

- [Automated semantic matching of existing generations to human labels](AUTOMATED_SEMANTIC_MATCHING.md)
- [Exact-text, word-overlap and OpenAI embedding baselines versus the semantic judge](MATCHING_BASELINES.md)
- [Setup, hosted API configuration, source gates, and live-run instructions](FEATURE_SPACE_PILOT.md)
- [Artifact and data dictionary](DATA_DICTIONARY.md)
- [Product requirements](PRD_Feature_Space_Human_Norm_Pilot.md)
- [Incremental implementation and test record](scratchpad_feature_space_pilot.md)
- [Human reference source registry](data/feature_space_pilot/source_registry.json)

For live runs, put your key in a project-root `.env` file:

```dotenv
OPENAI_API_KEY=your-openai-api-key-here
```

The live runner loads it automatically. An existing exported `OPENAI_API_KEY` takes precedence. `.env` is ignored by Git; `.env.example` provides the blank template. Offline demos and dry runs do not read credentials.

Future generator and judge runs default to user-selected **GPT-6 Luna (`gpt-6-luna`), low reasoning, using OpenAI Batch**. The current Mini run preserves completed results and continues unfinished judgments through Batch under its original model and frozen inputs. Model IDs remain configurable; unsupported seed/temperature parameters are omitted. Paid execution requires verified pricing/capabilities, a fresh dry-run hash and explicit `--execute-live`. Three repetitions and the shared **$100** cumulative cap remain. The original runner retains its 150-feature baseline; the full workflow above has no feature cap. The new Luna Batch human+task completion ceiling is **$21.56625**, plus existing liabilities, with 16,000-input/700-generation-output/400-judge-output envelopes and two attempts.

All 60 supplied human stimuli and 15 task targets are integrated. Source/workbook/P0 verification remains enforced; English abstract generation is exploratory and pending reviews suppress affected comparisons. Mini diagnostics passed previously; Luna and Batch have offline mocked validation, with no paid diagnostic performed during this change.

Monitor the ongoing run without restarting it:

```bash
.venv/bin/python -m feature_space_pilot.status --watch
```

See [Batch and progress instructions](BATCH_AND_PROGRESS.md) for asynchronous submission/collection, and [the preflight workflow](PREFLIGHT_WORKFLOW.md) for source preparation, fresh stage hashes and explicit execution. New runner processes print progress to stderr; `--quiet` hides it. Visualization never makes API calls.

## Validation and reference data

This is a standalone pilot project. Execution uses hosted APIs, and the offline workflow needs only Python, NumPy, SciPy, and pytest.

The pilot's offline tests cover schemas, source gates, prompts, independent jobs, budget controls, binary judgments, metrics, compositionality, checkpoint recovery, and full CLI runs. CI runs without credentials.

Generated run directories are excluded from Git. Reproduce the sample report with the offline command above. Pilot fixtures and references are committed directly with their byte hashes preserved. Downloaded human norm files retain their source-specific terms recorded in the source registry; this project does not grant additional rights to those data.


For the current human pilot, [the visualization guide](VISUALIZATION_EXPLORER.md#human-only-empirical-pilot-analysis) covers verified analysis snapshots, the human-target matrix, judgment coverage, full data exports and the local explorer. These analyses do not make paid model calls and retain every unresolved cell explicitly.
