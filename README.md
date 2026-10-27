# Abstract task feature space

A reproducible human-norm feature-space pilot and two-task compositionality study using configurable hosted frontier-model APIs.

The pipeline provides isolated feature discovery, conservative candidate consolidation, strict binary applicability judgments, resumable execution, cumulative API budget controls, and auditable reports. The case studies cover creative writing and synthetic financial reconciliation.

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

Generator and judge now default to user-selected **GPT-5 mini (`gpt-5-mini-2025-08-07`) with low reasoning**; unsupported seed/temperature parameters are omitted. Model IDs remain configurable. Paid execution requires verified model pricing/capabilities, a successful dry run, its matching preflight hash, and explicit `--execute-live`. The current configured cumulative API cap is **$100** (hard maximum $500). The label-blind 150-feature bank and 7,500-input/700-generation-output/400-judge-output envelopes give a combined human+task ceiling of **$64.60875**, plus existing liabilities. Stage allocations sum to $100. Visualization never makes API calls.

All 60 supplied human stimuli and 15 task targets are integrated. Raw sources, workbook crosswalk and separate P0 protocols are verified as documented; English abstract generation is explicitly exploratory while pending reviews suppress translated human comparisons. A one-call GPT-5-mini diagnostic passed; total recorded diagnostic spending is $0.003158. No full live pilot has run.

```bash
python -m feature_space_pilot.preflight --config configs/feature_space_pilot.json --campaign
```

Follow [the preflight workflow](PREFLIGHT_WORKFLOW.md) for preparation, fresh stage hashes and explicit execution. The older mini snapshot is marked deprecated by the provider; its account access passed the isolated diagnostic, and the code never substitutes another model silently.

## Validation and reference data

This is a standalone pilot project. Execution uses hosted APIs, and the offline workflow needs only Python, NumPy, SciPy, and pytest.

The pilot's offline tests cover schemas, source gates, prompts, independent jobs, budget controls, binary judgments, metrics, compositionality, checkpoint recovery, and full CLI runs. CI runs without credentials.

Generated run directories are excluded from Git. Reproduce the sample report with the offline command above. Pilot fixtures and references are committed directly with their byte hashes preserved. Downloaded human norm files retain their source-specific terms recorded in the source registry; this project does not grant additional rights to those data.
