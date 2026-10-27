# Feature-space research explorer

The explorer reads exactly one existing frozen run. It validates its manifest and joins stable item, feature, job, prompt-family and replicate IDs. It never imports the API execution path, reads `.env`, generates features, judges new cells, changes references, or reseals source artifacts. It works without internet after dependencies are installed.

## Installation and launch

Use Python 3.11 from the repository root:

```bash
python -m pip install -r requirements-feature-space-viz.txt
streamlit run feature_space_pilot/viz/app.py -- --run-dir artifacts/feature_space_pilot/YOUR_RUN_ID
```

Open the local URL printed by Streamlit. The run path can also be changed in the sidebar. For scripted headless tests, `FEATURE_SPACE_RUN_DIR` provides the initial selection. Streamlit analytics are disabled in `.streamlit/config.toml`; the app loads no external assets or models.

If you have no run, create one explicitly synthetic engineering fixture using the ordinary offline demo first:

```bash
python -m feature_space_pilot.run --config configs/feature_space_pilot.json --stage all --demo
```

The current three-repetition design produces 675 synthetic generation records, not the original PRD snapshot's 2,250. These are test fixtures, not empirical LLM or human results. Opening the explorer never creates a run for you.

## Research pages

1. **Task Atlas** opens by default. Select domain, task type, concrete instance or one of three situations. Mirrored writing/reconciliation panes and a center shared column show full proposition text. Grouping uses independent binary judgments. Standalone situations remain an independent axis. An evidence button selects that exact feature in the inspector. Full target descriptions include the complete synthetic invoice input.
2. **Composition** shows all six frozen task/situation comparisons and independently checks their algebra against original generations/candidates/judgments. Component-applicable, incremental-applicable and interaction-candidate counts are separate, overlapping categories; bars are never stacked. Named manual reviews distinguish unreviewed, rejected and confirmed candidates. Full T/S/TS evidence and balanced call/token budgets are downloadable. `S_both` is one combined situation, not a factorial estimate.
3. **Matrix** shows explicit `1`, `0`, `?` and `N/A`. Filtered vocabulary denominators remain visible. Optional feature clustering uses binary Jaccard distance and average linkage, on complete selected cells only; incomplete cells disable clustering rather than become zeros. All labels, stable IDs and evidence are inspectable.
4. **Feature Discovery** switches among raw normalized phrases, selected canonical vocabulary, and independently judged-applicable selected features. Curves pool actual generation records by replicate, frozen item order and job ID. They never sum overlapping per-item unique counts. Facets retain target levels and prompt-specific caps/call/token denominators. Excluded proposals remain unjudged rather than negative. Contribution tables preserve unique/shared, selected/excluded and cross-item applicability distinctions.
5. **Human Baseline** displays source-specific real-data recovery only when source provenance, hashes, language alignment and appropriate source-level records are verified. Generated category profiles require independently reviewed source-compatible coding. Participant split-half estimates require participant records. The demo shows honest unavailable cards, not synthetic numbers presented as human findings.
6. **Geometry** supports exploratory generated-feature RDMs plus verified source-specific human and pinned raw embedding RDMs. Binary representations use Jaccard (0–1); dense vectors use cosine (0–2). Summaries cannot reconstruct missing vectors. Raw vector files require pinned model/revision/documentation, file hashes and exact item/text alignment. Correlations compare upper triangles in the same item order, with item-bootstrap intervals excluding copied self-pairs.
7. **Feature Inspector** shows the canonical proposition/ID, every original phrase/item/prompt/replicate, literal generation request, 1/0/?/N/A evidence, merge decisions and manual review provenance.
8. **Overview** shows run identity, completion, model revisions, inventory/source hashes and target scope. It is not the default page: inspecting actual features is the primary task.

Sidebar filters apply to **discovery origin**: prompt family, source items, source level, minimum independent discovery count, text/ID and optional unreviewed candidates. They do not rejudge applicability. The composition counts and full-vocabulary discovery denominators remain tied to their frozen primary comparison; their pages disclose this rather than silently changing the estimand.

Every chart has underlying rows and CSV/JSON download controls. Plot tooltips carry stable IDs and full proposition text. The shared inspector provides an explicit selection path where heatmap click events are not supported by the UI framework.

## Static gallery and publication figures

Open **Publication figures and offline gallery** on any app page and select **Export publication figures**. This renders the named full comparison outside the source run and offers a downloadable ZIP containing the offline gallery, figures, evidence and data. Sidebar vocabulary filters do not alter this frozen publication set.

```bash
python -m feature_space_pilot.viz.cli --run-dir artifacts/feature_space_pilot/YOUR_RUN_ID --inventory
python -m feature_space_pilot.viz.cli --run-dir artifacts/feature_space_pilot/YOUR_RUN_ID --dry-run
python -m feature_space_pilot.viz.cli --run-dir artifacts/feature_space_pilot/YOUR_RUN_ID
```

Default output:

```text
artifacts/feature_space_visualizations/YOUR_RUN_ID/v1/
    index.html
    feature_inspector.html
    figure_manifest.json
    figures_readme.md
    synthetic_figure_01_task_atlas.{png,svg,pdf}
    synthetic_figure_01_task_atlas_deadline.{png,svg,pdf}
    synthetic_figure_02_compositionality.{png,svg,pdf}
    synthetic_figure_03_binary_heatmap.{png,svg,pdf}
    synthetic_figure_04_discovery.{png,svg,pdf}
    synthetic_figure_05_human_baseline.html
    synthetic_figure_06_geometry.html
    data_exports/
```

The `synthetic_` prefix and **SYNTHETIC DEMONSTRATION — NOT EMPIRICAL RESULTS** banner appear for fixture runs. Verified live scientific panels are rendered only when their input gates pass; otherwise a source-specific unavailable or integrity-blocked page is written. Primary figures and the gallery remain useful in their absence. A partial matrix uses `?`/`N/A`, blocks inappropriate composition/geometry analyses, and does not crash the entire exporter.

The HTML gallery is meaningful offline without Streamlit. It uses local images/styles only, links SVG/PDF/PNG and data files, and includes a companion proposition table that links composition examples to exact target descriptions, origins and independent evidence. No external JavaScript, analytics, fonts or CDN resources are needed.

Options:

```bash
python -m feature_space_pilot.viz.cli --run-dir RUN --output-dir SEPARATE_OUTPUT --max-features 24 --dpi 220 --seed 42 --cluster
```

- PNG defaults to 220 DPI (minimum accepted 200); SVG/PDF remain vector.
- Atlas selection is deterministic round-robin shared/poem-only/reconciliation-only, then canonical ID. If fewer than 12 applicable propositions exist, no synthetic padding is added. Full rows remain exported.
- Heatmap defaults to columns applicable to at least one displayed task, conceptual row order and canonical-text columns, with deterministic optional Jaccard/average clustering. Figure metadata discloses selection and full bank counts.
- Plot fonts are pinned to DejaVu Sans. Data/ordering/captions/config/seed/source/evaluation fingerprints are deterministic; renderer file bytes need not be identical across platforms.
- Source runs, norms and configs are read-only. The exporter refuses output paths inside or enclosing the selected source run.

## Scientific limits and verification

Discovery `G(X)` and independent applicability `A(X,F)` remain distinct. The composition checker verifies `gain = directly-discovered-and-TS-applicable minus component-discovered-and-TS-applicable`, equivalent to the existing union-gain audit. Operational candidates also require T=0, S=0, TS=1. Confirmation requires a named independent reviewer and note; rejected/unreviewed candidates cannot appear confirmed.

A category-count-only human reference cannot produce an item-feature RDM. Unlisted human production is never automatically incorrect. Correlation or clustering does not demonstrate internal representations, preferences, neural encoding or causality. Matched call counts do not imply matched token consumption. All missing evidence is displayed as unavailable rather than fabricated.

Run all tests with the visualization environment installed:

```bash
python -m pytest -q tests
```

Tests include typed adapters, four states, toy set identities, independent atlas truth, pooled discovery, external alignment gates, exact Jaccard/cosine distances, partial-run exports, deterministic/source-preserving outputs, rendered layout bounds and headless navigation/feature-selection interactions.

The append-only QA/TDD record is [scratchpad_feature_space_visualizations.md](scratchpad_feature_space_visualizations.md). It includes commands, source IDs, failed attempts and manual inspection notes for the primary figures. Native browser integration was unavailable in the implementation session; Streamlit's official headless AppTest exercised the app, and exported PNG/PDF figures were visually inspected.
