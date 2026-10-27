# PRD — Feature-Space Visualization and Intuition Explorer

**Status:** Implementation specification for coding agent  
**Project:** Abstract Task Feature Space — Human-Norm and Two-Task Pilot  
**Priority:** Build scientifically faithful, inspectable figures for understanding the generated feature space; do not run a new feature-generation study.  
**Repository basis:** User-provided `abstract-task-feature-space-main.zip` (standalone `feature_space_pilot` project, 2026-10-26 source snapshot).  
**Companion documents:** `README.md`, `FEATURE_SPACE_PILOT.md`, `DATA_DICTIONARY.md`, `PRD_Feature_Space_Human_Norm_Pilot.md`, `scratchpad_feature_space_pilot.md`.

## 1. Objective

Create an offline, reproducible **visual exploration environment** that lets researchers understand:

1. What actual binary features were generated and judged applicable to each item, with inspectable text and provenance.
2. What changes between **domain → task type → task instance** for creative writing and financial reconciliation.
3. What features come from a **task T**, a standalone **situation S**, and the **situated task TS**; what direct TS generation discovers beyond its components; and which feature combinations may represent genuine task–situation interactions.
4. How feature-generation prompt family (`P0`, `P1`, `P2`) and replicate affect diversity, redundancy, and coverage.
5. How LLM feature representations relate to **verified human feature norms** and, where available, embedding-space geometry—without treating human norms or embeddings as ground truth.

The principal output is **a researcher-facing local interactive explorer plus paper/slide-quality static figures**. Prioritize legibility and opportunities to form/disconfirm intuitions over maximizing the number of plots. Every plotted element should support drill-down to the *actual item text, feature proposition, and underlying judgment/provenance*. Use these charts to learn what the candidate representation captures, not to claim that a target model internally uses these features to make choices.

## 2. Grounding in the existing repository (do not reinvent the pipeline)

The uploaded source contains these relevant modules:

- `feature_space_pilot/report.py`: generates evaluation artifacts, including composition audit and feature-level records.
- `feature_space_pilot/composition.py`: `compare_composition(...)`, which explicitly separates discovery and applicability.
- `feature_space_pilot/metrics.py`: `rdm`, `geometry_comparison`, `discovery_curves`, and human-norm comparisons.
- `feature_space_pilot/bank.py` and `judge.py`: canonical feature bank and binary judgments.
- `feature_space_pilot/io.py`: canonical JSON and file hashes.
- `feature_space_pilot/registry.py` and `fixtures.py`: hierarchical item inventory and synthetic financial case fixture.

Existing run artifacts live in `artifacts/feature_space_pilot/<run_id>/` and include:

| Primary inputs | Role in visuals |
|---|---|
| `manifest.json`, `matrix_scope.json`, `stimuli.jsonl` | Run identity, completeness, item type/level/hierarchy, display names, scope |
| `feature_bank.csv`, `feature_candidates.csv` | Canonical feature IDs/text; raw source phrase and provenance |
| `generations.jsonl`, `generation_jobs.jsonl` | Per-item/prompt/replicate discovery; generation prompt and job metadata |
| `binary_judgments.csv`, `binary_matrix.csv`, `judgments.jsonl` | Resolved binary applicability and supporting primary-judge evidence |
| `task_level_feature_audit.csv` | Feature provenance by task and abstraction level, inherited/new flags |
| `compositionality_audit.json`, `interaction_review.csv`, `manual_relational_audits.json` | Six T/S/TS comparisons and independent review status |
| `discovery_curves.csv` | Prompt-specific cumulative feature discovery |
| `human_norm_comparison.json`, `embedding_rdm_comparison.json` | Verified human comparisons and optional embedding comparison, with availability flags |
| `cost_report.json`, `pilot_report.md` | Run completeness, costs and caveats |

**Current confirmed source state:** The repository ZIP does not include its generated `artifacts/` runs. A fresh offline demo was successfully generated from the provided source: **2,250 synthetic generation jobs, 1,200 resolved binary cells, a 75-item × 16-feature matrix, and six composition records**. These are **fixture outputs only**, not empirical human or LLM results. The README and scratchpad state that the abstract human supplement and exact Vinson participant protocol remain scientific-source blockers; model IDs/pricing have not been configured for live runs. Therefore the visualization tool must run gracefully against the demo and explicitly distinguish unavailable scientific comparisons.

**Do not** change generation prompts, reference data, source gates, candidate selection, prior judgments, or feature matrices to improve the look of figures. Keep the visualization layer downstream and read-only.

## 3. Audiences and use cases

A researcher should be able to answer, within a minute of opening the tool:

- “What does the model think characterizes writing a poem versus reconciling invoices?”
- “Which exact features first arise at the task-type, instance, or situated-instance level?”
- “Did the deadline modifier add a generic time feature or create a task-specific conflict?”
- “Was this apparent interaction only *newly generated*, or independently *judged to apply only to TS*?”
- “What happens if I select just features generated by P0 instead of P2?”
- “How stable is the feature inventory as more generations are sampled?”
- “Which comparisons with humans are currently supported by data, and which are not?”

A slide-deck reader should be able to understand each primary figure **without** using the interactive app.

## 4. Non-goals and interpretive safeguards

- No new paid API calls, preference elicitation, retraining, new human-norm acquisition, or new feature judgments.
- No fabricated scientific or human results; use the demo only with a prominent **SYNTHETIC DEMONSTRATION — NOT EMPIRICAL RESULTS** banner and filename prefix.
- Do **not** treat absent human production as a false semantic property.
- Do **not** infer that semantic features are causally used in LLM preferences; this pipeline identifies candidate features and relations.
- Do **not** describe unreviewed interaction candidates as confirmed, emergent, causal, or mechanistic.
- Do **not** interpret an unmentioned/not-generated feature as absent: discovery and judged applicability are distinct.
- Do **not** silently treat missing, ambiguous, out-of-scope, or failed judgments as `0`.
- Do **not** merge human datasets into a single “human RDM” when their stimuli/features/codebooks are not aligned.
- Do **not** infer that prompt P0/P1/P2 have exactly comparable human caps without checking run/source metadata.
- No UMAP/t-SNE as a primary scientific result; may be an explicitly exploratory appendix with fixed random seed.

## 5. Core research figures (prioritized)

### Figure 1 — Interactive two-task **feature atlas** (highest priority)

**Question:** What precisely characterizes our two very different real-world task families, and what is shared?

Layout: mirrored panes for **creative writing** and **financial record reconciliation**, with a shared center column for overlapping applicable features. Show the hierarchy as tabs or a stepper:

- Creative writing: `poem_domain` → `poem_type` → `poem_T` → `poem_S_deadline`, `poem_S_review`, `poem_S_both`.
- Financial reconciliation: `finance_domain` → `finance_type` → `finance_T` → `finance_S_deadline`, `finance_S_review`, `finance_S_both`.
- Situation controls: `S_deadline`, `S_review`, `S_both` independently.

For each selected pair, show actual **feature proposition text**, not only embeddings, abbreviated codes, or feature IDs. Classify displayed features into “applies to both,” “poem only,” “finance only,” and optionally “neither (candidate bank)” based on judged binary cells for the **same chosen item level and complete scope**. Allow filtering by original prompt family, raw/generated source, repeated discovery, feature ID, and free-text search; clearly distinguish *filter on discovery origin* from *filter on applicability*.

Clicking a feature reveals its full proposition; canonical feature ID; originally generated source item, prompt family and replicate; raw source wording; which item(s) have `1`/`0`/unresolved; judge evidence; and any merge/manual-audit provenance. Show **complete target text**, including the invoice fixture reference but not inventing actual customer data. Display an explicit label when rows belong to different scopes or when any selected pair lacks complete cells.

Default static export: side-by-side profiles for **concrete poem and reconciliation** and a second for **both situated under deadline**, with around 12–25 well-chosen, fully legible feature labels; deterministic selection by disclosed ranking (e.g., shared plus discriminative plus reviewed interaction candidates), not cherry-picked for a desired story. Include feature-selection rule in caption.

### Figure 2 — **Compositionality decomposition** for all six T + S cases (primary result)

**Question:** Is directly generating features for the composed item useful beyond separately generating T and S? Which apparent novelties survive applicability checking?

Display a 2 × 3 comparison: task family (poem/reconciliation) × situation (`deadline`, `review`, `both`). In each panel, show:

1. Features discovered from T and S independently under the repository's balanced-generation rule.
2. Features discovered directly from TS under matched **number of generation calls**.
3. Number of applicable TS features available through component discovery.
4. Incremental applicable TS features recovered only by direct discovery (`incremental_applicable`).
5. Candidate relational interactions: direct-only applicable TS features judged `0` on T and S, `1` on TS (`interaction_candidates`); distinguish **unreviewed**, **rejected**, and **manually confirmed** using independent `manual_relational_audits` and/or `interaction_review`.

Use a paired bar / dot plot for *component-applicable*, *incremental-applicable*, and *interaction candidates*, with actual counts and disclosure of balanced call budgets; **do not add overlapping categories into one bar**. Keep a separate evidence/status strip for confirmed interactions. A companion **feature provenance table** should show 3–8 illustrative actual candidate propositions per condition, e.g. a deadline-versus-accuracy constraint, but always loaded from artifacts rather than hard-coded as results.

Define the data semantics explicitly in captions:

- `G_T` / `G_S` / `G_TS`: **discovered** feature sets from separate generation jobs.
- `A_TS(f)`: independent binary applicability of feature *f* to the composed item.
- `B = {f ∈ G_T ∪ G_S : A_TS(f)=1}`: baseline **component-discovered, TS-applicable** features.
- `D = {f ∈ G_TS : A_TS(f)=1}`: directly discovered TS-applicable features.
- `Δ = D \ B`: incremental applicable direct discoveries (reported by the existing code as incremental gain, with equivalent set logic; verify against source).
- `I = {f ∈ Δ : A_T(f)=0, A_S(f)=0, A_TS(f)=1}`: *operational interaction candidates*, not causal emergence.

The code's `incremental_applicable` is `applicable(component ∪ direct) − applicable(component)`; it is algebraically equivalent to `D \ B` given a complete frozen bank. Test that equivalence. Do not call other newly discovered but inapplicable features “additional applicable features.”

Include a small tooltip or footnote that **call counts are balanced, but token budgets may differ**, and show gain per 1,000 direct-generation tokens when actual token values exist. For `S_both`, the composed situation is **one standalone item** with deadline + review, not a two-factor statistical interaction estimated from a factorial model; label it accurately.

Optional exploratory UpSet graphic of membership in binary `A_T`, `A_S`, `A_TS` for the **same global bank**; use only if it clarifies relationships, and put set-membership patterns next to exact feature text.

### Figure 3 — **Clustered binary item × feature heatmap** (primary exploratory map)

**Question:** Which task features co-occur, and which features discriminate different tasks or abstraction levels?

Rows: displayed task and situation items (15 task-stage targets) grouped by task family, then domain/type/instance/situated and standalone situations. Columns: canonical features, with full labels accessible on hover, stable colors, and optional clustering. Values: `1 = judged applicable`, `0 = judged inapplicable`, visually distinct `? = missing/unresolved` and `— = out of scope`.

Default heatmap should prioritize **features present in ≥1 displayed row** and readable labels; provide filters for prompt of origin, level, prevalence, applicable-to-both, and interaction candidates. Allow deterministic binary Jaccard clustering (state distance/linkage) or switch to fixed conceptual ordering. Add column-side metadata indicating where a feature was first discovered; row annotation for source level and case family. Include small summary bars for number of applicable features by item, noting selection/coverage limitations. Do not imply that clusters identify learned internal model representations.

If any required cells are unresolved, either render missing symbols and a completeness banner or disable analyses that require complete matrices; **never** use zero fill. A sorted reduced static heatmap, with all feature labels and legend, is required.

### Figure 4 — **Where the feature vocabulary comes from** (prompt and abstraction audit)

**Question:** Which prompts and abstraction levels are actually contributing useful, nonredundant features?

Two linked views:

- **Discovery/rarefaction:** cumulative unique normalized candidate features versus replicate/call count, lines by `P0`/`P1`/`P2`, faceted by human target kind and task item level. If using `valid_unique`, label it **independently judged-applicable selected features** and show when it is unavailable rather than calling all unseen proposals invalid. Plot source-wise call counts and token use where possible; clearly label normalization/consolidation policy and whether curves are item-level or aggregated. Aggregated curves must be computed by pooling raw generation records in an explicit deterministic order, not summing already-overlapping unique counts.
- **Contribution table/stacked bars:** features discovered only from a particular prompt or target level, versus discovered elsewhere, with an additional separate indicator of how many judged applicable on cross-item targets. Preserve whether a feature was selected into the capped bank; do not misrepresent excluded candidates as judged 0.

Include a toggle between **raw generated candidate phrases**, **canonical bank features**, and **judged-applicable features**, each with the appropriate denominator. Add selected example feature texts to aid interpretation.

### Figure 5 — **Human-norm correspondence** (scientific figure, gated)

**Question:** Are generated features human-recognizable, and do they recover human feature-organization patterns?

Show a source-specific view for abstract concepts, events/actions and concrete objects **only where verified real reference data permit**:

- Feature-list recovery for compatible languages and approved semantic-equivalence maps: per-item or per-family distribution of *relative recovery of listed human productions*. Never call missing human-listed features “false positives,” or generated-only features “incorrect.”
- Human-vs-generated category profile chart only when model features have **independently reviewed category codes** compatible with that human source. Show coding coverage numerically and make uncoded cases explicitly unavailable. Preserve multi-label coding if supported.
- Optional paired per-item display of **human-listed**, **shared with generated**, and **generated-only** feature propositions; include short examples and matching rule.
- Optional human split-half ceiling only if participant-level records are actually present; otherwise display *unavailable*.

**Do not show quantitative human panels using synthetic fake reference data as if scientific.** In demo mode, render a separate explicitly labeled synthetic engineering example or an “awaiting verified human sources” card. Never mix language-unmatched corpora into a single benchmark metric.

### Figure 6 — **Representational geometry comparison** (conditional supplementary)

**Question:** Do human norms, generated binary features, and optional embeddings organize aligned items similarly?

For each **properly aligned subset** with enough items, show side-by-side item-labeled or small-multiple RDMs: human feature RDM, generated feature RDM, and pinned embedding RDM if configured. Use **binary Jaccard** for human/generated multi-hot features and **cosine distance** for dense embeddings; explicitly disclose differences in metrics and use a common, declared item order across panels. Display Spearman upper-triangle correlations with item-bootstrap uncertainty where valid. Avoid interpreting RDM agreement as proof of behavioral or neural encoding. Keep RDM scales labeled (Jaccard 0–1; cosine 0–2 possible) rather than forcing one unlabeled color normalization; an optional shared percentile-based display must disclose transformation.

**Implementation caveat:** `embedding_rdm_comparison.json` currently has summary correlations and model info, not necessarily the raw embedding vectors. To render an embedding RDM, read the **verified pinned embedding input** specified by config and its digest; otherwise render a clear `unavailable` reason. Do not infer a matrix from summary coefficients. For category-count-only human sources, do not invent a binary human feature RDM.

## 6. Interactive explorer: interaction contract

Recommend **Streamlit + Plotly** for local browsing (or an equivalent lightweight offline stack with equal capabilities), plus deterministic static PNG/SVG/PDF export using Matplotlib or a pinned image renderer. The app must work on a laptop with **no API keys or internet connection** after dependencies are installed. No analytics, network requests, or accidental live pipeline execution.

Layout:

- Run selector: explicit path `--run-dir`; show `run_id`, `demo/live`, stage, status, resolved fraction, model revisions, candidate-bank hash, and any missing-source warnings.
- Left navigation: **Overview**, **Task Atlas**, **Composition**, **Matrix**, **Feature Discovery**, **Human Baseline**, **Geometry**, **Feature Inspector**.
- Shared filters: task family, abstraction level, situation, prompt family, feature type/source, min discovery count, feature text search, include/exclude unreviewed interaction candidates.
- Each plot has a `View underlying rows` control and exports a CSV or JSON with stable feature IDs. Clicking or selecting a feature should show provenance and evidence. No orphaned visual marks.
- Tooltips contain full readable feature descriptions; compact static figures use line wrapping rather than truncating to unreadable IDs.
- Counts, coverage, denominators, source availability, and synthetic/live labels are present on every view.
- Provide **one-click export** of a named set of static figures and an index HTML / Markdown page with figure captions, run hashes, notes, and limitations.

UI is a means of exploring the data, not a dashboard optimized for generic business KPIs. Default navigation should open the **two-task atlas or compositionality view**, not a wall of aggregate numbers.

## 7. Data adapter and semantic correctness

Create a **single read-only visualization data adapter** to prevent each plot from implementing its own definitions. Suggested modules:

```
feature_space_pilot/viz/
    __init__.py
    load.py              # manifest verification and typed data model
    derive.py            # reusable semantics, filters and groupings
    figures.py           # pure chart specs/builders
    exports.py           # static exports and caption manifests
    app.py               # Streamlit/Plotly explorer
    cli.py               # batch render / dry-run / inventory

tests/feature_space_pilot/test_viz_*.py
```

Required contracts:

1. Select exactly **one** run directory, verify its manifest with the repository's existing `verify_manifest(...)`, and report if files are incomplete or hashes mismatch. If an app maintains a cache, key it by run path, manifest hash, and evaluation-input fingerprint; do not load stale data after manual review.
2. Join strictly on `item_id`, `feature_id`, `prompt_family`, `replicate`, `job_id`, and `parent_id`/`situation_ids` as appropriate; fail on accidental many-to-many joins or duplicate binary cells.
3. Use frozen canonical feature text for labels and preserve source phrases separately. Semantic-merging status must be visible; a feature not selected into the bank is **not** assumed inapplicable.
4. Preserve four states in visualization model: applicable (`1`), judged inapplicable (`0`), unresolved/missing (`?`), and out-of-scope (`N/A`). The final `binary_matrix.csv` is emitted only for complete declared scope; infer other states from judgment attempts and `matrix_scope.json`, not from nonexistent zero-filled cells.
5. Distinguish **discovered by a prompt** (generation provenance) from **applies to an item** (judge result). Allow filtering by either without changing underlying truth labels.
6. Use the **existing** composition audit rather than reinterpreting results; independently recompute critical identities as integrity checks and raise a flagged failure if they diverge. Do not blindly trust a previously generated summary if underlying bank/judgment inputs changed.
7. For figure analyses comparing prompts, hold denominator and item selection constant; when prompt conditions have unequal replicate/cap/token counts, plot or annotate those differences.
8. Construct external human/embedding panels only with verifiable source fields and item alignment. Treat `available=false` and missing review coverage as missing evidence, not negative results.
9. Format all outputs with deterministic sorting, pinned analysis configuration, seed, and run provenance. Figures should never silently change with file discovery order.
10. **Do not mutate** the run directory or reseal its manifest merely by opening the explorer. Write derived outputs to `artifacts/feature_space_visualizations/<run_id>/<viz_version>/` or user-designated separate directory. Keep the source artifacts and human references read-only.

## 8. Visualization design standards

- High-resolution exports suitable for publication: PNG ≥200 dpi, plus vector SVG/PDF wherever supported. Aim for 16:9 overview slides and one-column/two-column print alternatives.
- Accessible, colorblind-safe, *semantically consistent* palette across plots: e.g. one color per task family, a neutral shared-feature color, a distinct **candidate interaction** outline. Do not communicate 0/1 solely through hue; use symbols or explicit labels.
- Clear titles phrased as research questions (“What changes when we add a deadline?”); informative subtitles describing scope.
- Show natural-language feature text where interpretation matters. Avoid top-200 unreadable label walls; create a legible top-N view + inspector for all rows.
- Avoid pie charts for sparse feature banks, arbitrary 2D t-SNE “islands,” decorated word clouds, and unqualified novelty claims.
- Keep `n_items`, `n_features`, `n_selected`, `n_resolved`, prompt sample sizes, source, and any filter restrictions on figure caption or metadata.
- Use stable ordering and pinned fonts; avoid hidden axis clipping and figure labels that overlap. Human-readable names in charts; exact IDs available on click/export.
- Provide clear text alternatives / concise captions for each static figure.

## 9. Test-driven implementation: mandatory RED → GREEN → REFACTOR

This project explicitly requires **TDD**. For each stage: write focused failing tests **before** implementation; record the failing test and why it fails; implement the minimum to pass; rerun tests; refactor with regression tests green. Maintain all existing 42 pilot tests green.

**Stage 0 — Run fixture and integrity baseline:**
- Read `README.md`, `FEATURE_SPACE_PILOT.md`, `DATA_DICTIONARY.md`, `composition.py`, and `report.py`.
- If no run exists, create exactly one explicitly **synthetic offline** demo with:
  `python -m feature_space_pilot.run --config configs/feature_space_pilot.json --stage all --demo`
- Check `python -m pytest -q tests/feature_space_pilot` before modifications. Do not make any paid API calls.

**Stage 1 — Adapter RED/GREEN:**
- RED: reject mutated hashes, unknown item/feature IDs, duplicate judged cells, incorrect joins, missing scope metadata.
- RED: preserve 0 vs missing vs out-of-scope; the numeric feature matrix must remain strictly 0/1 only on resolved in-scope cells.
- GREEN: read-only typed adapter; REFACTOR: centralize selectors and provenance.

**Stage 2 — Compositionality RED/GREEN:**
- RED: toy fixture with component-only, direct-only, shared, and inapplicable direct features; exact expected `B`, `D`, `Δ`, `I`.
- RED: candidate not confirmed until named reviewer flag exists; a rejected candidate cannot appear confirmed.
- RED: unmatched call counts, missing applicability, incorrect `S_both` factor interpretation cause visible warnings/plot blocking.
- GREEN: composition cards, paired bars and proposition drill-down; compare derived outputs to existing `compositionality_audit.json`.

**Stage 3 — Feature atlas and matrix RED/GREEN:**
- RED: cross-case shared/unique features built from **judgments**, not feature-generation frequency.
- RED: parent/child relations and situation IDs follow `stimuli.jsonl`; feature labels never derive from arbitrary list positions.
- RED: incomplete judgments never render as 0; feature filters disclose reduced denominator.
- GREEN: task atlas and clustered heatmap; refactor reusable filtering/labeling logic.

**Stage 4 — Discovery RED/GREEN:**
- RED: fixture with repeated synonyms/exact duplicates and unequal replicates verifies cumulative unique curves and denominators; do not sum per-item unique counts as global uniqueness.
- RED: selected and excluded bank candidates remain distinguishable.
- GREEN: accumulation plots and prompt/level contribution plots.

**Stage 5 — Human/embedding RED/GREEN:**
- RED: demo data must show prominent synthetic banner; absent source-specific human labels/participant records/embeddings result in an honest unavailable state.
- RED: misaligned item IDs, text hashes, language mismatches, category coding without reviewer, incompatible datasets, unequal RDM orders all fail explicitly.
- RED: known toy 0/1 matrices produce expected Jaccard distances; pinned dense toy vectors produce expected cosine distances; correlations use upper triangles and the same aligned item order.
- GREEN: data-gated human panels and RDMs.

**Stage 6 — Export and visual QA RED/GREEN:**
- RED: deterministic report index, output paths, captions, exact source run digest; export twice and verify same data/ordering/metadata (PNG byte equality is not required when renderer metadata differs).
- RED: missing scientific data never creates fabricated panels or crashes the full explorer.
- GREEN: batch exports and offline app; inspect all default plots visually for clipping, legible labels, misleading legends, and correct emphasis.

At minimum, add unit tests for all pure derivations and integration tests that render from (a) minimal hand-built fixtures, (b) the full synthetic demo, and (c) partial/missing-source runs. Include an automated headless app smoke test if the UI framework supports it. Avoid brittle pixel-perfect golden tests; assert mark counts, underlying plot datasets, caption provenance, and export existence/dimensions instead.

## 10. Required scratchpad usage

**Mandatory:** Continue the existing `scratchpad_feature_space_pilot.md` or add an explicitly linked `scratchpad_feature_space_visualizations.md` (preferred to keep this feature independently auditable). **Update throughout implementation—not only at the end.** Each entry must include:

```
Date/time and stage
Goal / research question served
Files inspected or changed
RED: exact new test, failing command/output, failure reason
GREEN: implementation change, test command/output
REFACTOR: change and regression result
Figure sanity-check: expected vs observed, data IDs used
Uncertainty / missing-evidence / synthetic-status decisions
Known blockers or open issues
Next smallest testable step
```

The scratchpad must capture a manual visual QA pass for each primary figure with a short note of what it revealed or failed to reveal. Never overwrite prior entries. Keep commands sufficient for another researcher to reproduce each plot. Do not log API keys or sensitive datasets.

## 11. Deliverables and file layout

On a successful run, produce:

```
artifacts/feature_space_visualizations/<run_id>/v1/
    index.html                 # clickable figure gallery / captions; no hidden external calls
    figure_manifest.json       # source run hash, figure IDs, filters, seed, analysis version, caveats
    figure_01_task_atlas.png
    figure_01_task_atlas.svg
    figure_02_compositionality.png
    figure_02_compositionality.svg
    figure_03_binary_heatmap.png
    figure_03_binary_heatmap.svg
    figure_04_discovery.png
    figure_04_discovery.svg
    figure_05_human_baseline.png      # only if verified; else unavailable page/status
    figure_06_geometry.png            # only if verified; else unavailable page/status
    data_exports/              # tidy tables for each figure, stable IDs
    figures_readme.md          # figure-by-figure how to read them, limitations
```

The explorer can be started locally, e.g. `streamlit run feature_space_pilot/viz/app.py -- --run-dir PATH`; add a documented no-server static-render CLI such as `python -m feature_space_pilot.viz.cli --run-dir PATH --output-dir PATH`. Choose and document exact final commands rather than hardcoding invocation inconsistently with the selected framework. Provide **README usage instructions** and a dependency lock/pinned requirements file for visualization-only dependencies.

**Deliverable quality gate:** Opening `index.html` must be meaningful even without the app; each main figure has an explanatory caption, figures do not depend on synthetic fake claims, and the app can trace a plotted compositionality candidate back to its precise proposition and source judgments.

## 12. Acceptance criteria

The PR is ready for research review only if all of these hold:

1. Fresh offline demo renders all four primary figures with **explicit synthetic labeling** and no API credentials or calls.
2. User can select poem versus reconciliation at each available abstraction level and inspect exact applicable features and prompt/replicate provenance.
3. User can inspect all **six** compositions and distinguish feature discovery, binary applicability, interaction *candidate*, and manually confirmed interaction (if any).
4. Main composition metrics agree exactly with the existing audit, and published chart categories do not double-count.
5. Interactive matrix correctly distinguishes `1`, `0`, missing and out-of-scope; a missing feature is never automatically a negative.
6. Prompt/level plots represent actual discovery records and preserve source-specific denominators, selected/excluded candidate distinctions, and sample counts.
7. Human and embedding visualizations appear **only when supported by verified input data**; otherwise clear, source-specific unavailable panels appear.
8. Every static figure has legible proposition text, a caption, a provenance footnote, and accessible export. Feature IDs remain available in tidy data export.
9. Read-only execution leaves the original run, config, source norms, and manifest unchanged; no implicit resealing, regeneration, or paid actions.
10. Tests were written RED first and documented through GREEN/REFACTOR, **all existing pilot tests remain green**, the scratchpad contains reproducible commands and visual-QA findings, and the output gallery is linked from the README.

## 13. Recommended implementation order

Deliver value in this order, rather than implementing all charts at once:

1. **Read-only adapter and provenance-aware feature inspector.** This is the foundation for every trustworthy figure.
2. **Two-task feature atlas and six-panel compositionality display.** These best serve the current intuition-building goal.
3. **Binary feature matrix with clustering/search.** This reveals shared structure and feature families.
4. **Prompt/abstraction contribution and saturation.** This tells us how feature generation actually works.
5. **Source-gated human norms and optional geometry comparison.** This makes the results scientifically interpretable when verified scientific inputs become available.
6. **Publication exports, caption gallery, regression suite, and final visual QA.**

**Final principle:** Make it easy for a human to move from an aggregate pattern to a specific feature, a specific target description, the source prompt/replicate, and the independent binary judgment. The point of the visualizations is to *inspect the representation and expose its limitations*, not merely to make it look clean.
