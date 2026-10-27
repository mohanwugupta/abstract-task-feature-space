# Feature-space visualization implementation record

## 2026-10-26 — Stage 0: frozen source and scope

- Goal: a local, offline, read-only researcher explorer and reproducible publication figures, as specified in `PRD_Feature_Space_Visualization_Explorer.md`. Build downstream of existing run artifacts; do not regenerate or alter discovery, norms, judgments or matrices for presentation.
- Inspected PRD, config, setup/data docs, composition and reporting implementation. Existing completed source: `artifacts/feature_space_pilot/demo_all_cebc8c4d0142b011` (675 synthetic generations, 75×16 binary scope, 1,200 resolved cells). The PRD's 2,250-call snapshot predates the user-authorized reduction to three repetitions. Reuse this completed fixture; do not start a new feature study.
- Plan: verify source manifest, baseline 50 tests; test-first adapter; composition/atlas/matrix/discovery derivations; source-gated external geometry; exports; offline app; visual QA and full regression. Use Streamlit/Plotly locally and Matplotlib for PNG/SVG/PDF, with outputs outside source runs. No hosted Site registration/deployment is needed for this local-only request.
- Additional user request: default GPT Sol, low reasoning. Official model and GPT-6 guides inspected: interpret the unversioned name as current `gpt-6.1-sol`, low reasoning, strict structured outputs, no temperature/seed. Configure both generation/judging, verified standard prices, and preserve the $100 cap and three repetitions. No live call or key access is needed to implement defaults. Provider lists the model ID but no dated snapshot; record that limitation rather than inventing a revision.
- Missing evidence: demo is synthetic, not empirical; human source/protocol and embedding availability remain explicitly gated. Unsupported panels must show reasons, not fake scientific values.
- Costs: zero paid calls, $0 billed. Next: baseline and adapter RED.

## 2026-10-26 — Stage 0 GREEN / Stage 1 RED

- Baseline: `python3 -m pytest -q tests` → **50 passed**. Existing demo manifest verified without mutation. No new study run created.
- Installed Plotly 7.1.0 into ignored project `.venv` (other installed visualization versions: Streamlit 1.57.0, Matplotlib 3.10.5, pandas 2.3.2); versions will be pinned separately from core requirements.
- RED: new `test_openai_defaults.py` and `test_viz_load.py` exercise model config and read-only adapter acceptance; expected failures are unselected model and absent visualization package. Duplicate cells, unknown IDs, source hash mutation, missing scope and all four cell states have explicit tests.
- Next: configure verified Sol low defaults and implement one typed visualization adapter. No original run, source norms or judgments modified; $0 API spending.

## 2026-10-26 — Stage 1 GREEN / Stages 2–5 RED

- GREEN: `python3 -m pytest -q tests/feature_space_pilot/test_openai_defaults.py tests/feature_space_pilot/test_viz_load.py` → **4 passed**. Typed read-only adapter verifies manifests, strict ID joins, scope, canonical bank, provenance, CSV/matrix agreement and four cell states; inspection preserves raw wording/evidence.
- Model default is `gpt-6.1-sol`, low reasoning, standard verified $2 input/$10 output per million, no temperature or seed, strict structured outputs. Both roles use the existing environment variable. A snapshot note explicitly discloses the provider's undated identifier. No request executed; conservative $100/bank-scope preflight may still block a full billed plan.
- New RED tests in `test_viz_derive.py` and `test_viz_external.py`: exact composition identity and audit agreement; named/rejected reviews; judgment-based atlas; unresolved matrix; pooled discovery union/exclusions; synthetic human gating; verified raw vectors/alignment and correctly scaled cosine/Jaccard geometry. Expected absent derivation module before production code.
- Scientific decision: external scientific panels will remain unavailable on synthetic runs; generated-only geometry is explicitly exploratory. Keep original artifact bytes unchanged.

## 2026-10-26 — Derivations GREEN / figure-export and app RED

- Initial derive GREEN attempt exposed a syntax typo (extra closing parenthesis in the pooled-curve row constructor); fixed production code, retaining assertions. Targeted derivation/external suite then → **6 passed**.
- Exact source audit identities, matched calls, manual review status, independent applicability atlas, four-state matrix, pooled unique discovery and excluded-candidate handling pass. Embeddings require actual vectors, model/revision/documentation, file digest and exact text/item alignment; cosine uses 0–2, binary uses Jaccard 0–1.
- Figure/export RED: missing `figures` and `exports` modules before implementation. Tests require full labels and stable IDs in plot customdata; nonstacked composition traces; deterministic figure metadata; PNG/SVG/PDF export, synthetic filename/banner, gated human panels and unchanged source files.
- App RED: headless Streamlit smoke must load the synthetic fixture, show the banner, and navigate every research page without API/network calls. Test is written before app implementation.
- PDF skill is used for figure/PDF visual QA; Browser skill will verify the local interactive app. Static gallery and app remain local-only; no Site deployment or paid requests.

## 2026-10-26 — First export/app GREEN and visual QA failures

- Plot/export integration → **2 passed**; headless Streamlit all-eight-page navigation → **1 passed**. Full suite at this checkpoint → **63 passed**. Exports include five rendered primary figures (two atlas variants), vector PDF/SVG plus >=200-DPI PNG, tidy data, inspector and gallery; human/geometry scientific exports correctly show unavailable on the demo.
- Manual QA: Figure 1 revealed a real axes-autoscaling layout bug: column headings overlaid the main title and propositions fell near the footer. Figure 2 PDF rendered successfully, but second-row review notes overlapped its caption. Added a new failing layout test checking atlas normalized axes/text bounds and review-note body bounds before fixing production layout.
- Figure 3 QA: full feature propositions are legible; 15 rows, 13 task-present features, explicit 0/1 and row-family separators. Underlying bank includes 16 features; no padding of absent human-only proposals.
- Figure 4 QA: facets/call denominators and overlapping synthetic prompt curves are clear; constant-count facets need integer-count limits instead of tiny fractional auto axes.
- Browser integration unavailable: its mandatory bootstrap failed before execution with missing `sandboxPolicy` metadata. Read the installed troubleshooting guidance; no browser-control bypass used. Continue with supported Streamlit AppTest coverage, local-server readiness, static/PDF visual QA, and report this verification limit. Local app serves at 127.0.0.1:8501.
- PDF rendering: system Poppler unavailable; used installed PyMuPDF to render the actual exported PDF for inspection. Source run verified unchanged. $0 paid API calls.

## 2026-10-26 — Visual layout GREEN / interaction RED

- Layout RED reproduced atlas axes auto-limits outside 0–1. Production fix pins the prose-card axes to normalized coordinates; moves composition plot grid upward to reserve review/caption space; discovery count axes now start at zero with integer ranges.
- GREEN: `test_viz_figures.py` → **2 passed**, including actual rendered text-bounds checks. Re-export and re-inspection pending after final refinements.
- Added explicit partial-export and external-data regression cases: unknown cells survive as ?, composition blocks without zero fill; known toy Jaccard distances, unequal RDM item orders, language mismatch and missing category reviewer are tested.
- New interaction RED requires clicking an atlas feature to select that exact ID in the evidence inspector, then verifies shared search discloses the reduced vocabulary denominator. This tests the actual researcher drill-down rather than page navigation alone.

## 2026-10-26 — Interaction and external-source GREEN / final scope RED

- Atlas-button RED caught stale Streamlit widget selection: the clicked proposition did not become the evidence-inspector selection. Fixed the explicit widget state before rendering the selector; rerun passed. Search still preserves/discloses reduced vocabulary. No assertion relaxed.
- Positive external-source RED caught exporting only the first verified source. Geometry export now keeps every verified source in separate rows and item orders; targeted test passed. No datasets are pooled into one human RDM.
- Added reviewed source-category bars/coverage in the app and prevented blank feature-recovery plots when a source only has category summaries. Export gallery now includes actual incremental proposition links to the evidence inspector and full target descriptions.
- Full regression after these changes: `.venv/bin/python -m pytest -q tests` → **69 passed**. App navigation and exact feature selection are tested headlessly; native Browser bootstrap remains unavailable due session metadata, not application failure.
- Final small adapter acceptance test adds explicit in-scope denominator and path-safe run identity before export. This prevents a run-ID string from escaping the separate visualization output tree; default UI will show resolved/declared fraction alongside revisions and bank hash.
- Layout/export regeneration and final manual QA are in progress. No source changes or API calls.

## 2026-10-26 — Final primary-figure QA and conditional-source RED

- Latest full suite: **70 passed**; scope denominator/path-safe run identity and app header revisions pass. Export metadata now includes visualization-module hashes and renderer versions.
- Manual QA after fixes: baseline atlas shows six actual applicable propositions with clean mirrored columns; deadline atlas shows nine, including two exact task-specific conflict hypotheses. No padding to an arbitrary feature count. All labels/headings/captions are inside the figure.
- Composition PDF re-render: six 9-versus-9-call panels, separate overlapping bar categories, source-loaded examples, 0 confirmed versus unreviewed strip, no second-row caption collision. Heatmap PDF: 15 targets × 13 present task features from a 16-feature bank, readable full labels/0/1 and source-level grouping. Discovery PDF: eight level facets, integer counts from zero, actual pooled call counts and coincident synthetic prompt curves; no artificial separation.
- All five primary PDFs render as one vector page and retain the synthetic banner. Gallery uses local assets only and links each incremental proposition to full evidence/target text. Source manifest still verifies complete (675 generations, 1,200 cells).
- Final conditional-source RED covers a verified category-count-only human reference with independent category coding: it should render the supported category profile rather than suppress the entire human figure just because feature-string recovery is unavailable. This case matters for restricted human supplements; no human data are acquired or fabricated for production.

## 2026-10-26 — Conditional source GREEN / final acceptance export control RED

- Category-only source RED reproduced `fig is None`; static human figure now independently renders reviewed category composition and coding coverage while keeping feature-list recovery unavailable. Source-specific geometry remains blocked for category counts. `test_viz_external.py` → **6 passed**; full suite → **71 passed in 45.42s**.
- Refreshed the read-only preflight example for the newly configured Sol-low defaults. No model/source gate was removed: full task worst-case bound is still above the unchanged $100 cumulative cap. Historical demo metadata is not relabeled as Sol.
- Final acceptance cross-check found a missing UI affordance: batch export CLI existed, but the requested one-click figure bundle control did not. Added an AppTest before implementing it, checking actual gallery creation, downloadable ZIP, no source mutation and valid source manifest. First test draft lacked its local AppTest import; fixed the test setup before exercising the expected missing-button RED.
- Re-exported five primary figures and two honest unavailable pages; source manifest verifies complete with 1,200 resolved/in-scope cells. No paid calls or key reads. Next: add the export control and rerun regression.

## 2026-10-26 — One-click export GREEN and release regression

- Expected RED: `test_one_click_publication_export_is_downloadable_and_read_only` failed with `StopIteration` because no publication export button existed. Added the explicit control on every loaded app page; it renders the full frozen comparison and offers a ZIP with current-manifest assets, captions, data and feature evidence. It does not silently apply sidebar filters to the publication estimand.
- GREEN: `test_viz_app.py` → **3 passed in 21.01s**, including every page, exact evidence selection, actual ZIP-download control, real gallery generation and unchanged verified source. Bundle state is keyed by run fingerprint to prevent stale exports after review changes.
- Refactor/style: Black checks all 15 visualization/default/test modules. Updated README and visualization guide with one-click bundle usage. Current offline task dry run succeeds as a read-only command while correctly retaining live budget blockers.
- Primary manual QA and native-browser limitation remain as recorded above. Final full regression and source-preserving delivery export are running; no API charges or credentials read.

## 2026-10-26 — Final release GREEN

- `.venv/bin/python -m pytest -q tests` → **72 passed in 48.27s**. All 50 baseline tests remain green. New checks cover source integrity, four states, composition identities/reviews, provenance drill-down, pooled discovery, external-source gates, exact distances/alignment, partial exports, deterministic data and actual one-click UI export.
- Final export: `.venv/bin/python -m feature_space_pilot.viz.cli --run-dir artifacts/feature_space_pilot/demo_all_cebc8c4d0142b011`. Delivery gallery: `artifacts/feature_space_visualizations/demo_all_cebc8c4d0142b011/v1/index.html`. Five rendered PNG/SVG/PDF figure records (including both atlas variants), two unavailable scientific panels, feature inspector and tidy data. Source manifest hash and all seven visualization-code hashes match the final export metadata; original complete 675-generation / 1,200-cell run verifies unchanged.
- Manual visual QA outcome: all primary PNG/PDF labels, legends, proposition cards and captions are legible and unclipped after the recorded layout fixes. Native live-browser QA remains unavailable due tool bootstrap metadata; official Streamlit AppTest verifies navigation and actual interactions. Owned preview server stopped after checks.
- Default generator/judge: GPT-6.1 Sol, low reasoning, configurable; official identifier is undated and this limitation is recorded. Three repetitions and $100 cap retained. Conservative full paid task design remains budget-blocked, and human-source/protocol/embedding gates remain unresolved until verified inputs exist. No implicit paid execution, no key reads, **$0 API spending**.
- Publication: code/config/docs/tests/PRD are staged; local `.env`, `.venv` and generated artifacts remain ignored. Original PRD Markdown hard-line-break spaces are retained verbatim. Next: commit/push this implementation to the existing private repository.

## 2026-10-27 — Start empirical human-pilot visual analyses / RED

- User authorized starting visualization analyses. Current empirical Mini human run has 60 targets, 150 frozen features, 540 successful generations and 7,208 resolved cells; 1,759 ambiguous and 33 missing/failed cells remain unknown. All confirmed Batch outputs have been collected; one uncertain retry remains held by the independent collector. Do not change the scientific run, fill unknowns with zeros, relabel Mini as Luna, or silently run a task pilot. No paid calls are part of analysis.
- Inventory verifies Vinson human alignment is available; Harpaintner translated human comparisons remain gated by pending named review; category coding/raw embeddings/complete judged binary geometry are unavailable. Existing discovery and supported human recovery can be analyzed now.
- Human-only exporter/Matrix RED: **4 failed** before implementation. Reproduced an empty task-only matrix crash, absent source-preserving snapshot helper, missing human-target UI filters and absent human diagnostic panels. Added exact verified checkpoint copying outside source runs, human matrix defaults/filtering, explicit missing-state preservation, coverage/applicability diagnostics, truthful task-only unavailable panels, and full matrix data exports. Additional coverage test checks unresolved denominators rather than zero-filling. Source collector code pins remain unchanged; modifications are confined to nested visualization modules.
- Plan: frozen snapshot, actual empirical exports and scientific availability inventory, visual PNG/PDF QA, headless/browser UI verification and a running local explorer. Snapshot/source hashes and output manifests must verify throughout.

## 2026-10-27 — Empirical export and visual QA / layout RED → GREEN

- Initial human-only tests: **5 passed in 4.82s** after the four expected RED failures. Snapshot created at artifacts/feature_space_analysis_snapshots/live_human_74608725aae4e0dc/d76ce6eff476ac64, exact source manifest SHA-256 d76ce6eff476ac64acfb216e342e401f494124f7108b9d5607c9ce276428c269, all 60 sealed artifacts verified. Original run and collector scientific/execution pins still validate unchanged. No paid model calls or credential reads.
- Actual empirical gallery exported under artifacts/feature_space_visualizations/live_human_74608725aae4e0dc/v1: five PNG/SVG/PDF panels (partial matrix, discovery, verified Vinson lexical recovery, coverage, top feature applicability), four truthful unavailable task/geometry panels, full evidence inspector, complete 9,000-cell export, and initial analysis summary. Counts: 971 applies, 6,237 inapplicable, 1,759 ambiguous, 33 missing. Discovery includes 5,405 unique normalized proposals before bank150 selection.
- Visual QA caught overlapping full heatmap labels on the actual 60-target figure. Added a failing rendered-bounds test for dense 24-column long labels, then rotated/wrapped the labels. Reinspection shows no label overlap; the actual vector PDF was rendered separately and inspected. Full test initially **148 passed / 1 failed** due a category-caption wording regression; restored the explicit independently coded category qualification. Full regression then **149 passed in 58.89s**.
- Vinson's 108 item/family lexical-recovery entries are all zero under the frozen normalized exact-string rule; figures and summary now explicitly distinguish this from semantic/paraphrase correspondence. No equivalences, translation approvals or category coding were invented. Complete binary geometry remains unavailable.
- In-app Browser bootstrap again fails before execution because session metadata lacks sandboxPolicy. Read mandatory Browser skill and installed troubleshooting; no browser-control bypass used. Supported Streamlit AppTest initially timed out once on the full snapshot during concurrent work; isolated repeat passed all seven actual-data research pages, each 0.3–0.7s after a 2.1s initial load. Local explorer is serving the Matrix page at http://127.0.0.1:8501 (PID 46538); health endpoint returns ok.

## 2026-10-27 — Empirical analysis delivery verification

- Final actual export regenerated with current visualization code hashes and the unchanged snapshot manifest; all hashes match. Five empirical figures rendered, four unavailable panels preserved, and full declared-scope CSV/JSON contains exactly **9,000 cells, 1,792 unresolved**. Exported 24-column matrix remains explicitly a display subset; full bank150 is retained in downloadable data.
- Actual PDF heatmap rendered through PyMuPDF and inspected: full target/proposition labels are readable, rotated columns no longer overlap, unknown cells remain distinct and source fingerprint/footer are intact. Discovery, human lexical matching, review coverage and top-feature PNGs were also visually inspected.
- Final targeted human/export regressions after the full declared-scope data export adjustment: **8 passed in 13.69s**. Prior full regression: **149 passed**. Full-data headless navigation passed Matrix/Coverage/Discovery/Human/Geometry/Inspector/Overview. Server health is ok; interactive explorer remains running at http://127.0.0.1:8501.
- Delivery artifact: offline gallery/index.html, feature inspector, analysis_summary.md/JSON and all figure/data exports under artifacts/feature_space_visualizations/live_human_74608725aae4e0dc/v1/. Download bundle: human_pilot_visualizations_v1.zip, **6,874,430 bytes**, adjacent to v1/. Generated study/results files stay ignored by Git. Publishing visualization code/tests/guide and this audit; source run, references, API settings and collector implementation remain unchanged. No new paid calls.
