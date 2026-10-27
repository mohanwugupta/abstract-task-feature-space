# Feature-space pilot implementation record

## 2026-10-26 — Workspace and source audit

- Goal/hypothesis: implement the frozen, binary feature-space pipeline in the PRD, with an offline reproducible demonstration before any live work.
- Workspace: initially contains only `PRD_Feature_Space_Human_Norm_Pilot.md`; no upstream code, Git checkout, tests, or materialized human norms are present.
- Source assumptions: the referenced upstream repository and primary norm supplements must be verified. Human production norms are reference observations, not ontological truth. Synthetic demonstration sources must never pass the live human-data gate.
- Implementation status: no production code written; RED tests are next. This record is append-only.
- Run provenance/spending: zero API calls, zero billed tokens, $0 spent; hard pilot maximum $500. No live credentials or prices assumed.
- Risks/next action: inspect upstream availability and primary sources, then record RED/GREEN/REFACTOR evidence for each delivered slice. Unavailable sources remain explicit live-run blockers.

## 2026-10-26 — Upstream integration base

- The user-provided `LLM_judge_item_expansion-main.zip` arrived after the initial workspace inspection. Safely extracted its contents into this workspace; original experiment code/data/history are preserved. No `AGENTS.md` is present in the archive.
- Audited existing generation/prompt/schema/client boundaries: existing judgments use 0–4; the vLLM client caches responses and does not provide fail-closed commercial accounting. The pilot will use a separate strict binary namespace and uncached adapter, reusing the existing text-normalization helper only where compatible.
- Baseline regression suite started before production changes. Source papers and official API model/schema documentation are being inspected; no benchmark data availability is inferred from article abstracts.
- Calls/spending remain 0/$0. Next: first registry RED tests and frozen synthetic task fixtures.

## 2026-10-26 — Slice 1 RED: item registry and task fixtures

- Acceptance: stable hashes; reject duplicate IDs, missing parents/situations, invalid target types, cycles and LFS pointers; exactly 15 case targets; full 20-invoice synthetic fixture with a reproducible reference.
- RED: `python3 -m pytest -q tests/feature_space_pilot/test_registry.py` fails collection with `ModuleNotFoundError: No module named 'feature_space_pilot'`, as expected before implementation.
- Baseline regression: `python3 -m pytest -q tests` yielded **207 passed, 1 failed**, before pilot code. Existing `test_v4_atomic_resources_and_frozen_shards` expects array `0-31%4`; upstream launcher differs. Preserve evidence and investigate separately.
- Decision: keep German accents and modality/negation intact; the Leuven accent-stripping word helper is inappropriate for proposition equivalence, so do not reuse it.
- Source audit: primary articles opened; Harpaintner methods specify 60 participants, 10 per word, about four properties, no time limit. Actual supplements still need inspection. Zero paid calls, $0 spent.

## 2026-10-26 — Slice 1 GREEN / REFACTOR and API clarification

- GREEN: `python3 -m pytest -q tests/feature_space_pilot/test_registry.py` → **4 passed**. Added `io.py`, `registry.py`, `fixtures.py`, and package initialization; no prior experiment code modified.
- REFACTOR: registry validation centralized; deterministic fixture reference uses integer cents and documented duplicate precedence; the same targeted suite remains green.
- User clarification: execution uses hosted frontier-model APIs, **not cluster/vLLM models**. User chose to leave model IDs configurable. Therefore live config will intentionally have no default billed model/price; fake runs remain available. No live calls will be attempted during implementation.
- Frozen case design: 15 targets; five-minute deadline; ABAB CDCD eight-line poem; 20 synthetic invoices. Financial target includes full inputs/policy/schema but excludes reference answers.
- Spend: $0. Next slices: reference gate, isolated prompts, planning/parsing.

## 2026-10-26 — Slices 2–4 RED: reference gate, prompts, planning/parser

- RED command: `python3 -m pytest -q tests/feature_space_pilot/test_references.py tests/feature_space_pilot/test_generation.py` → collection errors for absent references module and prompt builder, expected before code.
- Tests cover read-only/hash/schema/language/source verification, synthetic-source rejection, deterministic stratified held-out selection, source metadata isolation, separate P0 protocol routing, 2,250 independent jobs, paired seeds, resume, malformed/duplicate/bundled features, and meaningful relational predicates.
- Decision: missing Vinson participant protocol blocks live P0, rather than inventing historic instructions. Harpaintner adaptation is explicitly labeled as an adaptation with cap 4. All prompts and seed policy will be frozen in each design fingerprint.
- User selected configurable hosted frontier models. No default commercial model or cluster dependency will be introduced. Costs remain $0.

## 2026-10-26 — Slices 2–4 GREEN / REFACTOR

- Initial GREEN attempt found `test_selection_is_label_blind_stratified_and_frozen` failing: no holdout for three-item strata. Fixed the implementation to reserve one item when a stratum has at least three, preserving the assertion. No out-of-sample claims for smaller strata.
- GREEN / REFACTOR: `python3 -m pytest -q tests/feature_space_pilot` → **16 passed**. Modules: `references.py`, `prompts.py`, `planner.py`; five literal prompt files. Selection sorts labels before seeded sampling; generation never imports evaluation code.
- Public source-page network access succeeded. Vinson supplement ZIP located at the publisher; checking actual contents next. No paid model calls, $0 spent.

## 2026-10-26 — Slices 5–7 RED: consolidation, binary judge, budget

- RED: `python3 -m pytest -q tests/feature_space_pilot/test_bank_judge.py tests/feature_space_pilot/test_budget.py` → expected missing `bank` and `budget` modules.
- Acceptance: preserve role/negation/modality/conditions and complete raw provenance; freeze label-blind candidate cap; isolate judge input; reject 0–4 values, bools and missing cells; cumulative/stage caps and durable worst-case reservations on interruptions.
- Source progress: publisher Vinson ZIP downloaded verbatim and chmod 444. Actual word-category, feature-type and frequency matrices inspected. Harpaintner supplemental XLSX/DOCX links located via PMC. These files are not yet declared validated adapters.
- API calls/spending remain 0/$0. Public file downloads are not model calls.

## 2026-10-26 — Slices 5–7 GREEN / REFACTOR; slices 8–9 RED

- GREEN/REFACTOR: `python3 -m pytest -q tests/feature_space_pilot` → **20 passed**. New bank/judge/budget modules, conservative exact equivalence and reviewed merge guard; primary judgments are strictly binary, ambiguity retained; process-locked durable budget ledger.
- RED: `python3 -m pytest -q tests/feature_space_pilot/test_analysis.py` → missing metrics module. Acceptance: item-resampled geometry excludes repeated-original self-pairs; zero vectors have a declared distance policy; recovery is relative to human lists; category vocabulary is source-specific; interactions require manual confirmation and equal generation-call budgets.
- Frozen fair-comparison rule: with 10 replicates per family, use first 5 task + first 5 situation versus all 10 direct-combination replicates. All full discoveries are retained separately. Equal calls do not imply equal tokens; report tokens and gain per token as well. No unbudgeted extra generation calls.
- No paid calls; $0 spent.

## 2026-10-26 — Slices 8–9 GREEN / REFACTOR; slice 10 RED: end-to-end controller

- GREEN/REFACTOR: targeted pilot suite → **23 passed**; added metrics and compositionality functions. Item-level resampling, missing-data availability flags, audited-interaction distinction and equal-call comparisons pass.
- RED: `python3 -m pytest -q tests/feature_space_pilot/test_pipeline.py` → missing engine module. Tests now require 60+15 synthetic targets, dry-run with zero writes/calls, no synthetic live execution, complete 0/1 artifacts, exact resume, charged parse retries, partial interruption, unresolved judgment audits, stale artifact rejection and uncached hosted adapter.
- Source constraint: PMC supplement URLs returned HTML challenges rather than office files; rejected these bytes instead of saving them as norms. Vinson terms permit non-commercial research/education with citation to both paper and downloaded materials; preserve the ZIP and terms verbatim.
- Next: orchestrator, fake/hosted clients and reports. Real paid benchmark remains gated, not claimed complete. Spending $0.

## 2026-10-26 — Slice 10 GREEN: hosted adapter and pipeline

- GREEN: `python3 -m pytest -q tests/feature_space_pilot/test_pipeline.py` → **5 passed**. Full fake integration produces 75 target rows, strict binary matrix, six balanced composition records, source/category/recovery availability records, hashes, token/cost ledger, and exact no-call resume.
- Interrupt/retry integration retains the paid-attempt-equivalent failed parse record and succeeds on retry; unresolved binary cells prevent matrix construction and produce separate audits.
- Files: hosted/fake clients, orchestrator, deterministic reports. Hosted adapter is uncached HTTPS, with no implicit retry and explicit capability-based seed/temperature omissions. Frontier model IDs/prices remain unset for user selection.
- REFACTOR/robustness next: validate journals and interrupted manifest recovery, CLI/config/source adapters, regression suite. No paid API calls, $0.

## 2026-10-26 — Slice 10 robustness RED → GREEN / REFACTOR

- New RED `test_resume_after_crash_appends_journal_beyond_partial_manifest`: simulated power loss after durable response append caused stale-manifest rejection on resume (`Stale artifact hash: generations.jsonl`). This exposed a real interruption boundary.
- GREEN: added checkpointed journal prefixes and per-record/response hashes; incomplete runs permit only append growth, preserve frozen-input hashes, and regenerate derived reports. Complete runs still require all artifact hashes. `test_missing_provider_usage_halts_and_keeps_liability` verifies missing token usage halts after one retained reservation.
- REFACTOR: `python3 -m pytest -q tests/feature_space_pilot` → **30 passed**. No test assertions relaxed. Failed historical records remain append-only.
- Next: command entry points and verified raw-data adapters. Spending $0, no model API calls.

## 2026-10-26 — Slice 11 RED: CLI, source adapter and delivered artifacts

- RED: `python3 -m pytest -q tests/feature_space_pilot/test_cli_sources.py` fails on absent source adapter. Tests require a feature×item → item×feature Vinson adapter with original category vocabulary, strict item counts, both requested dry-run entry points, complete 2,250-job offline CLI demo and report integrity checks.
- Publisher/PMC Harpaintner supplement endpoints failed (HTML challenge/404). Checking the official NCBI open-access package once before declaring acquisition blocked. No source values are reconstructed from article prose.
- Remaining regression issue is the pre-existing old launcher concurrency header. Any repair will be documented separately and will not connect the API pilot to cluster execution.
- Paid calls 0; spend $0.

## 2026-10-26 — Slice 11 GREEN; scope-estimate RED

- CLI/source GREEN: `python3 -m pytest -q tests/feature_space_pilot/test_cli_sources.py` → **2 passed**, including a complete 2,250-generation offline CLI run and verified report.
- Actual Vinson raw source: **456 items, 1,029 features**; adapter verifies dimensions, indices, nonnegative integer weights and original feature category membership. Derived evaluation-only CSV and raw ZIP are read-only and hashed. Frozen 12 action nouns + 12 action verbs + 12 object labels (9 held out). Harpaintner's 24 slots remain explicitly unfilled; no fabricated abstract labels enter the live registry.
- New RED `test_missing_human_data_does_not_underestimate_declared_matrix_scope`: declared 60×200 matrix was underestimated as 36×200 when only 36 labels are available. The test also requires the unavailable primary source to be named in blockers. Fix preflight to expose declared and plannable scopes separately.
- Decisions: exact Vinson P0 method still unverified, so registry verified=false even though raw data are available. No live calls or unsupported dollar claims; configurable models/prices remain null. Spending $0.

## 2026-10-26 — Slice 11 scope GREEN and upstream regression repair; review/preflight RED

- Scope GREEN: declared human matrix upper bound is 12,000 cells even while only 36 of 60 stimuli are materialized; blockers now identify each source and its actual acquisition/protocol problem.
- Existing baseline regression repair: changed only old `run_leuven_v4_atomic.sh` array header from `0-31` to the test's frozen concurrency limit `0-31%4`. This preserves 32 shards and reduces concurrency; the API pilot neither imports nor executes this script. `python3 -m pytest -q ...test_missing_human_data_does_not_underestimate_declared_matrix_scope tests/test_v4_cluster_launchers.py` → **4 passed**. Original RED was captured before any pilot code.
- New RED tests: cumulative liabilities in preflight; rejecting inline secrets before artifacts; adding named manual interaction audits to an existing completed run without model calls. These expose preflight's omitted prior balance, permissive settings and stale review report on resume.
- Spend: $0. Next: fix these acceptance gaps and complete source/data dictionary documentation.

## 2026-10-26 — Review and cumulative-preflight GREEN / REFACTOR

- GREEN: pipeline suite → **10 passed**. Preflight now reads cumulative liabilities without creating files; inline credentials are rejected before artifact creation; named relational audits regenerate reports without repeating model calls. Prior manifests are appended to a history journal.
- Manual interaction confirmations require a named reviewer, explicit boolean and substantive note. A bare automatic label cannot confirm an interaction. Review inputs have a separate evaluation fingerprint; primary generations/bank remain frozen.
- Refactor: limit quadratic wording-similarity review to selected inventory phrases (cap 200); retain all excluded raw candidates and selection flags for coverage reporting.
- Source gate remains blocked for Harpaintner acquisition and exact Vinson P0 instructions; hosted frontier model/price/capability settings remain intentionally unselected. Paid calls 0, cost $0.

## 2026-10-26 — Correction to preceding GREEN record

- The preceding entry was written before the command finished; actual first rerun was **9 passed, 1 failed**, not 10 passed. Existing crash-recovery test caught a regression: rewriting dynamic preflight balance (`0` → `0.0`) changed a frozen file before a simulated crash.
- Fixed by writing frozen config/preflight exactly once and storing changing review inputs separately. Actual rerun `python3 -m pytest -q tests/feature_space_pilot/test_pipeline.py` → **10 passed**. This append preserves the failed approach and corrects the premature entry.
- Spending remains $0. No paid calls.

## 2026-10-26 — Evaluation completion RED

- New RED tests require reviewed source-category coding to become quantitative profiles (with unknown categories rejected), and an aligned pinned embedding artifact to compare with **both** generated and human feature geometry. Current reporting only exports unavailable flags for those paths; tests expose missing behavior.
- RED command: targeted tests in `test_analysis.py` and `test_pipeline.py`; missing `generated_profiles` / `human_comparisons` keys are the intended failures.
- Verified the prior crash-recovery rerun actually completed: **10 passed**. No paid calls, $0.

## 2026-10-26 — Evaluation completion GREEN / REFACTOR

- GREEN/REFACTOR: `python3 -m pytest -q tests/feature_space_pilot/test_analysis.py tests/feature_space_pilot/test_pipeline.py` → **15 passed**.
- Reviewed category CSVs require item/feature IDs, source-provided category and reviewer; profile coverage is reported and unknown categories rejected. Embedding imports require model/revision/documentation, whole-file hash, exact item IDs and text hashes; both generated and source-specific human geometries are computed where supported.
- These evaluation-only additions update an existing run's reports and manifest history without model calls. Fair family comparisons now derive the common cap from the actual configured P0 protocol rather than assuming ten.
- Refactor: source reference loading shared between comparison paths; composition tables now render as contiguous Markdown rows with examples below the table. Full regression suite started.
- Spending: zero paid model calls, $0. Next: final documentation, default full offline artifact, final integrity/regression checks.

## 2026-10-26 — Full regression and final provenance/budget RED

- Regression at this checkpoint: `python3 -m pytest -q tests` → **246 passed**, four pre-existing pandas FutureWarnings. No failures or skips introduced.
- New RED: `test_demo_reference_is_materialized_read_only_and_hashed` fails because synthetic reference data were constructed in evaluation memory instead of preserved as a read-only source artifact. Materialize them and exercise the same loader as real reference data, while retaining kind=synthetic.
- New RED: `test_completed_resume_preflight_reserves_only_remaining_work` requires a completed run's pending-cost bound to be zero. Full-plan-plus-prior estimates otherwise overstate resume spending and can block a zero-call report update near the cap.
- Zero paid model calls/$0 spent; fake usage records are engineering fixtures only.

## 2026-10-26 — Provenance/resume GREEN; explicit smoke RED

- GREEN: `python3 -m pytest -q tests/feature_space_pilot/test_pipeline.py` → **13 passed**. Synthetic references now exist as read-only CSVs, pass the ordinary evaluation loader and have source hashes. Resume preflight subtracts successful/exhausted attempts and uses actual frozen judgment scope; completed runs have a zero remaining-spend bound.
- Documentation added: `FEATURE_SPACE_PILOT.md`, `DATA_DICTIONARY.md`, pinned offline requirements and an offline CI workflow.
- New RED `test_opt_in_smoke_freezes_one_per_class_and_one_task_situation_pair`: requires a separate frozen eight-target scope (one synthetic/human item per class plus the poem task/situation pair and its hierarchy parents), two replicates, 48 generation calls, and an explicit smoke manifest flag. This supplies the PRD's opt-in billed-smoke path without running a paid request.
- Spending $0. Next: smoke helper and final artifact/test verification.

## 2026-10-26 — Smoke GREEN / REFACTOR and final manifest RED

- Smoke GREEN: targeted test **1 passed**; full pilot regression **41 passed**. New `--smoke` scope uses eight targets, 48 independent generation calls, cap 10 and explicit smoke provenance. Live scope remains source/model/price gated; no billed smoke executed.
- A full 2,250-generation offline artifact run has been started for final delivery. CLI/source/report tests have already independently exercised the same full plan in temporary directories.
- New RED `test_manifest_exposes_code_hashes_and_unfrozen_scope_upper_bound`: code hashes currently enter the design digest but are not exposed for review. Require explicit module hashes and the declared judgment upper bound before a bank is frozen. This is the final provenance refinement before final checks.
- Paid calls 0; billed tokens 0; spend $0; remaining live cap $500.


## 2026-10-26 — Final GREEN / REFACTOR, delivery and resume record

- Final provenance RED test now GREEN: explicit module hashes, declared judgment upper bound and bank-freeze status are recorded in manifests. Targeted test passed.
- REFACTOR: formatted only new pilot modules/tests with Black. Final regression `python3 -m pytest -q tests` → **250 passed**, 4 pre-existing pandas FutureWarnings, no failures/skips. The preceding targeted pilot run had 41 passed; final suite contains 42 pilot tests plus 208 existing tests.
- Final offline run: `demo_all_dbaa90fe6ce81f2e`; **2,250/2,250 independent generation jobs**, **1,200/1,200 resolved binary cells**; 75×16 complete matrix; values only 0/1. All 30 artifact hashes verified. Six compositions, full level audit, fake human recovery/category availability/geometry, discovery curves and review queues exist. Embedding/category-review quantitative paths are covered by separate offline integration tests; the delivered unreviewed demo correctly marks missing inputs unavailable.
- Report: `/Users/mg9965/Desktop/abstract-task-feature-space/artifacts/feature_space_pilot/demo_all_dbaa90fe6ce81f2e/pilot_report.md`. Manifest: `/Users/mg9965/Desktop/abstract-task-feature-space/artifacts/feature_space_pilot/demo_all_dbaa90fe6ce81f2e/manifest.json`. Matrix SHA-256: `fa06fa6d6c1b02a7a212c2feeaea196db850214f106bd5d23a3fb08d6ffefe3c`. Config SHA-256: `d3929d3b3fb41ceb71abc11fcc29768b6d6809f74a5b712a98d67e3285b9c1b2`.
- Exact CLI resume rerun returned the identical result/manifest without appending provider attempts. Cumulative demo ledger also preserves the earlier pre-refactor fake run; per-run and cumulative token counts are explicitly separated.
- Archive comparison: original files unchanged except `run_leuven_v4_atomic.sh`'s documented concurrency-header repair; no original files missing. Original `scratchpad.md`, Leuven/V4 data and model files remain untouched.
- Added final validation summary at `artifacts/feature_space_pilot/implementation_validation.json`, actual sample manifest at `data/feature_space_pilot/sample_run_manifest.json`, and current dry-run examples at `data/feature_space_pilot/preflight_example.json`.
- Spending/provenance: **0 paid API calls, 0 billed tokens, $0 spent; $500 live budget remains**. Demo usage tokens are synthetic fixture counters, not billed usage. No API key was requested or read for model execution. Public publisher downloads were the only network acquisition.
- Delivery boundary: hosted frontier model IDs, prices and capabilities remain configurable/unselected per the user. The actual Harpaintner supplement and exact Vinson P0 participant protocol remain explicit scientific-source blockers; 24 abstract slots are not fabricated. No live benchmark or real case-study scientific outcome is claimed.
- Resume instructions: read `FEATURE_SPACE_PILOT.md`; choose/pin hosted generator/judge and verify prices/capabilities; acquire/verify remaining human source/protocol, freeze all 60 human stimuli, complete source-language mappings as needed; inspect dry-run blockers and spending; use the explicit source-stratified smoke/preflight-hash gates before any billed full run. Human/task outputs remain separate; do not tune prompts on held-out human results or downstream case outcomes. Manual relational/categorical/adjudication reviews and optional pinned embeddings can update reports without repeating completed calls.

## 2026-10-26 — GitHub repository preparation

- User explicitly requested a new GitHub repository for this project. Default: private `mohanwugupta/abstract-task-feature-space`, named after the workspace.
- Connected GitHub profile confirmed `mohanwugupta`. Connector has no repository-creation tool; use authenticated GitHub CLI. Sandboxed CLI auth initially reported an invalid token due to restricted access; network-enabled auth verified the existing keychain login. No credentials were printed or copied into project files.
- Added a repository README with setup, scope, source blockers, provenance and the existing 250-test validation result. Excluded the original 54 MB distribution ZIP and local environment files. Pilot CSVs explicitly bypass inherited LFS rules so the offline CI fixtures are materialized on a fresh checkout.
- Read-only secret-pattern scan found no matching private keys or API credentials in source files. Generated artifacts remain ignored and reproducible locally. No production-code changes or paid model calls were needed for publication.
- Next: initialize local main branch, inspect the staged source inventory, commit, create the private remote, push, and verify the published branch.

## 2026-10-26 — Publication byte-integrity check

- Initial Git staging exposed CRLF normalization of the read-only derived norm CSV: staged SHA-256 differed from the source registry. Fixed the pilot CSV attributes to disable both LFS filtering and line-ending conversion. This preserves reference-byte hashes across clones.
- Staged inventory contains 191 source/documentation/fixture files. Original ZIP, generated run directories and environment files are excluded. Existing Git identity is configured and retained.
- Legacy data LFS pointers are intentionally retained from the supplied base; their materialized objects were never present. The repository documentation identifies these limits.

## 2026-10-26 — User-directed standalone cleanup before publication

- User explicitly requested removal of everything inherited from the old ISC-CI repository before committing/publishing.
- The first push had failed on a missing legacy LFS object, so no source branch was published. Stopped the in-progress read-only legacy LFS fetch. The new private GitHub repository is still empty.
- Removed 537 inherited files: experiment code, datasets/LFS pointers, tests, validation/training directories, cluster launchers, papers, old PRDs, historical scratchpad/todo files, and generated legacy artifacts. Replaced shared Git ignore/attribute and pytest configuration with minimal standalone settings.
- Kept only the newly implemented pilot, its new tests, frozen pilot fixtures/references, requested PRD, pilot documents, CI and the pilot scratchpad. Original distribution archive and earlier synthetic outputs were moved outside the workspace to `/private/tmp/abstract-task-feature-space-precleanup` for local recovery.
- Updated current documentation to describe a standalone project and its 42 pilot tests. Historical entries above intentionally retain the implementation audit; they do not mean old experiment files are still included.
- Next: rebuild the unpublished local Git history without any legacy commit/upstream remote, run the standalone tests and full offline demo, then publish the cleaned initial commit. No paid model calls; live budget untouched.


## 2026-10-26 — Standalone cleanup GREEN and publication inventory

- `python3 -m pytest -q tests` → **42 passed**. All remaining tests belong to the pilot; no legacy test/dependency remains.
- Fresh full offline run `demo_all_3104758f653fa4f9` completed **2,250 generation jobs and 1,200 binary judgments** with all artifact hashes verified. Source/model live gates remain unchanged; no paid calls.
- Compared the workspace against every original archive path: only `.gitignore`, `.gitattributes`, and `pyproject.toml` overlap, and all three were rewritten for this standalone project. All 537 other inherited files were removed; the original archive is outside the workspace.
- Previous unpublished `.git` metadata is backed up outside the workspace. The fresh Git repository has only `origin`, no upstream remote or inherited commit history.
- Updated standalone sample manifest and dry-run examples. Current report: `/Users/mg9965/Desktop/abstract-task-feature-space/artifacts/feature_space_pilot/demo_all_3104758f653fa4f9/pilot_report.md`.
- New private repository exists at `https://github.com/mohanwugupta/abstract-task-feature-space`; verified empty before publishing the cleaned initial commit. Next: stage the allowlisted pilot files, verify reference-byte hashes and zero LFS entries, commit, push, and inspect the remote branch/CI.


## 2026-10-26 — Standalone GitHub publication complete

- Published cleaned standalone source to private repository `https://github.com/mohanwugupta/abstract-task-feature-space`, branch `main`, initial source commit `ac46294`. Remote repository is verified nonempty/private with main as its default branch.
- Initial Git history contains only the new standalone commit; no inherited ISC-CI commit or upstream remote was retained. The repository contains 51 pilot source/configuration/documentation/fixture files and zero Git LFS entries.
- Removed inherited experiment material from both the working tree and the published history. The original archive and discarded unpublished Git metadata remain only in the local `/private/tmp/abstract-task-feature-space-precleanup` recovery directory, outside this repository.
- Validation: 42 standalone tests pass; fresh full synthetic demo completes 2,250 generations and 1,200 resolved binary cells; reference byte hashes are preserved. GitHub Actions offline workflow was triggered by publication.
- Whitespace inspection flags the deliberately preserved CRLF reference CSV and the user's original Markdown line breaks; those source bytes were retained to preserve reference hashes and the requested PRD.
- No paid API calls or credentials were uploaded. Model/price selection and human-source scientific gates remain pending as documented.

## 2026-10-26 — User-requested reduction to three repetitions: RED

- User amended the sampling design from ten to three independent repetitions per target/prompt family to reduce API costs. Preserve all targets, prompt families, binary judging and source gates.
- New acceptance tests in `test_sampling.py`: defaults plan 675 combined / 135 task / 540 human generation requests; odd repetition counts execute successfully; matched comparisons use all three direct samples with equal per-family call budgets; even-count designs remain supported.
- Frozen odd-count rule: split each family's component budget as 2 task + 1 situation or 1 task + 2 situation, alternating the extra sample by frozen composition index and prompt-family index. This uses all three direct samples, costs no extra calls, and balances the extra component allocation across the six compositions. Retain all discoveries separately. Do not select allocations using generated outcomes.
- Expected RED: current defaults remain ten; preflight rejects odd counts; allocation helper does not exist. No paid calls; cumulative live spending remains $0.

## 2026-10-26 — Three-repetition reduction GREEN and refactor

- Observed RED: `python3 -m pytest -q tests/feature_space_pilot/test_sampling.py` → **3 failed** for the intended reasons: demo default 10 instead of 3; preflight rejected odd 3; missing matched-allocation helper.
- GREEN: the same command → **3 passed**. Defaults in live config, demo config and job planner are now three; preflight accepts integer counts >=2; all three direct replicates enter matched comparisons. At n=3, family-specific component allocations alternate 2+1 / 1+2, independently of outcomes; six compositions balance task and situation component calls at 27 each. Even n=2/4/10 behavior remains supported.
- Updated CLI/default-count assertions to the user-authorized new counts. Kept the explicit 10-repetition planner test as a supported nondefault regression. Small smoke remains two repetitions to avoid increasing its cost.
- Added `data/feature_space_pilot/sampling_amendment.json`, updated current setup/data documentation and README; preserved the original PRD and historical scratchpad records. Generation calls decrease 70%; binary judgment scope/cost is not assumed to decrease by that amount.
- REFACTOR: formatted changed source/tests with Black; full standalone regression is being checked. No paid model calls; $0 billed, $500 live cap remains.


## 2026-10-26 — Three-repetition final verification and publication

- REFACTOR/regression: `python3 -m pytest -q tests` → **45 passed**. New default/odd-count tests and existing even-count, retry, resume, source, binary, metric and CLI regressions all pass.
- Fresh offline run `demo_all_cebc8c4d0142b011` completed **675/675 generation jobs** and **1200/1200 binary cells**. All artifact hashes verified. Each of the six composition comparisons uses 9 component calls versus 9 direct calls; all three direct repetitions per family are included.
- Updated example preflights and sample run manifest. Human/task generation plans are now 540 and 135. Judgment upper bounds remain 12,000 and 3,000 respectively.
- Detected a concurrent user configuration edit setting `api_cap_usd` from 500 to **100**. Preserved it; the sampling implementation did not change that field. This supersedes earlier default-cap statements in this record. Live spend remains $0, so the current configured live cap is $100.
- Recorded the sampling amendment before any live evaluation. Original PRD and earlier completed artifacts are preserved as historical designs; the new default and sample report use three repetitions. No paid requests were made.
- Report: `/Users/mg9965/Desktop/abstract-task-feature-space/artifacts/feature_space_pilot/demo_all_cebc8c4d0142b011/pilot_report.md`; manifest: `/Users/mg9965/Desktop/abstract-task-feature-space/artifacts/feature_space_pilot/demo_all_cebc8c4d0142b011/manifest.json`; config SHA-256: `8d2ab30c00ef0c87ce04802c13126dcfcc029d5e56da66aaadaa90014afc06ba`.
- Next: commit and publish the reviewed sampling/config/documentation update to the existing private repository.

## 2026-10-26 — API environment-variable integration RED

- User requested adding the API key variable to the scripts. Implement automatic project-root `.env` loading for explicit live execution and provide `.env.example` with `OPENAI_API_KEY=`. Preserve any existing local `.env` and exported values; never put actual credentials into config, prompts, artifacts or Git.
- New `test_environment.py` criteria: named credentials only; exported values win; quotes/comments/export syntax; literal values with no shell execution; malformed-file errors omit credential text; missing keys fail before artifacts/reservations; live script resolves `.env` from configured root even from another directory; offline execution never reads credentials; no key values in captured output artifacts.
- Tests use fake sentinel credentials and a mocked provider only. No actual key is needed, read or submitted during implementation; no paid calls are authorized by this setup request.
- Next: capture RED, implement minimal loader, run targeted/regression checks, update setup docs and publish the template/code only.

## 2026-10-26 — API environment-variable integration GREEN / REFACTOR

- RED observed: new environment suite → **5 failed**: missing loader, missing early key check, live `.env` not loaded, and absent offline-loader hook. Fake fixtures only; no network requests.
- GREEN: `python3 -m pytest -q tests/feature_space_pilot/test_environment.py` → **5 passed**. Added a small standard-library loader for configured generator/judge key names, preserving exported values and treating `.env` values literally. No dependency added.
- The explicit live path validates preflight first, then loads keys from the configured project root before creating artifacts or reserving budget. Missing keys now fail before a request/reservation. Credential values are not returned or serialized; selected-entry parse errors contain names/line numbers only. Dry runs/offline/report paths do not load credentials.
- Added `.env.example` containing empty `OPENAI_API_KEY=`. Created a local ignored `.env` with the same empty variable only because no file existed; did not read an existing user key. The new local file uses owner-only permissions. No actual key was entered, read, committed or submitted.
- Updated setup/README/data dictionary. Verified `.env` is ignored/untracked and `.env.example` is publishable with no value. Existing three-repetition default and $100 live cap are preserved.
- Refactor: formatted new loader/tests with Black. Full offline regression verification is recorded next. No paid API calls; live spending $0.


## 2026-10-26 — API environment setup final verification

- Final REFACTOR/regression: `python3 -m pytest -q tests` → **50 passed**. Environment integration covers mocked live entry, different working directory, exported-key precedence, literal parsing, sanitized errors, no credential artifacts and no offline credential reads.
- Refreshed the read-only preflight example; key loading is never part of preflight and does not alter its config/design identity. Actual `.env` remains ignored and untracked.
- Publishing only the loader, integration tests, blank `.env.example`, docs, scratchpad and nonsecret preflight example. Model selections, source gates, three repetitions and the user's $100 cap are unchanged. No paid API requests were made.

## 2026-10-26 — Visualization explorer implementation started

- New user PRD is implemented downstream with a separate append-only record: `scratchpad_feature_space_visualizations.md`. It tracks TDD stages, source integrity and visual QA. User also requested the current GPT Sol default with low reasoning; official model verification is recorded there. Three repetitions and the $100 cap remain unchanged. No paid requests are part of this work.

## 2026-10-27 — User-authorized live API connectivity diagnostic

- User requested a quick real request before the full pilot. Executed the existing HostedClient with the existing project-local credential loader; key values/headers were never printed or saved. Official Sol documentation checked for Chat Completions/low reasoning; no model substitution or config change.
- Isolated ignored diagnostic script: `artifacts/feature_space_pilot/api_connection_test.py --execute-live`. At most one generation plus one judgment, no retries; $0.02 reserved per call against the shared cumulative live ledger and unchanged $100 cap. Full human-study source gates were not disabled or claimed satisfied.
- Actual outcome: **PASS**. P1 on the existing `poem_T` fixture produced seven features; original strict parser accepted them. One feature was judged `applies=1`, `status=ok`, with evidence; original judgment parser accepted it. Both responses completed normally using `gpt-6.1-sol`, low reasoning. 602 input tokens + 161 output tokens total; estimated spend **$0.002814**, settled in `budget_live.jsonl`, no unsettled reservations, remaining **$99.997186**.
- Evidence: `artifacts/feature_space_pilot/connection_test_4c27b62cd382/result.json` contains safe request IDs, actual response model, usage, parsed features/judgment and ledger summary. Not a scientific benchmark run and not included in source run matrices. User's new CSVs remain untouched/untracked; full integration, human source/protocol/translation verification and full-plan budget sizing remain outstanding.

## 2026-10-27 — CSV integration Stage 0 / test-first scope

- User requested integrating `human_norm_pilot_60_stimuli.csv` and `human_norm_pilot_abstract_24.csv` into the ordinary workflow. Inspected registry, preflight/planning, prompts, reference loading, human report/category analyses and source registry.
- Inputs: 60 unique IDs, with existing 36 Vinson labels and 24 abstract targets; abstract companion has the same 24 IDs. UTF-8 BOM/CSV input; English stimulus labels, German originals, publication row/hash metadata, four rounded category proportions and ratings. All 24 English abstract translations explicitly unreviewed. Preserve uploaded bytes and metadata; do not manufacture feature strings, verification or participant data.
- Implementation plan: hash-pinned CSV adapter plus companion cross-check; canonical source IDs and original source metadata; freeze outcome-independent held-outs for previously unassigned abstract rows, preserving all existing Vinson splits; category-proportion reference adapter; default config/source-registry/selection/docs updates. Dry runs must inspect all 60 targets/540 jobs and report remaining source/language/protocol/budget gates without crashing. Preserve low reasoning, three repetitions and $100 cumulative cap. No new paid calls.
- Baseline full suite started. Next RED: full inventory, exact metadata/category proportions, cross-file disagreement/invalid values/hash mutation, split determinism, prompt-label isolation and truthful live gates.

## 2026-10-27 — CSV integration RED and first implementation

- Baseline `.venv/bin/python -m pytest -q tests` → **72 passed in 51.24s**.
- New `test_human_csv.py` RED → **13 failed**: default config still selected JSONL, absent CSV adapter/hash pins, absent category-share adapter, and preflight only 36 targets. Added exact inventory/provenance and label-isolation assertions, deterministic abstract holdouts, eight malformed/cross-CSV input cases, byte-hash mutation guards, category-proportion unpivoting and truthful 60-target live gates before implementation.
- Implemented a hash-pinned CSV inventory adapter with strict schema/quotas/source/ID/numeric validation and full 24-ID companion agreement. Frozen abstract held-outs use sorted IDs, seed 42, one of six per each of four strata; existing 36 source splits are preserved. Uploaded bytes remain unchanged; human ratings/proportions never enter generation request fields.
- Added evaluation-only wide-to-long category-share loading, retaining German source labels, original rounded values and source rows; `category_counts` is the existing container name, with explicit `measurement=rounded_category_proportions`, not invented integer counts. Registry now records available user-supplied summary but retains `verified=false`, missing-publication/codebook/terms limits and unreviewed translations. Original supplement hash remains reported provenance rather than claimed verified bytes.
- Default config reads both supplied CSVs with SHA-256 pins. Inspectable nonlive planning uses an explicitly awaiting-review P0 placeholder for English abstract items; actual live P0 still requires reviewed translated instructions. Source, model, stage budget and cumulative cap gates remain enforced. Next: targeted GREEN and report/alignment integration tests.

## 2026-10-27 — CSV/report GREEN and explorer alignment RED

- Targeted importer/source/prompt/report suite → **35 passed in 6.54s**. Ordinary preflight now loads all **60** targets and all **540** plannable generation jobs; obsolete missing-sample blocker is gone. Remaining source/protocol, English/German translation and budget blockers are explicit. Supplied bytes and full prior ledger remain unchanged.
- Report/audit RED reproduced missing `item_availability` and absent audit CLI output (**3 failed**) before implementation. Added item-specific source/alignment reasons, unavailable recovery records, and no quantitative category profile until both item translation and exact source mapping are reviewed. Isolated reviewed fixture passes category shares, but still has no lexical recovery, feature RDM or participant reliability. Category coding remains independently reviewed; no generated labels are inferred.
- New offline audit CLI writes deterministic normalized 60-item JSONL, 96-row long-form category proportions and an input manifest, outside raw files. It does not load credentials or call APIs. Actual audit generated at `artifacts/human_input_audit/v1/` with 47 development / 13 held-out targets; original CSV abstract split tags remain unassigned in source bytes and are separately preserved as `source_split`.
- Explorer alignment RED (**1 failed, 1 passed**) caught its old blanket language rejection even for a valid reviewed mapping. Reused the same item/source alignment helper for report, preflight and explorer; unreviewed mappings remain blocked. No scientific input verification flags were changed in production.
- Selection rules explicitly describe supplied inventory selection, rather than claiming a new seeded draw from full source norms. Only abstract held-out assignment is seeded; all 36 prior splits remain frozen. Next: documentation, complete regression and final audit/dry-run verification. $0 additional API spending.

## 2026-10-27 — Complete CSV regression GREEN / exact-byte publication RED

- First full regression after CSV/report/explorer integration: **92 passed in 50.36s**. Includes all 72 prior tests plus inventory, cross-file/hash/numeric validation, category semantics, reviewed/unreviewed report and explorer cases, exact mapping rejection and deterministic offline audit export.
- Source-label ambiguity checks now reject duplicate abstract originals/translations before a many-to-many reference join. Existing language-gate regression initially expected the word `language` in its reason; clarified the production error to explicitly name stimulus language and required reviewed translation, preserving the original test.
- Actual 60-target audit and preflight verified; the eight-target human smoke now constructs **48 calls** and an existing conservative **$17.92** bound, with scientific source/protocol/translation blockers still enforced. No budget blocker for that smaller plan. Full 540-call pilot retains the conservative $1,755.60 bound and unchanged $100 cap; existing diagnostic ledger remains $0.002814 with no new paid calls.
- Publication RED found a substantive reproducibility bug: repository `* text=auto` would normalize the uploaded CRLF CSVs and invalidate their pinned hashes on checkout. Added `test_uploaded_csv_hashes_survive_git_checkout_without_line_ending_conversion` first; it failed with different filtered/raw Git object hashes. Added explicit `-text` attributes for both supplied CSV paths, preserving BOM and CRLF without rewriting uploaded bytes. Final regression running after this fix.
- Original completed demo manifest still verifies unchanged (1,200 resolved cells). Input-audit output hashes verify and original CSV SHA-256 values remain exactly pinned. Next: final regression, stage/review, commit/push and inspect CI. No paid study started.

## 2026-10-27 — CSV integration final GREEN and publication

- Final `.venv/bin/python -m pytest -q tests` → **93 passed in 49.90s**. All 72 prior tests remain green. Black checks all seven changed Python modules/tests; staged whitespace check accepts only the raw CSVs' intentional CRLF endings. Both staged Git blobs are byte-identical to the uploaded files, including BOM/CRLF, so hashes survive checkout.
- Default full human dry run: 60 selected items, 540 actual/planned generations, 12,000 judgment-cell bound. Missing-inventory gate removed; source/protocol/translation and full-budget blockers remain explicit. The existing eight-target smoke now plans 48 generation calls and $17.92 conservative bound; no scientific gate was weakened.
- Offline input audit delivered at `artifacts/human_input_audit/v1/`: 60 sealed stimuli, 96 exact rounded category-share rows, manifest/input/output hashes and 47/13 development/held-out assignment. Hashes verify, original CSV bytes and historical completed source run remain unchanged. Documentation covers commands, mappings, source limits and revision policy in `HUMAN_INPUTS.md`, README, setup guide and data dictionary.
- Zero new API calls during integration; cumulative diagnostic spend still $0.002814. Models, three repetitions, cap/bank and source verification status preserved. User-supplied data/code/config/tests/docs staged for the existing private repository; `.env`, environments, journals and generated audit outputs remain ignored.
- Next: commit/push the tested integration and inspect GitHub CI. Scientific full-run readiness still requires source/protocol/translation verification and budget sizing; integration itself is complete.
