# Feature-space pilot artifact dictionary

All run artifacts live under `artifacts/feature_space_pilot/<run_id>/`. Text uses UTF-8. Hashes are SHA-256. Structured objects are canonical JSON (sorted keys, compact separators, Unicode preserved, no NaN) when hashed; file hashes cover verbatim bytes. Human reference content is evaluation-only and never used as model context.

| Artifact | Fields / meaning |
|---|---|
| `stimuli.jsonl` | Frozen item order; `item_id,text,target_type,level,parent_id,task_id,situation_ids,source_dataset,language,selection_stratum,translation_status,content_hash`; optional `split,source_text`. |
| `generation_jobs.jsonl` | `job_id,item_id,item_hash,prompt_family,replicate,seed,seed_policy,model_id,model_revision,decoding,system,user,cap,protocol,prompt_version,prompt_hash`. Each row is one independent request. Reservation amount/ticket is attached per attempt in the ledger and response journal. |
| `generations.jsonl` | Append-only attempts: job/prompt/model/replicate/seed provenance; `raw_response,response_hash,parse_status,error,usage,latency_seconds,budget_ticket,attempt,record_hash`; successful rows add `features`. Empty lists are valid; invalid responses are retained. |
| `feature_candidates.csv` | Every raw phrase and `normalized,job_id,origin_item,prompt_family,replicate,raw_feature_index,feature_id,selected,normalization`. Selected=false records candidate-cap exclusions. |
| `feature_bank.csv` | Frozen `feature_id,text,cluster_id,inventory_hash,merge_decisions`. IDs derive from conservative canonical phrases. Inventory order is label-blind hash order, capped at 200. |
| `merge_review.csv` | `left,right,decision,reviewer,reason`. Similar wording is a pending review, never an automatic semantic merge. |
| `judgment_jobs.jsonl` | Frozen complete selected scope; item and feature ID, job ID, literal isolated prompt, prompt hash and rubric version. |
| `judgments.jsonl` | Append-only primary judge attempts with raw responses, usage and hashes; successful JSON adds `item_id,feature_id,applies,status,evidence`. `ambiguous/null` is retained. |
| `judgment_audit.jsonl` | Unresolved primary judgments. Invalid responses remain in `judgments.jsonl` with parse_status=error. |
| `binary_judgments.csv` | Unique resolved `(item_id,feature_id,rubric_version,judge_model)` rows; integer `applies` 0/1, `evidence,prompt_hash`. Human adjudications name the reviewer in judge_model. |
| `matrix_scope.json` | Frozen `item_ids,feature_ids`, declared policy and `excluded_combinations`. Scope is the complete stage cross product. Different stages are excluded, not padded with zeros. |
| `binary_matrix.csv` | `item_id` then columns by feature ID; every value is 0/1. File exists only after the complete scope is resolved. |
| `frozen_config.json` | Initial config snapshot; stores key environment variable names only. Never contains credential values. Local `.env` credentials are loaded only for explicit live execution and are never included as run artifacts. |
| `frozen_prompts.json` | Literal versioned template texts; constructed per-request texts also appear in job plans. |
| `frozen_source_registry.json` | Reference kind, source paths/URLs, codebook, language, instructions, permission constraints and hashes. Synthetic sources remain labeled synthetic. |
| `synthetic_human_reference.csv` | Offline-only read-only fake norm table, hashed and ingested by the reference loader. It does not represent human participants. |
| `preflight.json` | Initial read-only planning result copied into the run after execution begins: CSV/JSONL input byte hashes in `input_hashes`, complete/plannable counts, worst-case token and USD bounds, blockers, model capability deviations and design/preflight hashes. |
| `evaluation_inputs.json` | Current manual adjudications, relational review, category coding and embedding artifact references. Separate from frozen generation inputs. |
| `human_norm_comparison.json` | Per-item/family cap-normalized list-relative recovery, source-specific human/category profiles and reviewed generated profiles with coding coverage; source/subset geometry and participant split-half availability flags; `item_availability` reports missing source or unreviewed alignment per item. |
| `embedding_rdm_comparison.json` | Pinned vector-model provenance and generated/human geometry comparisons, or explicit unavailable reason. No automatic paid embedding calls. |
| `discovery_curves.csv` | Item/family/replicate cumulative produced and unique discoveries, lexical redundancy and confirmed valid unique yield. Unjudged propositions are not classified false. |
| `task_level_feature_audit.csv` | Every case feature at every level: source item/task/parent/situations, phrase, family/replicate/canonical ID, selection/applicability, inherited discovery, newly generated and newly applicable indicators. |
| `compositionality_audit.json` | Six explicit compositions. Component/direct banks, applicable baseline and union, discovered novelty, applicable gain, interaction candidates, manually confirmed interactions, overlap, equal-call counts, per-family `sampling_allocation` (task/situation/direct replicate counts), actual tokens and gain per call/token. Odd extra component samples alternate by frozen composition/family order. Also preserves full discoveries before equal-budget subsampling. |
| `interaction_review.csv` | Candidate task×situation relations requiring manual confirmation, reviewer and note. |
| `manual_relational_audits.json` | Explicit submitted reviewer/boolean/note decisions; a boolean without a reviewer is rejected. |
| `novel_feature_audit.csv` | Reproducible hash sample of generated propositions for blinded manual clarity/applicability/nonredundancy review. Sample is not automatically judged valid or false. |
| `judgment_disagreement_review.csv` | Sample of item/feature cells for independent manual binary review; primary judgment labels and source model are withheld. |
| `cost_report.json` | Cumulative accounted USD, remaining cap, unsettled full reservations, stage subtotals, provider-reported input/output tokens, billing/demo status and estimate limitations. |
| `reconciliation_fixture.json` | Synthetic inputs, integer-cent exact-ID/currency matching policy, expected report schema and deterministic reference output. No real customer data. |
| `decision_memo.json` | Scale-up recommendation and rationale; synthetic/unresolved runs cannot establish scientific success. |
| `pilot_report.md` | Human-readable results, six composition summaries/examples, assumptions, limits, spending and decision. |
| `manifest.json` | Design/config/source/prompt/fixture/model/seed provenance, optional git commit, planned/completed generation/judgment counts, inventory hash, journal checkpoints and every artifact hash. Status is complete, partial or needs_adjudication; demo flag distinguishes engineering from scientific completion. |
| `manifest_history.jsonl` | Append-only prior manifest snapshots when a partial run resumes or reviewed evaluation inputs change. |

`budget_live.jsonl` and `budget_demo.jsonl` sit above individual run folders. `reserve` entries bind a ticket/job/stage to a worst-case USD liability; `settle` entries replace the reservation with provider-token cost; `overage` records actual cost and stops execution. Unknown usage remains reserved. The adjacent lock file coordinates processes. Keep a single live ledger across the entire pilot.

Source inventory and selection files live in `data/feature_space_pilot/`; retain raw norm files read-only and separately from model prompts.


## Read-only visualization artifacts

`feature_space_pilot.viz` reads one verified run and writes separately under `artifacts/feature_space_visualizations/<run_id>/v1/`. `figure_manifest.json` records the exact source-manifest digest, evaluation fingerprint, visualization module hashes, renderer versions, analysis config/seed, selected counts, captions and underlying-data digests. Source runs and references are never resealed or altered.

The offline `index.html` gallery links PNG/SVG/PDF, tidy JSON/CSV, and `feature_inspector.html` (canonical propositions, source phrases/items/prompts/replicates, full target text, independent evidence and merge/manual-review provenance). Synthetic exports have `synthetic_` filenames and an explicit nonempirical banner. Gated or integrity-blocked figures get status pages rather than fabricated values.

Interactive/export state semantics: `1` applicable, `0` independently judged inapplicable, `?` unresolved/missing, `N/A` outside declared scope. Excluded candidates remain unjudged proposals; origin filters do not relabel applicability. See `VISUALIZATION_EXPLORER.md` for exact commands and interpretation rules.


## Supplied human CSV inputs

`human_norm_pilot_60_stimuli.csv` is the default model-target inventory; `human_norm_pilot_abstract_24.csv` is its cross-checked evaluation-only abstract companion. Both byte hashes are pinned in config and recorded as `input_hashes` in preflight/design/manifest provenance. Human category values and ratings are excluded from model prompts.

Imported registry items retain `source_dataset_original`, `source_language`, `source_text`, publication row/reported digest, translation status, raw CSV hash and `source_split`; canonical source IDs and deterministic frozen `split` assignments are separate. The source files themselves are never rewritten. The historical 36-item JSONL remains available for older configurations.

The offline input-audit CLI writes `stimuli.jsonl` (60 hashed items), `abstract_category_shares.csv` (96 source-item/category rows) and `input_manifest.json` (input/output hashes, counts and scientific limits) under a separate output directory. Its `count` field contains rounded proportions, explicitly marked `rounded_category_proportions`, rather than integer production counts. An audit is not a scientific run or publication verification. See [HUMAN_INPUTS.md](HUMAN_INPUTS.md).


## Scientific and budget preflight amendment

`Data Sheet 1.xlsx` is the pinned, read-only user-supplied Harpaintner workbook. `references/harpaintner_crosswalk.json` records the exact 24-row match against its 296-concept sheet; `publisher_download_verified=false` preserves chain-of-custody limits. Vinson raw/derived rows are independently cross-checked. The historical `budget_amendment.json` records English, GPT-5-mini-low, three repetitions, label-blind bank150, role token envelopes, <=$100 summed stage allocations and combined 75×150 scope.

`pilot_abstract_translation_review.csv` holds original German, published loose English, reviewed English, named reviewer and pending/approved/rejected decisions. It is hash-pinned; no pending entry is treated as reviewed. Registry `stimulus_condition`, `english_text`, `original_german` and `translation_review` preserve the declared cross-language condition. Vinson `text` adds the published noun/verb marker while `source_text` retains the norm lookup key.

`preflight --campaign` returns both stage plans, combined remaining liability plus actual cumulative prior spend, stage/cumulative blockers and readiness. Individual stage entrypoints also enforce combined budgets. Token audits distinguish payload bytes, model-tokenizer payload tokens, explicit framing reserve and conservative input bound; detailed tables are exported separately. Unknown mappings and oversize payloads fail before reservations/transport. Journals record `token_budget`; actual provider usage settles cost and envelope breaches stop execution.

`human_norm_comparison.json.category_geometry` contains conditional supplementary Euclidean RDMs of reviewed category proportions, common item/category order and item-bootstrap comparison. It requires >=4 aligned items and complete independent coding for compared generated profiles; it is separate from `geometry` binary-feature Jaccard. Unsupported feature recall/participant reliability remain unavailable.

`calibration_<id>/attempt.jsonl` and `result.json` are isolated explicit one-call diagnostics with real schema/usage/ledger settlement, not scientific pilot rows. The original supplied `pilot_prompt_size_audit.json` remains historical byte-only data, not the current tokenizer audit. Full preparation/execution instructions: [PREFLIGHT_WORKFLOW.md](PREFLIGHT_WORKFLOW.md).


## Progress and Batch execution

`configs/feature_space_pilot.json` now selects future Luna-low Batch runs; `feature_space_pilot_mini_online.json` preserves the earlier model settings. A frozen run's config remains its provenance. The new execution amendment is in `batch_execution_amendment.json`; the scientific v2 amendment is historical and unchanged.

`batch_state.json` is atomically written, content-hashed state with frozen-config hash, per-request task metadata/custom IDs, ledger tickets, exact payloads/token counts/prices, provider batch/file IDs, lifecycle status and ingestion markers. `batch_input_<key>.jsonl` is the hash-pinned API submission. `batch_output_file_id_<key>.jsonl` and `batch_error_file_id_<key>.jsonl` preserve raw provider responses. These are included in the ordinary sealed run inventory. Do not edit or delete state to retry an uncertain submission.

Primary `generations.jsonl`/`judgments.jsonl` retain the original schemas and hashes, adding `execution_mode`, `batch_id` and `batch_custom_id` for Batch attempts. IDs join unordered provider output to frozen jobs. Usage settles the existing shared ledger once. Provider-counted `token_budget` records distinguish exact provider counts plus reserve from offline byte-based planning bounds.

The `status` command produces read-only approximate live snapshots rather than sealed artifacts. Completed calls include ambiguous results; resolved 0/1 cells are counted separately. Ledger spending is cumulative across runs and includes outstanding reservations. New process progress messages use stderr; final JSON uses stdout. See [BATCH_AND_PROGRESS.md](BATCH_AND_PROGRESS.md).
