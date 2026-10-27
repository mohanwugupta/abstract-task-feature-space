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
| `preflight.json` | Initial read-only planning result copied into the run after execution begins: complete/plannable counts, worst-case token and USD bounds, blockers, model capability deviations and design/preflight hashes. |
| `evaluation_inputs.json` | Current manual adjudications, relational review, category coding and embedding artifact references. Separate from frozen generation inputs. |
| `human_norm_comparison.json` | Per-item/family cap-normalized list-relative recovery, source-specific human/category profiles and reviewed generated profiles with coding coverage; source/subset geometry and participant split-half availability flags. |
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
