# Expanded pilot: state audit before implementation

Audit recorded 27 October 2026 (America/New_York). Running collectors are allowed to advance, so provider counts and accounted liabilities will change.

## Data that can be reused

| Source | Existing evidence | Expanded-pilot treatment |
|---|---|---|
| Human run `live_human_74608725aae4e0dc` | 60 frozen items; 540 successful generation jobs; 5,653 phrases / 5,405 normalized strings | Reuse every accepted generated feature, including uncapped Vinson P0 outputs beyond the separate ten-feature comparison cap. Preserve generation provenance and audit rejected attempts. |
| Original human applicability | 9,000 planned cells; 7,208 resolved, 1,759 ambiguous, 33 missing | Keep as the original 150-feature baseline. Mini judgments cannot be relabeled Luna; no cross-model reuse. Ambiguous/missing remain unresolved. |
| Task run `live_tasks_96fb5b19b4c8b9ef` | 15 frozen targets; 135 generation requests already submitted | Latest provider read: 133/135 completed, batch still in progress. No terminal response file collected yet; no task propositions can yet be claimed as observed. Preserve the original forthcoming 150-feature run. |
| Human-norm matching `semantic_8f8897426b82de20` | Target-conditioned generated-to-human-label judgments; 406/606 requests collected | Evaluation-only correspondence, not feature-bank consolidation. It covers the original first-ten-features cap and 36 Vinson targets. Do not use it to merge the global vocabulary. |
| Similarity baseline `baseline_320fefced4bd9a02` | 4,184 target-context embeddings; lexical scores | Reference diagnostics only. Target-context embeddings and high similarity are not proof of proposition equivalence. |

Existing `bank.py` pools normalized strings, orders them by hash, and selects 150. Its semantic-review suggestions only inspect the selected inventory. The full pilot needs a separate uncapped bank, whole-vocabulary provenance, conservative equivalence decisions, and explicit unresolved proposals. Existing primary artifacts and code pins must remain untouched.

## Running work and liabilities

Four detached collectors were verified alive: original human **51688**, human-norm matching **51700**, task **53088**, baseline **55545**. Read-only provider audit found 19 locally known chunks: 16 complete, the task-generation chunk and one 200-request semantic-matching chunk in progress, plus the original unconfirmed ten-request Mini chunk with no matching provider acknowledgement. Preserve its reservations; never resubmit it blindly.

Shared ledger at this checkpoint: **$2.89144988 accounted**, comprising **$2.68382713 settled** and **$0.20762275 held**. Stage allocations remain generation10 / judgment65 / adjudication10 / embeddings5 / reserve10 = **$100**. These are conservative ledger estimates; provider invoices remain authoritative. No new paid jobs are authorized to bypass the new complete-bank/cost preflight.

## Work still required

1. Snapshot the existing studies read-only, preserving complete and failed attempts and all frozen item identities. The eventual global scope is exactly **75 items**, with one shared feature vocabulary.
2. Build a deterministic uncapped bank. Separate safe normalization/equivalence rules, explicitly reviewed equivalences, non-equivalent relations, and uncertain proposals. Keep rare features and every raw origin. Embedding similarity alone cannot merge anything.
3. Plan all item × canonical-feature cells. Reuse only resolved evidence whose item content, proposition, rubric, actual model/request configuration and literal prompt are compatible. Detect externally pending jobs and avoid duplicate submissions. Keep incompatible Mini evidence as a baseline.
4. Use measured provider input-count evidence where available and realistic existing output envelopes. Report primary and retry bounds, prior liabilities, eligible reuse, unresolved cells and source-completion blockers. Preserve the $100 cap and existing allocations; an unaffordable whole-bank workload remains blocked, with no new subset.
5. Extend source-preserving execution, resume and reports for the global matrix. Keep binary values exactly 0/1 and represent incomplete data separately.
6. Report full and equal-call composition comparisons, explicit feature text, discovery gaps versus applicability interactions, human-reference availability, saturation and geometry only where supported. A proposed interaction still needs named interpretation before confirmation.

The initial vocabulary is provisional until the already-submitted task generations are collected. A final consolidated count or full completion price cannot honestly be claimed before those source data and equivalence decisions are available. Interim costed plans must clearly identify this limitation.

## Implementation boundary

Use `feature_space_pilot/full/` and a separate full-pilot config/output root. Do not modify the old `bank.py`, active configuration, frozen manifests, original bank IDs or running collectors. New tests must demonstrate RED → GREEN before substantial production changes. Updated analyses remain candidate descriptive representations, not evidence of features causally used by models in decisions.

## Updated preparation checkpoint

During implementation, the original task-generation Batch completed and its collector saved **130 accepted generations out of 135**. The collector submitted the remaining five invalid-generation retries as `batch_6ae16751e3548190aba9de54f2e3bdd0`; a read-only provider check found it in progress, zero of five completed. Nothing was interrupted, resubmitted by the full workflow, or overwritten.

The complete available vocabulary now contains **6,518 accepted phrases / 6,046 normalized and provisional canonical features**, spanning human and task discoveries. Narrow rules have not reduced the column count at this checkpoint; automated semantic assistance remains unrun. The shared scope is **75 × 6,046 = 453,450 cells**, all explicitly unresolved under the global Luna configuration. Historical Mini labels are incompatible for reuse. Their absence from the new matrix is not evidence that the original data were lost.

Measured-template first-pass bound: **$58.48511935**. Diagnostic all-two-attempt envelope: **$116.97023870**. The full workflow explicitly funds all primary cells first and only then preflights the entire known invalid-retry stage; it does not promise that every possible retry is funded. Current proposed equivalence assistance: 1,996 requests / up to 3,992 attempts, **$2.92521220** byte-bound estimate pending exact provider counts and completed source generations. Final task additions and semantic merges can change these figures. No expanded paid job was submitted during this preparation checkpoint.

Shared accounted liability is now **$2.74630463**, comprising **$2.73774988 settled** and **$0.00855475 held** across 18 tickets. The human-norm matcher finished **606/606** requests; all **2,944/2,944** propositions are scored. The embedding baseline also completed its analysis. Their collectors exited after completion; original human PID51688 and task PID53088 remain alive. Their artifacts and code pins remain untouched.

Prepared reports include complete proposition/provenance tables for both task studies, the explicit blocked-cell list, automated human correspondence, and bank-coverage/discovery figures. Global geometry and confirmed composition effects remain unavailable without their independent binary cells and reviews. These are data-completion gates, not assertions of scientific success.
