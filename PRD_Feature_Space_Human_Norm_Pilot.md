# PRD — Human-Norm Feature-Space Pilot and Two-Task Compositionality Study

**Status:** Proposed implementation specification (not implemented)  
**Date:** 2026-10-26  
**Repository:** `mohanwugupta/LLM_judge_item_expansion` (user-provided ZIP, `main`)  
**Scope:** Human-grounded feature-generation validation, followed by intensive studies of **creative writing** and **financial record reconciliation**.  
**Operating constraint:** Up to **$500** in API credits; use test-driven development (TDD) and explicit **RED → GREEN → REFACTOR** cycles.  
**Required working record:** `scratchpad_feature_space_pilot.md`, maintained throughout implementation.

## 1. Product goal and motivating question

Build a reusable pipeline for generating an interpretable **binary feature space** for concepts, tasks, and situations. Establish how its output corresponds to **human semantic-feature norms**, without treating those norms as ground truth or assuming that human-reported features determine an LLM's preferences. Then inspect two markedly different, realistic LLM workflows at multiple abstraction levels to assess **feature inheritance** and **task–situation compositionality**.

**Core scientific questions:**

1. Do free-generated feature sets for abstract concepts and event/action concepts recover interpretable patterns in human norms, including category composition and concept geometry?
2. Do different prompt families discover additional valid, reusable features rather than merely generate more synonyms or generic associations?
3. What features are discovered at the **domain**, **task type**, **specific task**, **situation**, and **situated task** levels?
4. Can the applicable features of a situated task be reconstructed from the features generated for its task and situation separately? What valid **relational or conjunctive** features are gained by prompting directly on the combination?
5. Do simple embedding-based semantic similarities agree with the resulting human and generated feature geometries? This is a **reference comparison**, not an optimization target.

**Not the goal in this PRD:** eliciting task preferences, determining LLM-internal feature use, training or retraining ISC-CI, extracting SAE features, modeling fMRI, causal tests of evaluations, or building a general large-scale task benchmark. Generated decision-relevant features are **hypotheses**, not demonstrations of model preferences.

## 2. Locked design choices

- **Final representation:** A binary, multi-hot `item × feature` matrix, exactly `0` (does not apply) or `1` (applies). Do **not** introduce necessary/typical/possible labels or numeric degrees into that matrix.
- **Feature content:** Short, specific propositions. Allow *genuinely relational/conjunctive properties*, e.g., `accuracy conflicts with satisfying the user's expressed preference`, as single testable binary predicates. Split only independent, accidentally bundled claims (`requires rhyme and has a deadline`); do not split a meaningful relationship (`X conflicts with Y`).
- **Scope of feature discovery:** Characteristics **and potentially decision-relevant** properties (operations, constraints, feedback, social context, rewards, goals, conflicts). A property is not positive or negative merely because it might matter to an agent.
- **Abstraction structure:** `domain → task type → concrete task instance` is a hierarchy. `situation` is an **independent axis**, and `situated task` combines an instance with a situation; it is not simply a fourth level of the same hierarchy.
- **Human role:** Benchmark human feature *production* and semantic organization, not ontological truth. Generated features absent from human lists can remain valid.
- **Pilot execution:** Freeze selected stimuli, reference-data provenance, prompt texts, seed policy, and reporting rules **before** evaluating outcomes. Do not tune prompts using held-out human results.

## 3. Repo audit and integration strategy

The uploaded repository contains the following existing components:

| Existing component | Pilot use / boundary |
|---|---|
| `leuven_expansion/generate_features.py`, `feature_prompts.py` | Reuse generation orchestration, independent samples, prompt hashes, validation patterns, CSV outputs, manifest, resume and preflight; **extend or adapt** for stable item IDs and longer descriptions. Do not disguise task descriptions as single Leuven words. |
| `leuven_expansion/prompts/feature_generation_prompt_A_v3_1_faithful.txt`, `...B_v3_1_first_to_mind.txt`, `...C_v3_1_individual_participant.txt` | Baselines and examples for human-style feature listing; do not claim exact replication of other studies' instructions without checking their methods. |
| `leuven_expansion/prompts/feature_generation_prompt_*_v4.txt`, `configs/v4_discovery.json` | Adapt V4 breadth/functional/situational discovery prompt ideas. |
| `build_v4_candidate_bank.py`, `ISC-CI_LLM_validation/consolidate_v3.py` | Reuse transparent normalization, provenance, merge-review concepts. Pilot gets its **own** frozen feature inventory, no mutation of previous V4 banks. |
| `leuven_expansion/feature_judge.py`, `feature_schema.py`, `run_v4_judgments.py` | Reuse client/robust job mechanics as appropriate, but implement **new binary applicability rubric/schema**. Existing Leuven `0–4` and production-frequency thresholds are not appropriate to copy unchanged. |
| `vllm_client.py` | Existing OpenAI-compatible **vLLM** client. Add a configurable API backend or verified adapter for billed models; do not assume this existing class is a general-purpose commercial API cost meter. |
| `tests/`, `scratchpad.md`, `todo.md` | Preserve existing tests/history. Add dedicated tests and **new append-only pilot scratchpad**; do not overwrite the original scratchpad. |

**Source-data warning:** The ZIP's Leuven CSVs under `data/` are Git LFS pointer files, not materialized tables. New scripts must detect and reject LFS pointers with an actionable error. Do not silently substitute them for human norms.

**Proposed new namespace:** `feature_space_pilot/` (source), `feature_space_pilot/prompts/`, `tests/feature_space_pilot/`, `configs/feature_space_pilot.json`, `data/feature_space_pilot/`, `artifacts/feature_space_pilot/<run_id>/`. Prefer thin adapters around existing utilities; preserve old experiment behavior.

## 4. Pilot A — Human-norm benchmark

### 4.1 Candidate human datasets and acquisition gate

1. **Vinson & Vigliocco (2008)**: English semantic feature norms for **456 words** including event verbs, event nouns, and object nouns. Target sample: **24 events/actions** (aim for 12 verbs and 12 event nouns, if norm coverage permits) plus **12 concrete object nouns**. Paper: https://doi.org/10.3758/BRM.40.1.183.
2. **Harpaintner, Trumpp & Kiefer (2018)**: human property-listing study of **296 German abstract nouns**, with property-category summaries and supplements. Target sample: **24 abstract concepts** distributed across the published semantic categories. Paper: https://doi.org/10.3389/fpsyg.2018.01748.
3. **Wiemer-Hastings & Xu (2005)**: conceptual reference for intrinsic/item versus contextual/situational responses; use quantitatively **only if suitable item-level source data can be verified**. Paper: https://doi.org/10.1207/s15516709cog0000_33.

**Mandatory gate before a live benchmark:** Locate and inspect the actual downloadable human data and codebook for each primary dataset, record URLs, release/license restrictions, exact columns, number of participants, response instructions, and whether raw response strings, per-item frequencies, or only category counts are available. Preserve verbatim reference files **read-only** with hashes. If item-level ground truth is unavailable, **do not fabricate or reconstruct it from article prose**. Run the supported category-level analysis and report what cannot be computed; substitute a suitable verified benchmark or reduce that stratum only with a documented amendment.

**Translation:** Keep Harpaintner source strings in German, preserve any published translations, and store a reviewed English translation mapping with ambiguous/polysemous items excluded or flagged. Record whether models see German originals or English translations; never mix them silently. Human/LLM lexical feature overlap is not meaningful without aligned language and documented mapping.

### 4.2 Freeze 60 stimuli

- 24 abstract concepts (Harpaintner), stratified across category profiles where available.
- 24 event/action concepts (Vinson & Vigliocco).
- 12 concrete object nouns (Vinson & Vigliocco) as continuity controls.
- Stable `item_id`, `item_text`, `source_dataset`, `source_language`, `item_class`, `selection_stratum`, `translation_status`, `parent_id=null`, `item_hash`.
- No cherry-picking after observing generated features. Reserve some stimuli per class for **held-out descriptive generalization** when sample sizes allow; otherwise avoid claiming out-of-sample validation.

### 4.3 Three prompt families

Run three independent conditions for each human target. Store **literal frozen texts** and hashes.

- **P0: Human-style free listing.** Adapt each study's documented participant instructions as closely as feasible, maintaining original feature limits where matched to humans; no human norm features supplied. For tasks in Pilot B use one predeclared general free-listing analogue. Tag separate datasets' matched protocols distinctly.
- **P1: Broad neutral feature discovery.** Up to 10 descriptive/relational features; encourages coverage without supplying individual features or assuming an agent's preferences.
- **P2: Decision-relevant candidate features.** Up to 10 properties that *could* matter to an agent choosing or executing the activity; operations, difficulty-related demands, feedback, risks, incentives, constraints and conflicts. Do not ask if the model likes the target, nor ask it to report its own preferences.

**Generation contract:** One item, one condition, one independent replicate per request; no previously generated features, other targets, normative feature columns, source category labels, evaluator outputs, or human similarity ratings in the prompt. Use 10 independent replicates initially per item/condition, with configurable temperature (e.g. `0.8`), paired seed design where supported, strict JSON output, max token and feature caps, failed-response retry, prompt version and complete provenance. Dataset-specific P0 output limits are recorded; **normalize or downsample counts for fair comparisons**, never compare raw counts as if identical instructions had been used.

### 4.4 Human-baseline analyses

- **Human recovery:** Match generated propositions to human features with frozen semantic-matching rules and sampled blinded/manual audits. Report precision/recall relative to *listed human features* only, explicitly **not truth**.
- **Feature-category profiles:** Compare sensorimotor, social, internal/emotional, verbal-associative, etc. **only using categories actually provided by the reference**. For events compare types of relations/roles if supported; do not force Harpaintner labels onto Vinson items without a separate declared coding rubric.
- **Geometry:** Construct matched item-by-feature representations where the source supports them. Compute item RDMs (document distance metric: cosine or Jaccard for binary matrices); Spearman correlations and bootstrap CIs over item resamples, with caveats about dependent item pairs. Report geometric results only when sufficiently item-level human data exist.
- **Embedding comparator (secondary):** Encode identical stimulus labels/descriptions with one pinned, documented embedding model. Compare RDMs with human- and LLM-feature geometries as an external reference. **Do not** optimize prompt selection to match embedding geometry.
- **LLM-only features:** Randomly audit novel generated propositions for clarity, applicability and nonredundancy instead of assigning them false solely because absent from human lists.
- **Reliability/saturation:** Feature counts, unique valid feature yield, semantic redundancy, overlap by prompt, discovery curves by replicate. Include human split-half estimates **only where participant-level source records allow**.

**Interpretation limit:** Matching human norm structure establishes correspondence to human semantic descriptions; it does not establish an LLM's internal feature usage or preference structure.

## 5. Pilot B — Two realistic LLM task case studies

**Case 1: Creative writing.** A user asks for an **eight-line poem about memory and change**, with a consistent rhyme scheme. The output is prose/verse assessed by task-appropriate quality criteria.

**Case 2: Financial reconciliation.** A user supplies a **small synthetic, internally consistent** set of invoices (e.g. 20) and payment records and asks for a discrepancy report listing matched, missing, partial and duplicate payments. The full input, matching policy, reference reconciliation, and expected report schema must be generated and version-controlled as fixtures. This is a workflow an assistant could genuinely perform; do not use real private financial records.

**Abstraction hierarchy for each task:**

| Level | Creative writing | Financial reconciliation |
|---|---|---|
| Domain | Creative production | Business administration / record analysis |
| Task type | Writing a poem | Reconciling invoices against payments |
| Concrete instance (`T`) | Eight-line rhyming poem on memory/change | Reconcile supplied 20-invoice/payment fixture and report discrepancies |

**Shared independent situation modifiers (`S`):**

- `S_deadline`: The user must receive the finished result within a specified short deadline.
- `S_review`: The work will be independently reviewed for quality against task-appropriate criteria.
- `S_both`: Both conditions together.

For each task, create `T`, `T+S_deadline`, `T+S_review`, `T+S_both`. The baseline `T` is the same frozen concrete-instance record, **not a duplicate stimulus**. Generate standalone features for the three situation descriptions as well. Total case-study targets = **15**: 6 domain/type/instance targets, 3 standalone situation targets, 6 non-baseline compositions. Apply P0 analogue, P1 and P2 to all targets in independent replicates.

No task-preference judgments or task executions are required for this phase. Realistic fixtures and prompts are used to make candidate properties concrete, not to test actual performance.

### 5.1 Compositionality hypotheses and tests

Let `G(X)` be the set of features **discovered** by generating for item `X`, and let `A(X, F)` be the subset of frozen candidate features `F` judged to apply to `X` (binary). Keep these operations distinct.

- **Component bank:** `C = G(T) ∪ G(S)`, consolidated under the frozen equivalence rules.
- **Direct-combination bank:** `D = G(T+S)` independently generated, without showing the component features.
- **Fair comparison:** Evaluate `A(T+S, C)` versus `A(T+S, C ∪ D)` using exactly the **same** independent binary judge and the same situated target. The incremental applicable features in the latter condition quantify what **direct composition discovers** beyond independent component generation.
- **Interaction candidate (more stringent):** A new relational/conjunctive proposition discovered from `T+S`, applicable to `T+S`, **not applicable to T alone or S alone under the same rubric**, and confirmed as describing a relationship between them (manual audit). This is an operational indicator, **not proof** of mathematical emergence or of a model's internal representation.
- **Comparators:** Hold generation-call counts, prompt families, and total sampling budgets constant between component and direct paths; analyze gain per call/token. Also quantify overlap, unique valid yield, Jaccard similarity, redundancy and error rates. Report each of the six compositions, rather than only aggregate means.
- **Conjunctions:** Preserve meaningful relational features as binary units (`meeting the deadline can conflict with checking every discrepancy`), not flattened into independent `deadline` + `checking` features. Prevent judge leakage of expected interactions.

**Manual case-study output:** For each of the two tasks, an auditable table showing every abstraction level's generated features, source prompt, merged canonical ID, binary applicability, inherited/shared features, new features at each level, and **component-only versus direct-combination** examples. Distinguish newly *generated* from newly *applicable* features.

## 6. Canonical data contracts

Required entities:

- `stimuli.csv` or `stimuli.jsonl`: `item_id`, `text`, `target_type` (`concept`, `event`, `object`, `domain`, `task_type`, `task_instance`, `situation`, `situated_task`), `level`, `parent_id`, `task_id`, `situation_ids`, `source_dataset`, `language`, `selection_stratum`, `content_hash`.
- `generation_jobs.jsonl`: `job_id`, `item_id`, `prompt_family`, `prompt_version/hash`, `model_id/revision`, `replicate`, `seed`, decoding settings, budget reservation.
- `generations.jsonl`: raw response, parsed features, parse status, usage tokens, latency when available, response/job hashes; append-only, idempotent resume.
- `feature_candidates.csv`: phrase, canonical candidate ID, origin item, prompt, replicate, normalization and raw provenance.
- `feature_bank.csv`: immutable, canonical feature IDs/text, cluster assignments, merge decision/provenance, frozen bank hash. **Do not silently merge** features with differing agent roles, negation, modality or conditions; keep human review of uncertain merges.
- `binary_judgments.csv`: unique `(item_id, feature_id, rubric_version, judge_model)` records, strict `0` or `1`, evidence note, prompt hash; failures and disagreements retained in separate audit files. Reject missing/ambiguous judgments from final matrices until adjudicated; never default missing to zero.
- `binary_matrix.csv`: rows by frozen item order, columns by feature ID, values **only 0/1**, complete for explicitly declared item×feature scope. Store excluded out-of-scope combinations separately rather than filling them with zeros.
- `manifest.json`: source and config hashes, model snapshots, prompt hashes, run seed, selected items, planned/completed jobs, price assumptions, token usage, bill estimate, git commit if available, generated artifact hashes.

Store human norms **separately and read-only**. Model generators and applicability judges must not receive their content. Keep human reference/category mapping in evaluation-only code.

## 7. Production phases and delivery order

**Phase 0 — Source audit and frozen design:** verify norm data availability, language and human instructions; create source registry, 60-item selection rules, 15 case targets, prompts, budgets, config and a sample expected run manifest. Clearly flag any inaccessible primary norms as a blocker, not a successful pilot.

**Phase 1 — Pure-function infrastructure:** implement item schema, hashing, deterministic job planning, record validation, independent prompt construction, budgets and unit-test fixtures. No paid API calls.

**Phase 2 — Generator adapter and smoke:** extend existing generation mechanics for arbitrary target descriptions and stable IDs. One fake-client integration test, then a tiny paid smoke test (one human item per class plus one task/situation pair) only after test and budget gates.

**Phase 3 — Candidate consolidation and binary judging:** pooled candidate bank, audited merges, binary rubric, frozen cap with preregistered selection independent of human match, judge harness, disagreement audit; pilot on a tiny matrix before production.

**Phase 4 — Human pilot:** freeze selected sample and run P0/P1/P2. Produce human recovery, category profiles, saturation, geometry (where supported) and embedding reference.

**Phase 5 — Two-task case studies:** run all 15 targets; judge features and produce level-by-level and compositionality reports with concrete examples.

**Phase 6 — Reproducibility and decision memo:** automatic report, manifests, data dictionaries, tests, source constraints, cost accounting, and a succinct **go / revise / stop** recommendation for scale-up.

**Important:** Phase 4 must not be tuned using downstream case-study results; store each pilot's outputs separately.

## 8. Budget and execution safeguards

- Hard configured cumulative API cap **≤ $500** for this pilot, covering generation, similarity/merge assistance, judging and adjudication. No live calls until a **model-specific price table**, input/output token assumptions, per-stage budget and fail-closed token accounting are specified. Check model support for `seed` and structured outputs; log deviations.
- Initial planned generation: `(60 human + 15 case-study items) × 3 prompt families × 10 independent replicates = 2,250 calls` **before** judgements or retries. P0 may have dataset-specific feature caps. Emit accurate per-stage token/cost estimates from dry-run, not unsupported dollar-price claims.
- Suggested allocation ceilings: up to $120 generation, $220 primary binary judgments, $80 adjudication/semantic merge auditing, $30 embeddings/analysis if billed, $50 reserve. If estimated total exceeds cap, shrink the **frozen candidate-bank size or the number of audited cells** through documented, label-blind rules; don't silently skip pairs after starting. Keep a maximum number of candidates (configurable, initially 200) and report the cost/coverage trade-off.
- Never perform a paid full cross-product without explicit `--execute-live` and a successful `--dry-run` preflight showing expected jobs and worst-case spending estimate. Idempotent resume, retries, rate limits, partial-failure reporting and provider-usage audit required.
- Real invoices/payment fixtures must be synthetic, not uploaded customer data. Do not log API keys or secrets.

## 9. TDD: mandatory RED → GREEN → REFACTOR protocol

**Every implementation slice starts with a failing automated test** that expresses its acceptance criterion. Demonstrate that it fails for the intended reason (**RED**); write the minimum production code needed (**GREEN**); rerun targeted and relevant existing tests; refactor with tests green (**REFACTOR**). No test-first claim without actual failing-test evidence. Never alter assertions merely to make a failure disappear, and never mask failures with skips or `xfail` after the fact.

**Test order and key fixtures:**

| Slice | RED test(s) to write first | Required GREEN behavior |
|---|---|---|
| Item registry | Duplicate IDs, missing parent, unknown situation, wrong target type, LFS pointer | Versioned, validated inputs; deterministic ordering; explicit errors |
| Human reference loader | Missing file, unsupported columns, misaligned language, synthetic reference masquerading as human | Documented source adaptation, provenance and source hashes |
| Prompt builders | Leakage of human norms / source labels / other target; over-cap output; missing role distinctions | Single-item isolated, stable prompt/hash, correct P0–P2 routing |
| Job planner | Same replicate mistakenly deduped; paired seeds inconsistent; resume after interruption | Stable unique IDs, reproducible plan, correct 2,250-call default and exact resume |
| Response parser | Malformed JSON; wrong item; duplicate phrases; unintended multi-claim; empty valid output | Strict schema, preserved raw response, transparent normalization and error log |
| Consolidation | Negation/agent-role/conditional mismatch merging; provenance lost | Conservative canonical IDs, reviewed ambiguity, stable inventory hash |
| Binary judge | 0–4 Leuven values accepted; missing judged as 0; irrelevant source leakage; duplicate cells | Exactly 0/1, isolated `(item, feature)` prompts, explicit adjudication and scope |
| Budget/controller | Missing prices, budget overshoot, partial retries, dry-run writes or charges | Fail closed, token accounting, dry-run no API calls, safe resume |
| Human metrics | No item-level source for RDM; unmatched translation; zero-vector distance; repeated item-pair CI error | Correct metric availability flags, documented geometry and uncertainty |
| Compositionality | Treats `G(T+S)` discovery novelty as emergent; unequal sample budgets; duplicated baseline | Separate discovered/applicable sets, valid union comparison, interaction audit |
| Report integration | Missing fixture or manifest, stale hashes, incomplete cells | Deterministic complete sample report and reproducible artifacts |

**Test layers and commands:** Use `pytest -q tests/feature_space_pilot` for targeted tests and `pytest -q tests` for regression checks. Use fake API clients and tiny synthetic fixtures for all unit/integration tests; add one opt-in billed smoke test with live output capture. Tests must run offline and in CI without credentials. Track stage-specific snapshots/golden fixtures for small expected feature matrices and compositionality sets. Do not require ISC-CI training or GPUs.

**Definition of TDD completion:** For each delivered slice, the scratchpad contains the name of a newly added test, observed RED failure and why it is expected, observed GREEN pass, any REFACTOR rerun, and the changed files/commit when applicable. A final all-green suite alone does not substitute for the RED record.

## 10. Mandatory scratchpad protocol

Create **`scratchpad_feature_space_pilot.md`** at the repository root on the *first implementation step*; leave the existing `scratchpad.md` unchanged except optionally adding a link. It must be maintained **incrementally** throughout every coding/work session, not reconstructed at the end. At minimum, each dated entry includes:

1. **Goal and current hypothesis** (what the slice is testing and why).
2. **Source assumptions and unresolved scientific decisions** (including dataset availability and human/LLM norm distinctions).
3. **RED:** exact test name/command, expected behavior, captured failure reason.
4. **GREEN:** smallest production change and exact passing command/result.
5. **REFACTOR:** cleanup and regression command/result.
6. **Files changed**, config/prompt versions, input/output hashes and decisions, including rejected alternatives.
7. **Run provenance and spending:** preflight count, calls completed, billed/predicted token usage, cumulative cost, remaining budget, manifest location.
8. **Bugs/risks and next action**. Record failed approaches and partial runs as well as successes.

Use dated headings and append rather than rewriting old history. If the work stops midway, the scratchpad must tell the next contributor exactly how to resume. No generated result should be described as completed unless an artifact and manifest exist.

## 11. Acceptance criteria (must pass before scale-up)

**Engineering acceptance:**

- Existing pytest regression suite remains green, plus complete pilot tests with documented RED→GREEN cycles.
- A full offline end-to-end run works on fake human datasets and fake model responses; strict 0/1 matrix, correct manifest, idempotent resume, complete cost accounting.
- Live human benchmark and case-study stages run only if their source and budget preflights pass, and all calls/versions are traceable.
- No prior Leuven/V4 data or model checkpoints changed; source human data retained read-only.

**Scientific/reporting acceptance:**

- Human sample and reference-data availability documented; P0/P1/P2 comparisons and limitations reported without portraying human norms as truth.
- Comparative feature-category profiles and reliability/discovery coverage reported for all supported strata; geometric/embedding comparisons reported where source data support them, otherwise clearly marked unavailable.
- For **both real-world task families**, complete level-by-level feature tables and all three situation conditions are presented with examples.
- Compositionality report explicitly separates **newly generated** features from **newly applicable relational** features; includes component-union baseline, fixed judge comparisons, and a manual audit of claimed interaction features.
- Deliver a concise decision memo: what seems valid, what fails, which prompt/level combinations are worth scaling, and what data would be necessary for the later preferences study. Do not require an arbitrary geometry threshold to declare scientific success.

## 12. Planned deliverables / CLI UX

Target (proposed; implementations may use equivalent names with documented mapping):

```bash
python -m feature_space_pilot.preflight --config configs/feature_space_pilot.json
python -m feature_space_pilot.run --config configs/feature_space_pilot.json --stage human --dry-run
python -m feature_space_pilot.run --config configs/feature_space_pilot.json --stage tasks --dry-run
pytest -q tests/feature_space_pilot
pytest -q tests
# Explicit billed execution, only after approved preflight:
python -m feature_space_pilot.run --config configs/feature_space_pilot.json --stage human --execute-live
python -m feature_space_pilot.run --config configs/feature_space_pilot.json --stage tasks --execute-live
python -m feature_space_pilot.report --config configs/feature_space_pilot.json
```

Expected artifacts: validated `stimuli`, frozen prompts, human-source registry, raw `generations`, canonical `feature_bank`, `binary_judgments`, `binary_matrix`, `human_norm_comparison`, `embedding_rdm_comparison`, `task_level_feature_audit`, `compositionality_audit`, `cost_report`, `manifest`, `pilot_report.md`, and **`scratchpad_feature_space_pilot.md`**.

## 13. References and source constraints

- Vinson & Vigliocco (2008), *Semantic feature production norms for a large set of objects and events*, https://doi.org/10.3758/BRM.40.1.183.
- Harpaintner, Trumpp & Kiefer (2018), *The Semantic Content of Abstract Concepts*, https://doi.org/10.3389/fpsyg.2018.01748.
- Wiemer-Hastings & Xu (2005), *Content Differences for Abstract and Concrete Concepts*, https://doi.org/10.1207/s15516709cog0000_33.
- Source repository: https://github.com/mohanwugupta/LLM_judge_item_expansion.

**Dataset acquisition, exact historic prompt wording, use permissions, model selection, and current provider pricing require verification during Phase 0.** Do not claim that norms, API credits, or an end-to-end run are already available merely because this PRD specifies them.
