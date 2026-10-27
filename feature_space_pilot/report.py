"""Deterministic reports, availability flags and artifact integrity verification."""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from .io import canonical, digest, file_hash, read_jsonl, write_csv, write_json
from .metrics import (
    category_profile,
    discovery_curves,
    geometry_comparison,
    human_recovery,
    rdm,
    split_half,
    unavailable,
)
from .composition import compare_composition, matched_sampling_allocation
from .prompts import normalize_phrase


def seal_manifest(out):
    out = Path(out)
    m = json.loads((out / "manifest.json").read_text())
    m["artifacts"] = {
        p.name: file_hash(p)
        for p in sorted(out.iterdir())
        if p.is_file() and p.name != "manifest.json"
    }
    write_json(out / "manifest.json", m)
    return m


def verify_manifest(out):
    out = Path(out)
    m = json.loads((out / "manifest.json").read_text())
    actual = {
        p.name for p in out.iterdir() if p.is_file() and p.name != "manifest.json"
    }
    if actual != set(m["artifacts"]):
        raise ValueError("Artifact inventory/hash mismatch")
    for name, h in m["artifacts"].items():
        if file_hash(out / name) != h:
            raise ValueError(f"Stale artifact hash: {name}")
    return m


def _references(c, items):
    from .engine import _source_specs, resolve
    from .references import load_reference
    from .clients import FakeClient

    human = [i for i in items if i["target_type"] in {"concept", "event", "object"}]
    refs = {}
    if c.get("demo"):
        if c.get("_evaluation_reference_spec"):
            refs["synthetic_demo"] = load_reference(
                c["_evaluation_reference_spec"], live=False
            )
        else:
            refs["synthetic_demo"] = dict(
                kind="synthetic",
                data_level="item_features",
                language="en",
                rows=[
                    dict(
                        item=i["text"],
                        feature=f,
                        category=i["target_type"],
                        frequency=1,
                    )
                    for i in human
                    for f in FakeClient.properties(i["text"])
                ],
            )
    else:
        for spec in _source_specs(c):
            if spec.get("path"):
                spec = dict(spec, path=str(resolve(c, spec["path"])))
                try:
                    refs[spec["dataset_id"]] = load_reference(spec, live=True)
                except (ValueError, OSError):
                    continue
    return refs


def _human_analysis(c, items, generations, bank, judgments):
    from .engine import resolve
    from .prompts import generation_prompt

    human = [i for i in items if i["target_type"] in {"concept", "event", "object"}]
    refs = _references(c, items)
    coding = {}
    if c.get("category_coding_file"):
        path = resolve(c, c["category_coding_file"])
        if (
            not c.get("category_coding_sha256")
            or file_hash(path) != c["category_coding_sha256"]
        ):
            raise ValueError("Category coding artifact hash mismatch")
        by_item = {i["item_id"]: i for i in human}
        valid_features = {f["feature_id"] for f in bank}
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                item = by_item.get(row.get("item_id"))
                if (
                    not item
                    or row.get("feature_id") not in valid_features
                    or not row.get("reviewer")
                ):
                    raise ValueError(
                        "Invalid reviewed category item, feature or reviewer"
                    )
                ref = refs.get(item["source_dataset"])
                allowed = (
                    {r.get("category") for r in ref["rows"] if r.get("category")}
                    if ref
                    else set()
                )
                if row.get("category") not in allowed:
                    raise ValueError("Unknown source category in reviewed coding")
                coding.setdefault((row["item_id"], row["feature_id"]), set()).add(
                    row["category"]
                )
    phrase_ids = {normalize_phrase(f["text"]): f["feature_id"] for f in bank}
    recoveries = []
    profiles = []
    geometry = []
    reliability = []
    cells = {
        (r["item_id"], r["feature_id"]): r["applies"]
        for r in judgments
        if r.get("status") == "ok"
    }
    # Same output cap for within-item family comparisons; no outcome-based prompt choice.
    for item in human:
        ref = refs.get(item["source_dataset"])
        if not ref:
            continue
        rows = [
            r for r in ref["rows"] if r["item"] == item.get("source_text", item["text"])
        ]
        if ref["data_level"] == "category_counts":
            profiles.append(
                dict(
                    item_id=item["item_id"],
                    human_profile=category_profile(
                        {r["category"]: float(r["count"]) for r in rows},
                        allowed_categories={r["category"] for r in rows},
                    ),
                    generated_profile=unavailable(
                        "Requires blinded manual coding in the source category vocabulary"
                    ),
                )
            )
            for family in ["P0", "P1", "P2"]:
                recoveries.append(
                    dict(
                        item_id=item["item_id"],
                        prompt_family=family,
                        **unavailable(
                            "Only category summaries available; no feature-string recovery"
                        ),
                    )
                )
            continue
        cap = min(
            10, generation_prompt(item, "P0", protocols=c.get("protocols"))["cap"]
        )
        for family in ["P0", "P1", "P2"]:
            group = [
                r
                for r in generations
                if r["item_id"] == item["item_id"] and r["prompt_family"] == family
            ]
            generated = [f for r in group for f in r["features"][:cap]]
            aligned = item["language"] == ref["language"]
            if aligned:
                recovery = human_recovery(
                    generated,
                    [r["feature"] for r in rows],
                    aligned_language=True,
                    equivalences=c.get("semantic_equivalences", {}),
                )
            else:
                recovery = unavailable(
                    "Feature-language alignment/reviewed feature mapping unavailable"
                )
            recoveries.append(
                dict(
                    item_id=item["item_id"],
                    prompt_family=family,
                    comparison_cap=cap,
                    split=item.get("split", "unspecified"),
                    **recovery,
                )
            )
        categories = {r.get("category") for r in rows if r.get("category")}
        if categories:
            counts = {
                cat: sum(
                    float(r.get("frequency", 1))
                    for r in rows
                    if r.get("category") == cat
                )
                for cat in categories
            }
            profiles.append(
                dict(
                    item_id=item["item_id"],
                    human_profile=category_profile(
                        counts, allowed_categories=categories
                    ),
                    generated_profile=unavailable(
                        "Requires blinded manual coding; no automatic label transfer across sources"
                    ),
                )
            )
    for profile in profiles:
        profile.pop("generated_profile", None)
        iid = profile["item_id"]
        profile["generated_profiles"] = {}
        profile["coding_coverage"] = {}
        for family in ["P0", "P1", "P2"]:
            ids = {
                phrase_ids[normalize_phrase(f)]
                for r in generations
                if r["item_id"] == iid and r["prompt_family"] == family
                for f in r["features"]
                if normalize_phrase(f) in phrase_ids
            }
            valid = {f for f in ids if cells.get((iid, f)) == 1}
            coded = {f for f in valid if (iid, f) in coding}
            counts = {}
            for fid in coded:
                for category in coding[iid, fid]:
                    counts[category] = counts.get(category, 0) + 1
            profile["generated_profiles"][family] = (
                category_profile(counts, allowed_categories=set(counts))
                if counts
                else unavailable(
                    "No reviewed source-category labels for applicable generated features"
                )
            )
            profile["coding_coverage"][family] = dict(
                coded_applicable_features=len(coded),
                applicable_features=len(valid),
                interpretation="Profiles cover independently coded features; overlapping category memberships retained",
            )
    for dataset, ref in sorted(refs.items()):
        selected = [i for i in human if i["source_dataset"] == dataset]
        reliability.append(
            dict(
                source_dataset=dataset,
                **split_half(
                    ref["rows"] if ref["data_level"] == "participant_features" else None
                ),
            )
        )
        for subset in ["all", "development", "held_out"]:
            group = [i for i in selected if subset == "all" or i.get("split") == subset]
            if ref["data_level"] == "category_counts":
                result = unavailable(
                    "Category summaries do not support item-by-feature geometry"
                )
            else:
                features = sorted({r["feature"] for r in ref["rows"]})
                observed = {
                    (r["item"], r["feature"])
                    for r in ref["rows"]
                    if float(r.get("frequency", 1)) > 0
                }
                h = [
                    [
                        int((i.get("source_text", i["text"]), f) in observed)
                        for f in features
                    ]
                    for i in group
                ]
                g = [
                    [cells.get((i["item_id"], f["feature_id"])) for f in bank]
                    for i in group
                ]
                if any(v is None for row in g for v in row):
                    result = unavailable("Incomplete judged cells")
                else:
                    result = geometry_comparison(
                        h,
                        g,
                        bootstrap=c.get("bootstrap_resamples", 200),
                        seed=c["seed"],
                    )
            geometry.append(
                dict(
                    source_dataset=dataset,
                    subset=subset,
                    item_ids=[i["item_id"] for i in group],
                    **result,
                )
            )
    return dict(
        reference_role="Human production norms, not ontological truth",
        synthetic=bool(c.get("demo")),
        recovery=recoveries,
        category_profiles=profiles,
        geometry=geometry,
        split_half=reliability,
        unavailable_reason=(
            None if refs else "Verified human references not available in this stage"
        ),
        category_coding_rule="Only supplied source categories; generated-category coding requires independent review",
    )


def _embedding_analysis(c, items, bank, judgments):
    if not c.get("embedding_file"):
        return unavailable(
            "No pinned embedding model/vector artifact configured; secondary comparator not run"
        )
    from .engine import resolve

    p = resolve(c, c["embedding_file"])
    if not c.get("embedding_sha256") or file_hash(p) != c["embedding_sha256"]:
        raise ValueError("Embedding artifact hash mismatch")
    record = json.loads(p.read_text())
    if (
        not record.get("model_id")
        or not record.get("revision")
        or not record.get("source_url")
    ):
        raise ValueError("Pinned documented embedding model required")
    by_id = {r["item_id"]: r for r in record["items"]}
    if len(by_id) != len(record["items"]) or set(by_id) != {
        i["item_id"] for i in items
    }:
        raise ValueError("Embedding item alignment mismatch")
    vectors = []
    for item in items:
        if by_id[item["item_id"]]["text_hash"] != digest(item["text"]):
            raise ValueError("Embedding stimulus text mismatch")
        vectors.append(by_id[item["item_id"]]["vector"])
    cells = {
        (r["item_id"], r["feature_id"]): r["applies"]
        for r in judgments
        if r.get("status") == "ok"
    }
    if any((i["item_id"], f["feature_id"]) not in cells for i in items for f in bank):
        return unavailable("Incomplete feature geometry")
    g = [[cells[i["item_id"], f["feature_id"]] for f in bank] for i in items]
    human_comparisons = []
    for dataset, ref in sorted(_references(c, items).items()):
        subset = [i for i in items if i["source_dataset"] == dataset]
        if ref["data_level"] == "category_counts":
            comparison = unavailable("No item-level human feature strings")
        else:
            features = sorted({r["feature"] for r in ref["rows"]})
            observed = {
                (r["item"], r["feature"])
                for r in ref["rows"]
                if float(r.get("frequency", 1)) > 0
            }
            human_matrix = [
                [
                    int((i.get("source_text", i["text"]), f) in observed)
                    for f in features
                ]
                for i in subset
            ]
            comparison = geometry_comparison(
                human_matrix,
                [by_id[i["item_id"]]["vector"] for i in subset],
                metric="cosine",
                seed=c["seed"],
            )
        human_comparisons.append(
            dict(
                source_dataset=dataset,
                item_ids=[i["item_id"] for i in subset],
                **comparison,
            )
        )
    return dict(
        model_id=record["model_id"],
        revision=record["revision"],
        artifact_hash=c["embedding_sha256"],
        generated_comparison=geometry_comparison(
            vectors, g, metric="cosine", seed=c["seed"]
        ),
        human_comparisons=human_comparisons,
        interpretation="Secondary external comparison; never a prompt-selection objective",
    )


def write_reports(out, c, generations, items, bank, judgments):
    out = Path(out)
    m = json.loads((out / "manifest.json").read_text())
    cells = {
        (r["item_id"], r["feature_id"]): r["applies"]
        for r in judgments
        if r.get("status") == "ok"
    }
    feature_map = {f["feature_id"]: f["text"] for f in bank}
    text_to_id = {normalize_phrase(f["text"]): f["feature_id"] for f in bank}
    discovered = {i["item_id"]: set() for i in items}
    for r in generations:
        discovered[r["item_id"]] |= {
            text_to_id[normalize_phrase(f)]
            for f in r["features"]
            if normalize_phrase(f) in text_to_id
        }
    phrase_judgments = {
        (i, text): cells.get((i, fid))
        for (i, fid), v in cells.items()
        for text in [feature_map[fid]]
    }
    curves = discovery_curves(generations, phrase_judgments)
    write_csv(
        out / "discovery_curves.csv",
        curves,
        [
            "item_id",
            "prompt_family",
            "replicate",
            "unique_discovered",
            "produced",
            "lexical_redundancy",
            "valid_unique",
        ],
    )
    task_rows = []
    by_id = {i["item_id"]: i for i in items}
    for r in generations:
        item = by_id[r["item_id"]]
        if item["target_type"] in {"concept", "event", "object"}:
            continue
        parent = discovered.get(item.get("parent_id"), set())
        components = set(parent)
        for sid in item["situation_ids"]:
            components |= discovered.get(sid, set())
        for phrase in r["features"]:
            fid = text_to_id.get(normalize_phrase(phrase))
            applies = cells.get((item["item_id"], fid))
            task_rows.append(
                dict(
                    item_id=item["item_id"],
                    task_id=item["task_id"],
                    level=item["level"],
                    parent_id=item["parent_id"],
                    situation_ids=item["situation_ids"],
                    phrase=phrase,
                    prompt_family=r["prompt_family"],
                    replicate=r["replicate"],
                    feature_id=fid or "",
                    selected=fid is not None,
                    applies="" if applies is None else applies,
                    inherited_discovery=bool(fid in parent),
                    newly_generated=bool(fid and fid not in components),
                    newly_applicable=bool(
                        fid
                        and applies == 1
                        and item.get("parent_id")
                        and cells.get((item["parent_id"], fid)) == 0
                    ),
                )
            )
    fields = [
        "item_id",
        "task_id",
        "level",
        "parent_id",
        "situation_ids",
        "phrase",
        "prompt_family",
        "replicate",
        "feature_id",
        "selected",
        "applies",
        "inherited_discovery",
        "newly_generated",
        "newly_applicable",
    ]
    write_csv(out / "task_level_feature_audit.csv", task_rows, fields)
    compositions = []
    interaction_rows = []
    manual_audits = {}
    for target, audits in c.get("relational_audit", {}).items():
        manual_audits[target] = {}
        for feature, decision in audits.items():
            if (
                not isinstance(decision, dict)
                or not decision.get("reviewer")
                or not decision.get("note")
                or type(decision.get("confirmed")) is not bool
            ):
                raise ValueError(
                    "A named manual relational reviewer, note and boolean confirmation are required"
                )
            manual_audits[target][feature] = decision["confirmed"]
    write_json(out / "manual_relational_audits.json", c.get("relational_audit", {}))
    for composition_index, item in enumerate(
        i for i in items if i["target_type"] == "situated_task"
    ):
        t = item["parent_id"]
        s = item["situation_ids"][0]
        ts = item["item_id"]
        allocation = matched_sampling_allocation(
            c["replicates"], composition_index=composition_index
        )

        def records(id, role):
            return [
                r
                for r in generations
                if r["item_id"] == id
                and r["replicate"] < allocation[r["prompt_family"]][role]
            ]

        def feats(rs):
            return {
                text_to_id[normalize_phrase(f)]
                for r in rs
                for f in r["features"]
                if normalize_phrase(f) in text_to_id
            }

        rt, rs, rd = records(t, "task"), records(s, "situation"), records(ts, "direct")
        js = {
            k: {f: cells[id, f] for f in feature_map if (id, f) in cells}
            for k, id in [("T", t), ("S", s), ("TS", ts)]
        }
        try:
            result = compare_composition(
                feats(rt),
                feats(rs),
                feats(rd),
                js,
                component_calls=len(rt) + len(rs),
                direct_calls=len(rd),
                component_tokens=sum(
                    r["usage"]["input_tokens"] + r["usage"]["output_tokens"]
                    for r in rt + rs
                ),
                direct_tokens=sum(
                    r["usage"]["input_tokens"] + r["usage"]["output_tokens"] for r in rd
                ),
                relational_audit=manual_audits.get(ts, {}),
            )
            result.update(
                item_id=ts,
                task_id=item["task_id"],
                situation_id=s,
                full_component_discovered=sorted(discovered[t] | discovered[s]),
                full_direct_discovered=sorted(discovered[ts]),
                balance_rule="equal_calls_alternating_component_extra_v1",
                sampling_allocation=allocation,
                incremental_examples=[
                    dict(feature_id=f, text=feature_map[f])
                    for f in result["incremental_applicable"]
                ],
            )
            for fid in result["interaction_candidates"]:
                interaction_rows.append(
                    dict(
                        item_id=ts,
                        feature_id=fid,
                        text=feature_map[fid],
                        relational_confirmed="",
                        reviewer="",
                        note="Manual review required; candidates are not confirmed interactions",
                    )
                )
        except ValueError as e:
            result = dict(
                item_id=ts,
                task_id=item["task_id"],
                available=False,
                reason=str(e),
                confirmed_interactions=[],
            )
        compositions.append(result)
    write_json(out / "compositionality_audit.json", compositions)
    write_csv(
        out / "interaction_review.csv",
        interaction_rows,
        ["item_id", "feature_id", "text", "relational_confirmed", "reviewer", "note"],
    )
    human = _human_analysis(c, items, generations, bank, judgments)
    write_json(out / "human_norm_comparison.json", human)
    write_json(
        out / "embedding_rdm_comparison.json",
        _embedding_analysis(c, items, bank, judgments),
    )
    # A reproducible hash sample for blinded manual clarity/applicability/nonredundancy review.
    all_audit = [
        dict(
            audit_id=digest([r["job_id"], idx])[:20],
            item_id=r["item_id"],
            phrase=f,
            clarity="",
            applicability="",
            nonredundancy="",
            reviewer="",
        )
        for r in generations
        for idx, f in enumerate(r["features"])
    ]
    sample = sorted(all_audit, key=lambda a: a["audit_id"])[: min(60, len(all_audit))]
    write_csv(
        out / "novel_feature_audit.csv",
        sample,
        [
            "audit_id",
            "item_id",
            "phrase",
            "clarity",
            "applicability",
            "nonredundancy",
            "reviewer",
        ],
    )
    # Manual review cells are sampled without showing primary labels/model provenance.
    audit_n = (
        max(1, int(len(cells) * c.get("judgment_audit_fraction", 0.05))) if cells else 0
    )
    judge_review = sorted(
        [
            dict(
                item_id=i,
                feature_id=f,
                proposition=feature_map[f],
                reviewer="",
                applies="",
                evidence="",
            )
            for i, f in cells
        ],
        key=lambda r: digest([r["item_id"], r["feature_id"]]),
    )[:audit_n]
    write_csv(
        out / "judgment_disagreement_review.csv",
        judge_review,
        ["item_id", "feature_id", "proposition", "reviewer", "applies", "evidence"],
    )
    write_json(
        out / "decision_memo.json",
        dict(
            recommendation=(
                "revise"
                if m["demo"] or m.get("smoke") or m["status"] != "complete"
                else "review"
            ),
            rationale="Offline engineering validation does not establish scientific correspondence. Complete source/protocol, blinded audits and model-price gates before scale-up.",
            later_preferences_study_needs=[
                "independent task-preference elicitation",
                "preregistered linkage of valid features to evaluations",
                "held-out tasks and controlled interventions",
            ],
            no_arbitrary_geometry_threshold=True,
        ),
    )
    lines = [
        "# Human-norm feature-space pilot",
        "",
        (
            "**SYNTHETIC DEMONSTRATION — engineering validation only.**"
            if m["demo"]
            else "**Live API pilot — candidate properties are hypotheses.**"
        ),
        "",
        f"Status: **{m['status']}**. Run: `{m['run_id']}`.",
        f"Generation jobs: {m['completed_generation_jobs']} / {m['planned_generation_jobs']}. Resolved binary cells: {m['resolved_judgments']} / {m['planned_judgments']}.",
        "",
        "Human lists measure feature production, not truth. Correspondence does not demonstrate internal feature use or model preferences.",
        "",
        "## Human reference comparison",
        "",
        f"{len(human['recovery'])} item/condition recovery records; {len(human['geometry'])} source/subset geometry records. Exact or frozen reviewed-equivalence matching only. Novel unlisted features remain candidates for review.",
        "Category profiles use each source’s actual vocabulary. Generated profiles require independent coding; missing annotations are unavailable, never inferred from another dataset.",
        "Geometry uses binary Jaccard distances and Spearman correlations with item-resampled intervals; copied self-pairs are excluded. Embedding comparisons are secondary and unavailable until a pinned, aligned vector artifact is supplied.",
        "",
        "## Abstraction and independent situations",
        "",
        "Domain → task type → task instance is the hierarchy. Situations are independent; baseline T appears once. Full invoice input, matching policy and reference report are in `reconciliation_fixture.json`.",
        f"All generated case features, provenance, selected canonical IDs, binary applicability, inherited discoveries, newly generated and newly applicable features: `task_level_feature_audit.csv` ({len(task_rows)} rows).",
        "",
        "## Each composition",
        "",
        "| Target | Component applicable | Applicable gain | Interaction candidates | Manually confirmed |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in compositions:
        lines.append(
            f"| {r['item_id']} | {len(r.get('component_applicable',[]))} | {len(r.get('incremental_applicable',[]))} | {len(r.get('interaction_candidates',[]))} | {len(r.get('confirmed_interactions',[]))} |"
        )
    lines.append("")
    for r in compositions:
        if r.get("incremental_examples"):
            lines.append(f"Applicable direct-discovery examples for `{r['item_id']}`:")
            lines.append("")
            lines.extend(
                f"- `{e['feature_id']}`: {e['text']}" for e in r["incremental_examples"]
            )
            lines.append("")
    lines += [
        "",
        "The balanced comparison uses equal generation-call budgets separately within each family, retaining all direct replicates. For odd counts, the extra component sample alternates between task and situation by frozen composition and prompt-family order, independently of outcomes (three repetitions use 2+1 or 1+2 component samples versus 3 direct samples). Even counts split component samples equally. The same independent judge cells for T+S evaluate both the component bank and its union with direct discoveries. Full discoveries are also retained. The exact sampling_allocation, token counts and gain per call/token appear in `compositionality_audit.json`.",
        "Interaction candidates require applicability to T+S, nonapplicability to T and S individually, and a named manual relational audit before confirmation. No mathematical emergence or internal representation claim is made.",
        "",
        "## Costs, missing evidence and decision",
        "",
        "See `cost_report.json` for cumulative reservations, provider-reported tokens, accounted USD and unknown-usage liabilities. Offline fake requests have zero billing. Model prices and capabilities must be verified before live execution.",
        "**Recommendation: revise before scale-up.** Finish source/protocol checks, source-language mapping where needed, novel-feature and merge audits, binary-judgment audits, and the embedding reference. Scientific success has no arbitrary geometry threshold.",
        "Synthetic outputs cannot identify which prompt/level combination merits scientific scale-up. Later preference work needs independent preference elicitation and held-out controlled tests.",
        "",
        "Artifact integrity and full run provenance are recorded in `manifest.json`.",
    ]
    (out / "pilot_report.md").write_text("\n".join(lines) + "\n")


def generate_report(out):
    verify_manifest(out)
    return (Path(out) / "pilot_report.md").read_text()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir")
    parser.add_argument("--config")
    args = parser.parse_args()
    if args.run_dir:
        targets = [Path(args.run_dir)]
    elif args.config:
        from .engine import load_config, resolve

        c = load_config(args.config)
        targets = sorted(
            resolve(c, c.get("artifact_root", "artifacts/feature_space_pilot")).glob(
                "*/manifest.json"
            )
        )
        targets = [p.parent for p in targets]
    else:
        parser.error("Supply --run-dir or --config")
    if not targets:
        parser.error(
            "No manifested run found; run the demo or an explicitly preflighted live stage first"
        )
    for p in targets:
        try:
            print(generate_report(p))
        except (ValueError, OSError) as e:
            parser.exit(2, f"Report verification failed: {e}\n")


if __name__ == "__main__":
    main()
