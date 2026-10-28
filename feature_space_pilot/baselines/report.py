"""Descriptive agreement and disagreements; never labels the judge as human truth."""

from pathlib import Path
import json
import numpy as np
from ..io import write_json, write_csv, digest, file_hash
from .core import calibrate, evaluate, methods, grid, metrics, REFERENCE


def evidence_hash(snapshot):
    return digest(
        [
            snapshot["manifest"]["analysis_id"],
            snapshot["manifest"]["grader"],
            snapshot["inventory"],
            [r["record_hash"] for r in snapshot["records"]],
            snapshot["manifest"]["control_checks"],
        ]
    )


def write_report(out, rows, inventory, snapshot, *, render=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    seal = out / "report_manifest.json"
    if seal.exists():
        previous = json.loads(seal.read_text())
        if previous["source_evidence_sha256"] != evidence_hash(snapshot):
            raise ValueError("Baseline report source changed")
        for name, expected in previous["artifacts"].items():
            if not (out / name).exists() or file_hash(out / name) != expected:
                raise ValueError("Frozen baseline report artifact changed: " + name)
        return json.loads((out / "summary.json").read_text())
    calibration = calibrate(rows, inventory)
    thresholds = out.parent.parent / "thresholds"
    if calibration["available"]:
        thresholds.mkdir(exist_ok=True)
        for name, entry in calibration["methods"].items():
            path = thresholds / (name + ".json")
            evidence = digest(
                [
                    {
                        k: r[k]
                        for k in [
                            "item_id",
                            "proposition_id",
                            "human_id",
                            "judge_relation",
                            name,
                        ]
                    }
                    for r in rows
                    if r["split"] == "development"
                ]
            )
            fixed = dict(
                method=name,
                development_evidence_hash=evidence,
                selection=calibration["selection"],
                development_item_ids=calibration["development_item_ids"],
                result=entry,
            )
            if path.exists():
                if json.loads(path.read_text()) != fixed:
                    raise ValueError(
                        "Frozen development threshold/evidence changed: " + name
                    )
            else:
                write_json(path, fixed)
    development = evaluate(rows, calibration, split="development")
    held_out = evaluate(rows, calibration, split="held_out")
    summary = dict(
        reference_kind=REFERENCE,
        human_reviewed=False,
        demo=snapshot["manifest"].get("demo", False),
        source_matching_analysis=snapshot["manifest"]["analysis_id"],
        source_manifest_sha256=snapshot["manifest_hash"],
        planned_pairs=len(rows),
        unscored_pairs=sum(r["judge_relation"] == "unscored" for r in rows),
        methods_available=methods(rows),
        calibration=calibration,
        development=development,
        held_out=held_out,
        control_checks=snapshot["manifest"]["control_checks"],
        caveat="Agreement with a model reference is not independent semantic accuracy. Similarity does not establish entailment. Related is negative and uncertain excluded in the strict comparison.",
    )
    write_json(out / "summary.json", summary)
    write_json(out / "source_manifest.json", snapshot["manifest"])
    write_json(out / "source_judge_records.json", snapshot["records"])
    write_csv(out / "pair_scores.csv", rows, list(rows[0]) if rows else ["status"])
    curves = []
    for split in ["development", "held_out"]:
        cohort = evaluate(rows, dict(available=False, methods={}), split=split)
        allowed = set(cohort["item_ids"])
        part = [r for r in rows if r["item_id"] in allowed]
        for name in methods(rows):
            for t in grid(name):
                curves.append(
                    dict(
                        split=split,
                        method=name,
                        n_items=len(allowed),
                        **metrics(part, name, t),
                    )
                )
    write_csv(
        out / "threshold_curves.csv", curves, list(curves[0]) if curves else ["status"]
    )
    disagreement = []
    if calibration["available"]:
        for row in rows:
            if row["judge_label"] is None:
                continue
            for name, s in calibration["methods"].items():
                predicted = int(row[name] >= s["threshold"])
                if predicted != row["judge_label"]:
                    disagreement.append(
                        dict(
                            row,
                            method=name,
                            threshold=s["threshold"],
                            baseline_match=predicted,
                        )
                    )
    disagreement.sort(
        key=lambda r: (
            r["split"],
            r["method"],
            digest([r["item_id"], r["proposition_id"], r["human_id"]]),
        )
    )
    write_csv(
        out / "disagreements.csv",
        disagreement,
        list(disagreement[0]) if disagreement else ["status"],
    )
    text = "SYNTHETIC FIXTURE — NOT EMPIRICAL\n\n" if summary["demo"] else ""
    text += "# Similarity baselines versus Luna — exploratory\n\n"
    text += f"Source: `{summary['source_matching_analysis']}` at `{snapshot['manifest_hash']}`. All comparisons stay within the same target. {len(rows):,} planned feature pairs; {summary['unscored_pairs']:,} pairs have no collected judge decision.\n\n"
    text += "Exact normalized text, token Jaccard, human-label token coverage, and (when available) target-context embedding cosine are independent baseline scores. Labels and propositions are never semantically merged. Embedding input is the literal `Target: <target>\\nFeature: <feature>` on both sides.\n\n"
    if not calibration["available"]:
        text += f"Threshold selection pending: {calibration['reason']}. No tuned held-out result is available.\n\n"
    else:
        text += f"Thresholds maximize development F1 against clear Luna matches, with ties resolved toward fewer false positives then higher thresholds. All {len(calibration['development_item_ids'])} development targets were completely judged before selection. Held-out comparisons currently include {len(held_out['item_ids'])} completely judged targets and {held_out['n_pairs']:,} non-uncertain pairs.\n\n"
        for name, s in held_out["methods"].items():
            text += f"- {name}: threshold {s['threshold']:.3f}; precision vs judge {s['precision_vs_judge']}; recall vs judge {s['recall_vs_judge']}; F1 vs judge {s['f1_vs_judge']}.\n"
    text += "\nRelated-but-different is negative for strict same-property matching; uncertain decisions are excluded and their counts reported. Missing judgments never become negatives. Overall agreement can look high because most pairs are nonmatches; prioritize positive-match precision/recall/F1 and inspect disagreements. Scores and feature pairs are dependent within targets; no independence-based confidence intervals are claimed. Threshold curves on held-out data are descriptive and never choose thresholds.\n\nThis is agreement with an automated reference, not independent accuracy. A blinded human audit is needed to establish which method is right. No reviewer decisions, translation approvals or semantic equivalences are created. Primary pilot scores, task studies and the Luna matching run are unchanged.\n"
    (out / "comparison.md").write_text(text)

    def finish_report():
        write_json(
            seal,
            dict(
                source_manifest_sha256=snapshot["manifest_hash"],
                source_evidence_sha256=evidence_hash(snapshot),
                artifacts={
                    p.name: file_hash(p)
                    for p in out.iterdir()
                    if p.is_file() and p != seal
                },
            ),
        )
        return summary

    if not render:
        return finish_report()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = methods(rows)
    fig, axes = plt.subplots(
        1, len(names), figsize=(4.3 * len(names), 4.7), squeeze=False
    )
    for ax, name in zip(axes[0], names):
        for value, label, color in [
            (1, "Luna clear match", "#167D8D"),
            (0, "Luna strict nonmatch", "#C67037"),
        ]:
            scores = [r[name] for r in rows if r["judge_label"] == value]
            if scores:
                ax.hist(
                    scores,
                    bins=np.linspace(-0.05, 1.05, 23),
                    histtype="step",
                    density=True,
                    label=label,
                    color=color,
                    lw=1.6,
                )
        ax.set(title=name, xlabel="Baseline similarity", ylabel="Within-class density")
        ax.legend(fontsize=8)
    fig.suptitle("Similarity scores versus collected Luna decisions — exploratory")
    fig.text(
        0.02,
        0.015,
        ("SYNTHETIC FIXTURE | " if summary["demo"] else "")
        + f"Automated reference; no human validation | {summary['unscored_pairs']:,} unscored pairs excluded",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(out / ("figure_01_score_distributions." + ext), dpi=220)
    plt.close(fig)
    if held_out["available"] and summary["control_checks"]["passed"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(names))
        width = 0.25
        for offset, key, label, color in [
            (-1, "precision_vs_judge", "Precision", "#167D8D"),
            (0, "recall_vs_judge", "Recall", "#C67037"),
            (1, "f1_vs_judge", "F1", "#526F9B"),
        ]:
            values = [held_out["methods"][n][key] for n in names]
            ax.bar(
                x + offset * width,
                [v if v is not None else np.nan for v in values],
                width,
                label=label,
                color=color,
            )
        ax.set(
            xticks=x,
            xticklabels=names,
            ylim=(0, 1.05),
            ylabel="Agreement metric against Luna",
            title=f"Fixed development thresholds on {len(held_out['item_ids'])} held-out targets",
        )
        ax.legend()
        fig.text(
            0.02,
            0.02,
            ("SYNTHETIC FIXTURE | " if summary["demo"] else "")
            + "Automated reference, no human validation. Undefined metrics omitted.",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(out / ("figure_02_held_out_agreement." + ext), dpi=220)
        plt.close(fig)
    return finish_report()
