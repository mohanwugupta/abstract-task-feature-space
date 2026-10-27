"""Deterministic, entirely synthetic case-study inputs; no task execution by models."""

from __future__ import annotations
from .io import canonical, digest
from .registry import seal_item, validate_items


def reconcile(fixture):
    """Exact invoice ID/currency match, integer cents; duplicates repeat an identical payment."""
    result = []
    for inv in fixture["invoices"]:
        rows = [
            p
            for p in fixture["payments"]
            if p["invoice_id"] == inv["invoice_id"] and p["currency"] == inv["currency"]
        ]
        unique = {(p["transaction_ref"], p["amount_cents"]): p for p in rows}
        paid = sum(p["amount_cents"] for p in unique.values())
        status = (
            "duplicate"
            if len(rows) > len(unique)
            else (
                "missing"
                if not rows
                else "matched" if paid == inv["amount_cents"] else "partial"
            )
        )
        result.append(
            dict(
                invoice_id=inv["invoice_id"],
                status=status,
                invoice_cents=inv["amount_cents"],
                paid_unique_cents=paid,
                balance_cents=inv["amount_cents"] - paid,
                payment_ids=[p["payment_id"] for p in rows],
            )
        )
    return result


def reconciliation_fixture():
    invoices = [
        dict(
            invoice_id=f"INV-{i:03}",
            amount_cents=10000 + i * 125,
            currency="USD",
            customer=f"Synthetic customer {i:02}",
        )
        for i in range(1, 21)
    ]
    payments = []
    for i, inv in enumerate(invoices, 1):
        if i in {5, 12, 20}:
            continue
        p = dict(
            payment_id=f"PAY-{len(payments)+1:03}",
            invoice_id=inv["invoice_id"],
            amount_cents=(
                inv["amount_cents"] // 2 if i in {7, 14} else inv["amount_cents"]
            ),
            currency="USD",
            transaction_ref=f"TXN-{i:03}",
        )
        payments.append(p)
        if i in {9, 18}:
            payments.append(dict(p, payment_id=f"PAY-{len(payments)+1:03}"))
    f = dict(
        version="1",
        synthetic=True,
        invoices=invoices,
        payments=payments,
        matching_policy="Exact invoice_id and currency. Integer cents. Identical transaction_ref and amount are duplicate records; count once. Duplicate status takes precedence; otherwise missing, matched, or partial. List every invoice and its payment IDs.",
        expected_report_schema=[
            "invoice_id",
            "status",
            "invoice_cents",
            "paid_unique_cents",
            "balance_cents",
            "payment_ids",
        ],
    )
    f["reference_report"] = reconcile(f)
    return f


def task_items():
    fixture = reconciliation_fixture()
    public = {k: v for k, v in fixture.items() if k != "reference_report"}
    situations = {
        "S_deadline": "The user must receive the finished result within five minutes.",
        "S_review": "The finished work will be independently reviewed for quality against task-appropriate criteria.",
    }
    situations["S_both"] = situations["S_deadline"] + " " + situations["S_review"]
    items = []

    def add(id, text, typ, task=None, parent=None, situations=None):
        items.append(
            seal_item(
                dict(
                    item_id=id,
                    text=text,
                    target_type=typ,
                    level=typ,
                    parent_id=parent,
                    task_id=task,
                    situation_ids=situations or [],
                    source_dataset="synthetic_case_study_v1",
                    language="en",
                    selection_stratum="case_study",
                    translation_status="original",
                )
            )
        )

    for sid, text in situations.items():
        add(sid, text, "situation")
    for task, domain, typ, text in [
        (
            "poem",
            "Creative production",
            "Writing a poem",
            "Write an eight-line poem about memory and change using a consistent ABAB CDCD rhyme scheme. Quality criteria: eight lines, coherent themes, vivid language, consistent rhyme.",
        ),
        (
            "finance",
            "Business administration and record analysis",
            "Reconciling invoices against payments",
            "Reconcile these 20 synthetic invoices and payments. Return a discrepancy report listing matched, missing, partial and duplicate payments, following the supplied policy and report schema. Quality criteria: correct matching, balances and complete traceability. Input: "
            + canonical(public),
        ),
    ]:
        add(task + "_domain", domain, "domain", task)
        add(task + "_type", typ, "task_type", task, task + "_domain")
        add(task + "_T", text, "task_instance", task, task + "_type")
        for sid, s in situations.items():
            add(
                task + "_" + sid,
                text + " Situation: " + s,
                "situated_task",
                task,
                task + "_T",
                [sid],
            )
    return validate_items(items)


def demo_human_items():
    """60 *synthetic* labels exercise the full plan; never claimed as human stimuli."""
    result = []
    for typ, n in [("concept", 24), ("event", 24), ("object", 12)]:
        for i in range(n):
            result.append(
                seal_item(
                    dict(
                        item_id=f"demo_{typ}_{i:02}",
                        text=f"Synthetic {typ} {i:02}",
                        target_type=typ,
                        level=typ,
                        parent_id=None,
                        task_id=None,
                        situation_ids=[],
                        source_dataset="synthetic_demo",
                        language="en",
                        selection_stratum=typ,
                        translation_status="original",
                        split="held_out" if i % 4 == 0 else "development",
                    )
                )
            )
    return validate_items(result)
