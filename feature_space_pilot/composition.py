"""Discovery G and applicability A are deliberately distinct operations."""

from __future__ import annotations


def matched_sampling_allocation(replicates, *, composition_index):
    """Equal per-family call budgets, including odd replicate counts.

    An odd extra component sample alternates by frozen composition/family order,
    independently of outcomes. Every direct replicate is used. Even counts retain
    the original equal task/situation split.
    """
    if type(replicates) is not int or replicates < 2:
        raise ValueError("Integer replicate count >=2 required")
    if type(composition_index) is not int or composition_index < 0:
        raise ValueError("Nonnegative integer composition index required")
    half, odd = divmod(replicates, 2)
    allocation = {}
    for family_index, family in enumerate(("P0", "P1", "P2")):
        task = half + int(bool(odd) and (composition_index + family_index) % 2 == 0)
        allocation[family] = dict(
            task=task, situation=replicates - task, direct=replicates
        )
    return allocation


def compare_composition(
    task,
    situation,
    direct,
    judgments,
    *,
    component_calls,
    direct_calls,
    relational_audit=None,
    component_tokens=None,
    direct_tokens=None,
):
    if component_calls != direct_calls or component_calls <= 0:
        raise ValueError("Unequal or empty generation-call budget")
    task, situation, direct = map(set, (task, situation, direct))
    component = task | situation
    union = component | direct
    for level in ["T", "S", "TS"]:
        if level not in judgments or union - set(judgments[level]):
            raise ValueError(f"missing binary cells for {level}")
        if any(
            type(judgments[level][f]) is not int or judgments[level][f] not in (0, 1)
            for f in union
        ):
            raise ValueError("Unresolved applicability")
    applicable = lambda features: {f for f in features if judgments["TS"][f] == 1}
    baseline = applicable(component)
    extended = applicable(union)
    gain = extended - baseline
    candidates = {f for f in gain if judgments["T"][f] == 0 and judgments["S"][f] == 0}
    audit = relational_audit or {}
    result = dict(
        component_bank=sorted(component),
        direct_bank=sorted(direct),
        component_applicable=sorted(baseline),
        union_applicable=sorted(extended),
        newly_discovered=sorted(direct - component),
        incremental_applicable=sorted(gain),
        interaction_candidates=sorted(candidates),
        confirmed_interactions=sorted(f for f in candidates if audit.get(f) is True),
        jaccard_discovery=len(component & direct) / len(union) if union else 1.0,
        component_calls=component_calls,
        direct_calls=direct_calls,
        gain_per_direct_call=len(gain) / direct_calls,
        component_tokens=component_tokens,
        direct_tokens=direct_tokens,
        gain_per_1000_direct_tokens=(
            1000 * len(gain) / direct_tokens if direct_tokens else None
        ),
        interpretation="Operational audited relational interaction; not proof of emergence or internal representation",
    )
    return result
