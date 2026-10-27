"""Independent deterministic jobs; job identity includes all decoding/prompt inputs."""

from .io import digest
from .prompts import generation_prompt


def plan_generation(
    items, settings, *, replicates=10, seed=42, live=False, protocols=None
):
    if type(replicates) is not int or replicates < 1:
        raise ValueError("replicates must be positive")
    jobs = []
    for item in items:
        for family in ("P0", "P1", "P2"):
            prompt = generation_prompt(item, family, live=live, protocols=protocols)
            for replicate in range(replicates):
                paired_seed = int(
                    digest([seed, item["item_id"], replicate])[:8], 16
                ) % (2**31)
                j = dict(
                    item_id=item["item_id"],
                    item_hash=item["content_hash"],
                    prompt_family=family,
                    replicate=replicate,
                    seed=paired_seed if settings.get("supports_seed") else None,
                    seed_policy="paired_item_replicate_v1",
                    model_id=settings.get("model_id"),
                    model_revision=settings.get("revision"),
                    decoding=dict(settings),
                    **prompt
                )
                j["job_id"] = digest(j)
                jobs.append(j)
    return jobs


def pending_jobs(jobs, records):
    complete = {r["job_id"] for r in records if r.get("parse_status") == "ok"}
    return [j for j in jobs if j["job_id"] not in complete]
