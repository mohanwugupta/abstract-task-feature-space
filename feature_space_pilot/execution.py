"""Explicit execution/pricing mode, separate from credentials and API transport."""

import copy


def execution_config(config, mode=None):
    result = copy.deepcopy(config)
    mode = mode or result.get("execution_mode", "online")
    if mode not in {"online", "batch"}:
        raise ValueError("Execution mode must be online or batch")
    result["execution_mode"] = mode
    for role in ["generator", "judge"]:
        settings = result[role]
        standard = settings.get("standard_prices") or settings["prices"]
        settings["standard_prices"] = copy.deepcopy(standard)
        if mode == "batch":
            if not settings.get("batch_prices"):
                raise ValueError("Verified model-specific Batch prices required")
            settings["prices"] = copy.deepcopy(settings["batch_prices"])
        else:
            settings["prices"] = copy.deepcopy(standard)
        settings["pricing_mode"] = mode
    return result
