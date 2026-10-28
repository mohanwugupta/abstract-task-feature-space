"""Provider-measured localized feature offsets for the frozen atomic payloads."""

from ..io import digest
from ..full.budget import validate_profile, template_payload
from ..full.reuse import model_identity


class MeasuredInputs:
    def __init__(self, profile, settings):
        if (
            profile.get("profile_hash")
            != digest({k: v for k, v in profile.items() if k != "profile_hash"})
            or profile.get("version") != "provider_item_feature_offsets_v1"
            or profile.get("model_identity") != model_identity(settings)
        ):
            raise ValueError("Immediate input profile changed or model differs")
        validate_profile(profile["base_profile"], settings)
        margin = profile["boundary_reserve_tokens"]
        if (
            type(margin) is not int
            or margin < 8
            or not profile["validation"]
            or any(r["residual_tokens"] > margin for r in profile["validation"])
        ):
            raise ValueError(
                "Immediate input profile lacks valid cross-item boundary evidence"
            )
        self.profile = profile
        self.settings = settings
        self.validated_items = set()

    def bound(self, item, feature):
        base = self.profile["base_profile"]["items"][item["item_id"]]
        if item["item_id"] not in self.validated_items:
            if (
                digest(template_payload(item, self.settings))
                != base["baseline_payload_sha256"]
                or item["content_hash"] != base["item_content_hash"]
            ):
                raise ValueError("Immediate input profile item changed")
            self.validated_items.add(item["item_id"])
        delta = self.profile["feature_counts"][feature["feature_id"]]["delta_tokens"]
        n = base["input_tokens"] + delta + self.profile["boundary_reserve_tokens"]
        if type(n) is not int or not 0 < n <= self.settings["max_input_tokens"]:
            raise ValueError("Immediate measured input exceeds frozen ceiling")
        return dict(
            input_token_bound=n,
            method="provider_item_feature_offsets_with_cross_item_validation_v1",
            exact_provider_count=False,
            profile_hash=self.profile["profile_hash"],
            boundary_reserve_tokens=self.profile["boundary_reserve_tokens"],
        )


def input_bound(item, feature, settings, profile):
    return MeasuredInputs(profile, settings).bound(item, feature)
