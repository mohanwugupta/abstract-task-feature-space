"""Published researcher coding statistics are not participant similarity ceilings."""


def reliability_summary(review):
    for r in review["records"]:
        if r.get("participant_similarity_ceiling") and (
            r["unit"] != "participant_feature_space"
            or not r.get("comparable_to_cosine_rsa")
        ):
            raise ValueError(
                "Researcher coding agreement, consensus rules and sample sizes cannot be a participant-similarity ceiling"
            )
    ceilings = [r for r in review["records"] if r.get("participant_similarity_ceiling")]
    return dict(
        records=review["records"],
        participant_similarity_ceiling=ceilings or None,
        loocv_performed=False,
        participant_level_reconstruction_performed=False,
        conclusion="Neither reviewed paper reports a numerical participant-level cosine/RSA, split-half feature-space or leave-one-participant-out ceiling suitable for these analyses. Reported coding agreement describes annotation reproducibility, not feature-generation similarity.",
    )
