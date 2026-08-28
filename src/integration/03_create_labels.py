
import os
import pandas as pd


INPUT_FILE = "data/unified/aligned_dataset.csv"
OUTPUT_FILE = "data/unified/labeled_dataset.csv"


def create_provisional_labels(df):
    """
    Create provisional bloom labels using the 75th percentile
    of chlorophyll-a.

    IMPORTANT:
    This is only for pipeline testing and must not be treated
    as final scientific ground truth.
    """

    threshold = df["chlorophyll_a"].quantile(0.75)

    df["bloom_label"] = (
        df["chlorophyll_a"] >= threshold
    ).astype(int)

    print(
        "Provisional chlorophyll-a threshold:",
        threshold
    )

    print("\nLabel distribution:")
    print(df["bloom_label"].value_counts())

    return df


def create_five_day_target(df):
    """
    Create a target representing whether a bloom occurs
    exactly 5 days after the feature date.
    """

    df = df.sort_values(
        "date"
    ).reset_index(drop=True)

    df["feature_date"] = df["date"]

    df["target_date"] = (
        df["feature_date"] +
        pd.Timedelta(days=5)
    )

    label_lookup = (
        df.set_index("date")["bloom_label"]
    )

    df["bloom_label_t_plus_5"] = (
        df["target_date"].map(label_lookup)
    )

    return df


def verify_target_alignment(df):
    """
    Verify that all available targets are exactly 5 days ahead
    and were retrieved from the correct target date.
    """

    usable = df.dropna(
        subset=["bloom_label_t_plus_5"]
    ).copy()

    usable["forecast_gap_days"] = (
        usable["target_date"] -
        usable["feature_date"]
    ).dt.days

    if not (
        usable["forecast_gap_days"] == 5
    ).all():
        raise ValueError(
            "Incorrect forecast gap detected."
        )

    verification_lookup = (
        df.set_index("date")["bloom_label"]
    )

    usable["actual_label_on_target_date"] = (
        usable["target_date"].map(
            verification_lookup
        )
    )

    labels_match = (
        usable["bloom_label_t_plus_5"]
        ==
        usable["actual_label_on_target_date"]
    ).all()

    if not labels_match:
        raise ValueError(
            "Target-label mismatch detected."
        )

    print(
        "✓ PASS: Every target is exactly "
        "5 days after its feature date."
    )

    print(
        "✓ PASS: All t+5 labels came from "
        "the correct target dates."
    )

    print(
        "\nUsable 5-day prediction samples:",
        len(usable)
    )

    print("\nVerification sample:")

    print(
        usable[
            [
                "feature_date",
                "chlorophyll_a",
                "target_date",
                "forecast_gap_days",
                "bloom_label_t_plus_5",
                "actual_label_on_target_date"
            ]
        ].head(10)
    )


def main():

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=[
            "date",
            "image_date"
        ]
    )

    df = create_provisional_labels(df)

    df = create_five_day_target(df)

    verify_target_alignment(df)

    os.makedirs(
        "data/unified",
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\n✓ Labeled dataset saved to {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
