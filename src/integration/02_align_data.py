
import os
import pandas as pd
import importlib.util


MASTER_SCHEMA_PATH = "src/integration/01_master_schema.py"
OUTPUT_FILE = "data/unified/aligned_dataset.csv"


def load_master_schema_module():
    spec = importlib.util.spec_from_file_location(
        "master_schema",
        MASTER_SCHEMA_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def align_satellite_with_environment(
    satellite,
    environment,
    max_lookback_days=10
):
    """
    Attach the most recent Sentinel-2 image available on or before
    each environmental observation date.

    Future images are never used.
    """

    satellite_sorted = satellite.sort_values(
        "date"
    ).reset_index(drop=True)

    def latest_image_on_or_before(row_date):
        valid = satellite_sorted[
            satellite_sorted["date"] <= row_date
        ]

        if valid.empty:
            return pd.Series([None, None, pd.NaT])

        latest = valid.iloc[-1]

        age_days = (
            row_date - latest["date"]
        ).days

        if age_days > max_lookback_days:
            return pd.Series([None, None, pd.NaT])

        return pd.Series([
            latest["image_path"],
            latest["cloud_percentage"],
            latest["date"]
        ])

    aligned = environment.copy()

    aligned[
        [
            "image_path",
            "cloud_percentage",
            "image_date"
        ]
    ] = aligned["date"].apply(
        latest_image_on_or_before
    )

    # Calculate age of matched image
    aligned["image_age_days"] = (
        aligned["date"] - aligned["image_date"]
    ).dt.days

    return aligned


def check_future_leakage(aligned):
    """
    Confirm that no satellite image comes from after
    the environmental feature date.
    """

    valid_matches = aligned.dropna(
        subset=["image_date"]
    ).copy()

    future_images = valid_matches[
        valid_matches["image_date"] >
        valid_matches["date"]
    ]

    if future_images.empty:
        print(
            "✓ PASS: No future satellite images were used."
        )
    else:
        print(
            "✗ ERROR: Future satellite images detected!"
        )

        print(
            future_images[
                [
                    "date",
                    "image_date",
                    "image_path"
                ]
            ]
        )

        raise ValueError(
            "Future-image leakage detected."
        )


def main():

    schema = load_master_schema_module()

    satellite, environment = (
        schema.load_and_validate_inputs()
    )

    aligned = align_satellite_with_environment(
        satellite,
        environment,
        max_lookback_days=10
    )

    check_future_leakage(aligned)

    os.makedirs(
        "data/unified",
        exist_ok=True
    )

    aligned.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\n✓ Aligned dataset saved to {OUTPUT_FILE}"
    )

    print("\nSample aligned rows:")

    print(
        aligned[
            [
                "date",
                "image_date",
                "image_age_days",
                "image_path"
            ]
        ].head(10)
    )


if __name__ == "__main__":
    main()
