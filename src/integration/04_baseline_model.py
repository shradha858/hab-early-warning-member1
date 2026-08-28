
import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


INPUT_FILE = "data/unified/labeled_dataset.csv"

METRICS_FILE = "outputs/metrics/baseline_metrics.txt"
PREDICTIONS_FILE = "outputs/metrics/baseline_predictions.csv"


FEATURES = [
    "chlorophyll_a",
    "sst",
    "rainfall",
    "wind_speed"
]


def prepare_data(df):
    """
    Keep only rows with a valid 5-day target and sort
    chronologically before splitting.
    """

    model_data = df.dropna(
        subset=["bloom_label_t_plus_5"]
    ).copy()

    model_data = model_data.sort_values(
        "feature_date"
    ).reset_index(drop=True)

    return model_data


def chronological_split(model_data, train_ratio=0.8):
    """
    Split chronologically:
    earlier dates for training,
    later dates for testing.

    Random splitting is avoided because nearby time-series
    observations can be correlated and cause information leakage.
    """

    split_point = int(
        len(model_data) * train_ratio
    )

    train = model_data.iloc[
        :split_point
    ].copy()

    test = model_data.iloc[
        split_point:
    ].copy()

    if train.empty or test.empty:
        raise ValueError(
            "Not enough samples for chronological train/test split."
        )

    if not (
        train["feature_date"].max()
        <
        test["feature_date"].min()
    ):
        raise ValueError(
            "Chronological train/test split is invalid."
        )

    return train, test


def train_baseline(train, test):
    """
    Train Random Forest using environmental features
    to predict bloom occurrence 5 days ahead.
    """

    X_train = train[FEATURES]
    y_train = train[
        "bloom_label_t_plus_5"
    ].astype(int)

    X_test = test[FEATURES]
    y_test = test[
        "bloom_label_t_plus_5"
    ].astype(int)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    # Model-estimated probability of bloom class (1)
    probability_matrix = model.predict_proba(X_test)

    if 1 in model.classes_:
        bloom_class_index = list(model.classes_).index(1)
        bloom_probabilities = probability_matrix[:, bloom_class_index]
    else:
        bloom_probabilities = [0.0] * len(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    return (
        model,
        predictions,
        bloom_probabilities,
        accuracy,
        precision,
        recall,
        f1,
        cm
    )


def save_results(
    train,
    test,
    predictions,
    bloom_probabilities,
    accuracy,
    precision,
    recall,
    f1,
    cm
):
    """
    Save predictions and baseline metrics.
    """

    os.makedirs(
        "outputs/metrics",
        exist_ok=True
    )

    results = pd.DataFrame({
        "feature_date":
            test["feature_date"].values,

        "target_date":
            test["target_date"].values,

        "actual_bloom":
            test[
                "bloom_label_t_plus_5"
            ].astype(int).values,

        "bloom_probability":
            bloom_probabilities,

        "bloom_probability_percent":
            [p * 100 for p in bloom_probabilities],

        "predicted_bloom":
            predictions
    })

    results.to_csv(
        PREDICTIONS_FILE,
        index=False
    )

    with open(
        METRICS_FILE,
        "w"
    ) as f:

        f.write(
            "HAB Early-Warning System - Baseline Model\n"
        )

        f.write(
            "-----------------------------------------\n"
        )

        f.write(
            "Model: Random Forest Classifier\n"
        )

        f.write(
            "Forecast horizon: 5 days\n"
        )

        f.write(
            "Train/test split: Chronological 80/20\n\n"
        )

        f.write(
            f"Training samples: {len(train)}\n"
        )

        f.write(
            f"Testing samples: {len(test)}\n\n"
        )

        f.write(
            f"Accuracy: {accuracy:.4f}\n"
        )

        f.write(
            f"Precision: {precision:.4f}\n"
        )

        f.write(
            f"Recall: {recall:.4f}\n"
        )

        f.write(
            f"F1 Score: {f1:.4f}\n\n"
        )

        f.write(
            "Confusion Matrix:\n"
        )

        f.write(
            str(cm)
        )

        f.write(
            "\n\nNOTE: If mock data is being used, "
            "these metrics are only for pipeline verification "
            "and are not scientific project results."
        )

    return results


def main():

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=[
            "date",
            "image_date",
            "feature_date",
            "target_date"
        ]
    )

    model_data = prepare_data(df)

    train, test = chronological_split(
        model_data
    )

    print(
        "Total usable samples:",
        len(model_data)
    )

    print(
        "Training samples:",
        len(train)
    )

    print(
        "Testing samples:",
        len(test)
    )

    print(
        "\nTraining period:",
        train["feature_date"].min(),
        "to",
        train["feature_date"].max()
    )

    print(
        "Testing period:",
        test["feature_date"].min(),
        "to",
        test["feature_date"].max()
    )

    print(
        "\n✓ PASS: Chronological split verified."
    )

    (
        model,
        predictions,
        bloom_probabilities,
        accuracy,
        precision,
        recall,
        f1,
        cm
    ) = train_baseline(
        train,
        test
    )

    print(
        "\n✓ Random Forest training completed."
    )

    print(
        "\n--- BASELINE RESULTS ---"
    )

    print(
        "Accuracy :",
        round(accuracy, 4)
    )

    print(
        "Precision:",
        round(precision, 4)
    )

    print(
        "Recall   :",
        round(recall, 4)
    )

    print(
        "F1 Score :",
        round(f1, 4)
    )

    print(
        "\nConfusion Matrix:"
    )

    print(cm)

    results = save_results(
        train,
        test,
        predictions,
        bloom_probabilities,
        accuracy,
        precision,
        recall,
        f1,
        cm
    )

    print(
        f"\n✓ Metrics saved to {METRICS_FILE}"
    )

    print(
        f"✓ Predictions saved to {PREDICTIONS_FILE}"
    )

    print(
        "\n--- 5-DAY BLOOM FORECAST ---"
    )

    display_results = results[[
        "feature_date",
        "target_date",
        "bloom_probability_percent",
        "actual_bloom",
        "predicted_bloom"
    ]].copy()

    display_results["bloom_probability_percent"] = (
        display_results["bloom_probability_percent"].round(2)
    )

    print(display_results)


if __name__ == "__main__":
    main()
