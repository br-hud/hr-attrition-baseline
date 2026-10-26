import pickle
from pathlib import Path
from random import Random

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
FEATURE_NAMES = [
    "overtime_hours",
    "satisfaction_score",
    "years_at_company",
    "recent_promotion",
]
ARTIFACT_PATH = Path("artifacts") / "attrition_pipeline.pkl"

def build_synthetic_dataset(n_samples=240):
    rng = Random(RANDOM_SEED)
    features = []
    labels = []

    for _ in range(n_samples):
        overtime_hours = rng.choice([0, 5, 10, 15])
        satisfaction_score = rng.randint(1, 5)
        years_at_company = rng.randint(0, 12)
        recent_promotion = int(rng.random() < 0.30)

        risk_points = (
            2 * int(overtime_hours >= 10)
            + 2 * int(satisfaction_score <= 2)
            + int(years_at_company <= 1)
            + int(recent_promotion == 0)
        )
        attrition = int(risk_points >= 4)

        features.append(
            [
                overtime_hours,
                satisfaction_score,
                years_at_company,
                recent_promotion,
            ]
        )
        labels.append(attrition)

    return features, labels

def main():
    X, y = build_synthetic_dataset()
    positive_count = sum(y)

    print(f"Generated {len(X)} synthetic employee records.")
    print(
        "Attrition labels: "
        f"{positive_count} yes, {len(y) - positive_count} no"
    )

    X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=RANDOM_SEED,
            stratify=y,
        )

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)

    print("\nNaive baseline")
    print(f"Accuracy: {accuracy_score(y_test, baseline_predictions):.2f}")
    print(
        "Attrition recall: "
        f"{recall_score(y_test, baseline_predictions, zero_division=0):.2f}"
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000),
    )
    model.fit(X_train, y_train)
    model_predictions = model.predict(X_test)

    print("\nTrained pipeline")
    print(f"Accuracy: {accuracy_score(y_test, model_predictions):.2f}")
    print(
        "Attrition recall: "
        f"{recall_score(y_test, model_predictions, zero_division=0):.2f}"
    )

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "feature_names": FEATURE_NAMES,
    }
    with ARTIFACT_PATH.open("wb") as artifact_file:
        pickle.dump(artifact, artifact_file, pickle.HIGHEST_PROTOCOL)

    print(f"\nSaved trusted local artifact to {ARTIFACT_PATH}")

if __name__ == "__main__":
    main()
        