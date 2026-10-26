import pickle
from pathlib import Path

ARTIFACT_PATH = Path("artifacts") / "attrition_pipeline.pkl"


def main():
    with ARTIFACT_PATH.open("rb") as artifact_file:
        artifact = pickle.load(artifact_file)

    model = artifact["model"]
    feature_names = artifact["feature_names"]

    synthetic_employee = [15, 1, 1, 0]
    probability = float(model.predict_proba([synthetic_employee])[0][1])
    prediction = int(model.predict([synthetic_employee])[0])

    print("Synthetic input")
    for name, value in zip(feature_names, synthetic_employee):
        print(f"{name}: {value}")

    print(f"\nAttrition probability: {probability:.2%}")
    print(f"Flagged for human review: {'yes' if prediction == 1 else 'no'}")
    print("Education only. Never use this model for real employment decisions.")


if __name__ == "__main__":
    main()