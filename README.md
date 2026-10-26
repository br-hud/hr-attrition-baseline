# HR Attrition Risk Baseline

A beginner MLOps foundation project that compares a naive majority-class baseline with a reproducible scikit-learn classification pipeline.

## What it demonstrates

- Deterministic synthetic HR data with no real employee PII
- A held-out, stratified train/test split
- Accuracy compared with positive-class recall
- A preprocessing and logistic-regression pipeline
- A locally generated, persisted model artifact
- A separate prediction program that reloads the artifact

## Run the project

1. Select the `sklearn-env` interpreter in VS Code.
2. Open `train.py` and use Run Python File.
3. Open `predict.py` and use Run Python File.

The generated `artifacts` folder is intentionally excluded from Git. Recreate the artifact by running `train.py`.

## Responsible-use limitations

This is a synthetic educational demonstration. It is not validated for real employees and must not be used for hiring, discipline, termination, promotion, compensation, or retention decisions. A real HR system would require representative data, consent and privacy controls, bias and fairness testing, explainability, legal review, human oversight, access controls, monitoring, and an appeal process.

## Next project

Serve the saved pipeline through a tested API and package it with Docker.