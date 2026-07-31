from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import URL

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
N_ITER_SEARCH = 20

OUTPUT_DIR = Path("model_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

server_name = r"AJ\ARIAN"
database_name = "bone_tumor"
driver = "ODBC Driver 17 for SQL Server"


# ---------------------------------------------------------------------
# Database loading
# ---------------------------------------------------------------------

connection_string = (
    f"DRIVER={{{driver}}};"
    f"SERVER={server_name};"
    f"DATABASE={database_name};"
    "Trusted_Connection=yes;"
    "Encrypt=yes;"
    "TrustServerCertificate=yes;"
)

connection_url = URL.create(
    "mssql+pyodbc",
    query={"odbc_connect": connection_string},
)

engine = sa.create_engine(connection_url, pool_pre_ping=True)

query = """
SELECT
    f.*,
    CASE b.outcome_status
        WHEN 'NED' THEN 0
        WHEN 'AWD' THEN 1
        WHEN 'D'   THEN 2
    END AS outcome_label
FROM dbo.vw_feature_matrix AS f
JOIN dbo.Bone_Tumor1 AS b
    ON f.patient_id = b.patient_id;
"""

df = pd.read_sql(query, engine)


# ---------------------------------------------------------------------
# Input validation and feature preparation
# ---------------------------------------------------------------------

required_columns = {"patient_id", "outcome_label"}
missing_columns = required_columns.difference(df.columns)

if missing_columns:
    raise ValueError(
        f"Required columns are missing from the query result: "
        f"{sorted(missing_columns)}"
    )

df = df.dropna(subset=["outcome_label"]).copy()

# Explicitly exclude identifiers and target from the feature matrix.
X = df.drop(columns=["patient_id", "outcome_label"])
y = df["outcome_label"].astype(int)

# The current feature view is expected to be numeric/encoded.
non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric_columns:
    raise TypeError(
        "Non-numeric feature columns require preprocessing: "
        f"{non_numeric_columns}"
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,
)


# ---------------------------------------------------------------------
# Models and hyperparameter spaces
# ---------------------------------------------------------------------

models = {
    "logistic_regression": (
        Pipeline(
            steps=[
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        multi_class="multinomial",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        {
            "model__C": np.logspace(-3, 2, 20),
            "model__solver": ["lbfgs", "newton-cg", "saga"],
            "model__class_weight": [None, "balanced"],
        },
    ),
    "svm": (
        Pipeline(
            steps=[
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf")),
            ]
        ),
        {
            "model__C": np.logspace(-3, 2, 20),
            "model__gamma": np.logspace(-4, 0, 20),
            "model__class_weight": [None, "balanced"],
            "model__kernel": ["rbf"],
        },
    ),
    "random_forest": (
        Pipeline(
            steps=[
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                (
                    "model",
                    RandomForestClassifier(
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        {
            "model__n_estimators": [100, 120, 150, 200, 300, 400],
            "model__max_depth": [3, 5, 8, 12, 20, None],
            "model__max_features": ["sqrt", "log2", None],
            "model__min_samples_leaf": [1, 2, 3, 4, 5],
            "model__min_samples_split": [2, 4, 5, 8, 9, 10, 11],
            "model__class_weight": [None, "balanced", "balanced_subsample"],
        },
    ),
    "gradient_boosting": (
        Pipeline(
            steps=[
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                (
                    "model",
                    GradientBoostingClassifier(
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        {
            "model__learning_rate": np.linspace(0.01, 0.30, 20),
            "model__max_depth": [2, 3, 4],
            "model__n_estimators": [60, 80, 100, 120, 150, 200, 240],
            "model__subsample": [0.7, 0.8, 0.9, 1.0],
        },
    ),
}


# ---------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------

summary_rows = []
detailed_results = {}

for model_name, (pipeline, parameter_space) in models.items():
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_space,
        n_iter=N_ITER_SEARCH,
        scoring="f1_macro",
        cv=CV_FOLDS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        verbose=0,
        return_train_score=False,
    )

    search.fit(X_train, y_train)

    y_pred = search.predict(X_test)

    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_test,
        y_pred,
        digits=3,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    summary_rows.append(
        {
            "model": model_name,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "f1_macro": f1_macro,
            "best_cv_f1_macro": search.best_score_,
        }
    )

    detailed_results[model_name] = {
        "model": model_name,
        "best_params": search.best_params_,
        "best_cv_score_f1_macro": float(search.best_score_),
        "test_metrics": {
            "accuracy": float(accuracy),
            "balanced_accuracy": float(balanced_accuracy),
            "f1_macro": float(f1_macro),
        },
        "classification_report": report_dict,
        "confusion_matrix": matrix.tolist(),
        "classes": [int(value) for value in search.classes_],
    }

    model_dir = OUTPUT_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    with (model_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump(
            detailed_results[model_name],
            file,
            indent=2,
            default=lambda value: value.item()
            if isinstance(value, np.generic)
            else value,
        )

    pd.DataFrame(report_dict).transpose().to_csv(
        model_dir / "classification_report.csv"
    )

    pd.DataFrame(
        matrix,
        index=[f"actual_{value}" for value in search.classes_],
        columns=[f"predicted_{value}" for value in search.classes_],
    ).to_csv(model_dir / "confusion_matrix.csv")

    with (model_dir / "classification_report.txt").open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report_text)

    with (model_dir / "best_params.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "best_params": search.best_params_,
                "best_cv_score_f1_macro": float(search.best_score_),
            },
            file,
            indent=2,
            default=lambda value: value.item()
            if isinstance(value, np.generic)
            else value,
        )


# ---------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------

summary = (
    pd.DataFrame(summary_rows)
    .sort_values("f1_macro", ascending=False)
    .reset_index(drop=True)
)

summary.to_csv(OUTPUT_DIR / "model_summary.csv", index=False)

with (OUTPUT_DIR / "all_model_results.json").open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        detailed_results,
        file,
        indent=2,
        default=lambda value: value.item()
        if isinstance(value, np.generic)
        else value,
    )

with (OUTPUT_DIR / "run_details.txt").open(
    "w",
    encoding="utf-8",
) as file:
    file.write(f"Dataset shape: {df.shape}\n")
    file.write(f"Feature shape: {X.shape}\n")
    file.write(f"Train shape: {X_train.shape}\n")
    file.write(f"Test shape: {X_test.shape}\n")
    file.write(f"Feature columns: {list(X.columns)}\n")
    file.write(f"Target distribution:\n{y.value_counts().sort_index()}\n\n")
    file.write("Final model summary:\n")
    file.write(summary.to_string(index=False))
    file.write("\n")


# The only normal console output from the full run.
print(summary.to_string(index=False))
print(f"\nDetailed reports saved under: {OUTPUT_DIR.resolve()}")
