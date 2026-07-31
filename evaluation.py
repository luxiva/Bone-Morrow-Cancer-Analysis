# evaluation.py
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


def evaluate_model(model, X_test, y_test, name="Model"):
    """
    Evaluate a trained model on test data.
    """
    y_pred = model.predict(X_test)

    results = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro")
    }

    print(f"\n===== {name} =====")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"Balanced Accuracy: {results['balanced_accuracy']:.3f}")
    print(f"F1 Macro: {results['f1_macro']:.3f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=3))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return results
