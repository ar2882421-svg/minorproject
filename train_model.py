"""Train and save classification models and evaluation metrics."""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "student_data.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_PATH = MODEL_DIR / "student_outcome_model.joblib"
METRICS_PATH = REPORT_DIR / "model_metrics.json"

def load_data():
    df = pd.read_csv(DATA_PATH, sep=None, engine="python")
    target_col = next((c for c in df.columns if c.strip().lower() == "target"), None)
    if target_col is None:
        raise ValueError("Dataset must contain a 'Target' column.")
    y = df[target_col].astype(str).str.strip()
    X = df.drop(columns=[target_col])
    for col in X.columns:
        converted = pd.to_numeric(X[col], errors="coerce")
        if converted.notna().sum() >= max(1, int(0.8 * len(X))):
            X[col] = converted
    return X, y

def train_and_save():
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    transformers = []
    if numeric_cols:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_cols))
    preprocessor = ColumnTransformer(transformers=transformers)
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1
        )
    }
    results = []
    best_name, best_pipe, best_accuracy = None, None, -1
    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        results.append({"model": name, "accuracy": round(float(accuracy), 4)})
        if accuracy > best_accuracy:
            best_name, best_pipe, best_accuracy = name, pipe, accuracy
    predictions = best_pipe.predict(X_test)
    labels = sorted(y.unique())
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    metrics = {
        "selected_model": best_name,
        "test_accuracy": round(float(best_accuracy), 4),
        "model_comparison": results,
        "class_labels": labels,
        "classification_report": report,
        "feature_columns": list(X.columns),
        "dataset_rows": int(len(X)),
        "test_rows": int(len(X_test)),
        "note": "Held-out test split; results are dataset-specific."
    }
    joblib.dump(best_pipe, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    cm = confusion_matrix(y_test, predictions, labels=labels)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix — {best_name}")
    plt.xlabel("Predicted outcome")
    plt.ylabel("Actual outcome")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()
    if best_name == "Random Forest":
        try:
            names = best_pipe.named_steps["preprocessor"].get_feature_names_out()
            values = best_pipe.named_steps["classifier"].feature_importances_
            imp = pd.Series(values, index=names).sort_values(ascending=False).head(15).sort_values()
            plt.figure(figsize=(9, 6))
            imp.plot(kind="barh")
            plt.title("Top 15 Feature Importances")
            plt.tight_layout()
            plt.savefig(REPORT_DIR / "feature_importance.png", dpi=160)
            plt.close()
        except Exception:
            pass
    return metrics

if __name__ == "__main__":
    metrics = train_and_save()
    print(f"Model: {metrics['selected_model']} | Test accuracy: {metrics['test_accuracy']:.1%}")
