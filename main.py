import os
import copy
import warnings
import csv as csv_module
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tabulate import tabulate

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.resolve()
K_MANUAL = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42
DATA_PATH = PROJECT_DIR / "diabetes.csv"
OUTPUT_DIR = PROJECT_DIR / "output"

MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

PARAM_GRID = {
    "n_neighbors": list(range(1, 31)),
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"],
}


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"\n{'='*60}")
    print(f"  Dataset dimuat : {path}")
    print(f"  Shape          : {df.shape}")
    print(f"  Kolom          : {list(df.columns)}")
    missing_summary = {col: (df[col] == 0).sum() for col in MISSING_COLS}
    print(f"\n  Missing values (nilai 0) per kolom:")
    for col, cnt in missing_summary.items():
        pct = cnt / len(df) * 100
        print(f"    {col:<30s}: {cnt:3d} ({pct:.1f}%)")
    print(f"{'='*60}")
    return df


def mark_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[MISSING_COLS] = df[MISSING_COLS].replace(0, np.nan)
    return df


def impute_knn(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    df = mark_missing(df)
    imputer = KNNImputer(n_neighbors=k)
    feature_cols = [c for c in df.columns if c != "Outcome"]
    df[feature_cols] = imputer.fit_transform(df[feature_cols])
    return df


def impute_mean(df: pd.DataFrame) -> pd.DataFrame:
    df = mark_missing(df)
    for col in MISSING_COLS:
        df[col] = df[col].fillna(df[col].mean())
    return df


def impute_median(df: pd.DataFrame) -> pd.DataFrame:
    df = mark_missing(df)
    for col in MISSING_COLS:
        df[col] = df[col].fillna(df[col].median())
    return df


def impute_median_per_class(df: pd.DataFrame) -> pd.DataFrame:
    df = mark_missing(df)
    for outcome in [0, 1]:
        mask = df["Outcome"] == outcome
        for col in MISSING_COLS:
            median_val = df.loc[mask, col].median()
            df.loc[mask, col] = df.loc[mask, col].fillna(median_val)
    return df


def train_evaluate(df: pd.DataFrame, label: str):
    X = df.drop(columns=["Outcome"]).values
    y = df["Outcome"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    knn_manual = KNeighborsClassifier(n_neighbors=K_MANUAL)
    knn_manual.fit(X_train, y_train)
    y_pred_manual = knn_manual.predict(X_test)
    acc_manual = accuracy_score(y_test, y_pred_manual)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        KNeighborsClassifier(),
        PARAM_GRID,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X_train, y_train)
    best_knn = gs.best_estimator_
    y_pred_gs = best_knn.predict(X_test)
    acc_gs = accuracy_score(y_test, y_pred_gs)
    best_params = gs.best_params_

    print(f"\n{'─'*60}")
    print(f"  [{label}]")
    print(f"  KNN Manual  (K={K_MANUAL:2d})  → Accuracy: {acc_manual*100:.2f}%")
    print(f"  KNN GridSearch      → Accuracy: {acc_gs*100:.2f}%")
    print(f"  Best params GS      → {best_params}")
    print(f"\n  Classification Report (KNN GridSearch):")
    print(
        classification_report(
            y_test, y_pred_gs, target_names=["Non-Diabetes", "Diabetes"]
        )
    )

    return {
        "label": label,
        "acc_manual": round(acc_manual * 100, 2),
        "acc_gridsearch": round(acc_gs * 100, 2),
        "best_k_gs": best_params["n_neighbors"],
        "best_weight_gs": best_params["weights"],
        "best_metric_gs": best_params["metric"],
    }


def save_csv(df: pd.DataFrame, filename: str) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False, sep=";", decimal=",")
    print(f"  → Saved : {path}")
    return str(path)


def main():
    print("\n" + "=" * 60)
    print("  DIABETES KNN — MULTI PREPROCESSING COMPARISON")
    print(f"  K Manual : {K_MANUAL}   |   Test Size : {TEST_SIZE*100:.0f}%")
    print("=" * 60)

    df_raw = load_data(DATA_PATH)

    print("\n[*] Membuat 5 copy data & melakukan preprocessing ...\n")

    copy1_raw = df_raw.copy()
    copy2_knn = impute_knn(df_raw.copy())
    copy3_mean = impute_mean(df_raw.copy())
    copy4_median = impute_median(df_raw.copy())
    copy5_medcls = impute_median_per_class(df_raw.copy())

    print("[*] Menyimpan copy ke folder output/ ...\n")
    save_csv(copy1_raw, "copy1_raw.csv")
    save_csv(copy2_knn, "copy2_knn_imputed.csv")
    save_csv(copy3_mean, "copy3_mean_imputed.csv")
    save_csv(copy4_median, "copy4_median_imputed.csv")
    save_csv(copy5_medcls, "copy5_median_per_class.csv")

    print("\n[*] Melatih model KNN & KNN GridSearch pada setiap copy ...\n")

    results = []
    datasets = [
        (copy1_raw, "Copy 1 — Raw (No Preprocessing)"),
        (copy2_knn, "Copy 2 — KNN Imputation"),
        (copy3_mean, "Copy 3 — Mean Imputation (Global)"),
        (copy4_median, "Copy 4 — Median Imputation (Global)"),
        (copy5_medcls, "Copy 5 — Median Per Class"),
    ]

    for df_copy, lbl in datasets:
        result = train_evaluate(df_copy, lbl)
        results.append(result)

    print("\n\n" + "=" * 60)
    print("  PERBANDINGAN AKURASI — RINGKASAN")
    print("=" * 60)

    table_data = []
    for r in results:
        table_data.append(
            [
                r["label"],
                f"{r['acc_manual']}%",
                f"{r['acc_gridsearch']}%",
                r["best_k_gs"],
                r["best_weight_gs"],
                r["best_metric_gs"],
            ]
        )

    headers = [
        "Dataset / Preprocessing",
        f"KNN Manual\n(K={K_MANUAL})",
        "KNN GridSearch\n(Best Acc)",
        "Best K\n(GS)",
        "Best Weight\n(GS)",
        "Best Metric\n(GS)",
    ]

    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

    best_manual = max(results, key=lambda x: x["acc_manual"])
    best_gs = max(results, key=lambda x: x["acc_gridsearch"])

    print(f"\n  KNN Manual terbaik     : {best_manual['label']}")
    print(f"     Accuracy            : {best_manual['acc_manual']}%")
    print(f"\n  KNN GridSearch terbaik : {best_gs['label']}")
    print(f"     Accuracy              : {best_gs['acc_gridsearch']}%")

    result_df = pd.DataFrame(results)
    result_df.columns = [
        "Preprocessing",
        f"Accuracy_KNN_K{K_MANUAL} (%)",
        "Accuracy_KNN_GridSearch (%)",
        "Best_K_GridSearch",
        "Best_Weight_GridSearch",
        "Best_Metric_GridSearch",
    ]
    comp_path = save_csv(result_df, "accuracy_comparison.csv")
    print(f"\n  Hasil perbandingan disimpan ke: {comp_path}")
    print("\n" + "=" * 60)
    print("  Selesai!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
