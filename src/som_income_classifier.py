from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

try:
    from .som_world_map_pipeline import (
        DATA_DIR,
        FEATURE_COLS,
        FIG_DIR,
        SEED,
        TRANSFORMED_FEATURE_COLS,
        SimpleSOM,
        build_country_feature_table,
        engineer_features,
    )
except ImportError:
    from som_world_map_pipeline import (
        DATA_DIR,
        FEATURE_COLS,
        FIG_DIR,
        SEED,
        TRANSFORMED_FEATURE_COLS,
        SimpleSOM,
        build_country_feature_table,
        engineer_features,
    )


INPUT_PATH = DATA_DIR / "world_bank_som_input_complete_cases.csv"
TARGET_COL = "income_level"
EXCLUDED_TARGETS = {"Not classified"}
CLASS_ORDER = ["Low income", "Lower middle income", "Upper middle income", "High income"]
CLASS_COLORS = {
    "Low income": "#d95f02",
    "Lower middle income": "#f0a202",
    "Upper middle income": "#1b9e77",
    "High income": "#2a6fbb",
    "Unlabeled": "#bbbbbb",
}


def load_static_snapshot(force_refresh: bool = False) -> pd.DataFrame:
    if INPUT_PATH.exists() and not force_refresh:
        df = pd.read_csv(INPUT_PATH)
    else:
        _, df = build_country_feature_table(force_refresh=force_refresh)
    df = df[~df[TARGET_COL].isin(EXCLUDED_TARGETS)].copy()
    df = df.dropna(subset=[TARGET_COL, *FEATURE_COLS]).reset_index(drop=True)
    return df


def make_test_split(df: pd.DataFrame, test_size: int, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, df[TARGET_COL]))
    return df.iloc[train_idx].copy().reset_index(drop=True), df.iloc[test_idx].copy().reset_index(drop=True)


def majority_label(labels: pd.Series) -> str:
    counts = Counter(labels)
    return counts.most_common(1)[0][0]


def fill_empty_node_labels(node_labels: dict[tuple[int, int], str], rows: int, cols: int, global_majority: str) -> dict[tuple[int, int], str]:
    filled = dict(node_labels)
    labeled_nodes = list(node_labels)
    for row in range(rows):
        for col in range(cols):
            key = (row, col)
            if key in filled:
                continue
            if not labeled_nodes:
                filled[key] = global_majority
                continue
            nearest = min(labeled_nodes, key=lambda node: abs(node[0] - row) + abs(node[1] - col))
            filled[key] = node_labels.get(nearest, global_majority)
    return filled


def train_som_classifier(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    rows: int,
    cols: int,
    iterations: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SimpleSOM, np.ndarray, np.ndarray]:
    transformed_train = engineer_features(train_df)
    transformed_test = engineer_features(test_df)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(transformed_train[TRANSFORMED_FEATURE_COLS])
    x_test = scaler.transform(transformed_test[TRANSFORMED_FEATURE_COLS])

    som = SimpleSOM(rows=rows, cols=cols, input_dim=x_train.shape[1], learning_rate=0.42, sigma=max(rows, cols) / 1.5)
    som.train(x_train, num_iterations=iterations)

    train_bmu = som.map_vectors(x_train)
    test_bmu = som.map_vectors(x_test)

    train_out = train_df.copy()
    train_out["split"] = "train"
    train_out["som_row"] = train_bmu[:, 0]
    train_out["som_col"] = train_bmu[:, 1]

    raw_node_labels = {}
    node_rows = []
    for (row, col), group in train_out.groupby(["som_row", "som_col"]):
        class_counts = group[TARGET_COL].value_counts()
        label = majority_label(group[TARGET_COL])
        raw_node_labels[(int(row), int(col))] = label
        node_rows.append(
            {
                "som_row": int(row),
                "som_col": int(col),
                "node_label": label,
                "train_count": int(len(group)),
                **{f"train_{label_name}": int(class_counts.get(label_name, 0)) for label_name in CLASS_ORDER},
            }
        )

    global_majority = majority_label(train_out[TARGET_COL])
    filled_labels = fill_empty_node_labels(raw_node_labels, rows, cols, global_majority)

    all_node_rows = []
    existing = {(row["som_row"], row["som_col"]): row for row in node_rows}
    for row in range(rows):
        for col in range(cols):
            base = existing.get((row, col), {"som_row": row, "som_col": col, "train_count": 0})
            base["node_label"] = filled_labels[(row, col)]
            for label_name in CLASS_ORDER:
                base.setdefault(f"train_{label_name}", 0)
            base["was_empty_in_train"] = base["train_count"] == 0
            all_node_rows.append(base)
    nodes = pd.DataFrame(all_node_rows).sort_values(["som_row", "som_col"]).reset_index(drop=True)

    test_out = test_df.copy()
    test_out["split"] = "test"
    test_out["som_row"] = test_bmu[:, 0]
    test_out["som_col"] = test_bmu[:, 1]
    test_out["predicted_income_level"] = [filled_labels[(int(row), int(col))] for row, col in test_bmu]
    test_out["correct"] = test_out[TARGET_COL].eq(test_out["predicted_income_level"])

    return train_out, test_out, nodes, som, x_train, x_test


def save_classifier_map(train_out: pd.DataFrame, test_out: pd.DataFrame, nodes: pd.DataFrame, rows: int, cols: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    for _, node in nodes.iterrows():
        color = CLASS_COLORS[node["node_label"]]
        alpha = 0.32 if node["was_empty_in_train"] else 0.58
        rect = plt.Rectangle((node["som_col"] - 0.5, node["som_row"] - 0.5), 1, 1, facecolor=color, alpha=alpha, edgecolor="white")
        ax.add_patch(rect)
        ax.text(
            node["som_col"],
            node["som_row"] - 0.22,
            node["node_label"].replace(" income", ""),
            ha="center",
            va="center",
            fontsize=9,
            weight="bold",
            color="#111111",
        )
        ax.text(
            node["som_col"],
            node["som_row"] + 0.18,
            f"n={int(node['train_count'])}",
            ha="center",
            va="center",
            fontsize=8,
            color="#333333",
        )

    rng = np.random.default_rng(SEED)
    ax.scatter(
        train_out["som_col"] + rng.uniform(-0.12, 0.12, len(train_out)),
        train_out["som_row"] + rng.uniform(-0.12, 0.12, len(train_out)),
        s=28,
        c=train_out[TARGET_COL].map(CLASS_COLORS),
        edgecolor="white",
        linewidth=0.45,
        alpha=0.82,
        label="Train",
    )

    for _, row in test_out.iterrows():
        marker = "P" if row["correct"] else "X"
        edge = "#111111" if row["correct"] else "#b00020"
        ax.scatter(row["som_col"], row["som_row"], s=155, marker=marker, color=CLASS_COLORS[row[TARGET_COL]], edgecolor=edge, linewidth=1.5)
        ax.text(row["som_col"] + 0.08, row["som_row"] + 0.34, row["country_code"], fontsize=8, weight="bold")

    handles = [mpatches.Patch(color=CLASS_COLORS[label], label=label) for label in CLASS_ORDER]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    ax.set_xlim(-0.55, cols - 0.45)
    ax.set_ylim(rows - 0.45, -0.55)
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_title("SOM majority-vote income classifier")
    ax.set_xlabel("SOM column")
    ax.set_ylabel("SOM row")
    ax.grid(color="white", linewidth=1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "som_income_classifier_map.png", bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(test_out: pd.DataFrame) -> None:
    labels = [label for label in CLASS_ORDER if label in set(test_out[TARGET_COL]) or label in set(test_out["predicted_income_level"])]
    cm = confusion_matrix(test_out[TARGET_COL], test_out["predicted_income_level"], labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([label.replace(" income", "") for label in labels], rotation=25, ha="right")
    ax.set_yticklabels([label.replace(" income", "") for label in labels])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("SOM classifier confusion matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="#111111")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "som_income_classifier_confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)


def run_som_income_classifier(
    test_size: int = 15,
    rows: int = 4,
    cols: int = 1,
    iterations: int = 6000,
    force_refresh: bool = False,
) -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_static_snapshot(force_refresh=force_refresh)
    if rows * cols != len(CLASS_ORDER):
        raise ValueError(f"This experiment expects rows * cols to equal {len(CLASS_ORDER)} income classes.")
    train_df, test_df = make_test_split(df, test_size=test_size, random_state=SEED)
    train_out, test_out, nodes, som, x_train, x_test = train_som_classifier(
        train_df, test_df, rows=rows, cols=cols, iterations=iterations
    )

    train_cols = [
        "country_code",
        "country_name",
        "region",
        "income_level",
        "som_row",
        "som_col",
        *FEATURE_COLS,
    ]
    test_cols = [
        "country_code",
        "country_name",
        "region",
        "income_level",
        "predicted_income_level",
        "correct",
        "som_row",
        "som_col",
        *FEATURE_COLS,
    ]
    train_out[train_cols].to_csv(DATA_DIR / "som_income_classifier_train.csv", index=False)
    test_out[test_cols].to_csv(DATA_DIR / "som_income_classifier_test_predictions.csv", index=False)
    nodes.to_csv(DATA_DIR / "som_income_classifier_node_labels.csv", index=False)

    labels = [label for label in CLASS_ORDER if label in set(test_out[TARGET_COL]) or label in set(test_out["predicted_income_level"])]
    metrics = {
        "target": TARGET_COL,
        "excluded_targets": sorted(EXCLUDED_TARGETS),
        "classes": CLASS_ORDER,
        "features": FEATURE_COLS,
        "countries_total_after_filter": int(len(df)),
        "train_countries": int(len(train_out)),
        "test_countries": int(len(test_out)),
        "som_grid": f"{rows}x{cols}",
        "som_nodes": rows * cols,
        "iterations": int(iterations),
        "accuracy": round(float(accuracy_score(test_out[TARGET_COL], test_out["predicted_income_level"])), 4),
        "macro_f1": round(float(f1_score(test_out[TARGET_COL], test_out["predicted_income_level"], labels=labels, average="macro", zero_division=0)), 4),
        "quantization_error_train": round(som.quantization_error(x_train), 4),
        "quantization_error_test": round(float(np.mean([np.linalg.norm(x - som.weights[som.find_bmu(x)]) for x in x_test])), 4),
        "node_labels": nodes.to_dict(orient="records"),
        "test_predictions": test_out[
            ["country_code", "country_name", "income_level", "predicted_income_level", "correct", "som_row", "som_col"]
        ].to_dict(orient="records"),
        "classification_report": classification_report(
            test_out[TARGET_COL],
            test_out["predicted_income_level"],
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }
    (DATA_DIR / "som_income_classifier_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    save_classifier_map(train_out, test_out, nodes, rows, cols)
    save_confusion_matrix(test_out)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a SOM majority-vote classifier for World Bank income level.")
    parser.add_argument("--test-size", type=int, default=15)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=6000)
    parser.add_argument("--force-refresh", action="store_true", help="Rebuild the static country snapshot from API/cache.")
    args = parser.parse_args()
    metrics = run_som_income_classifier(
        test_size=args.test_size,
        rows=args.rows,
        cols=args.cols,
        iterations=args.iterations,
        force_refresh=args.force_refresh,
    )
    print(json.dumps({k: metrics[k] for k in ["accuracy", "macro_f1", "som_grid", "train_countries", "test_countries"]}, indent=2))


if __name__ == "__main__":
    main()
