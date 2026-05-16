from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    from .som_world_map_pipeline import (
        CO2_TOTAL_CODE,
        DATA_DIR,
        FIG_DIR,
        POPULATION_CODE,
        SEED,
        SimpleSOM,
        fetch_country_metadata,
        fetch_indicator_long,
    )
except ImportError:
    from som_world_map_pipeline import (
        CO2_TOTAL_CODE,
        DATA_DIR,
        FIG_DIR,
        POPULATION_CODE,
        SEED,
        SimpleSOM,
        fetch_country_metadata,
        fetch_indicator_long,
    )


DYNAMIC_START_YEAR = 2014
DYNAMIC_END_YEAR = 2023
DYNAMIC_TARGET_CODE = "THA"

DYNAMIC_INDICATORS = {
    "gdp_per_capita": "NY.GDP.PCAP.CD",
    "life_expectancy": "SP.DYN.LE00.IN",
    "internet_users_pct": "IT.NET.USER.ZS",
    "urban_population_pct": "SP.URB.TOTL.IN.ZS",
    "unemployment_pct": "SL.UEM.TOTL.ZS",
    "electricity_access_pct": "EG.ELC.ACCS.ZS",
}
DYNAMIC_FEATURE_COLS = list(DYNAMIC_INDICATORS)
DYNAMIC_TRANSFORMED_FEATURE_COLS = [
    "log_gdp_per_capita",
    "life_expectancy",
    "internet_users_pct",
    "urban_population_pct",
    "unemployment_pct",
    "electricity_access_pct",
]
CO2_DYNAMIC_FEATURE = "co2_tons_per_capita"

CLUSTER_COLORS = ["#3366aa", "#dd4477", "#66aa44", "#ffbb33", "#8e63ce", "#00a6a6"]


def year_range(start_year: int, end_year: int) -> list[int]:
    return list(range(start_year, end_year + 1))


def date_range_text(start_year: int, end_year: int) -> str:
    return f"{start_year}:{end_year}"


def indicator_values(
    feature_name: str,
    indicator_code: str,
    valid_codes: set[str],
    start_year: int,
    end_year: int,
    force_refresh: bool,
) -> pd.DataFrame:
    values = fetch_indicator_long(indicator_code, date_range_text(start_year, end_year), force_refresh)
    if values.empty:
        return pd.DataFrame(columns=["country_code", "year", feature_name])
    values = values[values["country_code"].isin(valid_codes)].copy()
    values = values.rename(columns={"value": feature_name})
    return values[["country_code", "year", feature_name]]


def build_country_year_table(
    start_year: int = DYNAMIC_START_YEAR,
    end_year: int = DYNAMIC_END_YEAR,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    country_meta = fetch_country_metadata(force_refresh)
    valid_codes = set(country_meta["country_code"])
    years = year_range(start_year, end_year)

    skeleton = pd.MultiIndex.from_product(
        [country_meta["country_code"].sort_values(), years], names=["country_code", "year"]
    ).to_frame(index=False)
    full = skeleton.merge(country_meta, on="country_code", how="left")

    for feature_name, indicator_code in DYNAMIC_INDICATORS.items():
        values = indicator_values(feature_name, indicator_code, valid_codes, start_year, end_year, force_refresh)
        full = full.merge(values, on=["country_code", "year"], how="left")

    co2 = indicator_values("co2_mt_total", CO2_TOTAL_CODE, valid_codes, start_year, end_year, force_refresh)
    population = indicator_values("population", POPULATION_CODE, valid_codes, start_year, end_year, force_refresh)
    full = full.merge(co2, on=["country_code", "year"], how="left")
    full = full.merge(population, on=["country_code", "year"], how="left")
    full[CO2_DYNAMIC_FEATURE] = full["co2_mt_total"] * 1_000_000 / full["population"]

    full["core_feature_count"] = full[DYNAMIC_FEATURE_COLS].notna().sum(axis=1)
    full["core_features_complete"] = full["core_feature_count"].eq(len(DYNAMIC_FEATURE_COLS))

    model_table = full[full["core_features_complete"]].copy()
    return full, model_table


def engineer_dynamic_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = pd.DataFrame(index=df.index)
    engineered["log_gdp_per_capita"] = np.log10(df["gdp_per_capita"].clip(lower=1))
    engineered["life_expectancy"] = df["life_expectancy"]
    engineered["internet_users_pct"] = df["internet_users_pct"]
    engineered["urban_population_pct"] = df["urban_population_pct"]
    engineered["unemployment_pct"] = df["unemployment_pct"]
    engineered["electricity_access_pct"] = df["electricity_access_pct"]
    return engineered


def build_coverage_report(full: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    features = [*DYNAMIC_FEATURE_COLS, CO2_DYNAMIC_FEATURE]
    rows = []
    total_countries = full["country_code"].nunique()
    for feature in features:
        for year in year_range(start_year, end_year):
            subset = full[full["year"] == year]
            available = int(subset[feature].notna().sum())
            rows.append(
                {
                    "feature": feature,
                    "year": year,
                    "available_countries": available,
                    "total_countries": total_countries,
                    "coverage_pct": round(available / total_countries * 100, 1),
                }
            )
    return pd.DataFrame(rows)


def assign_dynamic_som(
    model_table: pd.DataFrame,
    rows: int,
    cols: int,
    iterations: int,
    n_clusters: int,
) -> tuple[pd.DataFrame, pd.DataFrame, SimpleSOM, np.ndarray, np.ndarray, StandardScaler]:
    transformed = engineer_dynamic_features(model_table)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(transformed[DYNAMIC_TRANSFORMED_FEATURE_COLS])

    som = SimpleSOM(rows=rows, cols=cols, input_dim=x_scaled.shape[1], learning_rate=0.42, sigma=max(rows, cols) / 2)
    som.train(x_scaled, num_iterations=iterations)
    bmu_coords = som.map_vectors(x_scaled)

    results = model_table.copy().reset_index(drop=True)
    results["som_row"] = bmu_coords[:, 0]
    results["som_col"] = bmu_coords[:, 1]

    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=30)
    node_clusters = kmeans.fit_predict(som.weights.reshape(-1, som.input_dim)).reshape(som.rows, som.cols)
    results["macro_cluster"] = [
        int(node_clusters[row, col]) for row, col in results[["som_row", "som_col"]].to_numpy()
    ]

    scaled = pd.concat(
        [
            results[
                [
                    "country_code",
                    "country_name",
                    "region",
                    "income_level",
                    "year",
                    "som_row",
                    "som_col",
                    "macro_cluster",
                ]
            ].reset_index(drop=True),
            transformed.reset_index(drop=True),
            pd.DataFrame(x_scaled, columns=[f"z_{col}" for col in DYNAMIC_TRANSFORMED_FEATURE_COLS]),
        ],
        axis=1,
    )
    return results, scaled, som, node_clusters, x_scaled, scaler


def dynamic_cluster_summary(results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results.groupby("macro_cluster")
        .agg(
            country_years=("country_code", "count"),
            unique_countries=("country_code", "nunique"),
            avg_gdp_per_capita=("gdp_per_capita", "mean"),
            avg_life_expectancy=("life_expectancy", "mean"),
            avg_internet_users_pct=("internet_users_pct", "mean"),
            avg_urban_population_pct=("urban_population_pct", "mean"),
            avg_unemployment_pct=("unemployment_pct", "mean"),
            avg_electricity_access_pct=("electricity_access_pct", "mean"),
        )
        .round(2)
    )
    examples = (
        results.sort_values(["macro_cluster", "country_name", "year"])
        .drop_duplicates(["macro_cluster", "country_code"])
        .groupby("macro_cluster")
        .head(10)
        .groupby("macro_cluster")["country_code"]
        .apply(lambda codes: ", ".join(codes))
        .rename("example_country_codes")
    )
    return summary.join(examples)


def cluster_year_counts(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["year", "macro_cluster"])["country_code"]
        .nunique()
        .rename("countries")
        .reset_index()
        .sort_values(["year", "macro_cluster"])
    )


def movement_scores(results: pd.DataFrame, scaled: pd.DataFrame, expected_years: int) -> pd.DataFrame:
    z_cols = [f"z_{col}" for col in DYNAMIC_TRANSFORMED_FEATURE_COLS]
    merged = results.merge(scaled[["country_code", "year", *z_cols]], on=["country_code", "year"], how="left")
    rows = []
    for country_code, group in merged.sort_values("year").groupby("country_code"):
        if len(group) < 2:
            continue
        group = group.sort_values("year")
        feature_steps = np.linalg.norm(np.diff(group[z_cols].to_numpy(), axis=0), axis=1)
        som_steps = (
            group["som_row"].diff().abs().fillna(0).to_numpy()
            + group["som_col"].diff().abs().fillna(0).to_numpy()
        )
        first = group.iloc[0]
        last = group.iloc[-1]
        rows.append(
            {
                "country_code": country_code,
                "country_name": last["country_name"],
                "region": last["region"],
                "income_level": last["income_level"],
                "start_year": int(first["year"]),
                "end_year": int(last["year"]),
                "years_observed": int(group["year"].nunique()),
                "start_som_row": int(first["som_row"]),
                "start_som_col": int(first["som_col"]),
                "end_som_row": int(last["som_row"]),
                "end_som_col": int(last["som_col"]),
                "complete_full_window": bool(group["year"].nunique() == expected_years),
                "total_feature_distance": round(float(feature_steps.sum()), 4),
                "mean_feature_step": round(float(feature_steps.mean()), 4),
                "max_single_year_feature_step": round(float(feature_steps.max()), 4),
                "total_som_steps": int(som_steps.sum()),
                "max_single_year_som_step": int(som_steps.max()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["complete_full_window", "total_feature_distance", "total_som_steps"],
        ascending=[False, False, False],
    )


def neighbor_drift(
    results: pd.DataFrame,
    scaled: pd.DataFrame,
    target_code: str,
    top_n: int = 10,
) -> pd.DataFrame:
    z_cols = [f"z_{col}" for col in DYNAMIC_TRANSFORMED_FEATURE_COLS]
    merged = results.merge(scaled[["country_code", "year", *z_cols]], on=["country_code", "year"], how="left")
    rows = []
    for year, year_df in merged.groupby("year"):
        target = year_df[year_df["country_code"] == target_code]
        if target.empty:
            continue
        target_row = target.iloc[0]
        target_vector = target_row[z_cols].to_numpy(dtype=float)
        year_df = year_df[year_df["country_code"] != target_code].copy()
        year_df["feature_distance"] = np.linalg.norm(year_df[z_cols].to_numpy(dtype=float) - target_vector, axis=1)
        year_df["som_grid_distance"] = (
            (year_df["som_row"] - target_row["som_row"]).abs()
            + (year_df["som_col"] - target_row["som_col"]).abs()
        )
        for rank, (_, row) in enumerate(year_df.sort_values("feature_distance").head(top_n).iterrows(), start=1):
            rows.append(
                {
                    "target_country_code": target_code,
                    "target_country_name": target_row["country_name"],
                    "year": int(year),
                    "rank": rank,
                    "neighbor_country_code": row["country_code"],
                    "neighbor_country_name": row["country_name"],
                    "neighbor_region": row["region"],
                    "neighbor_income_level": row["income_level"],
                    "feature_distance": round(float(row["feature_distance"]), 4),
                    "som_grid_distance": int(row["som_grid_distance"]),
                    "target_som_row": int(target_row["som_row"]),
                    "target_som_col": int(target_row["som_col"]),
                    "neighbor_som_row": int(row["som_row"]),
                    "neighbor_som_col": int(row["som_col"]),
                }
            )
    return pd.DataFrame(rows)


def time_shifted_similarity(
    results: pd.DataFrame,
    scaled: pd.DataFrame,
    target_code: str,
    target_year: int,
    top_n: int = 25,
) -> pd.DataFrame:
    z_cols = [f"z_{col}" for col in DYNAMIC_TRANSFORMED_FEATURE_COLS]
    merged = results.merge(scaled[["country_code", "year", *z_cols]], on=["country_code", "year"], how="left")
    target = merged[(merged["country_code"] == target_code) & (merged["year"] == target_year)]
    if target.empty:
        return pd.DataFrame()
    target_row = target.iloc[0]
    target_vector = target_row[z_cols].to_numpy(dtype=float)
    out = merged.copy()
    out["feature_distance"] = np.linalg.norm(out[z_cols].to_numpy(dtype=float) - target_vector, axis=1)
    out = out[~((out["country_code"] == target_code) & (out["year"] == target_year))]
    out["year_gap"] = out["year"] - target_year
    cols = [
        "country_code",
        "country_name",
        "region",
        "income_level",
        "year",
        "year_gap",
        "som_row",
        "som_col",
        "macro_cluster",
        "feature_distance",
        *DYNAMIC_FEATURE_COLS,
    ]
    return out.sort_values("feature_distance")[cols].head(top_n).reset_index(drop=True)


def jittered_positions(df: pd.DataFrame, amount: float = 0.18) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    return (
        df["som_col"].to_numpy(dtype=float) + rng.uniform(-amount, amount, len(df)),
        df["som_row"].to_numpy(dtype=float) + rng.uniform(-amount, amount, len(df)),
    )


def save_dynamic_component_planes(som: SimpleSOM) -> None:
    labels = {
        "log_gdp_per_capita": "log GDP pc",
        "life_expectancy": "life expectancy",
        "internet_users_pct": "internet users",
        "urban_population_pct": "urban population",
        "unemployment_pct": "unemployment",
        "electricity_access_pct": "electricity access",
    }
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.ravel()
    for idx, feature in enumerate(DYNAMIC_TRANSFORMED_FEATURE_COLS):
        ax = axes[idx]
        image = ax.imshow(som.weights[:, :, idx], cmap="viridis", origin="upper")
        ax.set_title(labels[feature], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Dynamic SOM component planes, standardized country-year features", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dynamic_som_component_planes.png", bbox_inches="tight")
    plt.close(fig)


def save_dynamic_som_year_facets(
    som: SimpleSOM,
    results: pd.DataFrame,
    node_clusters: np.ndarray,
    target_code: str,
    years: list[int],
) -> None:
    cmap = plt.matplotlib.colors.ListedColormap(CLUSTER_COLORS[: len(CLUSTER_COLORS)])
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, year in zip(axes, years):
        subset = results[results["year"] == year]
        ax.imshow(node_clusters, cmap=cmap, origin="upper", alpha=0.28)
        x, y = jittered_positions(subset)
        ax.scatter(
            x,
            y,
            s=24,
            c=subset["macro_cluster"].map(lambda c: CLUSTER_COLORS[int(c)]),
            edgecolor="white",
            linewidth=0.35,
            alpha=0.9,
        )
        target = subset[subset["country_code"] == target_code]
        if not target.empty:
            row = target.iloc[0]
            ax.scatter(row["som_col"], row["som_row"], s=140, color="#111111", marker="*", edgecolor="white")
            ax.text(
                row["som_col"],
                row["som_row"] - 0.35,
                target_code,
                fontsize=9,
                weight="bold",
                ha="center",
                color="#111111",
            )
        ax.set_title(str(year))
        ax.set_xticks(range(som.cols))
        ax.set_yticks(range(som.rows))
        ax.grid(color="white", alpha=0.22, linewidth=0.8)
    handles = [
        mpatches.Patch(color=CLUSTER_COLORS[i], label=f"Macro cluster {i}")
        for i in sorted(results["macro_cluster"].unique())
    ]
    fig.legend(handles=handles, loc="center right", frameon=False)
    fig.suptitle("Dynamic SOM snapshots: country-year positions by selected years", fontsize=13)
    fig.tight_layout(rect=(0, 0, 0.88, 0.97))
    fig.savefig(FIG_DIR / "dynamic_som_year_snapshots.png", bbox_inches="tight")
    plt.close(fig)


def save_country_trajectory(results: pd.DataFrame, target_code: str) -> None:
    target = results[results["country_code"] == target_code].sort_values("year")
    if target.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        target["som_col"],
        target["som_row"],
        c=target["year"],
        cmap="plasma",
        s=np.linspace(70, 160, len(target)),
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    ax.plot(target["som_col"], target["som_row"], color="#222222", linewidth=2.0, alpha=0.75, zorder=2)
    for _, row in target.iterrows():
        ax.text(row["som_col"] + 0.12, row["som_row"] + 0.12, str(int(row["year"])), fontsize=8)
    ax.invert_yaxis()
    ax.set_title(f"{target_code} trajectory across the dynamic SOM")
    ax.set_xlabel("SOM column")
    ax.set_ylabel("SOM row")
    ax.grid(color="#dddddd", linewidth=0.8)
    fig.colorbar(scatter, ax=ax, label="Year")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"dynamic_som_trajectory_{target_code}.png", bbox_inches="tight")
    plt.close(fig)


def save_neighbor_drift_chart(neighbors: pd.DataFrame, target_code: str) -> None:
    if neighbors.empty:
        return
    top = neighbors[neighbors["rank"] == 1].sort_values("year")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(top["year"], top["feature_distance"], marker="o", color="#3366aa", linewidth=2)
    for _, row in top.iterrows():
        ax.text(
            row["year"],
            row["feature_distance"] + 0.03,
            row["neighbor_country_code"],
            ha="center",
            fontsize=8,
        )
    ax.set_title(f"Nearest-neighbor drift for {target_code}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Distance to closest country-year peer")
    ax.set_xticks(sorted(top["year"].unique()))
    ax.grid(color="#dddddd", linewidth=0.8, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"neighbor_drift_{target_code}.png", bbox_inches="tight")
    plt.close(fig)


def save_movement_scores_chart(scores: pd.DataFrame) -> None:
    if scores.empty:
        return
    complete_scores = scores[scores["complete_full_window"]].copy()
    top = complete_scores.head(20).sort_values("total_feature_distance")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["country_code"], top["total_feature_distance"], color="#dd4477")
    ax.set_title("Countries with the largest complete-window profile movement")
    ax.set_xlabel("Cumulative movement in standardized feature space")
    for i, (_, row) in enumerate(top.iterrows()):
        ax.text(row["total_feature_distance"] + 0.05, i, row["country_name"], va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "country_movement_scores.png", bbox_inches="tight")
    plt.close(fig)


def save_world_cluster_scatter(results: pd.DataFrame, years: list[int]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, year in zip(axes, years):
        subset = results[(results["year"] == year) & results["longitude"].notna() & results["latitude"].notna()]
        ax.scatter(
            subset["longitude"],
            subset["latitude"],
            s=24,
            c=subset["macro_cluster"].map(lambda c: CLUSTER_COLORS[int(c)]),
            edgecolor="white",
            linewidth=0.35,
            alpha=0.9,
        )
        ax.set_title(str(year))
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)
        ax.grid(color="#dddddd", linewidth=0.7)
    handles = [
        mpatches.Patch(color=CLUSTER_COLORS[i], label=f"Macro cluster {i}")
        for i in sorted(results["macro_cluster"].unique())
    ]
    fig.legend(handles=handles, loc="center right", frameon=False)
    fig.suptitle("Geographic scatter of dynamic SOM macro clusters", fontsize=13)
    fig.tight_layout(rect=(0, 0, 0.88, 0.95))
    fig.savefig(FIG_DIR / "dynamic_world_cluster_scatter.png", bbox_inches="tight")
    plt.close(fig)


def save_time_shifted_similarity_chart(similarities: pd.DataFrame, target_code: str, target_year: int) -> None:
    if similarities.empty:
        return
    top = similarities.head(14).iloc[::-1]
    labels = top["country_code"] + " " + top["year"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels, top["feature_distance"], color="#00a6a6")
    ax.set_title(f"Most similar country-years to {target_code} {target_year}")
    ax.set_xlabel("Distance in standardized feature space")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"time_shifted_similarity_{target_code}_{target_year}.png", bbox_inches="tight")
    plt.close(fig)


def selected_snapshot_years(start_year: int, end_year: int) -> list[int]:
    years = year_range(start_year, end_year)
    picks = [years[0], years[len(years) // 3], years[(len(years) * 2) // 3], years[-1]]
    return sorted(set(picks))


def run_dynamic_pipeline(
    start_year: int = DYNAMIC_START_YEAR,
    end_year: int = DYNAMIC_END_YEAR,
    rows: int = 10,
    cols: int = 12,
    iterations: int = 14000,
    n_clusters: int = 6,
    target_code: str = DYNAMIC_TARGET_CODE,
    force_refresh: bool = False,
) -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    full, model_table = build_country_year_table(start_year, end_year, force_refresh)
    coverage = build_coverage_report(full, start_year, end_year)

    full.to_csv(DATA_DIR / "world_bank_country_year_indicators_full.csv", index=False)
    coverage.to_csv(DATA_DIR / "dynamic_data_coverage.csv", index=False)
    model_table.to_csv(DATA_DIR / "country_year_features.csv", index=False)

    results, scaled, som, node_clusters, x_scaled, _ = assign_dynamic_som(
        model_table, rows=rows, cols=cols, iterations=iterations, n_clusters=n_clusters
    )
    result_cols = [
        "country_code",
        "country_name",
        "region",
        "income_level",
        "capital_city",
        "latitude",
        "longitude",
        "year",
        "som_row",
        "som_col",
        "macro_cluster",
        *DYNAMIC_FEATURE_COLS,
        "co2_mt_total",
        "population",
        CO2_DYNAMIC_FEATURE,
    ]
    results[result_cols].to_csv(DATA_DIR / "country_year_som_results.csv", index=False)
    scaled.to_csv(DATA_DIR / "country_year_som_scaled_features.csv", index=False)

    cluster_summary = dynamic_cluster_summary(results)
    cluster_summary.to_csv(DATA_DIR / "dynamic_macro_cluster_summary.csv")
    cluster_counts = cluster_year_counts(results)
    cluster_counts.to_csv(DATA_DIR / "dynamic_cluster_year_counts.csv", index=False)

    expected_years = end_year - start_year + 1
    scores = movement_scores(results, scaled, expected_years=expected_years)
    scores.to_csv(DATA_DIR / "movement_scores.csv", index=False)

    neighbors = neighbor_drift(results, scaled, target_code=target_code, top_n=10)
    neighbors.to_csv(DATA_DIR / f"neighbor_drift_{target_code}.csv", index=False)

    target_year = min(end_year, int(results[results["country_code"] == target_code]["year"].max()))
    shifted = time_shifted_similarity(results, scaled, target_code=target_code, target_year=target_year, top_n=25)
    shifted.to_csv(DATA_DIR / f"time_shifted_similarity_{target_code}_{target_year}.csv", index=False)

    snapshot_years = selected_snapshot_years(start_year, end_year)
    save_dynamic_component_planes(som)
    save_dynamic_som_year_facets(som, results, node_clusters, target_code, snapshot_years)
    save_country_trajectory(results, target_code)
    save_neighbor_drift_chart(neighbors, target_code)
    save_movement_scores_chart(scores)
    save_world_cluster_scatter(results, snapshot_years)
    save_time_shifted_similarity_chart(shifted, target_code, target_year)

    target_years = results[results["country_code"] == target_code]["year"].astype(int).tolist()
    summary = {
        "year_range": f"{start_year}-{end_year}",
        "years": year_range(start_year, end_year),
        "countries_total": int(full["country_code"].nunique()),
        "country_years_total": int(len(full)),
        "country_years_complete_core_features": int(len(model_table)),
        "countries_with_all_years_complete": int(
            model_table.groupby("country_code")["year"].nunique().eq(end_year - start_year + 1).sum()
        ),
        "features_used_for_dynamic_som": DYNAMIC_FEATURE_COLS,
        "som_grid": f"{rows}x{cols}",
        "macro_clusters": int(n_clusters),
        "quantization_error": round(som.quantization_error(x_scaled), 3),
        "topographic_error": round(som.topographic_error(x_scaled), 3),
        "target_country_code": target_code,
        "target_years_available": target_years,
        "time_shifted_target_year": int(target_year),
        "top_neighbor_by_year": neighbors[neighbors["rank"] == 1][
            ["year", "neighbor_country_code", "neighbor_country_name", "feature_distance", "som_grid_distance"]
        ].to_dict(orient="records"),
        "largest_complete_window_movers": scores[scores["complete_full_window"]].head(10)[
            [
                "country_code",
                "country_name",
                "total_feature_distance",
                "total_som_steps",
                "years_observed",
                "max_single_year_feature_step",
            ]
        ].to_dict(orient="records"),
        "largest_partial_window_movers": scores[~scores["complete_full_window"]].head(5)[
            [
                "country_code",
                "country_name",
                "total_feature_distance",
                "total_som_steps",
                "years_observed",
                "max_single_year_feature_step",
            ]
        ].to_dict(orient="records"),
    }
    (DATA_DIR / "dynamic_project_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the dynamic country-year SOM pipeline.")
    parser.add_argument("--start-year", type=int, default=DYNAMIC_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DYNAMIC_END_YEAR)
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--cols", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=14000)
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--target-code", default=DYNAMIC_TARGET_CODE)
    parser.add_argument("--force-refresh", action="store_true", help="Ignore cached World Bank JSON and call the API.")
    args = parser.parse_args()

    summary = run_dynamic_pipeline(
        start_year=args.start_year,
        end_year=args.end_year,
        rows=args.rows,
        cols=args.cols,
        iterations=args.iterations,
        n_clusters=args.clusters,
        target_code=args.target_code.upper(),
        force_refresh=args.force_refresh,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
