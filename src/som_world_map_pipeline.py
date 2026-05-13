from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_JSON_DIR = DATA_DIR / "raw_world_bank_json"
FIG_DIR = ROOT / "figures"

WORLD_BANK_BASE = "https://api.worldbank.org/v2"
DATE_RANGE = "2018:2024"

INDICATORS = {
    "gdp_per_capita": "NY.GDP.PCAP.CD",
    "life_expectancy": "SP.DYN.LE00.IN",
    "internet_users_pct": "IT.NET.USER.ZS",
    "urban_population_pct": "SP.URB.TOTL.IN.ZS",
    "unemployment_pct": "SL.UEM.TOTL.ZS",
    "electricity_access_pct": "EG.ELC.ACCS.ZS",
}
CO2_TOTAL_CODE = "CC.CO2.EMSE.EL"
POPULATION_CODE = "SP.POP.TOTL"
CO2_FEATURE = "co2_tons_per_capita_2018"
FEATURE_COLS = list(INDICATORS) + [CO2_FEATURE]
TRANSFORMED_FEATURE_COLS = [
    "log_gdp_per_capita",
    "life_expectancy",
    "internet_users_pct",
    "urban_population_pct",
    "unemployment_pct",
    "electricity_access_pct",
    "log1p_co2_tons_per_capita_2018",
]


def cache_name(text: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in text)
    return f"{safe}.json"


def request_world_bank_json(url: str, cache_key: str, force_refresh: bool = False):
    RAW_JSON_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_JSON_DIR / cache_name(cache_key)
    if cache_path.exists() and not force_refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    last_error = None
    for attempt in range(1, 5):
        try:
            response = requests.get(url, timeout=(10, 90))
            response.raise_for_status()
            data = response.json()
            cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            return data
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"World Bank request failed: {url}") from last_error


def normalize_country_code(row: pd.Series) -> str:
    iso3 = row.get("countryiso3code", "")
    if isinstance(iso3, str) and iso3:
        return iso3
    return row.get("country", {}).get("id", "")


def fetch_country_metadata(force_refresh: bool = False) -> pd.DataFrame:
    url = f"{WORLD_BANK_BASE}/country?format=json&per_page=400"
    data = request_world_bank_json(url, "countries", force_refresh)
    rows = []
    for item in data[1]:
        if item["region"]["value"] == "Aggregates":
            continue
        rows.append(
            {
                "country_code": item["id"],
                "country_name": item["name"],
                "region": item["region"]["value"].strip(),
                "income_level": item["incomeLevel"]["value"],
                "capital_city": item["capitalCity"],
                "latitude": pd.to_numeric(item["latitude"], errors="coerce"),
                "longitude": pd.to_numeric(item["longitude"], errors="coerce"),
            }
        )
    return pd.DataFrame(rows)


def fetch_indicator_long(code: str, date_range: str = DATE_RANGE, force_refresh: bool = False) -> pd.DataFrame:
    url = f"{WORLD_BANK_BASE}/country/all/indicator/{code}?format=json&per_page=20000&date={date_range}"
    data = request_world_bank_json(url, f"{code}_{date_range}", force_refresh)
    if isinstance(data, list) and data and isinstance(data[0], dict) and "message" in data[0]:
        raise ValueError(f"World Bank API error for {code}: {data[0]['message'][0]['value']}")

    df = pd.DataFrame(data[1])
    if df.empty:
        return df
    df["country_code"] = df.apply(normalize_country_code, axis=1)
    df["country_name"] = df["country"].apply(lambda x: x["value"])
    df["year"] = df["date"].astype(int)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["country_code", "country_name", "year", "value"]]


def latest_values_for_indicator(feature_name: str, code: str, valid_codes: set[str], force_refresh: bool) -> pd.DataFrame:
    df = fetch_indicator_long(code, DATE_RANGE, force_refresh)
    df = df[df["country_code"].isin(valid_codes) & df["value"].notna()].copy()
    latest = df.sort_values("year").groupby("country_code", as_index=False).tail(1)
    return latest[["country_code", "value", "year"]].rename(
        columns={"value": feature_name, "year": f"{feature_name}_year"}
    )


def build_country_feature_table(force_refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    country_meta = fetch_country_metadata(force_refresh)
    valid_codes = set(country_meta["country_code"])
    full = country_meta.copy()

    for feature_name, code in INDICATORS.items():
        latest = latest_values_for_indicator(feature_name, code, valid_codes, force_refresh)
        full = full.merge(latest, on="country_code", how="left")

    co2 = fetch_indicator_long(CO2_TOTAL_CODE, "2018:2018", force_refresh)
    pop = fetch_indicator_long(POPULATION_CODE, "2018:2018", force_refresh)
    co2 = co2[co2["country_code"].isin(valid_codes) & co2["value"].notna()].rename(
        columns={"value": "co2_mt_total_2018"}
    )
    pop = pop[pop["country_code"].isin(valid_codes) & pop["value"].notna()].rename(
        columns={"value": "population_2018"}
    )
    co2_pc = co2[["country_code", "co2_mt_total_2018"]].merge(
        pop[["country_code", "population_2018"]], on="country_code", how="inner"
    )
    co2_pc[CO2_FEATURE] = co2_pc["co2_mt_total_2018"] * 1_000_000 / co2_pc["population_2018"]
    full = full.merge(
        co2_pc[["country_code", "co2_mt_total_2018", "population_2018", CO2_FEATURE]],
        on="country_code",
        how="left",
    )
    complete = full.dropna(subset=FEATURE_COLS).copy()
    return full, complete


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = pd.DataFrame(index=df.index)
    engineered["log_gdp_per_capita"] = np.log10(df["gdp_per_capita"].clip(lower=1))
    engineered["life_expectancy"] = df["life_expectancy"]
    engineered["internet_users_pct"] = df["internet_users_pct"]
    engineered["urban_population_pct"] = df["urban_population_pct"]
    engineered["unemployment_pct"] = df["unemployment_pct"]
    engineered["electricity_access_pct"] = df["electricity_access_pct"]
    engineered["log1p_co2_tons_per_capita_2018"] = np.log1p(df[CO2_FEATURE].clip(lower=0))
    return engineered


class SimpleSOM:
    def __init__(self, rows: int, cols: int, input_dim: int, learning_rate: float = 0.45, sigma: float = 4.5):
        self.rows = rows
        self.cols = cols
        self.input_dim = input_dim
        self.initial_learning_rate = learning_rate
        self.initial_sigma = sigma
        self.rng = np.random.default_rng(SEED)
        self.weights = self.rng.normal(0, 1, size=(rows, cols, input_dim))
        rr, cc = np.indices((rows, cols))
        self.grid = np.stack([rr, cc], axis=-1)

    def initialize_from_data(self, data: np.ndarray) -> None:
        picks = self.rng.choice(data.shape[0], size=self.rows * self.cols, replace=True)
        self.weights = data[picks].reshape(self.rows, self.cols, self.input_dim)
        self.weights += self.rng.normal(0, 0.01, size=self.weights.shape)

    def find_bmu(self, x: np.ndarray) -> tuple[int, int]:
        distances = np.linalg.norm(self.weights - x, axis=2)
        return np.unravel_index(np.argmin(distances), (self.rows, self.cols))

    def train(self, data: np.ndarray, num_iterations: int = 9000) -> None:
        self.initialize_from_data(data)
        time_constant = num_iterations / math.log(self.initial_sigma + 1)
        for step in range(num_iterations):
            x = data[self.rng.integers(0, data.shape[0])]
            bmu = self.find_bmu(x)
            learning_rate = self.initial_learning_rate * math.exp(-step / num_iterations)
            sigma = max(self.initial_sigma * math.exp(-step / time_constant), 1e-4)
            grid_distance_sq = (self.grid[..., 0] - bmu[0]) ** 2 + (self.grid[..., 1] - bmu[1]) ** 2
            influence = np.exp(-grid_distance_sq / (2 * sigma**2))[..., np.newaxis]
            self.weights += learning_rate * influence * (x - self.weights)

    def map_vectors(self, data: np.ndarray) -> np.ndarray:
        return np.array([self.find_bmu(x) for x in data])

    def quantization_error(self, data: np.ndarray) -> float:
        return float(np.mean([np.linalg.norm(x - self.weights[self.find_bmu(x)]) for x in data]))

    def topographic_error(self, data: np.ndarray) -> float:
        errors = 0
        flat_weights = self.weights.reshape(-1, self.input_dim)
        for x in data:
            first, second = np.argsort(np.linalg.norm(flat_weights - x, axis=1))[:2]
            r1, c1 = divmod(first, self.cols)
            r2, c2 = divmod(second, self.cols)
            if max(abs(r1 - r2), abs(c1 - c2)) > 1:
                errors += 1
        return errors / len(data)

    def u_matrix(self) -> np.ndarray:
        umat = np.zeros((self.rows, self.cols))
        for r in range(self.rows):
            for c in range(self.cols):
                neighbor_distances = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        neighbor_distances.append(np.linalg.norm(self.weights[r, c] - self.weights[nr, nc]))
                umat[r, c] = np.mean(neighbor_distances)
        return umat


def jittered_positions(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    return (
        df["som_col"].to_numpy(dtype=float) + rng.uniform(-0.22, 0.22, len(df)),
        df["som_row"].to_numpy(dtype=float) + rng.uniform(-0.22, 0.22, len(df)),
    )


def save_feature_distributions(transformed: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    axes = axes.ravel()
    for ax, col in zip(axes, TRANSFORMED_FEATURE_COLS):
        ax.hist(transformed[col], bins=24, color="#2f6f73", alpha=0.85, edgecolor="white")
        ax.set_title(col.replace("_", " "), fontsize=9)
        ax.tick_params(labelsize=8)
    axes[-1].axis("off")
    fig.suptitle("Feature distributions after log transforms", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_distributions.png", bbox_inches="tight")
    plt.close(fig)


def save_u_matrix(som: SimpleSOM, results: pd.DataFrame) -> None:
    income_colors = {
        "High income": "#2a6fbb",
        "Upper middle income": "#1b9e77",
        "Lower middle income": "#f0a202",
        "Low income": "#d95f02",
        "Not classified": "#7f7f7f",
    }
    focus_codes = ["THA", "MYS", "VNM", "IDN", "PHL", "CHN", "JPN", "KOR", "IND", "USA", "DEU", "BRA", "MEX", "ZAF", "QAT", "SGP"]
    fig, ax = plt.subplots(figsize=(11, 8))
    image = ax.imshow(som.u_matrix(), cmap="magma_r", origin="upper")
    fig.colorbar(image, ax=ax, shrink=0.75, label="Average neighbor distance")
    x, y = jittered_positions(results)
    ax.scatter(x, y, s=36, c=results["income_level"].map(income_colors).fillna("#7f7f7f"), edgecolor="white", linewidth=0.55, alpha=0.92)
    for _, row in results[results["country_code"].isin(focus_codes)].iterrows():
        ax.text(row["som_col"], row["som_row"], row["country_code"], fontsize=8, weight="bold", ha="center", va="center", color="white", bbox=dict(boxstyle="round,pad=0.18", facecolor="black", alpha=0.72, linewidth=0))
    handles = [mpatches.Patch(color=color, label=level) for level, color in income_colors.items() if level in set(results["income_level"])]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    ax.set_title("SOM U-Matrix: countries with similar profiles land near each other")
    ax.set_xlabel("SOM column")
    ax.set_ylabel("SOM row")
    ax.set_xticks(range(som.cols))
    ax.set_yticks(range(som.rows))
    ax.grid(color="white", alpha=0.18, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "som_u_matrix_country_map.png", bbox_inches="tight")
    plt.close(fig)


def save_component_planes(som: SimpleSOM) -> None:
    labels = {
        "log_gdp_per_capita": "log GDP pc",
        "life_expectancy": "life expectancy",
        "internet_users_pct": "internet users",
        "urban_population_pct": "urban population",
        "unemployment_pct": "unemployment",
        "electricity_access_pct": "electricity access",
        "log1p_co2_tons_per_capita_2018": "log CO2 pc",
    }
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.ravel()
    for idx, feature in enumerate(TRANSFORMED_FEATURE_COLS):
        ax = axes[idx]
        image = ax.imshow(som.weights[:, :, idx], cmap="viridis", origin="upper")
        ax.set_title(labels[feature], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    axes[-1].axis("off")
    fig.suptitle("SOM component planes, values are standardized feature weights", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "som_component_planes.png", bbox_inches="tight")
    plt.close(fig)


def save_macro_clusters(som: SimpleSOM, results: pd.DataFrame, node_clusters: np.ndarray) -> None:
    colors = ["#3366aa", "#dd4477", "#66aa44", "#ffbb33", "#8e63ce"]
    cmap = plt.matplotlib.colors.ListedColormap(colors[: len(colors)])
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(node_clusters, cmap=cmap, origin="upper", alpha=0.42)
    x, y = jittered_positions(results)
    ax.scatter(x, y, s=34, c=results["macro_cluster"].map(lambda c: colors[int(c)]), edgecolor="white", linewidth=0.55, alpha=0.95)
    for code in ["THA", "MYS", "CHN", "USA", "DEU", "KOR", "SGP", "QAT", "BRA", "MEX", "ZAF"]:
        focus = results[results["country_code"] == code]
        if not focus.empty:
            row = focus.iloc[0]
            ax.text(row["som_col"], row["som_row"], code, fontsize=8, weight="bold", ha="center", va="center", color="white", bbox=dict(boxstyle="round,pad=0.18", facecolor="black", alpha=0.72, linewidth=0))
    ax.legend(handles=[mpatches.Patch(color=colors[i], label=f"Macro cluster {i}") for i in range(5)], loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    ax.set_title("Macro clusters on top of the SOM grid")
    ax.set_xlabel("SOM column")
    ax.set_ylabel("SOM row")
    ax.set_xticks(range(som.cols))
    ax.set_yticks(range(som.rows))
    ax.grid(color="white", alpha=0.25, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "som_macro_clusters.png", bbox_inches="tight")
    plt.close(fig)


def similar_countries(target_code: str, results: pd.DataFrame, x_scaled: np.ndarray, top_n: int = 12) -> pd.DataFrame:
    target_idx = results.index[results["country_code"] == target_code][0]
    out = results.copy()
    out["feature_distance"] = np.linalg.norm(x_scaled - x_scaled[target_idx], axis=1)
    out["som_grid_distance"] = (
        (out["som_row"] - out.loc[target_idx, "som_row"]).abs()
        + (out["som_col"] - out.loc[target_idx, "som_col"]).abs()
    )
    cols = [
        "country_code",
        "country_name",
        "region",
        "income_level",
        "som_row",
        "som_col",
        "macro_cluster",
        "feature_distance",
        "som_grid_distance",
        *FEATURE_COLS,
    ]
    return out.sort_values("feature_distance")[cols].head(top_n)


def save_neighbor_chart(neighbors: pd.DataFrame, target_code: str = "THA") -> None:
    distances = neighbors.sort_values("feature_distance")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#d95f02" if code == target_code else "#2a6fbb" for code in distances["country_code"]]
    ax.barh(distances["country_code"], distances["feature_distance"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Distance in scaled multi-dimensional feature space")
    ax.set_title(f"Most similar countries to {target_code}")
    for i, (_, row) in enumerate(distances.iterrows()):
        label = f"{row['country_name']} | SOM ({row['som_row']}, {row['som_col']})"
        ax.text(row["feature_distance"] + 0.03, i, label, va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"similar_countries_{target_code}.png", bbox_inches="tight")
    plt.close(fig)


def run_pipeline(force_refresh: bool = False) -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    full_table, model_table = build_country_feature_table(force_refresh)
    full_table.to_csv(DATA_DIR / "world_bank_country_indicators_full.csv", index=False)
    model_table.to_csv(DATA_DIR / "world_bank_som_input_complete_cases.csv", index=False)

    transformed = engineer_features(model_table)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(transformed[TRANSFORMED_FEATURE_COLS])
    scaled_output = pd.concat(
        [
            model_table[["country_code", "country_name", "region", "income_level"]].reset_index(drop=True),
            transformed.reset_index(drop=True),
            pd.DataFrame(x_scaled, columns=[f"z_{col}" for col in TRANSFORMED_FEATURE_COLS]),
        ],
        axis=1,
    )
    scaled_output.to_csv(DATA_DIR / "world_bank_som_scaled_features.csv", index=False)
    save_feature_distributions(transformed)

    som = SimpleSOM(rows=9, cols=10, input_dim=x_scaled.shape[1])
    som.train(x_scaled, num_iterations=9000)
    bmu_coords = som.map_vectors(x_scaled)

    results = model_table.copy().reset_index(drop=True)
    results["som_row"] = bmu_coords[:, 0]
    results["som_col"] = bmu_coords[:, 1]

    kmeans = KMeans(n_clusters=5, random_state=SEED, n_init=20)
    node_clusters = kmeans.fit_predict(som.weights.reshape(-1, som.input_dim)).reshape(som.rows, som.cols)
    results["macro_cluster"] = [int(node_clusters[row, col]) for row, col in results[["som_row", "som_col"]].to_numpy()]

    cluster_summary = (
        results.groupby("macro_cluster")
        .agg(
            countries=("country_code", "count"),
            avg_gdp_per_capita=("gdp_per_capita", "mean"),
            avg_life_expectancy=("life_expectancy", "mean"),
            avg_internet_users_pct=("internet_users_pct", "mean"),
            avg_urban_population_pct=("urban_population_pct", "mean"),
            avg_unemployment_pct=("unemployment_pct", "mean"),
            avg_electricity_access_pct=("electricity_access_pct", "mean"),
            avg_co2_tons_per_capita_2018=(CO2_FEATURE, "mean"),
        )
        .round(2)
    )
    examples = (
        results.sort_values(["macro_cluster", "country_name"])
        .groupby("macro_cluster")
        .head(8)
        .groupby("macro_cluster")["country_code"]
        .apply(lambda codes: ", ".join(codes))
        .rename("example_country_codes")
    )
    cluster_summary = cluster_summary.join(examples)

    results[["country_code", "country_name", "region", "income_level", "som_row", "som_col", "macro_cluster", *FEATURE_COLS]].to_csv(
        DATA_DIR / "country_som_results.csv", index=False
    )
    cluster_summary.to_csv(DATA_DIR / "macro_cluster_summary.csv")

    thailand_neighbors = similar_countries("THA", results, x_scaled, top_n=12)
    thailand_neighbors.to_csv(DATA_DIR / "thailand_similar_countries.csv", index=False)

    save_u_matrix(som, results)
    save_component_planes(som)
    save_macro_clusters(som, results, node_clusters)
    save_neighbor_chart(thailand_neighbors, "THA")

    summary = {
        "countries_total": int(len(full_table)),
        "countries_complete": int(len(model_table)),
        "features": len(FEATURE_COLS),
        "som_grid": "9x10",
        "quantization_error": round(som.quantization_error(x_scaled), 3),
        "topographic_error": round(som.topographic_error(x_scaled), 3),
        "thailand_neighbors": thailand_neighbors[["country_code", "country_name", "feature_distance", "som_grid_distance"]]
        .round(3)
        .to_dict(orient="records"),
    }
    (DATA_DIR / "project_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the SOM World Map data, model outputs, and figures.")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore cached World Bank JSON and call the API again.")
    args = parser.parse_args()
    summary = run_pipeline(force_refresh=args.force_refresh)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
