# SOM World Map: Country Similarity with Self-Organizing Map

โปรเจกต์นี้ใช้ **Self-Organizing Map (SOM)** จัดกลุ่มประเทศจากตัวชี้วัดหลายมิติของ World Bank แล้วแปลงข้อมูลประเทศให้เป็นแผนที่ 2 มิติที่อ่านง่าย เหมาะสำหรับอธิบายประโยชน์ของ SOM ในงาน exploratory data analysis

## Project Idea

คำถามหลัก:

> ถ้าแต่ละประเทศมี profile หลายมิติ เช่น GDP, อายุคาดเฉลี่ย, internet usage, urbanization, unemployment, electricity access และ CO2 ต่อหัว ประเทศไหน “คล้ายกัน” บ้าง?

SOM ช่วยตอบคำถามนี้โดยจัดประเทศที่มี feature profile คล้ายกันให้อยู่ใกล้กันบน grid 2 มิติ

## Folder Structure

```text
SOM-World-Map/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── som_world_map_pipeline.py
│   ├── dynamic_som_pipeline.py
│   └── som_income_classifier.py
├── notebooks/
│   └── som-world-map.ipynb
├── data/
│   ├── raw_world_bank_json/
│   ├── world_bank_country_indicators_full.csv
│   ├── world_bank_som_input_complete_cases.csv
│   ├── world_bank_som_scaled_features.csv
│   ├── country_som_results.csv
│   ├── macro_cluster_summary.csv
│   ├── thailand_similar_countries.csv
│   └── project_summary.json
├── figures/
│   ├── feature_distributions.png
│   ├── som_u_matrix_country_map.png
│   ├── som_component_planes.png
│   ├── som_macro_clusters.png
│   └── similar_countries_THA.png
└── slides/
    └── som-world-map-country-similarity.pptx
```

## Data

Data source: World Bank Indicators API

Features used:

- GDP per capita
- Life expectancy
- Internet users (% population)
- Urban population (% total)
- Unemployment (% labor force)
- Access to electricity (% population)
- CO2 tons per capita in 2018

Note: CO2 per capita is computed from total CO2 (`CC.CO2.EMSE.EL`) divided by population (`SP.POP.TOTL`) in 2018 because the older `EN.ATM.CO2E.PC` endpoint is archived/deleted in the current World Bank API response.

## How To Run

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the full pipeline from cached World Bank JSON:

```bash
python src/som_world_map_pipeline.py
```

Force refresh from World Bank API:

```bash
python src/som_world_map_pipeline.py --force-refresh
```

Build the dynamic 10-year country-year SOM:

```bash
python src/dynamic_som_pipeline.py
```

The dynamic pipeline uses 2014-2023 country-year rows and trains the SOM on six high-coverage features: GDP per capita, life expectancy, internet users, urban population, unemployment, and electricity access. CO2 per capita is still pulled into the full country-year table where the World Bank endpoint has data, but it is not used as a 10-year SOM feature because recent years are missing.

Run the SOM majority-vote income classifier:

```bash
python src/som_income_classifier.py
```

This add-on keeps the classification method SOM-based: it trains a 4-node SOM on the static country snapshot, labels each SOM node by the majority `income_level` in the train set, then classifies 15 held-out countries by the label of their closest SOM node.

Open the notebook:

```bash
jupyter notebook notebooks/som-world-map.ipynb
```

## Latest Reproduced Results

From the current pipeline run:

- Countries in metadata: 217
- Countries with complete features: 171
- Number of model features: 7
- SOM grid: 9 x 10
- Quantization error: 0.687
- Topographic error: 0.018

Countries most similar to Thailand by scaled feature distance:

1. Ecuador
2. Jamaica
3. Azerbaijan
4. Mexico
5. China
6. Dominican Republic
7. Belarus
8. Hungary
9. Uzbekistan
10. Romania

## Main Outputs

- Notebook: `notebooks/som-world-map.ipynb`
- Final country map: `figures/som_u_matrix_country_map.png`
- Feature interpretation: `figures/som_component_planes.png`
- Macro clusters: `figures/som_macro_clusters.png`
- Thailand neighbors: `data/thailand_similar_countries.csv`
- Presentation deck: `slides/som-world-map-country-similarity.pptx`

Dynamic time-series outputs:

- Full country-year table: `data/world_bank_country_year_indicators_full.csv`
- Dynamic SOM input: `data/country_year_features.csv`
- Dynamic SOM positions: `data/country_year_som_results.csv`
- Thailand neighbor drift: `data/neighbor_drift_THA.csv`
- Movement scores: `data/movement_scores.csv`
- Time-shifted similarity: `data/time_shifted_similarity_THA_2023.csv`
- Dynamic summary: `data/dynamic_project_summary.json`
- Dynamic SOM snapshots: `figures/dynamic_som_year_snapshots.png`
- Thailand trajectory: `figures/dynamic_som_trajectory_THA.png`

SOM classifier outputs:

- Test predictions: `data/som_income_classifier_test_predictions.csv`
- Node labels: `data/som_income_classifier_node_labels.csv`
- Metrics: `data/som_income_classifier_metrics.json`
- Classifier map: `figures/som_income_classifier_map.png`
- Confusion matrix: `figures/som_income_classifier_confusion_matrix.png`

## Why SOM Is Useful Here

SOM is useful because it keeps the idea of “neighborhood” while reducing multi-dimensional data to a 2D grid. Instead of reading a large table of country indicators, we can visually inspect:

- Which countries have similar profiles
- Which regions of the map are high-income, digitally connected, or low-access
- Where group boundaries appear
- Which countries are local neighbors of Thailand in data space

## Sources

- World Bank Indicators API: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392
- World Development Indicators catalog: https://datacatalog.worldbank.org/infrastructure-data/search/dataset/0037712/World-Development-Indicators
