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
│   └── som_world_map_pipeline.py
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

## Why SOM Is Useful Here

SOM is useful because it keeps the idea of “neighborhood” while reducing multi-dimensional data to a 2D grid. Instead of reading a large table of country indicators, we can visually inspect:

- Which countries have similar profiles
- Which regions of the map are high-income, digitally connected, or low-access
- Where group boundaries appear
- Which countries are local neighbors of Thailand in data space

## Sources

- World Bank Indicators API: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392
- World Development Indicators catalog: https://datacatalog.worldbank.org/infrastructure-data/search/dataset/0037712/World-Development-Indicators
