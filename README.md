# Tube Usage and Weather Analysis

This repository contains a data analytics project investigating the relationship between weather conditions and public transport disruption on the London Underground, measured using Lost Customer Hours.

The project integrates multiple public datasets, performs exploratory data analysis (EDA), and applies regression models to assess whether weather conditions are associated with increased disruption.

## Data Sources

The project integrates two high-quality public datasets:

1. **Transport for London (TfL)**
   - Lost Customer Hours by financial period
   - Source: TfL Open Data
   - File: `tfl-tube-performance.xlsx`
  
2. **London Weather Data**
   - Daily observations of rainfall, temperature, sunshine, and humidity
   - Aggregated to monthly level
   - Source: Met Office / public climate data
   - https://www.kaggle.com/datasets/zongaobian/london-weather-data-from-1979-to-2023/data
   - File: `london_weather_data_1979_to_2023.csv`
  
## How to Run the Project

```bash
pip install -r requirements.txt
python data_analysis/tube_weather.py
python data_analysis/tube_weather.py
python data_analysis/baseline_model.py
pytest
```

## Use of AI Tools
Large Language Models (LLMs) were used to:
- Assist with Python debugging
- Suggest improvements to unit tests
- Help structure code and documentation
