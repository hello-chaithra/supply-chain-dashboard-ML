# AI-Driven Supply Chain Analytics Dashboard

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.54-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-green?logo=plotly)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

> **Live Demo:** [Launch Dashboard](https://chaithra-supply-chain-dashboard.streamlit.app/)  

---

## 📌 Project Overview

An end-to-end, AI-powered supply chain analytics dashboard built with Python and deployed publicly on Streamlit Cloud. The dashboard analyzes **10,000+ supply chain records** to surface actionable insights on inventory risk, logistics performance, and demand forecasting — the kind of analysis real operations and data teams run daily.

---

## Tech Stack
- **Data Processing**: Python, Pandas, NumPy
- **Database**: SQLite, SQLAlchemy
- **Machine Learning**: Scikit-learn (Random Forest, Linear Regression)
- **Visualization**: Plotly Express
- **Dashboard & Deployment**: Streamlit Community Cloud
- **Version Control**: Git, GitHub 

---

## ML Model Performance

Two models were trained and compared on an 80/20 train-test split:

| Model | MAE (units) | RMSE (units) |
|-------|-------------|--------------|
| Linear Regression (baseline) | ~145 | ~167 |
| **Random Forest (final)** | **~143** | **~165** |

**Top demand drivers (by feature importance):**
1. `shipping_cost` — operational cost correlates with order volume
2. `units_in_stock` — available inventory directly shapes ordering behavior  
3. `day_of_year` — seasonal patterns embedded in daily data
4. `lead_time_days` — longer supplier lead times drive anticipatory ordering

---

## Project Structure

```
supply-chain-dashboard/
│
├── app.py                  # Main Streamlit dashboard application
├── requirements.txt        # Python dependencies
└── README.md               # You are here

```
---

## Data Pipeline Architecture

```
Raw Data Generation (10k records)
        │
        ▼
  SQLite Database
  (SQLAlchemy ORM)
        │
        ▼
  SQL Aggregations          Pandas Transformations
  • Category performance    • Time-series features
  • Stockout detection      • Category encoding
  • Delivery rates          • Feature engineering
        │                         │
        └──────────┬──────────────┘
                   ▼
         Random Forest Model
         (demand forecasting)
                   │
                   ▼
        Streamlit Dashboard
        (interactive UI + Plotly charts)
```

---

##  Business Insights Discovered

- **Medical supplies & West region** had the highest stockout risk (151 at-risk orders, avg shortfall of 161 units): flagging a supplier or logistics gap
- **North region** outperformed all others on delivery reliability at 80.8% on-time rate vs. South at 77.9%
- **Shipping cost is the #1 demand predictor** suggesting high-value orders are systematically larger, useful for dynamic pricing strategy

---

## Future Improvements
- Build supplier scorecard tab with lead time variability metrics
- Add anomaly detection for sudden demand spikes

---

---

## Key Challenge Overcome
- **Migrating from Google Colab to a deployable Python app**: The project began in Google Colab, which uses notebook-specific magic commands like `%%writefile` and `!pip install` that are invalid in standard Python scripts. Transitioning to a clean, self-contained `app.py` required restructuring all code when moving from experimentation to production.

---

## About Me
- **Chaithra Arun**
- pursuing B.Tech AI & Data Science (2024-2028)
- [GitHub](https://github.com/hello-chaithra) · [LinkedIn](https://www.linkedin.com/in/chaithra-arun06/)
