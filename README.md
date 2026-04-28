# Applied Data Science Project 4: End-to-End Machine Learning

## Overview 

This project analyzes whether daily stock direction can be predicted for four large-cap technology stocks: **AAPL, NFLX, MSFT, and AMZN**. The workflow uses Yahoo Finance data, technical-indicator feature engineering, unsupervised PCA/k-means clustering, and supervised ML models.

The main supervised models are:

- Logistic Regression
- Random Forest
- XGBoost

The project also includes additional model evaluation analysis, a survival analysis extension, and an interactive Shiny app for exploring the results.

---

## Project Structure

```text
applied_datascience_project4/
│
├── README.md
│
├── Analysis/
│   ├── step1_analysis.R                  # Main R script: downloads data, engineers features, trains models, saves results
│   ├── summary_of_results.ipynb          # Additional model comparison, baseline analysis, and result interpretation
│   ├── survival_analysis_extension.ipynb # Cox/Kaplan-Meier survival analysis for downside-risk timing
│   │
│   ├── all_metrics.rds                   # Saved accuracy, sensitivity, specificity, precision, recall, F1, and AUC
│   ├── all_cm_data.rds                   # Saved confusion matrix data for each ticker/model
│   ├── all_roc_data.rds                  # Saved ROC curve data
│   ├── all_imp_rf.rds                    # Random Forest feature importance
│   ├── all_imp_xgb.rds                   # XGBoost feature importance
│   ├── all_pca_km.rds                    # PCA and k-means clustering outputs
│   └── stocks.rds                        # List of stock tickers used
│
├── Data/
│   ├── stock_original_data.xlsx          # Raw Yahoo Finance stock data
│   └── stock_processed_data.xlsx         # Cleaned feature-engineered modeling dataset
│
└── ShinyApp/
    └── step2_shinyapp.R                  # Interactive Shiny dashboard
```

---

## Team Members

- Nikhil Shanbhag (nvs2128)
- Charlene Shen (xs2546)
- Shreya Amalapurapu (sa4342)
- Yifan Wang (yw4663)

---
