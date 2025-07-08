#  ModelIQ: AI Copilot for Model Evaluation and Retuning

ModelIQ is a plug-and-play, AI-powered assistant that analyzes your machine learning models, diagnoses performance issues, and suggests intelligent tuning strategies — all in natural language.

>  Save hours of manual debugging by letting AI analyze, summarize, and guide your next modeling step.

---

## Features

- **AutoML Comparison**  
  Train and compare multiple models (XGBoost, LightGBM, RandomForest) on Snowflake-hosted data.

- **Smart Evaluation**  
  Compute metrics like accuracy, F1 score, confusion matrix, and SHAP feature importance automatically.

- **LLM-Powered Feedback**  
  Uses Snowflake Cortex with GPT-4.1 to generate plain-English summaries and actionable tuning recommendations (any model can be used given in Snowflake).

- **Optional Report Export**  
  Automatically generate a `.txt` report of all insights and feedback in one file.

---

## Tech Stack

| Component | Tool |
|----------|------|
| Data Source | Snowflake Table (`HEART_DATASET`) |
| Model Training | XGBoost, LightGBM, RandomForest |
| Evaluation | `scikit-learn`, `SHAP`, `confusion_matrix` |
| AI Assistant | Snowflake Cortex + `openai-gpt-4.1` |
| Output | Console + Optional `modeliq_report.txt` |

---

## How It Works

1. Connect to Snowflake and load your dataset
2. Train 3 ML models and evaluate on key metrics
3. Extract SHAP values for the best model
4. Pass structured summary to Cortex for LLM feedback
5. Print or save a complete report with insights

---

## Folder Structure

```bash
├── modeliq_main.py         # Full end-to-end script
├── modeliq_report.txt      # AI-generated report
├── README.md               # You're here
└── requirements.txt        # Dependencies
