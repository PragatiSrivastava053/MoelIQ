from snowflake.snowpark import Session
import pandas as pd
import numpy as np
import shap
import warnings
import json

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

warnings.filterwarnings("ignore")

# ---------------------------
# Snowflake Connection
# ---------------------------
connection_parameters = {
    "account": "<your_account>",
    "user": "<your_user>",
    "password": "<your_password>",
    "role": "<your_role>",
    "warehouse": "<your_warehouse>",
    "database": "<your_database>",
    "schema": "<your_schema>"
}

session = Session.builder.configs(connection_parameters).create()

# ---------------------------
# Load Data
# ---------------------------
df = session.table("HEART_DATASET").to_pandas()

# ---------------------------
# Preprocess
# ---------------------------
target = "HEARTDISEASE"
X = df.drop(columns=[target])
y = df[target]

for col in X.select_dtypes(include="object").columns:
    X[col] = LabelEncoder().fit_transform(X[col])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# Train Multiple Models
# ---------------------------
models = {
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42, force_col_wise=True),
    "RandomForest": RandomForestClassifier(random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        "model": model,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

# ---------------------------
# Pick Best Model (by F1)
# ---------------------------
best_model_name = max(results, key=lambda x: results[x]["f1_score"])
best_model = results[best_model_name]["model"]

# ---------------------------
# SHAP on Best Model
# ---------------------------
explainer = shap.Explainer(best_model)
shap_values = explainer(X_test)

# If multi-class/binary classification, select class 1 SHAP values
if shap_values.values.ndim == 3:
    # Binary classification, take SHAP values for class 1
    shap_values_class1 = shap_values.values[:, :, 1]
else:
    shap_values_class1 = shap_values.values  # Already 2D (n_samples x n_features)

# Now compute average absolute SHAP per feature
shap_df = pd.DataFrame({
    "feature": X_test.columns,
    "importance": np.abs(shap_values_class1).mean(axis=0)
}).sort_values(by="importance", ascending=False)

top_features = shap_df.head(5).to_dict(orient="records")

# ---------------------------
# Create Cortex Prompt
# ---------------------------
metrics_summary = "\n\n".join([
    f"{name}:\n" +
    f"  Accuracy: {res['accuracy']}\n" +
    f"  Precision: {res['precision']}\n" +
    f"  Recall: {res['recall']}\n" +
    f"  F1 Score: {res['f1_score']}\n" +
    f"  Confusion Matrix: {res['confusion_matrix']}"
    for name, res in results.items()
])

summary_input = f"""
We trained 3 classification models on a heart disease dataset:
{metrics_summary}

Top 5 Influential Features for the best model ({best_model_name}):
{json.dumps(top_features)}

Please answer the following:
1. Which model is best and why?
2. What potential issues or patterns do you observe?
3. What can we do to improve performance further?
"""

# ---------------------------
# Call Snowflake Cortex
# ---------------------------
summary_input_cleaned = summary_input.replace("'", "''")

query = f"""
SELECT SNOWFLAKE.CORTEX.COMPLETE(
    'openai-gpt-4.1',
    $$ {summary_input_cleaned} $$
) AS CORTEX_FEEDBACK
"""

df_result = session.sql(query).to_pandas()
cortex_output = df_result["CORTEX_FEEDBACK"].iloc[0]

# ---------------------------
# Final Output
# ---------------------------
print("\nModelIQ: Multi-Model Evaluation")
print("------------------------------------")

for name, res in results.items():
    print(f"\n{name}:")
    for k, v in res.items():
        if k != "model":
            print(f"{k}: {v}")

print(f"\nBest Model Based on F1: {best_model_name}")
print("\nTop Features (Best Model):")
for f in top_features:
    print(f"{f['feature']}: {round(f['importance'], 4)}")

print("\nCortex Feedback:")
print(cortex_output)

# ---------------------------
# Optional: Save Report to File
# ---------------------------
save_to_file = True  # Set to False if you don’t want to export

if save_to_file:
    with open("modeliq_report.txt", "w") as f:
        f.write(" ModelIQ - AutoML Evaluation Report\n")
        f.write("====================================\n\n")

        for name, res in results.items():
            f.write(f"\n{name}:\n")
            for k, v in res.items():
                if k != "model":
                    f.write(f"{k}: {v}\n")

        f.write(f"\nBest Model: {best_model_name}\n\n")
        f.write("Top Features (SHAP):\n")
        for f_item in top_features:
            f.write(f"- {f_item['feature']}: {round(f_item['importance'], 4)}\n")

        f.write("\nCortex Feedback:\n")
        f.write(cortex_output)
        f.write("\n")

    print("\n Report saved as: modeliq_report.txt")
