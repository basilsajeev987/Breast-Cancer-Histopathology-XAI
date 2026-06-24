# ==========================================
# evaluate_models.py
# Compare All BreakHis Models
# ==========================================

import os
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# CONFIG
# ==========================================

RESULTS_DIR = "results"

# ==========================================
# LOAD ALL RESULT FILES
# ==========================================

result_files = [

    "MobileNetV2_results.csv",

    "ResNet50_results.csv",

    "VGG16_results.csv",

    "DenseNet121_results.csv",

    "EfficientNetB0_results.csv"
]

all_results = []

for file in result_files:

    file_path = os.path.join(
        RESULTS_DIR,
        file
    )

    if os.path.exists(file_path):

        df = pd.read_csv(
            file_path
        )

        all_results.append(df)

        print(f"Loaded: {file}")

    else:

        print(f"Missing: {file}")

# ==========================================
# MERGE RESULTS
# ==========================================

comparison_df = pd.concat(
    all_results,
    ignore_index=True
)

comparison_df = comparison_df.sort_values(
    by="Accuracy",
    ascending=False
)

# ==========================================
# SAVE COMPARISON CSV
# ==========================================

comparison_csv = os.path.join(
    RESULTS_DIR,
    "Final_Model_Comparison.csv"
)

comparison_df.to_csv(
    comparison_csv,
    index=False
)

print("\nFinal Comparison Table")
print(comparison_df)

print(
    f"\nSaved:\n{comparison_csv}"
)

# ==========================================
# ACCURACY CHART
# ==========================================

plt.figure(figsize=(8,5))

plt.bar(
    comparison_df["Model"],
    comparison_df["Accuracy"]
)

plt.title(
    "Accuracy Comparison"
)

plt.ylabel(
    "Accuracy"
)

plt.xlabel(
    "Model"
)

plt.xticks(
    rotation=20
)

plt.tight_layout()

accuracy_chart = os.path.join(
    RESULTS_DIR,
    "Accuracy_Comparison.png"
)

plt.savefig(
    accuracy_chart,
    dpi=300
)

plt.show()

print(
    f"\nSaved:\n{accuracy_chart}"
)

# ==========================================
# ROC-AUC CHART
# ==========================================

plt.figure(figsize=(8,5))

plt.bar(
    comparison_df["Model"],
    comparison_df["ROC_AUC"]
)

plt.title(
    "ROC-AUC Comparison"
)

plt.ylabel(
    "ROC-AUC"
)

plt.xlabel(
    "Model"
)

plt.xticks(
    rotation=20
)

plt.tight_layout()

roc_chart = os.path.join(
    RESULTS_DIR,
    "ROC_AUC_Comparison.png"
)

plt.savefig(
    roc_chart,
    dpi=300
)

plt.show()

print(
    f"\nSaved:\n{roc_chart}"
)

# ==========================================
# PRINT BEST MODEL
# ==========================================

best_model = comparison_df.iloc[0]

print("\n=================================")
print("BEST MODEL")
print("=================================")

print(
    f"Model     : {best_model['Model']}"
)

print(
    f"Accuracy  : {best_model['Accuracy']:.4f}"
)

print(
    f"ROC_AUC   : {best_model['ROC_AUC']:.4f}"
)

print("=================================")