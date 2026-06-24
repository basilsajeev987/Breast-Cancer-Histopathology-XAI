# ==========================================
# evaluate_models.py
# BreakHis Model Comparison
# ==========================================

import os
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# PATHS
# ==========================================

RESULTS_DIR = "results"

# ==========================================
# RESULT FILES
# ==========================================

result_files = [

    "DenseNet121_results.csv",

    "ResNet50_results.csv",

    "EfficientNetB0_results.csv",

    "VGG16_results.csv",

    "MobileNetV2_results.csv"
]

# ==========================================
# LOAD RESULTS
# ==========================================

all_results = []

for file in result_files:

    file_path = os.path.join(
        RESULTS_DIR,
        file
    )

    if os.path.exists(file_path):

        print(f"Loading: {file}")

        df = pd.read_csv(file_path)

        all_results.append(df)

    else:

        print(f"Missing: {file}")

# ==========================================
# COMBINE RESULTS
# ==========================================

comparison_df = pd.concat(
    all_results,
    ignore_index=True
)

# ==========================================
# SORT BY ROC-AUC
# ==========================================

comparison_df = comparison_df.sort_values(
    by="ROC_AUC",
    ascending=False
)

print("\nFinal Comparison Table\n")

print(comparison_df)

# ==========================================
# SAVE FINAL TABLE
# ==========================================

comparison_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "Final_Model_Comparison.csv"
    ),
    index=False
)

print(
    "\nSaved: Final_Model_Comparison.csv"
)

# ==========================================
# CHART FUNCTION
# ==========================================

def create_chart(metric):

    plt.figure(figsize=(8,5))

    plt.bar(
        comparison_df["Model"],
        comparison_df[metric]
    )

    plt.title(
        f"{metric} Comparison"
    )

    plt.ylabel(metric)

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            f"{metric}_Comparison.png"
        ),
        dpi=300
    )

    plt.show()

# ==========================================
# ACCURACY
# ==========================================

create_chart(
    "Accuracy"
)

# ==========================================
# PRECISION
# ==========================================

create_chart(
    "Precision"
)

# ==========================================
# RECALL
# ==========================================

create_chart(
    "Recall"
)

# ==========================================
# F1 SCORE
# ==========================================

create_chart(
    "F1_Score"
)

# ==========================================
# ROC AUC
# ==========================================

create_chart(
    "ROC_AUC"
)

print("\nAll charts generated successfully.")