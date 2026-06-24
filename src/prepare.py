# ==========================================
# prepare_data.py
# BreakHis Stratified Patient-Level Split
# ==========================================

import os
import re
import pandas as pd

from sklearn.model_selection import train_test_split

# ==========================================
# CONFIG
# ==========================================

DATASET_PATH = "dataset_cancer_v1"

OUTPUT_DIR = "data"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ==========================================
# COLLECT IMAGE PATHS
# ==========================================

image_paths = []

for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        if file.endswith(".png"):

            image_paths.append(
                os.path.join(root, file)
            )

print(
    f"Total Images Found: {len(image_paths)}"
)

# ==========================================
# CREATE DATAFRAME
# ==========================================

df = pd.DataFrame(
    image_paths,
    columns=["image_path"]
)

# ==========================================
# LABEL EXTRACTION
# Benign = 0
# Malignant = 1
# ==========================================

def extract_label(path):

    filename = os.path.basename(path)

    if "SOB_B" in filename:
        return 0

    if "SOB_M" in filename:
        return 1

    return None

df["label"] = df["image_path"].apply(
    extract_label
)

# ==========================================
# PATIENT ID EXTRACTION
# Example:
# SOB_M_DC-14-2773-100-001.png
# Patient ID = 14-2773
# ==========================================

def extract_patient_id(path):

    filename = os.path.basename(path)

    match = re.search(
        r"(\d+-\d+)",
        filename
    )

    if match:

        return match.group(1)

    return None

df["patient_id"] = df["image_path"].apply(
    extract_patient_id
)

# ==========================================
# CLEAN DATA
# ==========================================

df = df.dropna()

print("\nDataset Shape:")
print(df.shape)

print("\nClass Distribution:")
print(
    df["label"].value_counts()
)

print("\nUnique Patients:")
print(
    df["patient_id"].nunique()
)

# ==========================================
# PATIENT-LEVEL TABLE
# ==========================================

patient_df = (
    df.groupby("patient_id")["label"]
      .max()
      .reset_index()
)

print("\nPatient-Level Distribution:")
print(
    patient_df["label"].value_counts()
)

# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

train_patients, test_patients = train_test_split(
    patient_df,
    test_size=0.20,
    random_state=42,
    stratify=patient_df["label"]
)

# ==========================================
# TRAIN / VALIDATION SPLIT
# ==========================================

train_patients, val_patients = train_test_split(
    train_patients,
    test_size=0.20,
    random_state=42,
    stratify=train_patients["label"]
)

# ==========================================
# IMAGE-LEVEL SPLITS
# ==========================================

train_df = df[
    df["patient_id"].isin(
        train_patients["patient_id"]
    )
]

val_df = df[
    df["patient_id"].isin(
        val_patients["patient_id"]
    )
]

test_df = df[
    df["patient_id"].isin(
        test_patients["patient_id"]
    )
]

# ==========================================
# VERIFY NO PATIENT LEAKAGE
# ==========================================

train_ids = set(
    train_df["patient_id"]
)

val_ids = set(
    val_df["patient_id"]
)

test_ids = set(
    test_df["patient_id"]
)

print("\nPatient Leakage Check")

print(
    "Train-Val:",
    len(train_ids.intersection(val_ids))
)

print(
    "Train-Test:",
    len(train_ids.intersection(test_ids))
)

print(
    "Val-Test:",
    len(val_ids.intersection(test_ids))
)

# ==========================================
# CLASS DISTRIBUTION CHECK
# ==========================================

def show_distribution(name, split_df):

    counts = split_df["label"].value_counts()

    total = len(split_df)

    benign = counts.get(0, 0)

    malignant = counts.get(1, 0)

    print(f"\n{name}")

    print(
        f"Total Images: {total}"
    )

    print(
        f"Benign: {benign} ({100*benign/total:.2f}%)"
    )

    print(
        f"Malignant: {malignant} ({100*malignant/total:.2f}%)"
    )

show_distribution(
    "TRAIN",
    train_df
)

show_distribution(
    "VALIDATION",
    val_df
)

show_distribution(
    "TEST",
    test_df
)

# ==========================================
# SAVE CSV FILES
# ==========================================

train_df[
    ["image_path", "label"]
].to_csv(
    os.path.join(
        OUTPUT_DIR,
        "train.csv"
    ),
    index=False
)

val_df[
    ["image_path", "label"]
].to_csv(
    os.path.join(
        OUTPUT_DIR,
        "val.csv"
    ),
    index=False
)

test_df[
    ["image_path", "label"]
].to_csv(
    os.path.join(
        OUTPUT_DIR,
        "test.csv"
    ),
    index=False
)

print("\nFiles Saved Successfully")

print(
    os.path.join(
        OUTPUT_DIR,
        "train.csv"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "val.csv"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "test.csv"
    )
)