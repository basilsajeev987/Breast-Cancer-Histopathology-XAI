import os
import re
import pandas as pd

from sklearn.model_selection import train_test_split

# =====================================
# CONFIG
# =====================================

DATASET_PATH = "dataset_cancer_v1"

# =====================================
# COLLECT IMAGE PATHS
# =====================================

image_paths = []

for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        if file.endswith(".png"):

            image_paths.append(
                os.path.join(root, file)
            )

print(f"Total Images Found: {len(image_paths)}")

# =====================================
# CREATE DATAFRAME
# =====================================

df = pd.DataFrame(
    image_paths,
    columns=["image_path"]
)

# =====================================
# LABEL EXTRACTION
# =====================================

def get_label(path):

    filename = os.path.basename(path)

    if "SOB_B" in filename:
        return 0

    elif "SOB_M" in filename:
        return 1

    return None

df["label"] = df["image_path"].apply(
    get_label
)

# =====================================
# PATIENT ID EXTRACTION
# =====================================

def extract_patient_id(path):

    filename = os.path.basename(path)

    match = re.search(
        r'(\d+-\d+)',
        filename
    )

    if match:
        return match.group(1)

    return None

df["patient_id"] = df["image_path"].apply(
    extract_patient_id
)

# =====================================
# REMOVE INVALID ROWS
# =====================================

df = df.dropna()

print("\nDataset Summary")
print(df.head())

print("\nUnique Patients:")
print(df["patient_id"].nunique())

# =====================================
# PATIENT LEVEL SPLIT
# =====================================

patients = df["patient_id"].unique()

train_patients, test_patients = train_test_split(
    patients,
    test_size=0.20,
    random_state=42
)

train_patients, val_patients = train_test_split(
    train_patients,
    test_size=0.20,
    random_state=42
)

# =====================================
# CREATE SPLITS
# =====================================

train_df = df[
    df["patient_id"].isin(
        train_patients
    )
]

val_df = df[
    df["patient_id"].isin(
        val_patients
    )
]

test_df = df[
    df["patient_id"].isin(
        test_patients
    )
]

# =====================================
# SAVE CSV FILES
# =====================================

train_df[["image_path", "label"]].to_csv(
    "train.csv",
    index=False
)

val_df[["image_path", "label"]].to_csv(
    "val.csv",
    index=False
)

test_df[["image_path", "label"]].to_csv(
    "test.csv",
    index=False
)

# =====================================
# STATS
# =====================================

print("\nTrain:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

print("\nCSV Files Saved:")
print("train.csv")
print("val.csv")
print("test.csv")