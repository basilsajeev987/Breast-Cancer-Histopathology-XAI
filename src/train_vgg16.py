# ==========================================
# train_vgg16.py
# BreakHis Binary Classification
# ==========================================

import os
import time
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator
)

from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dropout,
    Dense
)

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

# ==========================================
# PATHS
# ==========================================

TRAIN_CSV = "data/train.csv"
VAL_CSV = "data/val.csv"
TEST_CSV = "data/test.csv"

MODEL_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================
# PARAMETERS
# ==========================================

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

# ==========================================
# LOAD DATA
# ==========================================

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

for df in [train_df, val_df, test_df]:
    df["label"] = df["label"].astype(str)

# ==========================================
# DATA GENERATORS
# ==========================================

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col="image_path",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True
)

val_generator = test_datagen.flow_from_dataframe(
    val_df,
    x_col="image_path",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col="image_path",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# ==========================================
# MODEL
# ==========================================

base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dropout(0.3)(x)

output = Dense(
    1,
    activation="sigmoid"
)(x)

model = Model(
    inputs=base_model.input,
    outputs=output
)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================================
# CALLBACKS
# ==========================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    os.path.join(
        MODEL_DIR,
        "VGG16_BreakHis.keras"
    ),
    save_best_only=True,
    monitor="val_accuracy"
)

# ==========================================
# TRAIN
# ==========================================

print("\nTraining VGG16...")

start_time = time.time()

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[
        early_stop,
        checkpoint
    ]
)

training_time = (
    time.time() - start_time
)

# ==========================================
# EVALUATE
# ==========================================

pred_probs = model.predict(
    test_generator
)

y_pred = (
    pred_probs > 0.5
).astype(int).flatten()

y_true = test_generator.classes

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred
)

recall = recall_score(
    y_true,
    y_pred
)

f1 = f1_score(
    y_true,
    y_pred
)

roc_auc = roc_auc_score(
    y_true,
    pred_probs
)

cm = confusion_matrix(
    y_true,
    y_pred
)

tn, fp, fn, tp = cm.ravel()

print("\nClassification Report\n")

print(
    classification_report(
        y_true,
        y_pred
    )
)

print("\nConfusion Matrix")
print(cm)

# ==========================================
# SAVE RESULTS
# ==========================================

results_df = pd.DataFrame({

    "Model":["VGG16"],

    "Accuracy":[accuracy],

    "Precision":[precision],

    "Recall":[recall],

    "F1_Score":[f1],

    "ROC_AUC":[roc_auc],

    "True_Negative":[tn],

    "False_Positive":[fp],

    "False_Negative":[fn],

    "True_Positive":[tp],

    "Training_Time":[training_time]

})

results_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "VGG16_results.csv"
    ),
    index=False
)

# ==========================================
# SAVE CONFUSION MATRIX
# ==========================================

cm_df = pd.DataFrame(
    cm,
    columns=[
        "Pred_Benign",
        "Pred_Malignant"
    ],
    index=[
        "True_Benign",
        "True_Malignant"
    ]
)

cm_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "VGG16_confusion_matrix.csv"
    )
)

# ==========================================
# SAVE TRAINING HISTORY
# ==========================================

history_df = pd.DataFrame(
    history.history
)

history_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "VGG16_history.csv"
    ),
    index=False
)

print("\nResults Saved Successfully")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC AUC   : {roc_auc:.4f}")