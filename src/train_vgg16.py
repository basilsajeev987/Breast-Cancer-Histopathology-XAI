# ==========================================
# train_vgg16.py
# BreakHis Binary Classification using VGG16
# ==========================================

import os
import time
import pandas as pd
import numpy as np

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D
)

from tensorflow.keras.models import Model

from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator
)

from tensorflow.keras.callbacks import (
    EarlyStopping
)

# ==========================================
# CONFIG
# ==========================================

TRAIN_CSV = "data/train.csv"
VAL_CSV = "data/val.csv"
TEST_CSV = "data/test.csv"

MODEL_DIR = "models"
RESULTS_DIR = "results"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================
# LOAD DATA
# ==========================================

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

for df in [train_df, val_df, test_df]:

    df["label"] = df["label"].astype(str)

    df["image_path"] = (
        df["image_path"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
    )

print("\nTrain Samples:", len(train_df))
print("Validation Samples:", len(val_df))
print("Test Samples:", len(test_df))

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
    dataframe=train_df,
    x_col="image_path",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True
)

val_generator = test_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="image_path",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="image_path",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# ==========================================
# BUILD MODEL
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
# TRAIN
# ==========================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

print("\nTraining VGG16...")

start_train = time.time()

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

end_train = time.time()

training_time = end_train - start_train

print(
    f"\nTraining Time: {training_time:.2f} seconds"
)

# ==========================================
# EVALUATE
# ==========================================

test_loss, test_acc = model.evaluate(
    test_generator,
    verbose=1
)

predictions = model.predict(
    test_generator
)

y_pred = (
    predictions > 0.5
).astype(int)

y_true = test_generator.classes

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

roc = roc_auc_score(
    y_true,
    predictions
)

print("\nClassification Report\n")

print(
    classification_report(
        y_true,
        y_pred
    )
)

print(f"Accuracy  : {test_acc:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC AUC   : {roc:.4f}")

# ==========================================
# SAVE MODEL
# ==========================================

model_path = os.path.join(
    MODEL_DIR,
    "VGG16_BreakHis.h5"
)

model.save(model_path)

print(
    f"\nModel saved to:\n{model_path}"
)

# ==========================================
# SAVE RESULTS
# ==========================================

results = pd.DataFrame({

    "Model": ["VGG16"],

    "Accuracy": [test_acc],

    "Precision": [precision],

    "Recall": [recall],

    "F1_Score": [f1],

    "ROC_AUC": [roc],

    "Training_Time": [training_time]

})

results_path = os.path.join(
    RESULTS_DIR,
    "VGG16_results.csv"
)

results.to_csv(
    results_path,
    index=False
)

print(
    f"\nResults saved to:\n{results_path}"
)

print("\nVGG16 Training Completed Successfully.")