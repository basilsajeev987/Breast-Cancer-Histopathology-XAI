# ==========================================
# gradcam.py
# Grad-CAM for BreakHis Models
# ==========================================

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

# ==========================================
# CONFIG
# ==========================================

MODEL_NAME = "DenseNet121"

MODEL_PATH = "models/DenseNet121_BreakHis.h5"

IMAGE_PATH = "sample_image.png"

LAST_CONV_LAYER = "conv5_block16_concat"

OUTPUT_DIR = "gradcam"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ==========================================
# LOAD MODEL
# ==========================================

model = load_model(
    MODEL_PATH
)

print("Model Loaded")

# ==========================================
# LOAD IMAGE
# ==========================================

img = cv2.imread(
    IMAGE_PATH
)

if img is None:

    raise Exception(
        f"Cannot read image:\n{IMAGE_PATH}"
    )

img_rgb = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2RGB
)

img_resized = cv2.resize(
    img_rgb,
    (224,224)
)

input_image = np.expand_dims(
    img_resized,
    axis=0
)

input_image = tf.cast(
    input_image,
    tf.float32
)

# ==========================================
# PREDICTION
# ==========================================

prediction = model.predict(
    input_image,
    verbose=0
)[0][0]

predicted_label = (
    "Malignant"
    if prediction > 0.5
    else "Benign"
)

print(
    f"Prediction Score: {prediction:.4f}"
)

print(
    f"Predicted Class: {predicted_label}"
)

# ==========================================
# CREATE GRAD MODEL
# ==========================================

grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[
        model.get_layer(
            LAST_CONV_LAYER
        ).output,
        model.output
    ]
)

# ==========================================
# COMPUTE GRADIENTS
# ==========================================

with tf.GradientTape() as tape:

    conv_outputs, predictions = grad_model(
        input_image
    )

    loss = predictions[:,0]

grads = tape.gradient(
    loss,
    conv_outputs
)

pooled_grads = tf.reduce_mean(
    grads,
    axis=(0,1,2)
)

conv_outputs = conv_outputs[0].numpy()

pooled_grads = pooled_grads.numpy()

# ==========================================
# CREATE HEATMAP
# ==========================================

heatmap = np.zeros(
    shape=conv_outputs.shape[:2],
    dtype=np.float32
)

for i in range(
    pooled_grads.shape[0]
):

    heatmap += (
        pooled_grads[i]
        * conv_outputs[:,:,i]
    )

heatmap = np.maximum(
    heatmap,
    0
)

heatmap /= (
    np.max(heatmap) + 1e-8
)

# ==========================================
# RESIZE HEATMAP
# ==========================================

heatmap = cv2.resize(
    heatmap,
    (
        img_rgb.shape[1],
        img_rgb.shape[0]
    )
)

# ==========================================
# SMOOTH HEATMAP
# ==========================================

heatmap = cv2.GaussianBlur(
    heatmap,
    (21,21),
    0
)

heatmap /= (
    heatmap.max() + 1e-8
)

# ==========================================
# APPLY COLORMAP
# ==========================================

heatmap_uint8 = np.uint8(
    255 * heatmap
)

heatmap_color = cv2.applyColorMap(
    heatmap_uint8,
    cv2.COLORMAP_JET
)

heatmap_color = cv2.cvtColor(
    heatmap_color,
    cv2.COLOR_BGR2RGB
)

# ==========================================
# OVERLAY
# ==========================================

overlay = cv2.addWeighted(
    img_rgb,
    0.75,
    heatmap_color,
    0.25,
    0
)

# ==========================================
# DISPLAY
# ==========================================

plt.figure(figsize=(18,6))

plt.subplot(1,3,1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(heatmap_color)
plt.title("Grad-CAM Heatmap")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(overlay)
plt.title(
    f"{MODEL_NAME}\n{predicted_label}\nConfidence={prediction:.4f}"
)
plt.axis("off")

plt.tight_layout()

# ==========================================
# SAVE
# ==========================================

output_path = os.path.join(
    OUTPUT_DIR,
    f"GradCAM_{MODEL_NAME}.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    f"\nSaved:\n{output_path}"
)