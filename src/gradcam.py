# ==========================================
# gradcam.py
# Generate Grad-CAM Visualizations
# ==========================================

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_NAME = "DenseNet121"

IMAGE_PATH = "sample_image.png"

MODEL_CONFIG = {

    "DenseNet121": {
        "model_path": "models/DenseNet121_BreakHis.keras",
        "last_conv_layer": "conv5_block16_concat"
    },

    "ResNet50": {
        "model_path": "models/ResNet50_BreakHis.keras",
        "last_conv_layer": "conv5_block3_out"
    },

    "EfficientNetB0": {
        "model_path": "models/EfficientNetB0_BreakHis.keras",
        "last_conv_layer": "top_activation"
    },

    "VGG16": {
        "model_path": "models/VGG16_BreakHis.keras",
        "last_conv_layer": "block5_conv3"
    },

    "MobileNetV2": {
        "model_path": "models/MobileNetV2_BreakHis.keras",
        "last_conv_layer": "out_relu"
    }
}

# ==========================================
# VALIDATE MODEL
# ==========================================

if MODEL_NAME not in MODEL_CONFIG:
    raise ValueError(
        f"Unsupported model: {MODEL_NAME}"
    )

MODEL_PATH = MODEL_CONFIG[
    MODEL_NAME
]["model_path"]

LAST_CONV_LAYER = MODEL_CONFIG[
    MODEL_NAME
]["last_conv_layer"]

# ==========================================
# CREATE OUTPUT DIRECTORY
# ==========================================

OUTPUT_DIR = "gradcam"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ==========================================
# LOAD MODEL
# ==========================================

print(
    f"Loading {MODEL_NAME}..."
)

model = load_model(
    MODEL_PATH
)

print("Model loaded.")

# ==========================================
# LOAD IMAGE
# ==========================================

img = cv2.imread(
    IMAGE_PATH
)

if img is None:

    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
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
).astype(np.float32)

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

# ==========================================
# CREATE HEATMAP
# ==========================================

pooled_grads = tf.reduce_mean(
    grads,
    axis=(0,1,2)
)

conv_outputs = conv_outputs[0]

heatmap = tf.reduce_sum(
    pooled_grads * conv_outputs,
    axis=-1
)

heatmap = heatmap.numpy()

heatmap = np.maximum(
    heatmap,
    0
)

heatmap = heatmap / (
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
    0.7,
    heatmap_color,
    0.3,
    0
)

# ==========================================
# PREDICTION
# ==========================================

prediction = model.predict(
    input_image,
    verbose=0
)[0][0]

predicted_class = (
    "Malignant"
    if prediction > 0.5
    else "Benign"
)

print(
    f"Prediction Score: {prediction:.4f}"
)

print(
    f"Predicted Class: {predicted_class}"
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
    f"{MODEL_NAME} Grad-CAM"
)
plt.axis("off")

plt.tight_layout()

# ==========================================
# SAVE FIGURE
# ==========================================

save_path = os.path.join(
    OUTPUT_DIR,
    f"{MODEL_NAME}_GradCAM.png"
)

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    f"\nSaved: {save_path}"
)