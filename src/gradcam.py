# ==========================================
# gradcam.py
# Grad-CAM for DenseNet121
# ==========================================

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input

# ==========================================
# CONFIG
# ==========================================

MODEL_PATH = "models/DenseNet121_BreakHis.keras"

IMAGE_PATH = "sample_image.png"   # Replace with test image

OUTPUT_DIR = "gradcam"

LAST_CONV_LAYER = "conv5_block16_concat"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ==========================================
# LOAD MODEL
# ==========================================

print("Loading model...")

model = load_model(
    MODEL_PATH
)

print("Model loaded successfully.")

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
)

input_image = preprocess_input(
    input_image.astype(np.float32)
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

conv_outputs = conv_outputs[0]

pooled_grads = pooled_grads.numpy()

# ==========================================
# CREATE HEATMAP
# ==========================================

heatmap = np.zeros(
    conv_outputs.shape[:2]
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
plt.title("Grad-CAM Overlay")
plt.axis("off")

plt.tight_layout()

# ==========================================
# SAVE FIGURE
# ==========================================

save_path = os.path.join(
    OUTPUT_DIR,
    "DenseNet121_GradCAM.png"
)

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    f"\nGrad-CAM saved to: {save_path}"
)

# ==========================================
# PREDICTION
# ==========================================

prediction = model.predict(
    input_image
)[0][0]

print(
    f"\nPrediction Score: {prediction:.4f}"
)

if prediction > 0.5:

    print("Predicted Class: Malignant")

else:

    print("Predicted Class: Benign")