# ✅ Updated Streamlit Lung Cancer Detection App with Custom Grad-CAM and Explainable AI
# -----------------------------------------------------------------------------------
# This version integrates your exact Grad-CAM logic (from your Colab code)
# and supports prediction, heatmap visualization, and batch evaluation.
# -----------------------------------------------------------------------------------

import os
import io
import tempfile
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
import cv2
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import shap

# Streamlit config
st.set_page_config(page_title='🫁 Lung Cancer Detection — Explainable AI', layout='wide')

# ---------------------------
# Configuration
# ---------------------------
MODEL_PATH = 'trained_lung_cancer_model.keras'
IMAGE_SIZE = (350, 350)
CLASS_LABELS = [
    'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib',
    'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa',
    'normal',
    'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa'
]

# ---------------------------
# Utility functions
# ---------------------------
@st.cache_resource
def load_model_safe(model_path):
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}")
        return None
    return tf.keras.models.load_model(model_path)

def preprocess_image(img_pil, target_size=IMAGE_SIZE):
    img = img_pil.convert('RGB').resize(target_size)
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    return arr

def predict_image(model, img_array):
    preds = model.predict(img_array)
    return preds[0]

# ---------------------------
# Custom Grad-CAM logic (from your Colab code)
# ---------------------------
def prepare_gradcam_model(model):
    # Find Xception base
    xception_base = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            xception_base = layer
            break
    if xception_base is None:
        raise ValueError("❌ Could not find Xception base inside the model!")

    # Last conv layer
    last_conv_layer = xception_base.get_layer("block14_sepconv2_act")
    st.write(f"✅ Using `{last_conv_layer.name}` as last convolution layer for Grad-CAM.")

    # Reuse trained GAP + Dense layers
    gap_layer = [l for l in model.layers if isinstance(l, tf.keras.layers.GlobalAveragePooling2D)][0]
    dense_layer = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)][0]

    # Ensure a single tensor input
    x_base_output = xception_base.output
    if isinstance(x_base_output, (list, tuple)):
        x_base_output = x_base_output[0]

    x = gap_layer(x_base_output)
    outputs = dense_layer(x)
    functional_model = tf.keras.models.Model(inputs=xception_base.input, outputs=outputs)

    return functional_model, last_conv_layer




def display_gradcam(img_pil, model, last_conv_layer, class_labels, IMAGE_SIZE=(350,350), alpha=0.4):
    # Prepare input
    img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    img_pil.save(img_path)
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict class using original trained model
    preds = model.predict(img_array)
    pred_index = np.argmax(preds[0])
    predicted_label = class_labels[pred_index]

    # Grad-CAM model for getting last conv output
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap)+1e-8)
    heatmap = heatmap.numpy()

    img_orig = cv2.cvtColor(np.array(img_pil.resize(IMAGE_SIZE)), cv2.COLOR_RGB2BGR)
    heatmap = cv2.resize(heatmap, (img_orig.shape[1], img_orig.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img_orig, 1-alpha, heatmap, alpha, 0)

    return superimposed_img, predicted_label



def explain_with_shap(trained_model, img_pil, IMAGE_SIZE=(350, 350)):
    """Explain model prediction using SHAP (robust version)."""
    img_rgb = img_pil.convert("RGB").resize(IMAGE_SIZE)
    img_array = np.array(img_rgb, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Background sample for SHAP
    background = np.random.random((1, *IMAGE_SIZE, 3)).astype(np.float32)

    try:
        explainer = shap.GradientExplainer(trained_model, background)
        shap_values = explainer.shap_values(img_array)
    except Exception as e:
        st.error(f"⚠️ SHAP failed: {e}")
        return np.zeros((400, 400, 3), dtype=np.uint8)

    try:
        if isinstance(shap_values, list):
            pred_probs = trained_model.predict(img_array)[0]
            pred_class = int(np.argmax(pred_probs))
            shap_img = shap_values[min(pred_class, len(shap_values) - 1)][0]
        else:
            shap_img = shap_values[0]

        
        if shap_img.ndim == 2:
            shap_img = np.stack([shap_img] * 3, axis=-1)
        elif shap_img.shape[-1] != 3:
            shap_img = shap_img[..., :3]

        
        shap_img = (shap_img - shap_img.min()) / (shap_img.max() - shap_img.min() + 1e-8)
        shap_img = np.uint8(255 * shap_img)

        if shap_img.size == 0 or shap_img.shape[0] == 0 or shap_img.shape[1] == 0:
            shap_img = np.zeros((400, 400, 3), dtype=np.uint8)
        else:
            shap_img = cv2.resize(shap_img, (400, 400))

    except Exception as e:
        st.error(f"⚠️ SHAP post-processing error: {e}")
        shap_img = np.zeros((400, 400, 3), dtype=np.uint8)

    return shap_img

import tensorflow as tf
import numpy as np
import cv2

def integrated_gradients(model, img_tensor, baseline=None, steps=50):
    """Compute Integrated Gradients for a single image tensor."""
    if baseline is None:
        baseline = tf.zeros_like(img_tensor)

    interpolated_imgs = [baseline + (float(i) / steps) * (img_tensor - baseline) for i in range(steps + 1)]
    interpolated_imgs = tf.convert_to_tensor(tf.concat(interpolated_imgs, axis=0))  # ✅ ensure tensor

    with tf.GradientTape() as tape:
        tape.watch(interpolated_imgs)
        preds = model(interpolated_imgs)
        top_pred_index = tf.argmax(preds[-1])
        target = preds[:, top_pred_index]

    grads = tape.gradient(target, interpolated_imgs)
    avg_grads = tf.reduce_mean(grads, axis=0)
    integrated_grads = (img_tensor - baseline) * avg_grads

    return integrated_grads


def explain_with_ig(model, img_pil, IMAGE_SIZE=(350, 350)):
    """Explain prediction using Integrated Gradients."""
    
    img_rgb = img_pil.convert("RGB").resize(IMAGE_SIZE)
    img_array = np.array(img_rgb, dtype=np.float32) / 255.0
    img_tensor = tf.convert_to_tensor(img_array[np.newaxis, ...])  # ✅ convert to tensor

    try:
        ig_attributions = integrated_gradients(model, img_tensor, steps=30)
        ig_img = tf.reduce_sum(tf.math.abs(ig_attributions[0]), axis=-1).numpy()

        # Normalize to [0, 255]
        ig_img = (ig_img - ig_img.min()) / (ig_img.max() - ig_img.min() + 1e-8)
        ig_img = np.uint8(255 * ig_img)
        ig_img = cv2.applyColorMap(ig_img, cv2.COLORMAP_JET)
        ig_img = cv2.resize(ig_img, (400, 400))
    except Exception as e:
        st.error(f"⚠️ IG failed: {e}")
        ig_img = np.zeros((400, 400, 3), dtype=np.uint8)

    return ig_img

# ---------------------------
# Streamlit App
# ---------------------------
def main():
    st.title("🫁 Lung Cancer Detection — Explainable AI (Grad-CAM + Prediction)")

    # Sidebar
    st.sidebar.header("⚙️ Model Setup")
    uploaded_model = st.sidebar.file_uploader("Upload .keras or .h5 model", type=["keras", "h5"])
    if uploaded_model is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp:
            tmp.write(uploaded_model.read())
            model = load_model_safe(tmp.name)
    else:
        model = load_model_safe(MODEL_PATH)

    if model is None:
        st.stop()

    functional_model, last_conv_layer = prepare_gradcam_model(model)

    
    tab1, tab2, tab3 = st.tabs(["🩺 Predict & Explain", "📊 Batch Evaluation", "ℹ️ Model Info"])

    # ---------------------------
    # Tab 1: Single Image Prediction
    # ---------------------------
    with tab1:
      st.subheader("Single Image Prediction & Grad-CAM Visualization")
      uploaded_file = st.file_uploader("Upload a CT/X-ray Image", type=["png", "jpg", "jpeg"])

      if uploaded_file:
          img_pil = Image.open(uploaded_file)

          max_width = 500
          w, h = img_pil.size
          new_h = int(h * max_width / w)
          img_resized = img_pil.resize((max_width, new_h))
          st.image(img_resized, caption="Uploaded Image", use_container_width=False)

          # Predict
          arr = preprocess_image(img_pil)
          preds = predict_image(model, arr)
          pred_idx = np.argmax(preds)
          predicted_label = CLASS_LABELS[pred_idx]

          st.markdown(f"### 🧠 Predicted Class: **{predicted_label}** ({preds[pred_idx]*100:.2f}%)")
          st.bar_chart(pd.DataFrame({"Class": CLASS_LABELS, "Probability": preds}).set_index("Class"))

          # Grad-CAM
          st.markdown("---")
          st.subheader("🔍 Grad-CAM Heatmap")
          gradcam_img, gradcam_label = display_gradcam(img_pil, functional_model, last_conv_layer, CLASS_LABELS)

          # Resize Grad-CAM to max width 700px
          gradcam_img_resized = cv2.resize(
              gradcam_img,
              (500, int(500 * gradcam_img.shape[0] / gradcam_img.shape[1]))
          )

          st.image(gradcam_img_resized, caption=f"Grad-CAM Visualization ({gradcam_label})", use_container_width=False)

          # Download button
          buf = io.BytesIO()
          Image.fromarray(cv2.cvtColor(gradcam_img_resized, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")
          buf.seek(0)
          st.download_button("📥 Download Grad-CAM Image", buf, "gradcam.png", "image/png")          

          # Integrated Gradients Explanation
          st.markdown("---")
          st.subheader("Integrated Gradients Explanation")
          ig_img = explain_with_ig(model, img_pil, IMAGE_SIZE)
          st.image(ig_img, caption="Integrated Gradients Attribution Map", use_container_width=False)






    # ---------------------------
    # Tab 2: Batch Evaluation
    # ---------------------------
    with tab2:
        st.subheader("Batch Folder Evaluation")
        val_folder = st.text_input("Enter validation folder path (with subfolders per class):")

        if st.button("Run Evaluation"):
            if not os.path.exists(val_folder):
                st.error("Validation folder not found!")
            else:
                y_true, y_pred = [], []
                for idx, cls in enumerate(CLASS_LABELS):
                    class_dir = os.path.join(val_folder, cls)
                    if not os.path.isdir(class_dir):
                        continue
                    for file in os.listdir(class_dir):
                        if file.lower().endswith(('png', 'jpg', 'jpeg')):
                            img = Image.open(os.path.join(class_dir, file))
                            arr = preprocess_image(img)
                            pred = np.argmax(predict_image(model, arr))
                            y_true.append(idx)
                            y_pred.append(pred)

                if not y_true:
                    st.warning("No images found!")
                else:
                    cm = confusion_matrix(y_true, y_pred)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
                    plt.xlabel("Predicted")
                    plt.ylabel("True")
                    st.pyplot(fig)

                    cr = classification_report(y_true, y_pred, target_names=CLASS_LABELS, output_dict=True)
                    st.dataframe(pd.DataFrame(cr).transpose())

    # ---------------------------
    # Tab 3: Model Info
    # ---------------------------
    with tab3:
        st.subheader("Model Summary & Training Info")
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        st.text("\n".join(stringlist))

        st.markdown("---")
        st.markdown("""
        ### 💡 Tips:
        - Ensure class folder names match the model output labels.
        - Use balanced data for training.
        - Grad-CAM helps visualize which region influenced the model’s decision.
        """)

if __name__ == "__main__":
    main()
