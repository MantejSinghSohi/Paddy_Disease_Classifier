import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models

# Class names
VARIETY_CLASS_NAMES = [
    'AndraPonni', 'AtchayaPonni', 'IR20', 'KarnatakaPonni', 
    'Onthanel', 'Ponni', 'RR', 'Surya', 'Zonal'
]
DISEASE_CLASS_NAMES = [
    'bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight',
    'black_stem_borer', 'blast', 'brown_spot', 'downy_mildew', 'hispa', 
    'leaf_roller', 'normal', 'tungro', 'white_stem_borer', 'yellow_stem_borer'
]

# Model architecture
def create_inception_model(num_classes):
    model = models.inception_v3(pretrained=False, aux_logits=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
    return model

# Model loader
@st.cache_resource
def load_model(model_path, num_classes):
    try:
        model = create_inception_model(num_classes)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)

st.set_page_config(page_title="Paddy Disease & Variety Classifier", layout="wide")

# --- Info about the models ---
st.markdown("""
### 🧠 Model Information

This app uses two fine-tuned **Inception v3** models:
- **Disease Model:** Classifies paddy leaf/plant images into 13 disease categories.
- **Variety Model:** Identifies among 9 paddy varieties.

Inception v3 is a deep convolutional neural network designed for efficient and accurate image classification, using techniques like factorized convolutions, auxiliary classifiers, and batch normalization for improved performance and generalization[1][2][3][4][8].
""")

# --- Input instructions ---
st.markdown("""
### 📷 How to Use

1. **Upload** a clear, well-lit image of a paddy leaf or plant (JPG, JPEG, PNG).
2. **Wait** a moment while the models analyze your image.
3. **See** the top 3 predictions for both disease and variety.

*Tip: For best results, ensure the leaf/plant is in focus and fills most of the image.*
""")

# Model loading
disease_model = load_model("model_disease.pth", len(DISEASE_CLASS_NAMES))
variety_model = load_model("model_variety.pth", len(VARIETY_CLASS_NAMES))

if not disease_model or not variety_model:
    st.stop()

uploaded_file = st.file_uploader("Upload a paddy leaf or plant image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', width=350)
    input_tensor = preprocess_image(image)

    with torch.no_grad():
        disease_output = disease_model(input_tensor)
        disease_probs = torch.nn.functional.softmax(disease_output[0], dim=0)
        disease_top3 = torch.topk(disease_probs, 3)

        variety_output = variety_model(input_tensor)
        variety_probs = torch.nn.functional.softmax(variety_output[0], dim=0)
        variety_top3 = torch.topk(variety_probs, 3)

    st.markdown("### 📝 Prediction Results")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Diseases Possible: (Top 3)")
        for i in range(3):
            st.write(f"{DISEASE_CLASS_NAMES[disease_top3.indices[i]]}: {disease_top3.values[i].item()*100:.2f}%")
    with col2:
        st.subheader("Variety (Top 3)")
        for i in range(3):
            st.write(f"{VARIETY_CLASS_NAMES[variety_top3.indices[i]]}: {variety_top3.values[i].item()*100:.2f}%")

    st.info("Prediction complete. For best accuracy, use high-quality images.")

# --- Minimal sidebar ---
st.sidebar.header("About")
st.sidebar.info(
    "This app uses Inception v3 models for paddy disease and variety classification. "
    "Developed for research and educational use."
)
