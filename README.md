# Paddy Disease Classifier
# 🌾 Paddy Disease & Variety Classifier

# 🌾 Paddy Disease & Variety Classifier

Welcome to the **Paddy Disease & Variety Classifier**!  
Upload images of paddy leaves or plants and get instant predictions for:

---

## 🦠 Paddy Diseases

| Disease Name                | Disease Name            | Disease Name              |
|-----------------------------|------------------------|---------------------------|
| bacterial_leaf_blight       | black_stem_borer       | downy_mildew              |
| bacterial_leaf_streak       | blast                  | hispa                     |
| bacterial_panicle_blight    | brown_spot             | leaf_roller               |
| normal                      | tungro                 | white_stem_borer          |
| yellow_stem_borer           |                        |                           |

---

## 🌱 Paddy Varieties

| Variety Name      | Variety Name      | Variety Name  |
|-------------------|------------------|--------------|
| AndraPonni        | AtchayaPonni     | IR20         |
| KarnatakaPonni    | Onthanel         | Ponni        |
| RR                | Surya            | Zonal        |

---

*For best results, upload a clear, well-lit image of a paddy leaf or plant. The app will predict the top 3 most likely diseases and varieties for your sample!*


## 🚀 How It Works

- The app uses **fine-tuned Inception v3 neural networks** (PyTorch) trained on labeled paddy images.
- When you upload an image, it is preprocessed (resized, normalized) to match the model's training setup.
- The app predicts the top 3 most likely diseases and varieties for the uploaded image, showing both the class names and their probabilities.

## 🛠️ Features

- **Dual Prediction:** Simultaneously predicts both disease and variety.
- **Top-3 Results:** See the most probable classes for each prediction.
- **User-Friendly:** Upload images directly from your device.
- **Fast & Reliable:** Runs on CPU and does not require GPU.

## ⚙️ Installation

1. **Clone the repository** (or copy the files).
2. **Install dependencies:** ``` pip install -r requirements.txt ```
3. **Place your trained model files** (`model_disease.pth` and `model_variety.pth`) in the app directory.
4. **Run the app:** ``` streamlit run app.py ```


## 🖼️ Usage

- Click "Choose an image..." and upload a clear photo of a paddy leaf or plant.
- View the predictions for both disease and variety in real-time.

## 🧠 Model Details

- **Architecture:** Inception v3 (PyTorch, with custom final layers for your classes)
- **Preprocessing:** Images are resized to 299x299, normalized to ImageNet standards.
- **Class Names:** Custom class lists for both diseases and varieties.

## 📦 Dependencies

- `streamlit` – For the interactive web interface
- `torch`, `torchvision` – For model inference and image transformations
- `pillow` – For image loading and processing

## 💡 Notes

- For best results, upload high-quality, well-lit images.
- The app runs entirely locally; no data is sent to the cloud.

Enjoy classifying your paddy images! 🌾🦠🌱

---

*If you need to use `.pkl` files instead of `.pth`, simply update the `load_model` function to use `torch.load(model_path, ...)` as appropriate for your files.*


