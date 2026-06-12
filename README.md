# NN-flower-classification-
# 🌸 Flower Recognition

A deep learning web app that classifies flower images into 5 categories using three different model architectures — built with TensorFlow and Streamlit.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🌼 Demo

Upload or snap a photo of a flower and get an instant prediction with confidence scores across all 5 classes.

| 🌼 Daisy | 🌱 Dandelion | 🌹 Rose | 🌻 Sunflower | 🌷 Tulip |
|----------|-------------|---------|-------------|---------|

---

## 🧠 Models

| Model | Architecture | Notes |
|---|---|---|
| **CNN** | Residual blocks + SeparableConv | Skip connections + Global Average Pooling head |
| **MobileNetV2 TL** | MobileNetV2 + deeper 512→256 head | Fine-tunes top 50 layers of ImageNet weights |
| **DNN** | 4-layer Dense (512→384→256→128) | L2 regularization + LeakyReLU + label smoothing |

---

## 📁 Project Structure

```
flower-recognition/
│
├── app.py                        # Streamlit web app
├── flower_recognition_complete2.ipynb  # Training notebook
│
└── checkpoints/                  # Trained model files (generated after training)
    ├── cnn_best.keras
    ├── dnn_best.keras
    └── tl_best.keras
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/flower-recognition.git
cd flower-recognition
```

### 2. Install dependencies

```bash
pip install tensorflow streamlit opencv-python pillow numpy
```

### 3. Train the models

Open and run the Jupyter notebook to train and save the models:

```bash
jupyter notebook flower_recognition_complete2.ipynb
```

This will generate the `.keras` files inside the `checkpoints/` directory.

### 4. Run the app

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`.

---

## 📊 Dataset

[Flowers Recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition) from Kaggle — approximately **4,300 images** across 5 flower classes.

---

## ⚡ Performance Optimizations

The app is built for speed:

- **TensorFlow imported once** and cached — no reimport on reruns
- **Models cached per name** — loaded once, reused across all predictions
- **Graph warm-up** at load time — first prediction is instant
- **Preprocessed tensors cached by image hash** — same photo is never reprocessed twice

---

## 🖥️ App Features

- **Upload tab** — upload a JPG, PNG, or WebP image
- **Camera tab** — take a live photo directly in the browser
- **Top 3 predictions** with probability breakdown
- **Bar chart** of all class probabilities
- **Model selector** in the sidebar to switch between CNN, MobileNetV2 TL, and DNN

---

## 📦 Requirements

```
tensorflow>=2.10
streamlit
opencv-python
pillow
numpy
```

---

## 📄 License

This project is licensed under the MIT License.
