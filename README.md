

# 🔍 Image Forgery Detection System using Deep Learning Models

This repository contains a powerful and user-friendly image forgery detection system, designed using deep learning models. It helps identify tampered or manipulated images using advanced convolutional neural networks (CNNs) and image-processing techniques.


## 🎯 Project Objective

To develop an intelligent system that can detect forged regions in digital images by leveraging deep learning models trained on tampered image datasets.



## 🚀 Features

### 🧠 Deep Learning-Based Detection

* Utilizes CNN-based architectures to detect forged regions.
* Capable of identifying copy-move and splicing-based forgery.
* Heatmap/Mask output for visualizing tampered areas.

### 🧪 Forgery Types Detected

* **Copy-Move Forgery**: Parts of the same image are copied and pasted elsewhere in the image.
* **Splicing**: Content taken from different sources is combined into a single image.

### 🖼️ Input/Output Visualization

* Input: JPEG/PNG images
* Output: Mask or bounding box highlighting forged regions
* Option to save or display results with overlayed predictions



## 🧱 System Architecture

* **Frontend**: Command-line interface (CLI) for loading and processing images
* **Backend**: Deep learning models (CNNs) implemented in Keras/TensorFlow or PyTorch
* **Preprocessing**: Image resizing, patch extraction, and normalization
* **Postprocessing**: Forgery map generation using CNN outputs



## 🛠 Installation & Setup

### ✅ Prerequisites

* Python 3.8+
* pip (Python package manager)
* Virtual environment (recommended)
* TensorFlow or PyTorch

### 📦 Installation Steps

```bash
git clone https://github.com/your-username/image-forgery-detection.git
cd image-forgery-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```



## ▶️ Running the System

### 🎮 Detect Forgery in an Image

```bash
python main.py
```

You will be prompted to enter:

* ✅ Path to the image file
* ✅ Model to use (if multiple models are available)
* ✅ Output preference (save or display)



## 📊 Example Output

* Input: `tampered_image.jpg`
* Output: Displayed heatmap with forged region highlighted



## 🧪 Model Details

### CNN Architecture

* 3 Convolutional Layers (32, 64, 128 filters)
* Batch Normalization and Dropout
* MaxPooling layers for spatial reduction
* Dense layer with Softmax or Sigmoid output for classification

### Input

* Normalized image patches or full image: `(128x128x3)`
* Labels indicating forgery or original

### Training

* Dataset: CASIA v2, CoMoFoD, or synthetic datasets
* Loss Function: Binary Cross-Entropy or Categorical Cross-Entropy
* Optimizer: Adam



## 🗂 Project Structure

```
📁 Image-Forgery-Detection-System
├── main.py                     # Entry point of the system
├── models/                     # Saved trained models
├── utils/                      # Helper functions for preprocessing and visualization
├── test_images/                # Sample images to test
├── output/                     # Detected forged region masks
├── requirements.txt
└── README.md
```


## 📈 Evaluation Metrics

* Precision, Recall, and F1-Score
* Accuracy over tampered vs. authentic regions
* IOU (Intersection over Union) for mask prediction



## 👨‍💻 Developer Info

Project by **Chinmayi Porla**
A deep learning-based approach to image forgery detection with full integration of preprocessing, detection, and visualization.




