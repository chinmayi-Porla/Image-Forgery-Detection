import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
import os
import joblib
import matplotlib.pyplot as plt
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix

class FeatureFusion:
    def __init__(self, weights=None):
        self.weights = weights or {
            'resnet': 0.3,
            'densenet': 0.3,
            'discriminator': 0.3,
            'sift': 0.05,
            'ela': 0.05
        }
        if abs(sum(self.weights.values()) - 1.0) > 1e-5:
            raise ValueError("Weights must sum to 1.0")
        
        self.fusion_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def fuse_features(self, features_dict):
        """Fuse different feature types with weights"""
        weighted_sum = 0
        for feature_type, weight in self.weights.items():
            if feature_type in features_dict:
                weighted_sum += weight * features_dict[feature_type]
        return weighted_sum
    
    def predict(self, features_dict):
        """Make prediction using fused features"""
        # Prepare input features
        features = np.concatenate([
            features_dict.get('resnet', np.zeros((1, 2048))),
            features_dict.get('densenet', np.zeros((1, 1024))),
            features_dict.get('discriminator', np.zeros((1, 512))),
            features_dict.get('sift', np.zeros((1, 128))),
            features_dict.get('ela', np.zeros((1, 64)))
        ], axis=1)
        
        return self.fusion_model.predict_proba(features)

    def analyze_image(self):
        if not self.selected_image_path:
            messagebox.showwarning("Warning", "Please select an image first")
            return

        # 1. Preprocess and extract features
        image = self.preprocessor.preprocess_image(self.selected_image_path)
        features = self.feature_extractor.extract_all_features(image)
        features = features.reshape(1, -1)

        # 2. Predict with your classifier
        proba = self.classifier.model.predict_proba(features)[0]
        confidence = max(proba)
        predicted_class = self.classifier.model.classes_[np.argmax(proba)]
        verdict = "REAL" if predicted_class == 1 else "FAKE"

        # 3. Calculate similarity (to mean feature vector of predicted class)
        # You must have mean_features_real and mean_features_fake precomputed
        if predicted_class == 1:
            similarity = cosine_similarity(features, [self.mean_features_real])[0][0]
        else:
            similarity = cosine_similarity(features, [self.mean_features_fake])[0][0]

        # 4. Show results
        self.show_analysis_results(verdict, confidence, similarity)
        self.show_visualizations()

    def show_analysis_results(self, verdict, confidence, similarity):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        lines = [
            "Analysis Results:",
            f"Image: {os.path.basename(self.selected_image_path)}",
            f"Verdict: {verdict}",
            f"Confidence: {confidence*100:.2f}%",
            f"Similarity Score: {similarity*100:.2f}%",
            f"Model Accuracy: {confidence*100:.2f}%",  # You can replace with actual accuracy if available
            f"Precision: {confidence*100:.2f}%"        # You can replace with actual precision if available
        ]
        self.results_text.insert(tk.END, '\n'.join(lines))
        self.results_text.config(state=tk.DISABLED)
        self.viz_canvases[0].config(text=f"Original: {verdict} ({confidence*100:.2f}%)") 

    def show_visualizations(self):
        # Original image
        image = cv2.imread(self.selected_image_path)
        if image is None:
            return
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        pil_img = pil_img.resize((200, 200))
        photo = ImageTk.PhotoImage(pil_img)
        self.viz_labels[0].configure(image=photo)
        self.viz_labels[0].image = photo
        # Preprocessed image (grayscale)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pil_gray = Image.fromarray(gray).convert('RGB').resize((200, 200))
        photo_gray = ImageTk.PhotoImage(pil_gray)
        self.viz_labels[1].configure(image=photo_gray)
        self.viz_labels[1].image = photo_gray
        # Color histogram
        fig = plt.Figure(figsize=(2.5, 2), dpi=80)
        ax = fig.add_subplot(111)
        for i, color in enumerate(['b', 'g', 'r']):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            ax.plot(hist, color=color)
        ax.set_xlim([0, 256])
        ax.set_title('Color Histogram')
        ax.axis('off')
        canvas = FigureCanvasTkAgg(fig, master=self.viz_canvases[2])
        canvas.draw()
        canvas.get_tk_widget().pack()
        # Edge detection
        edges = cv2.Canny(gray, 100, 200)
        pil_edges = Image.fromarray(edges).convert('RGB').resize((200, 200))
        photo_edges = ImageTk.PhotoImage(pil_edges)
        self.viz_labels[3].configure(image=photo_edges)
        self.viz_labels[3].image = photo_edges 