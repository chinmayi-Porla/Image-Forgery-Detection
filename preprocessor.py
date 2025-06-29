import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

class DataPreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
    
    def load_image(self, image_path):
        """Load and preprocess an image"""
        try:
            image = Image.open(image_path)
            image = image.resize(self.target_size)
            image = np.array(image)
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return None
    
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        if isinstance(image, str):
            image = self.load_image(image)
        if image is None:
            return None
        
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def batch_preprocess(self, image_paths):
        """Preprocess a batch of images"""
        processed_images = []
        for path in image_paths:
            processed = self.preprocess_image(path)
            if processed is not None:
                processed_images.append(processed)
        return np.array(processed_images) 