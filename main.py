import os
import tkinter as tk
from gui.analyzer_gui import DeepFakeAnalyzerGUI
from utils.preprocessor import DataPreprocessor
from features.extractor import FeatureExtractor
from models.classifier import DeepfakeClassifier
from models.fusion import FeatureFusion

def main():
    # Create test_images directory if it doesn't exist
    if not os.path.exists('test_images'):
        os.makedirs('test_images')
        print("Please add some test images to the 'test_images' directory")
    
    # Initialize components
    preprocessor = DataPreprocessor(target_size=(224, 224))
    feature_extractor = FeatureExtractor()
    classifier = DeepfakeClassifier()
    fusion = FeatureFusion()
    
    # Create and run GUI
    root = tk.Tk()
    app = DeepFakeAnalyzerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 