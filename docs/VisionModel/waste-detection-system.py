import cv2
import numpy as np
import torch
from ultralytics import YOLO
from datetime import datetime
import json

class PetWasteDetectionSystem:
    def __init__(self):
        # Initialize YOLO model for object detection
        self.model = YOLO('yolov8x.pt')  # Using YOLOv8 large model
        self.waste_classifier = self.load_waste_classifier()
        self.frame_buffer = []
        self.detection_threshold = 0.75

    def load_waste_classifier(self):
        # Load fine-tuned waste classification model
        # This would be trained on specific pet waste dataset
        model = torch.load('waste_classifier.pth')
        model.eval()
        return model

    def preprocess_frame(self, frame):
        # Preprocess frame for better detection
        # Apply image enhancement techniques
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
        return frame

    def detect_waste(self, frame):
        """
        Main detection function using YOLO and custom classifier
        Returns: List of detections with coordinates and confidence
        """
        processed_frame = self.preprocess_frame(frame)
        results = self.model(processed_frame)
        detections = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                confidence = box.conf[0]
                class_id = box.cls[0]

                # If detection is above threshold, classify waste type
                if confidence > self.detection_threshold:
                    waste_roi = processed_frame[int(y1):int(y2), int(x1):int(x2)]
                    waste_type = self.classify_waste(waste_roi)
                    
                    detection = {
                        'coords': (x1, y1, x2, y2),
                        'confidence': float(confidence),
                        'waste_type': waste_type,
                        'timestamp': datetime.now().isoformat()
                    }
                    detections.append(detection)

        return detections

    def classify_waste(self, waste_roi):
        """
        Classify the type and characteristics of detected waste
        Returns: Dictionary with waste characteristics
        """
        # Resize ROI for classifier
        waste_roi = cv2.resize(waste_roi, (224, 224))
        
        # Convert to tensor and normalize
        tensor = torch.from_numpy(waste_roi).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0)

        # Get classifier prediction
        with torch.no_grad():
            prediction = self.waste_classifier(tensor)
            
        # Process prediction to get characteristics
        characteristics = {
            'consistency': self.analyze_consistency(waste_roi),
            'color': self.analyze_color(waste_roi),
            'size': self.calculate_size(waste_roi)
        }
        
        return characteristics

    def analyze_consistency(self, roi):
        """Analyze waste consistency using texture analysis"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        texture_features = self.calculate_texture_features(gray)
        return texture_features

    def analyze_color(self, roi):
        """Analyze color characteristics of waste"""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Calculate color histogram
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        
        return self.classify_color_profile(hist)

    def calculate_size(self, roi):
        """Calculate approximate size of waste"""
        # Using contour area with calibration factor
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            area = cv2.contourArea(contours[0])
            # Convert pixel area to real-world units (needs calibration)
            return self.convert_to_real_size(area)
        return 0

    def log_detection(self, detection):
        """Log detection data for health monitoring"""
        with open('detection_log.json', 'a') as f:
            json.dump(detection, f)
            f.write('\n')

    def get_health_insights(self, detections):
        """Analyze recent detections for health insights"""
        # Analyze patterns in waste characteristics
        # Return any concerning changes or patterns
        pass