import os
import cv2
import json
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

class ModelEvaluator:
    def __init__(self, model_path, test_dir, output_dir='evaluation_results'):
        """
        Initialize the model evaluator
        Args:
            model_path: Path to the trained model weights
            test_dir: Directory containing test images and labels
            output_dir: Directory to save evaluation results
        """
        self.model = YOLO(model_path)
        self.test_dir = Path(test_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for results
        (self.output_dir / 'detections').mkdir(exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        
        self.results = {
            'metrics': {},
            'detections': [],
            'false_positives': [],
            'false_negatives': []
        }

    def evaluate_on_test_set(self, confidence_threshold=0.5):
        """Evaluate model on test set and generate comprehensive metrics"""
        test_images = list(self.test_dir.glob('images/*.jpg')) + \
                     list(self.test_dir.glob('images/*.png'))
        
        total_detections = 0
        total_ground_truth = 0
        true_positives = 0
        
        for img_path in test_images:
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Could not load image: {img_path}")
                continue
                
            # Get ground truth annotations
            label_path = self.test_dir / 'labels' / f"{img_path.stem}.txt"
            ground_truth_boxes = self._load_ground_truth(label_path, image.shape)
            
            # Run detection
            results = self.model(image)[0]
            detections = []
            
            for r in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = r
                if conf > confidence_threshold:
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf
                    })
            
            # Calculate metrics for this image
            tp, fp, fn = self._calculate_metrics(detections, ground_truth_boxes)
            true_positives += tp
            total_detections += len(detections)
            total_ground_truth += len(ground_truth_boxes)
            
            # Save detection visualization
            self._save_detection_visualization(image, detections, ground_truth_boxes, img_path.name)
            
            # Log results
            self.results['detections'].append({
                'image': img_path.name,
                'detections': detections,
                'ground_truth': ground_truth_boxes,
                'metrics': {
                    'true_positives': tp,
                    'false_positives': fp,
                    'false_negatives': fn
                }
            })
        
        # Calculate final metrics
        precision = true_positives / total_detections if total_detections > 0 else 0
        recall = true_positives / total_ground_truth if total_ground_truth > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        self.results['metrics'] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'total_detections': total_detections,
            'total_ground_truth': total_ground_truth,
            'true_positives': true_positives
        }
        
        # Save results
        self._save_results()
        self._generate_plots()
        
        return self.results['metrics']

    def _load_ground_truth(self, label_path, image_shape):
        """Load ground truth bounding boxes from YOLO format label file"""
        boxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    # Convert YOLO format to pixel coordinates
                    img_height, img_width = image_shape[:2]
                    x1 = (x_center - width/2) * img_width
                    y1 = (y_center - height/2) * img_height
                    x2 = (x_center + width/2) * img_width
                    y2 = (y_center + height/2) * img_height
                    boxes.append([x1, y1, x2, y2])
        return boxes

    def _calculate_metrics(self, detections, ground_truth_boxes, iou_threshold=0.5):
        """Calculate true positives, false positives, and false negatives"""
        if not ground_truth_boxes:
            return 0, len(detections), 0
            
        if not detections:
            return 0, 0, len(ground_truth_boxes)
            
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(detections), len(ground_truth_boxes)))
        for i, det in enumerate(detections):
            for j, gt in enumerate(ground_truth_boxes):
                iou_matrix[i, j] = self._calculate_iou(det['bbox'], gt)
        
        # Find matches
        true_positives = 0
        matched_detections = set()
        matched_ground_truth = set()
        
        while True:
            max_iou = np.max(iou_matrix)
            if max_iou < iou_threshold:
                break
                
            det_idx, gt_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            true_positives += 1
            matched_detections.add(det_idx)
            matched_ground_truth.add(gt_idx)
            iou_matrix[det_idx, :] = 0
            iou_matrix[:, gt_idx] = 0
        
        false_positives = len(detections) - len(matched_detections)
        false_negatives = len(ground_truth_boxes) - len(matched_ground_truth)
        
        return true_positives, false_positives, false_negatives

    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0

    def _save_detection_visualization(self, image, detections, ground_truth_boxes, image_name):
        """Save visualization of detections and ground truth"""
        vis_image = image.copy()
        
        # Draw ground truth boxes in green
        for box in ground_truth_boxes:
            cv2.rectangle(vis_image, 
                        (int(box[0]), int(box[1])), 
                        (int(box[2]), int(box[3])), 
                        (0, 255, 0), 2)
        
        # Draw detection boxes in blue
        for det in detections:
            box = det['bbox']
            cv2.rectangle(vis_image, 
                        (int(box[0]), int(box[1])), 
                        (int(box[2]), int(box[3])), 
                        (255, 0, 0), 2)
            cv2.putText(vis_image, 
                       f"{det['confidence']:.2f}", 
                       (int(box[0]), int(box[1]-10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 0, 0), 2)
        
        cv2.imwrite(str(self.output_dir / 'detections' / image_name), vis_image)

    def _save_results(self):
        """Save evaluation results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f'evaluation_results_{timestamp}.json'
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=4)

    def _generate_plots(self):
        """Generate and save evaluation plots"""
        # Precision-Recall curve
        confidences = []
        true_positives = []
        false_positives = []
        
        for det in self.results['detections']:
            for d in det['detections']:
                confidences.append(d['confidence'])
                tp = det['metrics']['true_positives']
                fp = det['metrics']['false_positives']
                true_positives.append(tp)
                false_positives.append(fp)
        
        if confidences:
            plt.figure(figsize=(10, 6))
            plt.plot(confidences, true_positives, label='True Positives')
            plt.plot(confidences, false_positives, label='False Positives')
            plt.xlabel('Confidence Threshold')
            plt.ylabel('Count')
            plt.title('Detection Performance vs Confidence Threshold')
            plt.legend()
            plt.savefig(str(self.output_dir / 'plots' / 'performance_curve.png'))
            plt.close()

if __name__ == "__main__":
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path='runs/train/pet_waste_detector/weights/best.pt',
        test_dir='dataset/test',
        output_dir='evaluation_results'
    )
    
    # Run evaluation
    print("Starting model evaluation...")
    metrics = evaluator.evaluate_on_test_set(confidence_threshold=0.5)
    
    print("\nEvaluation Results:")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1_score']:.3f}")
    print(f"Total Detections: {metrics['total_detections']}")
    print(f"Total Ground Truth: {metrics['total_ground_truth']}")
    print(f"True Positives: {metrics['true_positives']}")
    
    print("\nResults saved in 'evaluation_results' directory") 