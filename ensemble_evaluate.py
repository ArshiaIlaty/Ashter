#!/usr/bin/env python3
"""
Ensemble YOLOv8n and YOLOv8s fine-tuned models using Weighted Box Fusion (WBF)
Evaluate ensemble on the test set and print metrics
"""
import os
import numpy as np
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from glob import glob
from tqdm import tqdm
import yaml

# Paths to models and data
yolov8n_path = "runs/detect/yolov8n_finetune_old_dataset/weights/best.pt"
yolov8s_path = "runs/detect/yolov8s_finetune_old_dataset/weights/best.pt"
test_images_dir = "dataset/test/images"
test_labels_dir = "dataset/test/labels"
data_yaml = "dataset/data.yaml"

# Load class names
with open(data_yaml, 'r') as f:
    data = yaml.safe_load(f)
    class_names = data['names'] if 'names' in data else [str(i) for i in range(data['nc'])]

# Load models
model_n = YOLO(yolov8n_path)
model_s = YOLO(yolov8s_path)

# Helper: Run inference and return boxes, scores, labels (normalized)
def run_inference(model, img_path):
    results = model(img_path, conf=0.001, iou=0.1, verbose=False)[0]
    boxes = results.boxes.xywhn.cpu().numpy() if results.boxes is not None else np.zeros((0, 4))
    scores = results.boxes.conf.cpu().numpy() if results.boxes is not None else np.zeros((0,))
    labels = results.boxes.cls.cpu().numpy().astype(int) if results.boxes is not None else np.zeros((0,), dtype=int)
    # xywhn to [x1, y1, x2, y2] normalized
    if len(boxes) > 0:
        x, y, w, h = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2
        # Clip to [0, 1] range
        x1 = np.clip(x1, 0, 1)
        y1 = np.clip(y1, 0, 1)
        x2 = np.clip(x2, 0, 1)
        y2 = np.clip(y2, 0, 1)
        boxes = np.stack([x1, y1, x2, y2], axis=1)
    return boxes, scores, labels

# Gather test images
image_paths = sorted(glob(os.path.join(test_images_dir, '*.jpg')) + glob(os.path.join(test_images_dir, '*.png')))

# Store all predictions and ground truths
all_pred_boxes = []
all_pred_scores = []
all_pred_labels = []
all_gt_boxes = []
all_gt_labels = []

for img_path in tqdm(image_paths, desc="Ensembling and evaluating"):
    # Inference for both models
    boxes_n, scores_n, labels_n = run_inference(model_n, img_path)
    boxes_s, scores_s, labels_s = run_inference(model_s, img_path)
    # Prepare for WBF: list of boxes, scores, labels for each model
    boxes_list = [boxes_n.tolist(), boxes_s.tolist()]
    scores_list = [scores_n.tolist(), scores_s.tolist()]
    labels_list = [labels_n.tolist(), labels_s.tolist()]
    # WBF
    boxes_wbf, scores_wbf, labels_wbf = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list, iou_thr=0.55, skip_box_thr=0.001
    )
    # Store ensemble predictions
    all_pred_boxes.append(boxes_wbf)
    all_pred_scores.append(scores_wbf)
    all_pred_labels.append(labels_wbf)
    # Load ground truth
    label_file = os.path.join(test_labels_dir, os.path.splitext(os.path.basename(img_path))[0] + '.txt')
    gt_boxes = []
    gt_labels = []
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, x, y, w, h = map(float, parts)
                    x1 = x - w/2
                    y1 = y - h/2
                    x2 = x + w/2
                    y2 = y + h/2
                    gt_boxes.append([x1, y1, x2, y2])
                    gt_labels.append(int(cls))
    all_gt_boxes.append(gt_boxes)
    all_gt_labels.append(gt_labels)

# Evaluation: Calculate mAP50, precision, recall manually
def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def evaluate_predictions(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5):
    """Evaluate predictions against ground truth"""
    if len(pred_boxes) == 0:
        return 0, 0, 0  # No predictions
    
    # Sort predictions by confidence score
    sorted_indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[sorted_indices]
    pred_scores = pred_scores[sorted_indices]
    pred_labels = pred_labels[sorted_indices]
    
    # Track which ground truth boxes have been matched
    gt_matched = [False] * len(gt_boxes)
    
    tp = 0  # True positives
    fp = 0  # False positives
    
    for i, (pred_box, pred_score, pred_label) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
        best_iou = 0
        best_gt_idx = -1
        
        # Find best matching ground truth box
        for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if gt_matched[j] or gt_label != pred_label:
                continue
            
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        # Check if prediction is correct
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp += 1
            gt_matched[best_gt_idx] = True
        else:
            fp += 1
    
    fn = sum(1 for matched in gt_matched if not matched)  # False negatives
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

# Evaluate ensemble predictions
total_precision = 0
total_recall = 0
total_f1 = 0
num_images = len(all_pred_boxes)

for i in range(num_images):
    pred_boxes = np.array(all_pred_boxes[i])
    pred_scores = np.array(all_pred_scores[i])
    pred_labels = np.array(all_pred_labels[i])
    gt_boxes = np.array(all_gt_boxes[i])
    gt_labels = np.array(all_gt_labels[i])
    
    precision, recall, f1 = evaluate_predictions(
        pred_boxes, pred_scores, pred_labels, 
        gt_boxes, gt_labels, iou_threshold=0.5
    )
    
    total_precision += precision
    total_recall += recall
    total_f1 += f1

# Calculate averages
avg_precision = total_precision / num_images
avg_recall = total_recall / num_images
avg_f1 = total_f1 / num_images

print("\nEnsemble Evaluation Results:")
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall:    {avg_recall:.4f}")
print(f"Average F1-Score:  {avg_f1:.4f}")

# Save results
with open("ensemble_evaluation_results.txt", "w") as f:
    f.write(f"Average Precision: {avg_precision:.4f}\n")
    f.write(f"Average Recall: {avg_recall:.4f}\n")
    f.write(f"Average F1-Score: {avg_f1:.4f}\n") 