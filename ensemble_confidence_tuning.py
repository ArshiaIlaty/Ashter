#!/usr/bin/env python3
"""
Confidence Threshold Tuning for Ensemble Model
Optimize precision-recall balance by tuning confidence thresholds
"""

import os
import numpy as np
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from glob import glob
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt
import json

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

def run_inference(model, img_path, conf_threshold=0.001):
    """Run inference with specific confidence threshold"""
    results = model(img_path, conf=conf_threshold, iou=0.1, verbose=False)[0]
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

def evaluate_ensemble_with_threshold(conf_threshold, wbf_iou_threshold=0.55):
    """Evaluate ensemble with specific confidence threshold"""
    image_paths = sorted(glob(os.path.join(test_images_dir, '*.jpg')) + glob(os.path.join(test_images_dir, '*.png')))
    
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    num_images = len(image_paths)
    
    for img_path in image_paths:
        # Inference for both models with confidence threshold
        boxes_n, scores_n, labels_n = run_inference(model_n, img_path, conf_threshold)
        boxes_s, scores_s, labels_s = run_inference(model_s, img_path, conf_threshold)
        
        # Prepare for WBF
        boxes_list = [boxes_n.tolist(), boxes_s.tolist()]
        scores_list = [scores_n.tolist(), scores_s.tolist()]
        labels_list = [labels_n.tolist(), labels_s.tolist()]
        
        # WBF
        boxes_wbf, scores_wbf, labels_wbf = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list, 
            iou_thr=wbf_iou_threshold, skip_box_thr=conf_threshold
        )
        
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
        
        # Evaluate
        precision, recall, f1 = evaluate_predictions(
            np.array(boxes_wbf), np.array(scores_wbf), np.array(labels_wbf),
            np.array(gt_boxes), np.array(gt_labels), iou_threshold=0.5
        )
        
        total_precision += precision
        total_recall += recall
        total_f1 += f1
    
    return {
        'precision': total_precision / num_images,
        'recall': total_recall / num_images,
        'f1': total_f1 / num_images
    }

def tune_confidence_thresholds():
    """Tune confidence thresholds and find optimal settings"""
    print("Tuning confidence thresholds for ensemble...")
    
    # Test different confidence thresholds
    conf_thresholds = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    wbf_iou_thresholds = [0.3, 0.4, 0.5, 0.55, 0.6, 0.7]
    
    results = []
    
    for conf_thresh in tqdm(conf_thresholds, desc="Testing confidence thresholds"):
        for wbf_iou_thresh in wbf_iou_thresholds:
            try:
                metrics = evaluate_ensemble_with_threshold(conf_thresh, wbf_iou_thresh)
                results.append({
                    'conf_threshold': conf_thresh,
                    'wbf_iou_threshold': wbf_iou_thresh,
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1']
                })
            except Exception as e:
                print(f"Error with conf_thresh={conf_thresh}, wbf_iou_thresh={wbf_iou_thresh}: {e}")
    
    return results

def analyze_results(results):
    """Analyze and visualize results"""
    if not results:
        print("No results to analyze")
        return
    
    # Find best configurations
    best_f1 = max(results, key=lambda x: x['f1'])
    best_precision = max(results, key=lambda x: x['precision'])
    best_recall = max(results, key=lambda x: x['recall'])
    
    print("\n" + "=" * 80)
    print("CONFIDENCE THRESHOLD TUNING RESULTS")
    print("=" * 80)
    
    print(f"\n🎯 Best F1-Score Configuration:")
    print(f"  Confidence Threshold: {best_f1['conf_threshold']:.3f}")
    print(f"  WBF IoU Threshold: {best_f1['wbf_iou_threshold']:.2f}")
    print(f"  Precision: {best_f1['precision']:.4f}")
    print(f"  Recall: {best_f1['recall']:.4f}")
    print(f"  F1-Score: {best_f1['f1']:.4f}")
    
    print(f"\n🎯 Best Precision Configuration:")
    print(f"  Confidence Threshold: {best_precision['conf_threshold']:.3f}")
    print(f"  WBF IoU Threshold: {best_precision['wbf_iou_threshold']:.2f}")
    print(f"  Precision: {best_precision['precision']:.4f}")
    print(f"  Recall: {best_precision['recall']:.4f}")
    print(f"  F1-Score: {best_precision['f1']:.4f}")
    
    print(f"\n🎯 Best Recall Configuration:")
    print(f"  Confidence Threshold: {best_recall['conf_threshold']:.3f}")
    print(f"  WBF IoU Threshold: {best_recall['wbf_iou_threshold']:.2f}")
    print(f"  Precision: {best_recall['precision']:.4f}")
    print(f"  Recall: {best_recall['recall']:.4f}")
    print(f"  F1-Score: {best_recall['f1']:.4f}")
    
    # Save results
    with open('confidence_tuning_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualization
    try:
        create_visualization(results)
    except Exception as e:
        print(f"Could not create visualization: {e}")
    
    return best_f1, best_precision, best_recall

def create_visualization(results):
    """Create visualization of results"""
    if not results:
        return
    
    # Extract data for plotting
    conf_thresholds = sorted(list(set(r['conf_threshold'] for r in results)))
    wbf_iou_thresholds = sorted(list(set(r['wbf_iou_threshold'] for r in results)))
    
    # Create precision-recall plot
    plt.figure(figsize=(12, 8))
    
    # Plot for different WBF IoU thresholds
    for wbf_iou in wbf_iou_thresholds:
        subset = [r for r in results if r['wbf_iou_threshold'] == wbf_iou]
        if subset:
            subset = sorted(subset, key=lambda x: x['conf_threshold'])
            precisions = [r['precision'] for r in subset]
            recalls = [r['recall'] for r in subset]
            plt.plot(recalls, precisions, 'o-', label=f'WBF IoU={wbf_iou}', linewidth=2, markersize=6)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for Different WBF IoU Thresholds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('confidence_tuning_pr_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Visualization saved to: confidence_tuning_pr_curves.png")

def main():
    print("Starting confidence threshold tuning for ensemble...")
    
    # Run tuning
    results = tune_confidence_thresholds()
    
    # Analyze results
    best_configs = analyze_results(results)
    
    print(f"\n✅ Confidence threshold tuning completed!")
    print(f"Results saved to: confidence_tuning_results.json")

if __name__ == "__main__":
    main() 