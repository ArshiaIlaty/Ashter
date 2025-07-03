#!/usr/bin/env python3
"""
Evaluate smaller YOLO models (YOLOv8n and YOLOv8s) on the test dataset
and compare with YOLOv8m results
"""

import os
import json
import time
from datetime import datetime
from ultralytics import YOLO
import numpy as np
from pathlib import Path

def evaluate_model_on_test(model_path, model_name, test_dataset_path, output_dir):
    """Evaluate a model on the test dataset"""
    print(f"\n{'='*80}")
    print(f"EVALUATING {model_name.upper()} ON TEST DATASET")
    print(f"{'='*80}")
    
    # Load model
    print(f"📦 Loading {model_name} model...")
    model = YOLO(model_path)
    
    # Test dataset path
    test_images_path = os.path.join(test_dataset_path, 'images')
    test_labels_path = os.path.join(test_dataset_path, 'labels')
    
    if not os.path.exists(test_images_path):
        print(f"❌ Test images path not found: {test_images_path}")
        return None
    
    # Create output directory
    model_output_dir = os.path.join(output_dir, f"{model_name}_test_evaluation")
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Run evaluation
    print(f"🔍 Running evaluation on test dataset...")
    print(f"📁 Test images: {test_images_path}")
    print(f"📁 Output: {model_output_dir}")
    
    start_time = time.time()
    
    # Run validation using the dataset config file
    results = model.val(
        data="shitspotter_dataset/data.yaml",
        split='test',  # Use test split
        save_txt=True,
        save_conf=True,
        save_json=True,
        project=output_dir,
        name=f"{model_name}_test_evaluation",
        verbose=True
    )
    
    evaluation_time = time.time() - start_time
    
    # Extract metrics
    metrics = {
        'model': model_name,
        'precision': float(results.results_dict.get('metrics/precision(B)', 0)),
        'recall': float(results.results_dict.get('metrics/recall(B)', 0)),
        'map50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
        'map50_95': float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
        'evaluation_time': evaluation_time
    }
    
    print(f"✅ Evaluation completed in {evaluation_time:.2f} seconds")
    print(f"📊 Results for {model_name}:")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall: {metrics['recall']:.4f}")
    print(f"   mAP50: {metrics['map50']:.4f}")
    print(f"   mAP50-95: {metrics['map50_95']:.4f}")
    
    return metrics

def run_custom_evaluation(model_path, model_name, test_dataset_path, output_dir):
    """Run custom evaluation similar to previous analysis"""
    print(f"\n🧪 Running custom evaluation for {model_name}...")
    
    # Load model
    model = YOLO(model_path)
    
    # Test images path
    test_images_path = os.path.join(test_dataset_path, 'images')
    
    if not os.path.exists(test_images_path):
        print(f"❌ Test images path not found: {test_images_path}")
        return None
    
    # Get all test images
    image_files = [f for f in os.listdir(test_images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"📸 Found {len(image_files)} test images")
    
    # Run predictions
    results = model.predict(
        source=test_images_path,
        save=True,
        save_txt=True,
        save_conf=True,
        project=output_dir,
        name=f"{model_name}_custom_test",
        verbose=False
    )
    
    print(f"✅ Custom evaluation completed for {model_name}")
    return True

def compare_with_previous_results(new_metrics, previous_results_file):
    """Compare new results with previous YOLOv8m results"""
    if not os.path.exists(previous_results_file):
        print(f"⚠️ Previous results file not found: {previous_results_file}")
        return
    
    try:
        with open(previous_results_file, 'r') as f:
            previous_data = json.load(f)
        
        # Extract previous metrics (assuming it's the YOLOv8m results)
        previous_metrics = {
            'model': 'YOLOv8m',
            'precision': 0.0755,  # From the summary you provided
            'recall': 0.0921,
            'f1_score': 0.0830
        }
        
        print(f"\n{'='*80}")
        print("COMPARISON WITH PREVIOUS RESULTS")
        print(f"{'='*80}")
        
        print(f"{'Model':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'mAP50':<12}")
        print("-" * 70)
        
        # Print previous results
        print(f"{previous_metrics['model']:<15} {previous_metrics['precision']:<12.4f} {previous_metrics['recall']:<12.4f} {previous_metrics['f1_score']:<12.4f} {'N/A':<12}")
        
        # Print new results
        for metric in new_metrics:
            if metric:
                f1_score = 2 * (metric['precision'] * metric['recall']) / (metric['precision'] + metric['recall']) if (metric['precision'] + metric['recall']) > 0 else 0
                print(f"{metric['model']:<15} {metric['precision']:<12.4f} {metric['recall']:<12.4f} {f1_score:<12.4f} {metric['map50']:<12.4f}")
        
    except Exception as e:
        print(f"❌ Error reading previous results: {e}")

def main():
    """Main evaluation function"""
    print("🚀 Starting evaluation of smaller models on test dataset...")
    
    # Configuration
    test_dataset_path = "shitspotter_dataset/test"
    output_dir = "smaller_models_test_evaluation"
    
    # Model paths
    models = {
        'yolov8n': 'runs/detect/shitspotter_yolov8n/weights/best.pt',
        'yolov8s': 'runs/detect/shitspotter_yolov8s/weights/best.pt'
    }
    
    # Previous results file
    previous_results_file = "shitspotter_evaluation_results/metrics/shitspotter_evaluation_results_20250630_164027.json"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate each model
    all_metrics = []
    
    for model_name, model_path in models.items():
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            continue
        
        # Run standard evaluation
        metrics = evaluate_model_on_test(model_path, model_name, test_dataset_path, output_dir)
        if metrics:
            all_metrics.append(metrics)
        
        # Run custom evaluation
        run_custom_evaluation(model_path, model_name, test_dataset_path, output_dir)
    
    # Compare with previous results
    compare_with_previous_results(all_metrics, previous_results_file)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"test_evaluation_results_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'models_evaluated': list(models.keys()),
            'metrics': all_metrics
        }, f, indent=2)
    
    print(f"\n✅ Test evaluation complete!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📄 Detailed results: {results_file}")

if __name__ == "__main__":
    main() 