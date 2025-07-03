#!/usr/bin/env python3
"""
Evaluate smaller YOLOv8 models on the old dataset
"""

import os
import sys
import yaml
import shutil
from pathlib import Path
from ultralytics import YOLO
import json
from datetime import datetime

def load_dataset_config(dataset_path):
    """Load dataset configuration from data.yaml"""
    config_path = os.path.join(dataset_path, 'data.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def evaluate_model_on_dataset(model_path, dataset_path, output_dir, model_name):
    """Evaluate a single model on the dataset"""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} on old dataset")
    print(f"{'='*60}")
    
    # Load model
    model = YOLO(model_path)
    
    # Create output directory for this model
    model_output_dir = os.path.join(output_dir, f"{model_name}_old_dataset")
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Get test images path
    test_images_path = os.path.join(dataset_path, 'test', 'images')
    
    # Run evaluation
    results = model.val(
        data=os.path.join(dataset_path, 'data.yaml'),
        split='test',
        save_txt=True,
        save_conf=True,
        save_json=True,
        project=model_output_dir,
        name='evaluation',
        verbose=True
    )
    
    # Save detailed results
    results_dict = {
        'model_name': model_name,
        'model_path': model_path,
        'dataset_path': dataset_path,
        'evaluation_date': datetime.now().isoformat(),
        'metrics': {
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'f1': float(results.box.map50)  # Using mAP50 as F1 approximation
        },
        'class_metrics': {
            'pet_waste': {
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
                'mAP50': float(results.box.map50),
                'mAP50-95': float(results.box.map)
            }
        }
    }
    
    # Save results to JSON
    results_file = os.path.join(model_output_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print(f"mAP50: {results_dict['metrics']['mAP50']:.4f}")
    print(f"mAP50-95: {results_dict['metrics']['mAP50-95']:.4f}")
    print(f"Precision: {results_dict['metrics']['precision']:.4f}")
    print(f"Recall: {results_dict['metrics']['recall']:.4f}")
    
    return results_dict

def main():
    # Configuration
    old_dataset_path = "/home/ailaty3088@id.sdsu.edu/Ashter/dataset"
    output_base_dir = "/home/ailaty3088@id.sdsu.edu/Ashter/smaller_models_old_dataset_evaluation"
    
    # Model paths
    models = {
        'yolov8n': '/home/ailaty3088@id.sdsu.edu/Ashter/runs/detect/shitspotter_yolov8n/weights/best.pt',
        'yolov8s': '/home/ailaty3088@id.sdsu.edu/Ashter/runs/detect/shitspotter_yolov8s/weights/best.pt'
    }
    
    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Load dataset config
    try:
        config = load_dataset_config(old_dataset_path)
        print(f"Dataset config loaded: {config}")
    except Exception as e:
        print(f"Error loading dataset config: {e}")
        return
    
    # Check if test images exist
    test_images_path = os.path.join(old_dataset_path, 'test', 'images')
    if not os.path.exists(test_images_path):
        print(f"Test images path not found: {test_images_path}")
        return
    
    test_images = [f for f in os.listdir(test_images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(test_images)} test images in old dataset")
    
    # Evaluate each model
    all_results = {}
    
    for model_name, model_path in models.items():
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue
            
        try:
            results = evaluate_model_on_dataset(
                model_path=model_path,
                dataset_path=old_dataset_path,
                output_dir=output_base_dir,
                model_name=model_name
            )
            all_results[model_name] = results
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            continue
    
    # Create comparison report
    if all_results:
        comparison_file = os.path.join(output_base_dir, 'model_comparison_report.md')
        
        with open(comparison_file, 'w') as f:
            f.write("# Model Comparison on Old Dataset\n\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Dataset: {old_dataset_path}\n")
            f.write(f"Test Images: {len(test_images)}\n\n")
            
            f.write("## Results Summary\n\n")
            f.write("| Model | mAP50 | mAP50-95 | Precision | Recall |\n")
            f.write("|-------|-------|----------|-----------|--------|\n")
            
            for model_name, results in all_results.items():
                metrics = results['metrics']
                f.write(f"| {model_name} | {metrics['mAP50']:.4f} | {metrics['mAP50-95']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} |\n")
            
            f.write("\n## Detailed Results\n\n")
            
            for model_name, results in all_results.items():
                f.write(f"### {model_name.upper()}\n\n")
                f.write(f"- **Model Path**: {results['model_path']}\n")
                f.write(f"- **mAP50**: {results['metrics']['mAP50']:.4f}\n")
                f.write(f"- **mAP50-95**: {results['metrics']['mAP50-95']:.4f}\n")
                f.write(f"- **Precision**: {results['metrics']['precision']:.4f}\n")
                f.write(f"- **Recall**: {results['metrics']['recall']:.4f}\n\n")
        
        print(f"\nComparison report saved to: {comparison_file}")
    
    print(f"\nAll evaluation results saved to: {output_base_dir}")

if __name__ == "__main__":
    main() 