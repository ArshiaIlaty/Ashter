#!/usr/bin/env python3
"""
Evaluate the hyperparameter-optimized model on test set
Compare with all previous models
"""

import os
import yaml
import json
from ultralytics import YOLO
from datetime import datetime

def load_class_names():
    """Load class names from data.yaml"""
    with open("dataset/data.yaml", 'r') as f:
        data = yaml.safe_load(f)
        return data['names'] if 'names' in data else ['pet_waste']

def evaluate_model(model_path, model_name, test_images_dir):
    """Evaluate a single model"""
    print(f"\n🔍 Evaluating {model_name}...")
    
    # Load model
    model = YOLO(model_path)
    
    # Run validation on test set
    results = model.val(
        data="dataset/data.yaml",
        split="test",
        imgsz=640,
        batch=8,
        conf=0.001,  # Low confidence to get all predictions
        iou=0.6,
        save_json=True,
        save_txt=True,
        verbose=False
    )
    
    # Extract metrics
    metrics = results.results_dict
    
    return {
        'model_name': model_name,
        'mAP50': metrics.get('metrics/mAP50(B)', 0),
        'mAP50-95': metrics.get('metrics/mAP50-95(B)', 0),
        'precision': metrics.get('metrics/precision(B)', 0),
        'recall': metrics.get('metrics/recall(B)', 0),
        'f1': metrics.get('metrics/f1(B)', 0),
        'model_path': model_path
    }

def main():
    """Main evaluation function"""
    print("🎯 EVALUATING HYPERPARAMETER-OPTIMIZED MODEL")
    print("=" * 60)
    
    # Define models to evaluate
    models = [
        {
            'path': 'hyperparameter_optimization/best_hyperparameters/weights/best.pt',
            'name': 'YOLOv8n Hyperparameter-Optimized'
        },
        {
            'path': 'runs/detect/yolov8n_finetune_old_dataset/weights/best.pt',
            'name': 'YOLOv8n Fine-tuned (Baseline)'
        },
        {
            'path': 'runs/detect/yolov8s_finetune_old_dataset/weights/best.pt',
            'name': 'YOLOv8s Fine-tuned'
        }
    ]
    
    # Test images directory
    test_images_dir = "dataset/test/images"
    
    # Evaluate all models
    results = []
    for model_info in models:
        if os.path.exists(model_info['path']):
            try:
                result = evaluate_model(
                    model_info['path'], 
                    model_info['name'], 
                    test_images_dir
                )
                results.append(result)
                print(f"✅ {model_info['name']}: mAP50 = {result['mAP50']:.4f}")
            except Exception as e:
                print(f"❌ Error evaluating {model_info['name']}: {e}")
        else:
            print(f"⚠️ Model not found: {model_info['path']}")
    
    # Create comparison table
    print(f"\n📊 MODEL COMPARISON RESULTS")
    print("=" * 80)
    print(f"{'Model':<35} {'mAP50':<8} {'mAP50-95':<10} {'Precision':<10} {'Recall':<8} {'F1':<8}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['model_name']:<35} {result['mAP50']:<8.4f} {result['mAP50-95']:<10.4f} "
              f"{result['precision']:<10.4f} {result['recall']:<8.4f} {result['f1']:<8.4f}")
    
    # Find best model
    best_model = max(results, key=lambda x: x['mAP50'])
    print(f"\n🏆 BEST MODEL: {best_model['model_name']}")
    print(f"   mAP50: {best_model['mAP50']:.4f}")
    print(f"   Precision: {best_model['precision']:.4f}")
    print(f"   Recall: {best_model['recall']:.4f}")
    print(f"   F1-Score: {best_model['f1']:.4f}")
    
    # Calculate improvements
    baseline = next((r for r in results if 'Baseline' in r['model_name']), None)
    if baseline and best_model != baseline:
        improvement = best_model['mAP50'] - baseline['mAP50']
        improvement_pct = (improvement / baseline['mAP50']) * 100
        print(f"\n📈 IMPROVEMENT OVER BASELINE:")
        print(f"   Absolute: {improvement:.4f}")
        print(f"   Relative: {improvement_pct:.2f}%")
    
    # Save results
    output_file = "optimized_model_evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'evaluation_date': datetime.now().isoformat(),
            'models': results,
            'best_model': best_model,
            'test_set_size': len(os.listdir(test_images_dir)) if os.path.exists(test_images_dir) else 0
        }, f, indent=2)
    
    print(f"\n✅ Evaluation completed!")
    print(f"📁 Results saved to: {output_file}")
    
    return results, best_model

if __name__ == "__main__":
    main() 