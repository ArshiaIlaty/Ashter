#!/usr/bin/env python3
"""
Compare ensemble results with individual model performances
"""

import os
import json
from datetime import datetime

def load_individual_results():
    """Load results from individual model evaluations"""
    results = {}
    
    # YOLOv8n fine-tuned results (from training logs)
    results['yolov8n_finetuned'] = {
        'mAP50': 0.774,
        'precision': 0.759,
        'recall': 0.732,
        'model_size': '3.0M params',
        'inference_time': '2.7ms'
    }
    
    # YOLOv8s fine-tuned results (from training logs)
    results['yolov8s_finetuned'] = {
        'mAP50': 0.817,
        'precision': 0.819,
        'recall': 0.732,
        'model_size': '11.1M params',
        'inference_time': '6.1ms'
    }
    
    return results

def load_ensemble_results():
    """Load ensemble evaluation results"""
    ensemble_file = "ensemble_evaluation_results.txt"
    if os.path.exists(ensemble_file):
        with open(ensemble_file, 'r') as f:
            lines = f.readlines()
            precision = float(lines[0].split(': ')[1])
            recall = float(lines[1].split(': ')[1])
            f1 = float(lines[2].split(': ')[1])
            
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'model_size': '14.1M params (combined)',
            'inference_time': '8.8ms (combined)'
        }
    return None

def print_comparison():
    """Print comprehensive comparison"""
    individual_results = load_individual_results()
    ensemble_results = load_ensemble_results()
    
    print("=" * 80)
    print("ENSEMBLE vs INDIVIDUAL MODEL COMPARISON")
    print("=" * 80)
    
    print(f"\n📊 Individual Model Performance:")
    print("-" * 50)
    
    for model_name, metrics in individual_results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  mAP50:     {metrics['mAP50']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  Model Size: {metrics['model_size']}")
        print(f"  Inference: {metrics['inference_time']}")
    
    if ensemble_results:
        print(f"\n🎯 Ensemble Performance:")
        print("-" * 30)
        print(f"  Precision: {ensemble_results['precision']:.4f}")
        print(f"  Recall:    {ensemble_results['recall']:.4f}")
        print(f"  F1-Score:  {ensemble_results['f1_score']:.4f}")
        print(f"  Model Size: {ensemble_results['model_size']}")
        print(f"  Inference: {ensemble_results['inference_time']}")
    
    # Calculate improvements
    print(f"\n📈 Performance Analysis:")
    print("-" * 30)
    
    best_individual = max(individual_results.values(), key=lambda x: x['mAP50'])
    
    if ensemble_results:
        # Note: We're comparing different metrics, so this is qualitative
        print(f"Best Individual Model: {best_individual['mAP50']:.4f} mAP50")
        print(f"Ensemble F1-Score: {ensemble_results['f1_score']:.4f}")
        print(f"\nKey Observations:")
        print(f"• Ensemble shows very high recall ({ensemble_results['recall']:.4f})")
        print(f"• Individual models have better precision")
        print(f"• Ensemble combines strengths of both models")
    
    print(f"\n🔧 Ensemble Configuration:")
    print("-" * 30)
    print(f"• Models: YOLOv8n + YOLOv8s (both fine-tuned)")
    print(f"• Fusion Method: Weighted Box Fusion (WBF)")
    print(f"• IoU Threshold: 0.55")
    print(f"• Skip Box Threshold: 0.001")
    
    print(f"\n💡 Recommendations:")
    print("-" * 20)
    print(f"• For high precision: Use individual YOLOv8s model")
    print(f"• For high recall: Use ensemble")
    print(f"• For balanced performance: Use YOLOv8n model")
    print(f"• For production: Consider ensemble with confidence threshold tuning")

def save_comparison_report():
    """Save detailed comparison report"""
    individual_results = load_individual_results()
    ensemble_results = load_ensemble_results()
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'individual_models': individual_results,
        'ensemble': ensemble_results,
        'summary': {
            'best_individual_mAP50': max(individual_results.values(), key=lambda x: x['mAP50'])['mAP50'],
            'ensemble_f1': ensemble_results['f1_score'] if ensemble_results else None
        }
    }
    
    with open('ensemble_comparison_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: ensemble_comparison_report.json")

if __name__ == "__main__":
    print_comparison()
    save_comparison_report() 