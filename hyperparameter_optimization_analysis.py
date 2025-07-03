#!/usr/bin/env python3
"""
Hyperparameter Optimization Analysis
Analyze the results of the hyperparameter optimization process
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def load_optimization_results():
    """Load optimization results from JSON file"""
    with open('hyperparameter_optimization/hyperparameter_optimization_results.json', 'r') as f:
        return json.load(f)

def analyze_optimization_performance():
    """Analyze the optimization performance"""
    results = load_optimization_results()
    
    print("🎯 HYPERPARAMETER OPTIMIZATION ANALYSIS")
    print("=" * 50)
    
    # Basic statistics
    trials = results['all_trials']
    scores = [trial['value'] for trial in trials]
    
    print(f"📊 Optimization Statistics:")
    print(f"   Total Trials: {len(trials)}")
    print(f"   Best mAP50: {results['optimization_info']['best_score']:.4f}")
    print(f"   Worst mAP50: {min(scores):.4f}")
    print(f"   Average mAP50: {np.mean(scores):.4f}")
    print(f"   Standard Deviation: {np.std(scores):.4f}")
    print(f"   Improvement from baseline: {results['optimization_info']['best_score'] - 0.774:.4f}")
    
    # Performance distribution
    print(f"\n📈 Performance Distribution:")
    print(f"   Trials with mAP50 > 0.80: {sum(1 for s in scores if s > 0.80)}")
    print(f"   Trials with mAP50 > 0.75: {sum(1 for s in scores if s > 0.75)}")
    print(f"   Trials with mAP50 < 0.70: {sum(1 for s in scores if s < 0.70)}")
    
    return results, scores

def analyze_best_hyperparameters():
    """Analyze the best hyperparameters found"""
    results = load_optimization_results()
    best_params = results['best_hyperparameters']
    
    print(f"\n🏆 BEST HYPERPARAMETERS (Trial {results['optimization_info']['best_trial_number']})")
    print("=" * 50)
    
    # Training parameters
    print("🎓 Training Parameters:")
    print(f"   Epochs: {best_params['epochs']}")
    print(f"   Batch Size: {best_params['batch']}")
    print(f"   Image Size: {best_params['imgsz']}")
    print(f"   Learning Rate (initial): {best_params['lr0']:.6f}")
    print(f"   Learning Rate (final): {best_params['lrf']:.6f}")
    print(f"   Momentum: {best_params['momentum']:.4f}")
    print(f"   Weight Decay: {best_params['weight_decay']:.6f}")
    print(f"   Warmup Epochs: {best_params['warmup_epochs']}")
    print(f"   Patience: {best_params['patience']}")
    
    # Loss weights
    print(f"\n⚖️ Loss Weights:")
    print(f"   Box Loss Weight: {best_params['box']:.4f}")
    print(f"   Classification Loss Weight: {best_params['cls']:.4f}")
    print(f"   DFL Loss Weight: {best_params['dfl']:.4f}")
    
    # Data augmentation
    print(f"\n🔄 Data Augmentation:")
    print(f"   HSV Hue: {best_params['hsv_h']:.4f}")
    print(f"   HSV Saturation: {best_params['hsv_s']:.4f}")
    print(f"   HSV Value: {best_params['hsv_v']:.4f}")
    print(f"   Rotation Degrees: {best_params['degrees']:.2f}")
    print(f"   Translation: {best_params['translate']:.4f}")
    print(f"   Scale: {best_params['scale']:.4f}")
    print(f"   Shear: {best_params['shear']:.2f}")
    print(f"   Perspective: {best_params['perspective']:.6f}")
    print(f"   Flip Up-Down: {best_params['flipud']:.4f}")
    print(f"   Mosaic: {best_params['mosaic']:.4f}")
    print(f"   Mixup: {best_params['mixup']:.4f}")
    print(f"   Copy-Paste: {best_params['copy_paste']:.4f}")

def analyze_parameter_importance():
    """Analyze parameter importance based on correlation with performance"""
    results = load_optimization_results()
    trials = results['all_trials']
    
    # Create DataFrame for analysis
    data = []
    for trial in trials:
        row = {'mAP50': trial['value']}
        row.update(trial['params'])
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Calculate correlations with mAP50
    correlations = df.corr()['mAP50'].abs().sort_values(ascending=False)
    
    print(f"\n🔍 PARAMETER IMPORTANCE (Correlation with mAP50)")
    print("=" * 50)
    
    # Top 10 most important parameters
    top_params = correlations[1:11]  # Exclude mAP50 itself
    for param, corr in top_params.items():
        print(f"   {param:20s}: {corr:.4f}")
    
    return df, correlations

def compare_with_baseline():
    """Compare optimized model with baseline"""
    print(f"\n📊 COMPARISON WITH BASELINE")
    print("=" * 50)
    
    baseline_map = 0.774  # Original fine-tuned YOLOv8n
    optimized_map = 0.8443
    
    improvement = optimized_map - baseline_map
    improvement_pct = (improvement / baseline_map) * 100
    
    print(f"   Baseline mAP50 (Fine-tuned YOLOv8n): {baseline_map:.4f}")
    print(f"   Optimized mAP50: {optimized_map:.4f}")
    print(f"   Absolute Improvement: {improvement:.4f}")
    print(f"   Relative Improvement: {improvement_pct:.2f}%")
    
    if improvement > 0.05:
        print(f"   🎉 Significant improvement achieved!")
    elif improvement > 0.02:
        print(f"   ✅ Moderate improvement achieved!")
    else:
        print(f"   ⚠️ Minimal improvement")

def generate_recommendations():
    """Generate recommendations based on optimization results"""
    print(f"\n💡 RECOMMENDATIONS")
    print("=" * 50)
    
    print("1. 🎯 Model Configuration:")
    print("   - Use larger image size (768x768) for better accuracy")
    print("   - Higher batch size (16) with proper learning rate")
    print("   - Longer training with patience=19 for convergence")
    
    print("\n2. 🔄 Data Augmentation Strategy:")
    print("   - High mosaic probability (0.89) for diverse training")
    print("   - Moderate mixup (0.18) for regularization")
    print("   - Balanced copy-paste (0.28) for object diversity")
    print("   - Significant geometric augmentations (rotation, shear)")
    
    print("\n3. ⚖️ Loss Balancing:")
    print("   - Higher box loss weight (5.61) for localization")
    print("   - Balanced classification weight (0.50)")
    print("   - Moderate DFL weight (1.03) for distribution learning")
    
    print("\n4. 🎓 Training Strategy:")
    print("   - Use AdamW optimizer with auto learning rate")
    print("   - Moderate weight decay (0.0007) for regularization")
    print("   - 3 warmup epochs for stable training start")

def main():
    """Main analysis function"""
    try:
        # Load and analyze results
        results, scores = analyze_optimization_performance()
        
        # Analyze best hyperparameters
        analyze_best_hyperparameters()
        
        # Analyze parameter importance
        df, correlations = analyze_parameter_importance()
        
        # Compare with baseline
        compare_with_baseline()
        
        # Generate recommendations
        generate_recommendations()
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📁 Results saved in: hyperparameter_optimization/")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")

if __name__ == "__main__":
    main() 