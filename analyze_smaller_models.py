#!/usr/bin/env python3
"""
Comprehensive Analysis of Smaller YOLO Models vs Original YOLOv8m
Analyzes training performance, evaluation metrics, and provides recommendations
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_training_results(model_path):
    """Load training results from CSV file"""
    try:
        df = pd.read_csv(model_path)
        return df
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None

def analyze_training_metrics(df, model_name):
    """Analyze training metrics for a model"""
    if df is None or df.empty:
        return None
    
    # Final epoch metrics
    final_metrics = {
        'model': model_name,
        'final_precision': df['metrics/precision(B)'].iloc[-1],
        'final_recall': df['metrics/recall(B)'].iloc[-1],
        'final_map50': df['metrics/mAP50(B)'].iloc[-1],
        'final_map50_95': df['metrics/mAP50-95(B)'].iloc[-1],
        'final_train_loss': df['train/box_loss'].iloc[-1] + df['train/cls_loss'].iloc[-1] + df['train/dfl_loss'].iloc[-1],
        'final_val_loss': df['val/box_loss'].iloc[-1] + df['val/cls_loss'].iloc[-1] + df['val/dfl_loss'].iloc[-1],
        'total_time_minutes': df['time'].iloc[-1] / 60,
        'convergence_epoch': None
    }
    
    # Find convergence epoch (when mAP50 stops improving significantly)
    map50_values = df['metrics/mAP50(B)'].values
    best_map50 = np.max(map50_values)
    convergence_threshold = best_map50 * 0.95  # 95% of best performance
    
    for i, map50 in enumerate(map50_values):
        if map50 >= convergence_threshold:
            final_metrics['convergence_epoch'] = i + 1
            break
    
    return final_metrics

def create_training_plots(models_data, output_dir):
    """Create comprehensive training plots"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Training Performance Comparison: YOLOv8n vs YOLOv8s vs YOLOv8m', fontsize=16, fontweight='bold')
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    model_names = list(models_data.keys())
    
    # Plot 1: mAP50 over epochs
    ax = axes[0, 0]
    for i, (model_name, df) in enumerate(models_data.items()):
        if df is not None:
            ax.plot(df['metrics/mAP50(B)'], label=model_name, color=colors[i], linewidth=2)
    ax.set_title('mAP50 Progress')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP50')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Precision over epochs
    ax = axes[0, 1]
    for i, (model_name, df) in enumerate(models_data.items()):
        if df is not None:
            ax.plot(df['metrics/precision(B)'], label=model_name, color=colors[i], linewidth=2)
    ax.set_title('Precision Progress')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Precision')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Recall over epochs
    ax = axes[0, 2]
    for i, (model_name, df) in enumerate(models_data.items()):
        if df is not None:
            ax.plot(df['metrics/recall(B)'], label=model_name, color=colors[i], linewidth=2)
    ax.set_title('Recall Progress')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Recall')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Training Loss
    ax = axes[1, 0]
    for i, (model_name, df) in enumerate(models_data.items()):
        if df is not None:
            total_train_loss = df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']
            ax.plot(total_train_loss, label=model_name, color=colors[i], linewidth=2)
    ax.set_title('Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Validation Loss
    ax = axes[1, 1]
    for i, (model_name, df) in enumerate(models_data.items()):
        if df is not None:
            total_val_loss = df['val/box_loss'] + df['val/cls_loss'] + df['val/dfl_loss']
            ax.plot(total_val_loss, label=model_name, color=colors[i], linewidth=2)
    ax.set_title('Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: mAP50-95 over epochs
    ax = axes[1, 2]
    for i, (model_name, df) in enumerate(models_data.items()):
        if df is not None:
            ax.plot(df['metrics/mAP50-95(B)'], label=model_name, color=colors[i], linewidth=2)
    ax.set_title('mAP50-95 Progress')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP50-95')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_summary(metrics_data, output_dir):
    """Create performance summary table and visualization"""
    if not metrics_data:
        return
    
    # Create summary DataFrame
    summary_data = []
    for model_name, metrics in metrics_data.items():
        if metrics:
            summary_data.append(metrics)
    
    if not summary_data:
        return
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Summary', fontsize=16, fontweight='bold')
    
    # Plot 1: Final Metrics Comparison
    ax = axes[0, 0]
    metrics_to_plot = ['final_precision', 'final_recall', 'final_map50', 'final_map50_95']
    x = np.arange(len(summary_df))
    width = 0.2
    
    for i, metric in enumerate(metrics_to_plot):
        values = summary_df[metric].values
        ax.bar(x + i*width, values, width, label=metric.replace('final_', '').replace('_', '-').upper())
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Final Training Metrics')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(summary_df['model'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Training Time Comparison
    ax = axes[0, 1]
    ax.bar(summary_df['model'], summary_df['total_time_minutes'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax.set_title('Training Time Comparison')
    ax.set_xlabel('Models')
    ax.set_ylabel('Time (minutes)')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Loss Comparison
    ax = axes[1, 0]
    x = np.arange(len(summary_df))
    width = 0.35
    
    ax.bar(x - width/2, summary_df['final_train_loss'], width, label='Training Loss', alpha=0.8)
    ax.bar(x + width/2, summary_df['final_val_loss'], width, label='Validation Loss', alpha=0.8)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Loss')
    ax.set_title('Final Loss Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df['model'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Convergence Analysis
    ax = axes[1, 1]
    ax.bar(summary_df['model'], summary_df['convergence_epoch'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax.set_title('Convergence Epoch')
    ax.set_xlabel('Models')
    ax.set_ylabel('Epoch')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save summary table
    summary_df.to_csv(f'{output_dir}/model_comparison_summary.csv', index=False)
    
    return summary_df

def generate_recommendations(summary_df, output_dir):
    """Generate recommendations based on analysis"""
    if summary_df is None or summary_df.empty:
        return
    
    recommendations = []
    
    # Find best performing model for each metric
    best_map50 = summary_df.loc[summary_df['final_map50'].idxmax(), 'model']
    best_precision = summary_df.loc[summary_df['final_precision'].idxmax(), 'model']
    best_recall = summary_df.loc[summary_df['final_recall'].idxmax(), 'model']
    fastest_training = summary_df.loc[summary_df['total_time_minutes'].idxmin(), 'model']
    
    recommendations.append(f"📊 **Best mAP50**: {best_map50} ({summary_df['final_map50'].max():.4f})")
    recommendations.append(f"🎯 **Best Precision**: {best_precision} ({summary_df['final_precision'].max():.4f})")
    recommendations.append(f"🔍 **Best Recall**: {best_recall} ({summary_df['final_recall'].max():.4f})")
    recommendations.append(f"⚡ **Fastest Training**: {fastest_training} ({summary_df['total_time_minutes'].min():.1f} minutes)")
    
    # Overall recommendations
    recommendations.append("\n💡 **Overall Recommendations:**")
    
    if summary_df['final_map50'].max() > 0.5:
        recommendations.append("✅ Models show good training performance with mAP50 > 0.5")
    else:
        recommendations.append("⚠️ Training performance is moderate, consider data augmentation or longer training")
    
    # Check for overfitting
    train_val_gaps = summary_df['final_train_loss'] - summary_df['final_val_loss']
    if any(abs(gap) > 0.5 for gap in train_val_gaps):
        recommendations.append("⚠️ Some models show signs of overfitting (large train-val loss gap)")
    
    # Efficiency recommendations
    if summary_df['total_time_minutes'].min() < 60:
        recommendations.append("✅ Training time is reasonable for all models")
    
    # Save recommendations
    with open(f'{output_dir}/recommendations.md', 'w') as f:
        f.write("# Model Training Analysis Recommendations\n\n")
        for rec in recommendations:
            f.write(f"{rec}\n")
    
    return recommendations

def main():
    """Main analysis function"""
    print("🔍 Analyzing smaller YOLO models training results...")
    
    # Define model paths
    model_paths = {
        'YOLOv8n': 'runs/detect/shitspotter_yolov8n/results.csv',
        'YOLOv8s': 'runs/detect/shitspotter_yolov8s/results.csv',
        'YOLOv8m': 'runs/detect/shitspotter_finetune/results.csv'
    }
    
    # Create output directory
    output_dir = 'smaller_models_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load training data
    models_data = {}
    metrics_data = {}
    
    for model_name, path in model_paths.items():
        print(f"📊 Loading {model_name} training data...")
        df = load_training_results(path)
        models_data[model_name] = df
        metrics_data[model_name] = analyze_training_metrics(df, model_name)
    
    # Create visualizations
    print("📈 Creating training comparison plots...")
    create_training_plots(models_data, output_dir)
    
    print("📊 Creating performance summary...")
    summary_df = create_performance_summary(metrics_data, output_dir)
    
    print("💡 Generating recommendations...")
    recommendations = generate_recommendations(summary_df, output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("📋 SMALLER MODELS TRAINING ANALYSIS SUMMARY")
    print("="*80)
    
    if summary_df is not None:
        print("\n📊 Final Training Metrics:")
        print(summary_df[['model', 'final_precision', 'final_recall', 'final_map50', 'final_map50_95']].to_string(index=False))
        
        print("\n⏱️ Training Time:")
        print(summary_df[['model', 'total_time_minutes', 'convergence_epoch']].to_string(index=False))
        
        print("\n💡 Key Recommendations:")
        for rec in recommendations:
            print(rec)
    
    print(f"\n✅ Analysis complete! Results saved to '{output_dir}' directory")
    print("📁 Files generated:")
    print("   - training_comparison.png: Training progress comparison")
    print("   - performance_summary.png: Final metrics comparison")
    print("   - model_comparison_summary.csv: Detailed metrics table")
    print("   - recommendations.md: Analysis recommendations")

if __name__ == "__main__":
    main() 