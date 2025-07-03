#!/usr/bin/env python3
"""
Comprehensive analysis of the fine-tuning results for the pet waste detection model.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_training_results(csv_path):
    """Load and clean training results from CSV."""
    df = pd.read_csv(csv_path)
    
    # Clean infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df

def plot_training_curves(df, save_path="training_analysis.png"):
    """Plot comprehensive training curves."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Pet Waste Detection Model Training Analysis', fontsize=16, fontweight='bold')
    
    # Loss curves
    axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', linewidth=2)
    axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', linewidth=2)
    axes[0, 0].set_title('Box Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(df['epoch'], df['train/cls_loss'], label='Train Class Loss', linewidth=2)
    axes[0, 1].plot(df['epoch'], df['val/cls_loss'], label='Val Class Loss', linewidth=2)
    axes[0, 1].set_title('Classification Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(df['epoch'], df['train/dfl_loss'], label='Train DFL Loss', linewidth=2)
    axes[0, 2].plot(df['epoch'], df['val/dfl_loss'], label='Val DFL Loss', linewidth=2)
    axes[0, 2].set_title('DFL Loss')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Metrics curves
    axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision', linewidth=2, color='green')
    axes[1, 0].set_title('Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(df['epoch'], df['metrics/recall(B)'], label='Recall', linewidth=2, color='orange')
    axes[1, 1].set_title('Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', linewidth=2, color='red')
    axes[1, 2].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', linewidth=2, color='purple')
    axes[1, 2].set_title('mAP Metrics')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('mAP')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def analyze_training_performance(df):
    """Analyze training performance and provide insights."""
    print("=" * 60)
    print("PET WASTE DETECTION MODEL TRAINING ANALYSIS")
    print("=" * 60)
    
    # Training duration
    total_time = df['time'].iloc[-1]
    print(f"\n📊 TRAINING SUMMARY:")
    print(f"   • Total training time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"   • Number of epochs: {len(df)}")
    print(f"   • Average time per epoch: {total_time/len(df):.1f} seconds")
    
    # Best performance metrics
    best_map50 = df['metrics/mAP50(B)'].max()
    best_map50_epoch = df.loc[df['metrics/mAP50(B)'].idxmax(), 'epoch']
    best_map50_95 = df['metrics/mAP50-95(B)'].max()
    best_map50_95_epoch = df.loc[df['metrics/mAP50-95(B)'].idxmax(), 'epoch']
    
    print(f"\n🏆 BEST PERFORMANCE:")
    print(f"   • Best mAP@0.5: {best_map50:.4f} (Epoch {best_map50_epoch:.0f})")
    print(f"   • Best mAP@0.5:0.95: {best_map50_95:.4f} (Epoch {best_map50_95_epoch:.0f})")
    
    # Final performance
    final_map50 = df['metrics/mAP50(B)'].iloc[-1]
    final_map50_95 = df['metrics/mAP50-95(B)'].iloc[-1]
    final_precision = df['metrics/precision(B)'].iloc[-1]
    final_recall = df['metrics/recall(B)'].iloc[-1]
    
    print(f"\n📈 FINAL PERFORMANCE:")
    print(f"   • Final mAP@0.5: {final_map50:.4f}")
    print(f"   • Final mAP@0.5:0.95: {final_map50_95:.4f}")
    print(f"   • Final Precision: {final_precision:.4f}")
    print(f"   • Final Recall: {final_recall:.4f}")
    
    # Loss analysis
    final_train_loss = df['train/box_loss'].iloc[-1] + df['train/cls_loss'].iloc[-1] + df['train/dfl_loss'].iloc[-1]
    final_val_loss = df['val/box_loss'].iloc[-1] + df['val/cls_loss'].iloc[-1] + df['val/dfl_loss'].iloc[-1]
    
    print(f"\n📉 LOSS ANALYSIS:")
    print(f"   • Final Training Loss: {final_train_loss:.4f}")
    print(f"   • Final Validation Loss: {final_val_loss:.4f}")
    print(f"   • Overfitting Check: {'⚠️ Potential overfitting' if final_val_loss > final_train_loss * 1.5 else '✅ Good generalization'}")
    
    # Convergence analysis
    last_10_map50 = df['metrics/mAP50(B)'].tail(10)
    map50_std = last_10_map50.std()
    map50_trend = (last_10_map50.iloc[-1] - last_10_map50.iloc[0]) / len(last_10_map50)
    
    print(f"\n🔄 CONVERGENCE ANALYSIS:")
    print(f"   • Last 10 epochs mAP@0.5 std: {map50_std:.4f}")
    print(f"   • mAP@0.5 trend (last 10 epochs): {'↗️ Improving' if map50_trend > 0.001 else '→ Stable' if abs(map50_trend) <= 0.001 else '↘️ Declining'}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if final_map50 < 0.5:
        print("   • Model performance is below 50% mAP@0.5 - consider:")
        print("     - Increasing training epochs")
        print("     - Adjusting learning rate")
        print("     - Data augmentation improvements")
        print("     - Model architecture changes")
    elif final_map50 < 0.7:
        print("   • Model performance is moderate - consider:")
        print("     - Fine-tuning hyperparameters")
        print("     - Additional data collection")
        print("     - Ensemble methods")
    else:
        print("   • Model performance is good! Consider:")
        print("     - Model deployment")
        print("     - Real-world testing")
        print("     - Performance optimization")
    
    if final_val_loss > final_train_loss * 1.5:
        print("   • Address potential overfitting with:")
        print("     - Early stopping")
        print("     - Regularization techniques")
        print("     - More validation data")

def analyze_evaluation_results(summary_path):
    """Analyze evaluation results on the test set."""
    if not Path(summary_path).exists():
        print("❌ Evaluation summary not found. Run evaluation first.")
        return
    
    with open(summary_path, 'r') as f:
        lines = f.readlines()
    
    # Extract metrics
    precision = float(lines[3].split(': ')[1])
    recall = float(lines[4].split(': ')[1])
    f1_score = float(lines[5].split(': ')[1])
    total_detections = int(lines[6].split(': ')[1])
    total_ground_truth = int(lines[7].split(': ')[1])
    true_positives = int(lines[8].split(': ')[1])
    false_positives = int(lines[9].split(': ')[1])
    false_negatives = int(lines[10].split(': ')[1])
    
    print(f"\n🧪 TEST SET EVALUATION:")
    print(f"   • Precision: {precision:.4f}")
    print(f"   • Recall: {recall:.4f}")
    print(f"   • F1-Score: {f1_score:.4f}")
    print(f"   • Total Detections: {total_detections}")
    print(f"   • Total Ground Truth: {total_ground_truth}")
    print(f"   • True Positives: {true_positives}")
    print(f"   • False Positives: {false_positives}")
    print(f"   • False Negatives: {false_negatives}")
    
    # Performance assessment
    if f1_score < 0.2:
        print(f"\n⚠️ PERFORMANCE ASSESSMENT: Poor")
        print("   The model needs significant improvement for practical use.")
    elif f1_score < 0.5:
        print(f"\n⚠️ PERFORMANCE ASSESSMENT: Fair")
        print("   The model shows promise but needs refinement.")
    elif f1_score < 0.7:
        print(f"\n✅ PERFORMANCE ASSESSMENT: Good")
        print("   The model performs reasonably well for deployment.")
    else:
        print(f"\n🎉 PERFORMANCE ASSESSMENT: Excellent")
        print("   The model performs very well and is ready for deployment.")

def main():
    """Main analysis function."""
    # Paths - Updated to use the correct fine-tuned model path
    training_csv = "runs/detect/shitspotter_finetune/results.csv"
    evaluation_summary = "shitspotter_evaluation_results/metrics/summary_20250627_190104.txt"
    
    print("🔍 Loading training results...")
    df = load_training_results(training_csv)
    
    # Generate plots
    print("📊 Generating training analysis plots...")
    plot_training_curves(df, "fine_tuned_training_analysis.png")
    
    # Analyze training performance
    analyze_training_performance(df)
    
    # Analyze evaluation results
    analyze_evaluation_results(evaluation_summary)
    
    print(f"\n📁 Analysis complete! Check 'fine_tuned_training_analysis.png' for visualizations.")
    print(f"📁 Best model saved at: /home/ailaty3088@id.sdsu.edu/Ashter/runs/detect/shitspotter_finetune/weights/best.pt")

if __name__ == "__main__":
    main() 