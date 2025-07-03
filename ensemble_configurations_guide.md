# Advanced Ensemble Configurations Guide

## 1. Confidence Threshold Tuning

The confidence threshold tuning script (`ensemble_confidence_tuning.py`) is currently running and will:

- Test confidence thresholds from 0.001 to 0.5
- Test WBF IoU thresholds from 0.3 to 0.7
- Find optimal configurations for F1, Precision, and Recall
- Generate precision-recall curves
- Save results to `confidence_tuning_results.json`

## 2. Advanced Ensemble Configurations

### A. Weighted Box Fusion (WBF) Variants

#### Basic WBF
```python
# Equal weights for both models
weights = [1.0, 1.0]
iou_thr = 0.55
```

#### Performance-Weighted WBF
```python
# Weight based on individual model performance
weights = [0.6, 0.4]  # YOLOv8n gets higher weight if it performs better
iou_thr = 0.55
```

#### Adaptive WBF
```python
# Dynamic weights based on confidence scores
base_weights = [0.6, 0.4]
confidence_factor = 0.3
# Weights adjust based on prediction confidence
```

### B. Multi-Scale Ensemble
```python
# Combine predictions from different input scales
scales = [0.8, 1.0, 1.2]  # 80%, 100%, 120% of original size
fusion_method = 'wbf'
# Better for objects of different sizes
```

### C. Confidence-Weighted Ensemble
```python
# Weight predictions based on confidence scores
confidence_power = 2.0  # Square the confidence scores
min_confidence = 0.1
# High-confidence predictions get more weight
```

### D. Class-Specific Ensemble
```python
# Different fusion strategies for different classes
class_weights = {
    'pet_waste': [0.7, 0.3],  # YOLOv8n better for small objects
    'other_classes': [0.5, 0.5]
}
# Optimized per class
```

### E. Cascade Ensemble
```python
# Multi-stage fusion with different thresholds
stages = [
    {'iou_thr': 0.7, 'skip_box_thr': 0.5},  # High precision stage
    {'iou_thr': 0.5, 'skip_box_thr': 0.3},  # Medium precision stage
    {'iou_thr': 0.3, 'skip_box_thr': 0.1}   # High recall stage
]
# Progressive refinement
```

### F. Temporal Ensemble (for video)
```python
# Combine predictions across video frames
window_size = 3  # Number of frames to consider
temporal_weight = 0.7  # Weight for temporal consistency
# Reduces temporal jitter
```

### G. Uncertainty-Aware Ensemble
```python
# Consider prediction uncertainty in fusion
uncertainty_threshold = 0.3
uncertainty_weight = 0.5
# More robust to uncertain predictions
```

### H. Bayesian Ensemble
```python
# Bayesian model averaging
prior_strength = 0.1
likelihood_method = 'gaussian'
# Probabilistic interpretation
```

## 3. Recommended Configurations by Use Case

### Production Systems
- **Weighted WBF**: Best balance of performance and simplicity
- **Confidence-Weighted**: Reduces false positives

### Research/Development
- **Multi-Scale Ensemble**: Maximum accuracy
- **Uncertainty-Aware**: Better understanding of model confidence

### Real-time Applications
- **Basic WBF**: Fastest execution
- **Confidence-Weighted**: Good precision-recall balance

### High Accuracy Requirements
- **Multi-Scale Ensemble**: Best overall performance
- **Cascade Ensemble**: Fine-tuned precision-recall trade-off

### Video Processing
- **Temporal Ensemble**: Reduces frame-to-frame jitter
- **Weighted WBF**: Good baseline performance

### Robust Systems
- **Uncertainty-Aware**: Handles uncertain predictions better
- **Bayesian Ensemble**: Probabilistic robustness

## 4. Implementation Steps

1. **Start with Basic WBF**: Use equal weights [1.0, 1.0]
2. **Tune Confidence Thresholds**: Run the confidence tuning script
3. **Optimize Weights**: Based on individual model performance
4. **Add Advanced Features**: Multi-scale, class-specific, etc.
5. **Validate**: Test on holdout set
6. **Deploy**: Monitor performance in production

## 5. Performance Metrics to Monitor

- **mAP50**: Mean Average Precision at IoU=0.5
- **Precision**: Ratio of correct detections to total detections
- **Recall**: Ratio of detected objects to total objects
- **F1-Score**: Harmonic mean of precision and recall
- **Inference Time**: Processing time per image
- **Memory Usage**: GPU/RAM requirements

## 6. Next Steps

1. Wait for confidence threshold tuning to complete
2. Analyze results from `confidence_tuning_results.json`
3. Implement the best configuration
4. Test on validation set
5. Consider advanced configurations based on requirements 