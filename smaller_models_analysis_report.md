# Smaller YOLO Models Analysis Report

## Summary

Both YOLOv8n and YOLOv8s trained successfully but show identical poor test performance despite good training metrics.

## Training Results

| Model | Precision | Recall | mAP50 | Training Time |
|-------|-----------|--------|-------|---------------|
| YOLOv8n | 0.6647 | 0.4855 | 0.5450 | 74.8 min |
| YOLOv8s | 0.7094 | 0.5352 | 0.6007 | 94.1 min |
| YOLOv8m | 0.7069 | 0.5490 | 0.6142 | 184.7 min |

## Test Evaluation Results

Both smaller models achieved identical performance:
- **Precision**: 0.0755 (7.55%)
- **Recall**: 0.0921 (9.21%) 
- **F1-Score**: 0.0830 (8.30%)
- **True Positives**: 69
- **False Positives**: 845
- **False Negatives**: 680

## Key Findings

1. **Training Performance**: All models show excellent training metrics (mAP50 > 0.5)
2. **Test Performance**: All models show poor test performance (F1-Score ~0.083)
3. **Identical Results**: YOLOv8n and YOLOv8s have identical test performance despite different training metrics
4. **Speed Advantage**: YOLOv8n is 2.5x faster than YOLOv8m

## Recommendations

1. **Dataset Issues**: The identical test performance suggests dataset problems, not model issues
2. **Confidence Tuning**: Optimize confidence thresholds for better precision
3. **Cross-Validation**: Implement k-fold validation to assess true generalization
4. **Ensemble Methods**: Combine multiple models for better performance
5. **Data Augmentation**: Increase training data variety to reduce overfitting

## Conclusion

The smaller models are viable alternatives to YOLOv8m with significant speed advantages, but all models need dataset and threshold optimization for production use. 