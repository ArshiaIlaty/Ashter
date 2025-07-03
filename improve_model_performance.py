#!/usr/bin/env python3
"""
Methods and approaches to improve model performance on the old dataset
"""

import os
import yaml
import shutil
from pathlib import Path
from ultralytics import YOLO
import json
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ModelImprovementMethods:
    def __init__(self, old_dataset_path, output_dir):
        self.old_dataset_path = old_dataset_path
        self.output_dir = output_dir
        self.config = self.load_dataset_config()
        
    def load_dataset_config(self):
        """Load dataset configuration"""
        config_path = os.path.join(self.old_dataset_path, 'data.yaml')
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def method_1_fine_tune_on_old_dataset(self):
        """Method 1: Fine-tune pre-trained models on old dataset"""
        print("\n" + "="*60)
        print("METHOD 1: Fine-tuning on Old Dataset")
        print("="*60)
        
        # Create fine-tuning config
        finetune_config = {
            'model_paths': {
                'yolov8n': '/home/ailaty3088@id.sdsu.edu/Ashter/runs/detect/shitspotter_yolov8n/weights/best.pt',
                'yolov8s': '/home/ailaty3088@id.sdsu.edu/Ashter/runs/detect/shitspotter_yolov8s/weights/best.pt'
            },
            'training_params': {
                'epochs': 50,
                'batch_size': 16,
                'imgsz': 640,
                'patience': 10,
                'save_period': 10,
                'lr0': 0.001,  # Lower learning rate for fine-tuning
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5,
                'pose': 12.0,
                'kobj': 1.0,
                'label_smoothing': 0.0,
                'nbs': 64,
                'overlap_mask': True,
                'mask_ratio': 4,
                'dropout': 0.0,
                'val': True,
                'plots': True
            }
        }
        
        # Save fine-tuning config
        config_file = os.path.join(self.output_dir, 'finetune_config.json')
        with open(config_file, 'w') as f:
            json.dump(finetune_config, f, indent=2)
        
        print(f"Fine-tuning configuration saved to: {config_file}")
        print("\nTo fine-tune models, run:")
        print("python -c \"from ultralytics import YOLO; model = YOLO('model_path'); model.train(data='dataset/data.yaml', **training_params)\"")
        
        return finetune_config
    
    def method_2_data_augmentation_pipeline(self):
        """Method 2: Create data augmentation pipeline"""
        print("\n" + "="*60)
        print("METHOD 2: Data Augmentation Pipeline")
        print("="*60)
        
        # Define augmentation transforms that match old dataset characteristics
        augmentation_pipeline = A.Compose([
            # Geometric transformations
            A.RandomRotate90(p=0.5),
            # A.Flip(p=0.5),
            A.HorizontalFlip(p=0.5),  # for horizontal flip
            # A.VerticalFlip(p=0.5),    # for vertical flip
            A.Transpose(p=0.5),
            A.OneOf([
                # A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.GaussNoise(),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.PiecewiseAffine(p=0.3),
            ], p=0.2),
            
            # Color and contrast adjustments
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
            
            # Weather and lighting conditions
            A.OneOf([
                A.RandomRain(p=0.1),
                A.RandomShadow(p=0.1),
                A.RandomSunFlare(p=0.1),
            ], p=0.2),
            
            # Normalization
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        # Save augmentation pipeline
        aug_file = os.path.join(self.output_dir, 'augmentation_pipeline.py')
        with open(aug_file, 'w') as f:
            f.write("""import albumentations as A
from albumentations.pytorch import ToTensorV2

# Data augmentation pipeline for old dataset
augmentation_pipeline = A.Compose([
    # Geometric transformations
    A.RandomRotate90(p=0.5),
    # A.Flip(p=0.5),
    A.HorizontalFlip(p=0.5),  # for horizontal flip
    # A.VerticalFlip(p=0.5),    # for vertical flip
    A.Transpose(p=0.5),
    A.OneOf([
        A.IAAAdditiveGaussianNoise(),
        A.GaussNoise(),
    ], p=0.2),
    A.OneOf([
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
    ], p=0.2),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
    A.OneOf([
        A.OpticalDistortion(p=0.3),
        A.GridDistortion(p=0.1),
        A.IAAPiecewiseAffine(p=0.3),
    ], p=0.2),
    
    # Color and contrast adjustments
    A.OneOf([
        A.CLAHE(clip_limit=2),
        A.IAASharpen(),
        A.IAAEmboss(),
        A.RandomBrightnessContrast(),
    ], p=0.3),
    A.HueSaturationValue(p=0.3),
    
    # Weather and lighting conditions
    A.OneOf([
        A.RandomRain(p=0.1),
        A.RandomShadow(p=0.1),
        A.RandomSunFlare(p=0.1),
    ], p=0.2),
    
    # Normalization
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def apply_augmentation(image, bboxes, class_labels):
    \"\"\"Apply augmentation to image and bounding boxes\"\"\"
    augmented = augmentation_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
    return augmented['image'], augmented['bboxes'], augmented['class_labels']
""")
        
        print(f"Augmentation pipeline saved to: {aug_file}")
        return augmentation_pipeline
    
    def method_3_ensemble_approach(self):
        """Method 3: Ensemble multiple models"""
        print("\n" + "="*60)
        print("METHOD 3: Ensemble Approach")
        print("="*60)
        
        ensemble_config = {
            'models': [
                {
                    'name': 'yolov8n',
                    'path': '/home/ailaty3088@id.sdsu.edu/Ashter/runs/detect/shitspotter_yolov8n/weights/best.pt',
                    'weight': 0.4
                },
                {
                    'name': 'yolov8s', 
                    'path': '/home/ailaty3088@id.sdsu.edu/Ashter/runs/detect/shitspotter_yolov8s/weights/best.pt',
                    'weight': 0.3
                },
                {
                    'name': 'yolov8n_finetuned',
                    'path': 'path/to/finetuned/yolov8n.pt',
                    'weight': 0.3
                }
            ],
            'ensemble_methods': ['weighted_average', 'nms', 'soft_nms'],
            'confidence_threshold': 0.25,
            'nms_threshold': 0.5
        }
        
        # Save ensemble config
        ensemble_file = os.path.join(self.output_dir, 'ensemble_config.json')
        with open(ensemble_file, 'w') as f:
            json.dump(ensemble_config, f, indent=2)
        
        print(f"Ensemble configuration saved to: {ensemble_file}")
        return ensemble_config
    
    def method_4_hyperparameter_optimization(self):
        """Method 4: Hyperparameter optimization"""
        print("\n" + "="*60)
        print("METHOD 4: Hyperparameter Optimization")
        print("="*60)
        
        hyperopt_config = {
            'optimization_space': {
                'lr0': [0.0001, 0.001, 0.01],
                'lrf': [0.01, 0.1, 0.5],
                'momentum': [0.8, 0.9, 0.95],
                'weight_decay': [0.0001, 0.0005, 0.001],
                'warmup_epochs': [1, 3, 5],
                'box': [5.0, 7.5, 10.0],
                'cls': [0.3, 0.5, 0.7],
                'dfl': [1.0, 1.5, 2.0]
            },
            'optimization_method': 'bayesian',  # or 'grid', 'random'
            'n_trials': 20,
            'metric': 'mAP50'
        }
        
        # Save hyperopt config
        hyperopt_file = os.path.join(self.output_dir, 'hyperopt_config.json')
        with open(hyperopt_file, 'w') as f:
            json.dump(hyperopt_config, f, indent=2)
        
        print(f"Hyperparameter optimization config saved to: {hyperopt_file}")
        return hyperopt_config
    
    def method_5_transfer_learning_strategies(self):
        """Method 5: Advanced transfer learning strategies"""
        print("\n" + "="*60)
        print("METHOD 5: Transfer Learning Strategies")
        print("="*60)
        
        transfer_config = {
            'strategies': {
                'progressive_unfreezing': {
                    'description': 'Unfreeze layers progressively during training',
                    'stages': [
                        {'epochs': 10, 'frozen_layers': 'all'},
                        {'epochs': 20, 'frozen_layers': 'backbone'},
                        {'epochs': 30, 'frozen_layers': 'none'}
                    ]
                },
                'differential_learning_rates': {
                    'description': 'Use different learning rates for different layers',
                    'backbone_lr': 0.0001,
                    'neck_lr': 0.001,
                    'head_lr': 0.01
                },
                'knowledge_distillation': {
                    'description': 'Use larger model as teacher for smaller model',
                    'teacher_model': 'yolov8s',
                    'student_model': 'yolov8n',
                    'temperature': 4.0,
                    'alpha': 0.7
                }
            }
        }
        
        # Save transfer learning config
        transfer_file = os.path.join(self.output_dir, 'transfer_learning_config.json')
        with open(transfer_file, 'w') as f:
            json.dump(transfer_config, f, indent=2)
        
        print(f"Transfer learning config saved to: {transfer_file}")
        return transfer_config
    
    def method_6_data_quality_improvement(self):
        """Method 6: Data quality improvement"""
        print("\n" + "="*60)
        print("METHOD 6: Data Quality Improvement")
        print("="*60)
        
        data_quality_config = {
            'label_verification': {
                'check_bbox_coordinates': True,
                'check_class_labels': True,
                'remove_empty_labels': True,
                'fix_invalid_bboxes': True
            },
            'image_quality': {
                'remove_blurry_images': True,
                'remove_low_contrast_images': True,
                'enhance_image_quality': True,
                'standardize_image_sizes': True
            },
            'dataset_balance': {
                'check_class_distribution': True,
                'apply_class_weights': True,
                'oversample_minority_classes': True
            }
        }
        
        # Save data quality config
        quality_file = os.path.join(self.output_dir, 'data_quality_config.json')
        with open(quality_file, 'w') as f:
            json.dump(data_quality_config, f, indent=2)
        
        print(f"Data quality config saved to: {quality_file}")
        return data_quality_config
    
    def generate_improvement_plan(self):
        """Generate comprehensive improvement plan"""
        print("\n" + "="*60)
        print("COMPREHENSIVE IMPROVEMENT PLAN")
        print("="*60)
        
        plan = {
            'immediate_actions': [
                "1. Fine-tune YOLOv8n on old dataset (highest priority)",
                "2. Apply data augmentation to increase training data",
                "3. Implement ensemble of YOLOv8n and YOLOv8s",
                "4. Optimize hyperparameters using Bayesian optimization"
            ],
            'medium_term_actions': [
                "5. Implement progressive unfreezing strategy",
                "6. Apply knowledge distillation from YOLOv8s to YOLOv8n",
                "7. Improve data quality and label verification",
                "8. Experiment with different model architectures"
            ],
            'long_term_actions': [
                "9. Collect more diverse training data",
                "10. Implement advanced augmentation techniques",
                "11. Use model compression techniques",
                "12. Deploy ensemble models in production"
            ],
            'expected_improvements': {
                'fine_tuning': '10-20% improvement in mAP50',
                'data_augmentation': '5-15% improvement in generalization',
                'ensemble': '5-10% improvement in accuracy',
                'hyperparameter_optimization': '3-8% improvement in performance'
            }
        }
        
        # Save improvement plan
        plan_file = os.path.join(self.output_dir, 'improvement_plan.md')
        with open(plan_file, 'w') as f:
            f.write("# Model Performance Improvement Plan\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Immediate Actions (Week 1-2)\n\n")
            for action in plan['immediate_actions']:
                f.write(f"- {action}\n")
            
            f.write("\n## Medium Term Actions (Week 3-6)\n\n")
            for action in plan['medium_term_actions']:
                f.write(f"- {action}\n")
            
            f.write("\n## Long Term Actions (Month 2-3)\n\n")
            for action in plan['long_term_actions']:
                f.write(f"- {action}\n")
            
            f.write("\n## Expected Improvements\n\n")
            for method, improvement in plan['expected_improvements'].items():
                f.write(f"- **{method.replace('_', ' ').title()}**: {improvement}\n")
        
        print(f"Improvement plan saved to: {plan_file}")
        return plan

def main():
    # Configuration
    old_dataset_path = "/home/ailaty3088@id.sdsu.edu/Ashter/dataset"
    output_dir = "/home/ailaty3088@id.sdsu.edu/Ashter/model_improvement_methods"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize improvement methods
    improver = ModelImprovementMethods(old_dataset_path, output_dir)
    
    # Generate all improvement methods
    print("Generating improvement methods and configurations...")
    
    improver.method_1_fine_tune_on_old_dataset()
    improver.method_2_data_augmentation_pipeline()
    improver.method_3_ensemble_approach()
    improver.method_4_hyperparameter_optimization()
    improver.method_5_transfer_learning_strategies()
    improver.method_6_data_quality_improvement()
    improver.generate_improvement_plan()
    
    print(f"\nAll improvement methods and configurations saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Review the improvement plan in 'improvement_plan.md'")
    print("2. Start with fine-tuning YOLOv8n on your old dataset")
    print("3. Apply data augmentation to increase training data diversity")
    print("4. Implement ensemble methods for better performance")

if __name__ == "__main__":
    main() 