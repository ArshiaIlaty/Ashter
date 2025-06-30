import yaml
import os
from ultralytics import YOLO
import shutil
import wandb
from datetime import datetime
from backup_training import TrainingBackup

class WasteDetectorTrainer:
    def __init__(self, dataset_dir):
        """
        Initialize trainer with pre-organized dataset directory
        """
        self.dataset_dir = dataset_dir
        
        # Initialize wandb first, before any logging operations
        wandb.init(project="pet-waste-detection", config={
            "architecture": "YOLOv8m",
            "dataset": "pet-waste",
            "epochs": 100,
            "batch_size": 16,
            "img_size": 640
        })
        
        # Then verify dataset structure
        self.verify_dataset_structure()
        self.backup = TrainingBackup()

    def verify_dataset_structure(self):
        """Verify dataset structure and content"""
        print("Verifying dataset structure...")
        
        # Check directories
        for split in ['train', 'val', 'test']:
            img_dir = os.path.join(self.dataset_dir, split, 'images')
            label_dir = os.path.join(self.dataset_dir, split, 'labels')
            
            if not os.path.exists(img_dir):
                raise ValueError(f"Missing images directory: {img_dir}")
            if not os.path.exists(label_dir):
                raise ValueError(f"Missing labels directory: {label_dir}")
                
            # Count files
            images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            labels = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
            
            print(f"\n{split} set:")
            print(f"Number of images: {len(images)}")
            print(f"Number of labels: {len(labels)}")
            
            # Log dataset statistics to wandb
            wandb.log({
                f"{split}_images": len(images),
                f"{split}_labels": len(labels)
            })
            
            # Verify matching files
            img_bases = {os.path.splitext(f)[0] for f in images}
            label_bases = {os.path.splitext(f)[0] for f in labels}
            
            missing_labels = img_bases - label_bases
            if missing_labels:
                print(f"Warning: {len(missing_labels)} images missing labels")
                print("First few missing:", list(missing_labels)[:5])
            
            # Verify label format
            if labels:
                sample_label = os.path.join(label_dir, labels[0])
                try:
                    with open(sample_label, 'r') as f:
                        content = f.read().strip()
                        if content:
                            print(f"Sample label content from {labels[0]}:")
                            print(content)
                        else:
                            print(f"Warning: Empty label file {labels[0]}")
                except Exception as e:
                    print(f"Error reading label file: {e}")

    def create_data_yaml(self):
        """Create YAML configuration file for training"""
        data_yaml = {
            'path': self.dataset_dir,
            'train': os.path.join('train', 'images'),
            'val': os.path.join('val', 'images'),
            'test': os.path.join('test', 'images'),
            'names': {
                0: 'pet_waste'  # Single class for pet waste
            },
            'nc': 1  # number of classes
        }

        yaml_path = os.path.join(self.dataset_dir, 'data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f)
        
        print(f"\nCreated data.yaml at {yaml_path}")
        return yaml_path

    def train_model(self, epochs=100, batch_size=16, img_size=640):
        """Train the YOLOv8 model"""
        print("\nStarting model training...")
        
        # Load YOLOv8 model
        model = YOLO('yolov8m.pt')  # medium size model

        # Create data.yaml
        yaml_path = self.create_data_yaml()

        # Configure training parameters with augmentation
        training_args = {
            'data': yaml_path,
            'epochs': epochs,
            'imgsz': img_size,
            'batch': batch_size,
            'name': f'pet_waste_detector_{datetime.now().strftime("%Y%m%d_%H%M%S")}',  # Add timestamp to run name
            'save': True,
            'patience': 15,  # Early stopping patience
            'device': '0',  # Use GPU if available
            # Data augmentation parameters
            'hsv_h': 0.015,  # HSV-Hue augmentation
            'hsv_s': 0.7,    # HSV-Saturation augmentation
            'hsv_v': 0.4,    # HSV-Value augmentation
            'degrees': 15.0,  # Rotation augmentation
            'translate': 0.1,  # Translation augmentation
            'scale': 0.2,    # Scale augmentation
            'shear': 0.0,    # Shear augmentation
            'perspective': 0.0,  # Perspective augmentation
            'flipud': 0.0,   # Flip up-down augmentation
            'fliplr': 0.5,   # Flip left-right augmentation
            'mosaic': 1.0,   # Mosaic augmentation
            'mixup': 0.0,    # Mixup augmentation
            'copy_paste': 0.0,  # Copy-paste augmentation
            'lr0': 0.001,    # Initial learning rate
            'lrf': 0.01,     # Final learning rate fraction
            'momentum': 0.937,  # SGD momentum
            'weight_decay': 0.0005,  # Optimizer weight decay
            'warmup_epochs': 3.0,  # Warmup epochs
            'warmup_momentum': 0.8,  # Warmup momentum
            'warmup_bias_lr': 0.1,  # Warmup bias learning rate
            'box': 7.5,      # Box loss gain
            'cls': 0.5,      # Class loss gain
            'dfl': 1.5,      # DFL loss gain
            'close_mosaic': 10,  # Disable mosaic for last epochs
            'save_period': 10,  # Save checkpoint every 10 epochs
            'cache': True,  # Cache images in memory
            'workers': 8,  # Number of worker threads
            'project': 'runs/train',  # Project name
            'exist_ok': True,  # Overwrite existing experiment
            'pretrained': True,  # Use pretrained weights
            'optimizer': 'auto',  # Optimizer (SGD, Adam, etc.)
            'verbose': True,  # Print verbose output
            'seed': 42,  # Random seed for reproducibility
            'deterministic': True,  # Deterministic training
            'single_cls': True,  # Single class detection
            'rect': False,  # Disable rectangular training
        }

        # Log training configuration to wandb
        wandb.config.update(training_args)

        try:
            results = model.train(**training_args)
            print("Training completed successfully!")
            
            # Log final metrics to wandb
            wandb.log({
                "mAP50": results.results_dict['metrics/mAP50(B)'],
                "mAP50-95": results.results_dict['metrics/mAP50-95(B)'],
                "precision": results.results_dict['metrics/precision(B)'],
                "recall": results.results_dict['metrics/recall(B)']
            })
            
            # After successful training, create a backup
            run_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.backup.create_backup(run_name)
            print(f"\nTraining backup created: {run_name}")
            
            return results
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise

    def evaluate_model(self, weights_path=None):
        """Evaluate the trained model"""
        if weights_path is None:
            weights_path = 'runs/detect/pet_waste_detector/weights/best.pt'
            
        if not os.path.exists(weights_path):
            print(f"Warning: Weights file not found at {weights_path}")
            return None
            
        print(f"\nEvaluating model using weights from: {weights_path}")
        model = YOLO(weights_path)
        
        try:
            metrics = model.val()
            print("\nValidation Results:")
            print(f"mAP@0.5: {metrics.box.map50}")
            print(f"mAP@0.5:0.95: {metrics.box.map}")
            
            # Log evaluation metrics to wandb
            wandb.log({
                "val_mAP50": metrics.box.map50,
                "val_mAP50-95": metrics.box.map
            })
            
            return metrics
        except Exception as e:
            print(f"Error during evaluation: {e}")
            raise

if __name__ == "__main__":
    # Initialize trainer with your dataset path
    dataset_path = '/home/ailaty3088@id.sdsu.edu/Ashter/dataset/'
    
    trainer = WasteDetectorTrainer(dataset_path)
    
    # Train model with specified parameters
    results = trainer.train_model(
        epochs=100,
        batch_size=16,
        img_size=640
    )
    
    # Evaluate the model
    metrics = trainer.evaluate_model()
    
    # Close wandb run
    wandb.finish()