import yaml
from ultralytics import YOLO
import os

class WasteDetectorTrainer:
    def __init__(self, dataset_dir):
        """
        Initialize trainer with pre-organized dataset directory
        dataset_dir should have train/val/test folders, each with images and labels subdirectories
        """
        self.dataset_dir = dataset_dir
        self.verify_dataset_structure()

    def verify_dataset_structure(self):
        """Verify that dataset follows the expected structure"""
        required_dirs = ['train', 'val', 'test']
        required_subdirs = ['images', 'labels']
        
        for dir_name in required_dirs:
            dir_path = os.path.join(self.dataset_dir, dir_name)
            if not os.path.exists(dir_path):
                raise ValueError(f"Missing required directory: {dir_path}")
            
            for subdir in required_subdirs:
                subdir_path = os.path.join(dir_path, subdir)
                if not os.path.exists(subdir_path):
                    raise ValueError(f"Missing required subdirectory: {subdir_path}")

    def create_data_yaml(self):
        """Create YAML configuration file for training"""
        data_yaml = {
            'path': self.dataset_dir,
            'train': os.path.join('train', 'images'),
            'val': os.path.join('val', 'images'),
            'test': os.path.join('test', 'images'),
            'names': {
                0: 'pet_waste'
            },
            'nc': 1  # number of classes
        }

        yaml_path = os.path.join(self.dataset_dir, 'data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f)
        
        return yaml_path

    def train_model(self, epochs=100, batch_size=16, img_size=640):
        """Train the YOLOv8 model"""
        # Load YOLOv8 model
        model = YOLO('yolov8m.pt')  # medium size model

        # Create data.yaml
        yaml_path = self.create_data_yaml()

        # Train the model
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            name='pet_waste_detector',
            save=True,
            patience=20,  # Early stopping patience
            device='0'  # Use GPU if available
        )

        return results

    def evaluate_model(self, weights_path=None):
        """Evaluate the trained model"""
        # If weights_path not provided, use the best weights from training
        if weights_path is None:
            weights_path = 'runs/train/pet_waste_detector/weights/best.pt'
            
        model = YOLO(weights_path)
        
        # Run validation
        metrics = model.val()
        
        print("\nModel Evaluation Metrics:")
        print(f"mAP@0.5: {metrics.box.map50}")
        print(f"mAP@0.5:0.95: {metrics.box.map}")
        print(f"Precision: {metrics.box.p}")
        print(f"Recall: {metrics.box.r}")
        
        return metrics

if __name__ == "__main__":
    # Initialize trainer with your pre-organized dataset
    trainer = WasteDetectorTrainer(
        dataset_dir="/Users/arshiailaty/Documents/Ashter/dataset"  # This should be the root directory containing train/val/test
    )

    # Train model
    print("Starting model training...")
    results = trainer.train_model(
        epochs=100,
        batch_size=16,
        img_size=640
    )
    print("Training completed!")

    # Evaluate model
    print("\nEvaluating model...")
    metrics = trainer.evaluate_model()