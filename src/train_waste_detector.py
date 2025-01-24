import yaml
import os
from ultralytics import YOLO
import shutil

class WasteDetectorTrainer:
    def __init__(self, dataset_dir):
        """
        Initialize trainer with pre-organized dataset directory
        """
        self.dataset_dir = dataset_dir
        self.verify_dataset_structure()

    def verify_dataset_structure(self):
        """Verify dataset structure and content"""
        print("Verifying dataset structure...")
        
        # Check directories
        for split in ['train_yolo', 'val_yolo', 'test_yolo']:
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
            'train': os.path.join('train_yolo', 'images'),
            'val': os.path.join('val_yolo', 'images'),
            'test': os.path.join('test_yolo', 'images'),
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

        # Train the model
        try:
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
            print("Training completed successfully!")
            return results
        except Exception as e:
            print(f"Error during training: {e}")
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
            return metrics
        except Exception as e:
            print(f"Error during evaluation: {e}")
            raise

if __name__ == "__main__":
    # Initialize trainer with your dataset path
    dataset_path = '/home/ailaty3088@id.sdsu.edu/Ashter/dataset_yolo'
    
    trainer = WasteDetectorTrainer(dataset_path)
    
    # Train model with specified parameters
    results = trainer.train_model(
        epochs=100,
        batch_size=16,
        img_size=640
    )
    
    # Evaluate the model
    metrics = trainer.evaluate_model()