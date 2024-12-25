# run this if the dataset is not organized
import yaml
from ultralytics import YOLO
import os
from sklearn.model_selection import train_test_split
import shutil

class WasteDetectorTrainer:
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.train_dir = os.path.join(output_dir, 'train')
        self.val_dir = os.path.join(output_dir, 'val')
        self.test_dir = os.path.join(output_dir, 'test')

    def prepare_dataset(self):
        """Organize dataset into train/val/test splits"""
        # Get all image files
        image_files = [f for f in os.listdir(self.data_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # Create train/val/test splits (70/20/10)
        train_files, temp_files = train_test_split(image_files, test_size=0.3, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.33, random_state=42)

        # Create directories
        for dir_path in [self.train_dir, self.val_dir, self.test_dir]:
            os.makedirs(os.path.join(dir_path, 'images'), exist_ok=True)
            os.makedirs(os.path.join(dir_path, 'labels'), exist_ok=True)

        # Move files to respective directories
        self._move_files(train_files, 'train')
        self._move_files(val_files, 'val')
        self._move_files(test_files, 'test')

    def _move_files(self, files, split_type):
        """Helper function to move files to split directories"""
        for f in files:
            # Move image
            src_img = os.path.join(self.data_dir, f)
            dst_img = os.path.join(self.output_dir, split_type, 'images', f)
            shutil.copy2(src_img, dst_img)

            # Move corresponding label if exists
            label_file = f.rsplit('.', 1)[0] + '.txt'
            src_label = os.path.join(self.data_dir, 'labels', label_file)
            if os.path.exists(src_label):
                dst_label = os.path.join(self.output_dir, split_type, 'labels', label_file)
                shutil.copy2(src_label, dst_label)

    def create_data_yaml(self):
        """Create YAML configuration file for training"""
        data_yaml = {
            'path': self.output_dir,
            'train': os.path.join('train', 'images'),
            'val': os.path.join('val', 'images'),
            'test': os.path.join('test', 'images'),
            'names': {
                0: 'pet_waste'
            },
            'nc': 1  # number of classes
        }

        yaml_path = os.path.join(self.output_dir, 'data.yaml')
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

    def evaluate_model(self, weights_path):
        """Evaluate the trained model"""
        model = YOLO(weights_path)
        
        # Run validation
        metrics = model.val()
        
        return metrics

if __name__ == "__main__":
    # Initialize trainer
    trainer = WasteDetectorTrainer(
        data_dir='path/to/your/6000/images',
        output_dir='path/to/output'
    )

    # Prepare dataset
    trainer.prepare_dataset()

    # Train model
    results = trainer.train_model(epochs=100)

    # Evaluate model
    metrics = trainer.evaluate_model('runs/train/pet_waste_detector/weights/best.pt')
    print(f"Validation metrics: {metrics}")