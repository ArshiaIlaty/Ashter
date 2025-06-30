import os
from ultralytics import YOLO
import tensorflow as tf
import tensorflowjs as tfjs

def convert_to_tfjs(model_path, output_dir):
    """
    Convert YOLOv8 model to TensorFlow.js format
    Args:
        model_path: Path to the trained YOLOv8 model (.pt file)
        output_dir: Directory to save the converted model
    """
    print(f"Loading YOLOv8 model from {model_path}")
    model = YOLO(model_path)
    
    # Export to TensorFlow SavedModel format
    print("Exporting to TensorFlow format...")
    model.export(format='tfjs', imgsz=640)
    
    # Get the path where the model was exported
    model_dir = os.path.dirname(model_path)
    saved_model_path = os.path.join(model_dir, 'best_web_model')
    
    if not os.path.exists(saved_model_path):
        raise FileNotFoundError(f"Exported model not found at {saved_model_path}")
    
    print(f"Model exported successfully to {saved_model_path}")
    return saved_model_path

if __name__ == "__main__":
    # Path to your trained model
    model_path = 'runs/train/pet_waste_detector/weights/best.pt'
    
    # Create output directory
    output_dir = 'runs/train/pet_waste_detector/weights'
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert model
    tfjs_path = convert_to_tfjs(model_path, output_dir)
    print(f"Conversion complete. Model saved at: {tfjs_path}") 