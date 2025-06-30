import os
import yaml
from ultralytics import YOLO
from pathlib import Path

def fine_tune_on_shitspotter():
    """Fine-tune the model on shitspotter dataset"""
    
    # Check if dataset exists
    if not os.path.exists('shitspotter_dataset'):
        print("Shitspotter dataset not found. Please run the download script first.")
        return
    
    # Check if base model exists
    base_model_path = 'runs/train/pet_waste_detector/weights/best.pt'
    if not os.path.exists(base_model_path):
        print(f"Base model not found at {base_model_path}")
        return
    
    print("Starting fine-tuning on shitspotter dataset...")
    print(f"Base model: {base_model_path}")
    print(f"Dataset: shitspotter_dataset")
    
    # Initialize model with pre-trained weights
    model = YOLO(base_model_path)
    
    # Fine-tune the model
    results = model.train(
        data='shitspotter_dataset/data.yaml',
        epochs=30,
        imgsz=640,
        batch=16,
        name='shitspotter_finetune',
        patience=10,
        save=True,
        save_period=5,
        device='0' if os.path.exists('/dev/nvidia0') else 'cpu'
    )
    
    print("Fine-tuning completed!")
    print(f"Results saved in: runs/train/shitspotter_finetune/")
    
    return results

if __name__ == "__main__":
    fine_tune_on_shitspotter() 