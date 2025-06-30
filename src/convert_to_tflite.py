import tensorflow as tf
from ultralytics import YOLO
import os
import shutil

def convert_to_tflite(model_path, output_dir='converted_models'):
    """
    Convert YOLO model to TensorFlow Lite format optimized for Raspberry Pi
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load YOLO model
        print("Loading YOLO model...")
        model = YOLO(model_path)
        
        # Export to TensorFlow format
        print("Exporting to TensorFlow format...")
        tf_model_path = os.path.join(output_dir, 'model.tf')
        model.export(format='tf', imgsz=640)
        
        # Move the exported model to our output directory
        exported_model_path = 'yolo11n_saved_model'
        if os.path.exists(exported_model_path):
            print(f"Moving exported model from {exported_model_path} to {tf_model_path}")
            if os.path.exists(tf_model_path):
                shutil.rmtree(tf_model_path)
            shutil.move(exported_model_path, tf_model_path)
        else:
            raise FileNotFoundError(f"Exported model not found at {exported_model_path}")
        
        # Convert to TFLite
        print("Converting to TFLite format...")
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
        
        # Optimize for Raspberry Pi
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        # Convert the model
        tflite_model = converter.convert()
        
        # Save the TFLite model
        tflite_model_path = os.path.join(output_dir, 'model.tflite')
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"Model converted and saved to {tflite_model_path}")
        return tflite_model_path
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        raise

if __name__ == "__main__":
    # Convert the model
    model_path = 'yolo11n.pt'  # Use your trained model
    convert_to_tflite(model_path) 