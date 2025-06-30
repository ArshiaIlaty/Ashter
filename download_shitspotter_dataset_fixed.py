import os
import shutil
from datasets import load_dataset
from PIL import Image
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml

def download_and_organize_dataset():
    """
    Download the shitspotter dataset and organize it in YOLO format
    """
    print("Downloading shitspotter dataset from Hugging Face...")
    
    # Load the dataset
    try:
        ds = load_dataset("erotemic/shitspotter")
        print(f"Dataset loaded successfully!")
        print(f"Available splits: {list(ds.keys())}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure you're logged in with: huggingface-cli login")
        return
    
    # Create output directory structure
    output_dir = Path("shitspotter_dataset")
    output_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for YOLO format
    for split in ['train', 'val', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split_name, dataset_split in ds.items():
        print(f"\nProcessing {split_name} split...")
        print(f"Number of samples: {len(dataset_split)}")
        
        # Determine target directory based on split name
        if 'train' in split_name.lower():
            target_split = 'train'
        elif 'val' in split_name.lower() or 'validation' in split_name.lower():
            target_split = 'val'
        elif 'test' in split_name.lower():
            target_split = 'test'
        else:
            # Default to train if unclear
            target_split = 'train'
        
        target_img_dir = output_dir / target_split / 'images'
        target_label_dir = output_dir / target_split / 'labels'
        
        processed_count = 0
        
        for i, sample in enumerate(dataset_split):
            try:
                # Get image and annotations with correct field names
                image = sample['jpg']  # Image is in 'jpg' field
                annotations_data = sample['json']  # Annotations are in 'json' field
                
                # Save image
                image_filename = f"{split_name}_{i:06d}.jpg"
                image_path = target_img_dir / image_filename
                image.save(image_path)
                
                # Process annotations and save in YOLO format
                label_filename = f"{split_name}_{i:06d}.txt"
                label_path = target_label_dir / label_filename
                
                yolo_annotations = []
                
                # Process annotations from the json field
                if 'annotations' in annotations_data and annotations_data['annotations']:
                    annotations = annotations_data['annotations']
                    img_width = annotations_data['width']
                    img_height = annotations_data['height']
                    
                    for annotation in annotations:
                        if 'bbox' in annotation:
                            # COCO format: [x_min, y_min, width, height]
                            x_min, y_min, width, height = annotation['bbox']
                            
                            # Calculate center coordinates
                            center_x = x_min + width / 2
                            center_y = y_min + height / 2
                            
                            # Normalize to [0, 1]
                            center_x_norm = center_x / img_width
                            center_y_norm = center_y / img_height
                            width_norm = width / img_width
                            height_norm = height / img_height
                            
                            # Class ID (assuming pet waste is class 0)
                            class_id = 0
                            
                            yolo_annotations.append(f"{class_id} {center_x_norm:.6f} {center_y_norm:.6f} {width_norm:.6f} {height_norm:.6f}")
                
                # Save YOLO annotations
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
                
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} samples...")
                    
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        print(f"Completed {split_name}: {processed_count} samples processed")
    
    # Create data.yaml file
    create_data_yaml(output_dir)
    
    # Print dataset statistics
    print_dataset_stats(output_dir)
    
    print(f"\nDataset successfully downloaded and organized in: {output_dir}")
    print("You can now use this dataset for training your YOLO model!")

def create_data_yaml(output_dir):
    """Create YAML configuration file for YOLO training"""
    data_yaml = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {
            0: 'pet_waste'
        },
        'nc': 1  # number of classes
    }
    
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"Created data.yaml at: {yaml_path}")

def print_dataset_stats(output_dir):
    """Print statistics about the organized dataset"""
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    
    total_images = 0
    total_labels = 0
    
    for split in ['train', 'val', 'test']:
        img_dir = output_dir / split / 'images'
        label_dir = output_dir / split / 'labels'
        
        if img_dir.exists():
            images = len(list(img_dir.glob('*.jpg'))) + len(list(img_dir.glob('*.png')))
            labels = len(list(label_dir.glob('*.txt')))
            
            print(f"{split.upper()}:")
            print(f"  Images: {images}")
            print(f"  Labels: {labels}")
            
            total_images += images
            total_labels += labels
    
    print(f"\nTOTAL:")
    print(f"  Images: {total_images}")
    print(f"  Labels: {total_labels}")
    print("="*50)

if __name__ == "__main__":
    # Check if huggingface is logged in
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        # This will raise an error if not logged in
        api.whoami()
        print("Hugging Face login verified.")
    except Exception as e:
        print("Please login to Hugging Face first:")
        print("Run: huggingface-cli login")
        print("Then enter your token when prompted.")
        exit(1)
    
    download_and_organize_dataset() 