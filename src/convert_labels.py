import os
import numpy as np
from pathlib import Path

def convert_to_yolo_format(points):
    """
    Convert polygon points to YOLO format (center x, center y, width, height)
    """
    # Convert string coordinates to float arrays
    coords = np.array([float(p) for p in points.split()][1:]).reshape(-1, 2)
    
    # Calculate bounding box
    min_x, min_y = np.min(coords, axis=0)
    max_x, max_y = np.max(coords, axis=0)
    
    # Calculate center coordinates and dimensions
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    width = max_x - min_x
    height = max_y - min_y
    
    return f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"

def process_directory(input_dir, output_dir):
    """
    Process all label files in a directory
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Counter for tracking
    total_files = 0
    processed_files = 0
    
    # Process each file
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            total_files += 1
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            try:
                # Read original file
                with open(input_path, 'r') as f:
                    lines = f.readlines()
                
                # Convert and write new format
                with open(output_path, 'w') as f:
                    for line in lines:
                        if line.strip():  # Skip empty lines
                            yolo_format = convert_to_yolo_format(line)
                            f.write(yolo_format + '\n')
                
                processed_files += 1
                
                # Print progress
                if processed_files % 100 == 0:
                    print(f"Processed {processed_files}/{total_files} files")
                    
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    print(f"\nCompleted: Processed {processed_files}/{total_files} files")
    return processed_files

def main():
    # Base paths
    base_dir = '/home/ailaty3088@id.sdsu.edu/Ashter/dataset'
    
    # Process each split (train/val/test)
    for split in ['train', 'val', 'test']:
        input_dir = os.path.join(base_dir, split, 'labels')
        output_dir = os.path.join(base_dir, f'{split}_yolo', 'labels')
        
        if os.path.exists(input_dir):
            print(f"\nProcessing {split} set...")
            files_processed = process_directory(input_dir, output_dir)
            print(f"Completed {split} set: {files_processed} files processed")
            
            # Create corresponding images directory and copy/link images
            src_img_dir = os.path.join(base_dir, split, 'images')
            dst_img_dir = os.path.join(base_dir, f'{split}_yolo', 'images')
            if os.path.exists(src_img_dir):
                os.makedirs(dst_img_dir, exist_ok=True)
                # Create symbolic links to images
                for img in os.listdir(src_img_dir):
                    src = os.path.join(src_img_dir, img)
                    dst = os.path.join(dst_img_dir, img)
                    if not os.path.exists(dst):
                        os.symlink(src, dst)
        else:
            print(f"Directory not found: {input_dir}")

if __name__ == "__main__":
    main()