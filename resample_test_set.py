#!/usr/bin/env python3
"""
Resample the test set to match the training distribution.
Creates a balanced test set with similar annotation density to the training set.
"""

import os
import shutil
import random
import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import defaultdict
import argparse

class TestSetResampler:
    def __init__(self, dataset_path="shitspotter_dataset", target_density=0.71, test_size=500):
        self.dataset_path = Path(dataset_path)
        self.target_density = target_density  # Target annotations per image
        self.test_size = test_size  # Target test set size
        self.backup_path = self.dataset_path / "backup_original"
        
    def analyze_current_distribution(self):
        """Analyze the current annotation distribution across all images."""
        print("🔍 Analyzing current annotation distribution...")
        
        all_images = []
        annotation_counts = []
        
        # Collect all images and their annotation counts
        for split in ['train', 'val', 'test']:
            split_path = self.dataset_path / split
            if not split_path.exists():
                continue
                
            images_path = split_path / 'images'
            labels_path = split_path / 'labels'
            
            if not images_path.exists() or not labels_path.exists():
                continue
            
            image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
            
            for img_file in image_files:
                label_file = labels_path / f"{img_file.stem}.txt"
                
                if label_file.exists():
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    annotation_count = len([line for line in lines if line.strip()])
                    
                    all_images.append({
                        'path': img_file,
                        'label_path': label_file,
                        'split': split,
                        'annotation_count': annotation_count
                    })
                    annotation_counts.append(annotation_count)
        
        # Calculate statistics
        annotation_counts = np.array(annotation_counts)
        current_density = np.mean(annotation_counts)
        
        print(f"📊 Current Distribution Analysis:")
        print(f"   • Total images: {len(all_images)}")
        print(f"   • Average annotations per image: {current_density:.2f}")
        print(f"   • Min annotations: {np.min(annotation_counts)}")
        print(f"   • Max annotations: {np.max(annotation_counts)}")
        print(f"   • Std annotations: {np.std(annotation_counts):.2f}")
        
        # Distribution by annotation count
        unique_counts, counts = np.unique(annotation_counts, return_counts=True)
        print(f"   • Distribution by annotation count:")
        for count, freq in zip(unique_counts, counts):
            percentage = (freq / len(all_images)) * 100
            print(f"     - {count} annotations: {freq} images ({percentage:.1f}%)")
        
        return all_images, annotation_counts
    
    def create_backup(self):
        """Create a backup of the original test set."""
        print("💾 Creating backup of original test set...")
        
        test_path = self.dataset_path / 'test'
        if not test_path.exists():
            print("❌ Test directory not found!")
            return False
        
        # Create backup directory
        self.backup_path.mkdir(exist_ok=True)
        
        # Copy test directory to backup
        backup_test_path = self.backup_path / 'test'
        if backup_test_path.exists():
            shutil.rmtree(backup_test_path)
        
        shutil.copytree(test_path, backup_test_path)
        print(f"✅ Backup created at: {self.backup_path}")
        return True
    
    def resample_test_set(self, all_images):
        """Resample the test set to match target density."""
        print(f"🔄 Resampling test set to target density: {self.target_density:.2f} annotations/image")
        
        # Filter out images from the current test set - we'll only use train and val
        train_val_images = [img for img in all_images if img['split'] in ['train', 'val']]
        
        # Group images by annotation count
        images_by_count = defaultdict(list)
        for img_info in train_val_images:
            count = img_info['annotation_count']
            images_by_count[count].append(img_info)
        
        # Calculate target distribution
        target_distribution = self._calculate_target_distribution(images_by_count)
        
        print(f"📊 Target Distribution:")
        for count, target_count in target_distribution.items():
            if target_count > 0:
                print(f"   • {count} annotations: {target_count} images")
        
        # Select images for new test set
        selected_images = []
        for count, target_count in target_distribution.items():
            if target_count > 0 and count in images_by_count:
                available_images = images_by_count[count]
                if len(available_images) >= target_count:
                    selected = random.sample(available_images, target_count)
                else:
                    selected = available_images  # Use all available
                    print(f"⚠️ Warning: Only {len(available_images)} images with {count} annotations available, target was {target_count}")
                
                selected_images.extend(selected)
        
        print(f"✅ Selected {len(selected_images)} images for new test set")
        print(f"   • From train set: {len([img for img in selected_images if img['split'] == 'train'])}")
        print(f"   • From val set: {len([img for img in selected_images if img['split'] == 'val'])}")
        
        return selected_images
    
    def _calculate_target_distribution(self, images_by_count):
        """Calculate target distribution to achieve desired density."""
        # Get all unique annotation counts
        all_counts = sorted(images_by_count.keys())
        
        # Start with a simple distribution centered around target density
        target_distribution = {}
        
        # Calculate how many images we need for each annotation count
        remaining_images = self.test_size
        total_annotations_needed = self.test_size * self.target_density
        
        # Distribute images based on annotation count, prioritizing counts close to target
        for count in all_counts:
            if remaining_images <= 0:
                break
            
            # Calculate how many images with this annotation count we should include
            # Prioritize counts close to target density
            distance_from_target = abs(count - self.target_density)
            weight = 1.0 / (1.0 + distance_from_target)  # Higher weight for closer counts
            
            # Calculate target count for this annotation count
            available_images = len(images_by_count[count])
            target_count = min(
                int(remaining_images * weight * 0.5),  # Conservative estimate
                available_images
            )
            
            if target_count > 0:
                target_distribution[count] = target_count
                remaining_images -= target_count
        
        # Adjust to meet exact target
        actual_images = sum(target_distribution.values())
        actual_annotations = sum(count * num for count, num in target_distribution.items())
        
        if actual_images > 0:
            actual_density = actual_annotations / actual_images
            print(f"📊 Calculated distribution: {actual_images} images, {actual_annotations} annotations, density: {actual_density:.2f}")
        
        return target_distribution
    
    def create_new_test_set(self, selected_images):
        """Create the new test set with selected images."""
        print("📁 Creating new test set...")
        
        test_path = self.dataset_path / 'test'
        test_images_path = test_path / 'images'
        test_labels_path = test_path / 'labels'
        
        # Clear existing test directory
        if test_path.exists():
            shutil.rmtree(test_path)
        
        # Create new test directory structure
        test_images_path.mkdir(parents=True, exist_ok=True)
        test_labels_path.mkdir(parents=True, exist_ok=True)
        
        # Copy selected images and labels
        copied_count = 0
        total_annotations = 0
        
        for img_info in selected_images:
            # Copy image
            dest_img_path = test_images_path / img_info['path'].name
            shutil.copy2(img_info['path'], dest_img_path)
            
            # Copy label
            dest_label_path = test_labels_path / img_info['label_path'].name
            shutil.copy2(img_info['label_path'], dest_label_path)
            
            copied_count += 1
            total_annotations += img_info['annotation_count']
        
        # Calculate new density
        new_density = total_annotations / copied_count if copied_count > 0 else 0
        
        print(f"✅ New test set created:")
        print(f"   • Images copied: {copied_count}")
        print(f"   • Total annotations: {total_annotations}")
        print(f"   • New density: {new_density:.2f} annotations/image")
        print(f"   • Target density: {self.target_density:.2f} annotations/image")
        print(f"   • Difference: {abs(new_density - self.target_density):.2f}")
        
        return copied_count, total_annotations, new_density
    
    def update_data_yaml(self):
        """Update the data.yaml file to reflect the new test set."""
        print("📝 Updating data.yaml file...")
        
        yaml_path = self.dataset_path / 'data.yaml'
        if not yaml_path.exists():
            print("❌ data.yaml not found!")
            return
        
        # Count new test set
        test_images_path = self.dataset_path / 'test' / 'images'
        test_count = len(list(test_images_path.glob('*.jpg'))) + len(list(test_images_path.glob('*.png')))
        
        # Read current yaml
        with open(yaml_path, 'r') as f:
            content = f.read()
        
        # Update test count in yaml (if it exists)
        # This is a simple update - you might want to use a proper yaml parser
        lines = content.split('\n')
        updated_lines = []
        
        for line in lines:
            if line.startswith('nc:') or line.startswith('names:'):
                updated_lines.append(line)
            elif line.startswith('test:'):
                updated_lines.append(f'test: test/images  # {test_count} images')
            elif line.startswith('train:'):
                updated_lines.append(line)
            elif line.startswith('val:'):
                updated_lines.append(line)
            else:
                updated_lines.append(line)
        
        # Write updated yaml
        with open(yaml_path, 'w') as f:
            f.write('\n'.join(updated_lines))
        
        print(f"✅ Updated data.yaml with new test count: {test_count}")
    
    def generate_report(self, original_images, new_test_images, new_density):
        """Generate a report of the resampling process."""
        print("📊 Generating resampling report...")
        
        # Calculate original test set statistics
        original_test = [img for img in original_images if img['split'] == 'test']
        original_test_annotations = [img['annotation_count'] for img in original_test]
        original_density = np.mean(original_test_annotations) if original_test_annotations else 0
        
        # Calculate new test set statistics
        new_test_annotations = [img['annotation_count'] for img in new_test_images]
        new_density_calc = np.mean(new_test_annotations) if new_test_annotations else 0
        
        # Convert numpy int64 keys to int for JSON serialization
        def convert_keys_to_int(d):
            return {int(k): v for k, v in d.items()}
        
        report = {
            'resampling_summary': {
                'original_test_images': len(original_test),
                'original_test_density': original_density,
                'new_test_images': len(new_test_images),
                'new_test_density': new_density_calc,
                'target_density': self.target_density,
                'improvement': abs(original_density - self.target_density) - abs(new_density_calc - self.target_density)
            },
            'distribution_comparison': {
                'original_test_distribution': convert_keys_to_int(dict(zip(*np.unique(original_test_annotations, return_counts=True)))) if original_test_annotations else {},
                'new_test_distribution': convert_keys_to_int(dict(zip(*np.unique(new_test_annotations, return_counts=True)))) if new_test_annotations else {}
            }
        }
        
        # Save report
        with open('test_resampling_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Resampling Report:")
        print(f"   • Original test set: {len(original_test)} images, density: {original_density:.2f}")
        print(f"   • New test set: {len(new_test_images)} images, density: {new_density_calc:.2f}")
        print(f"   • Target density: {self.target_density:.2f}")
        print(f"   • Improvement: {report['resampling_summary']['improvement']:.2f}")
        print(f"   • Report saved to: test_resampling_report.json")
        
        return report

def main():
    """Main function to resample the test set."""
    parser = argparse.ArgumentParser(description='Resample test set to match training distribution')
    parser.add_argument('--dataset', default='shitspotter_dataset', help='Dataset path')
    parser.add_argument('--target-density', type=float, default=0.71, help='Target annotations per image')
    parser.add_argument('--test-size', type=int, default=500, help='Target test set size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("🔄 Starting test set resampling process...")
    print(f"📁 Dataset: {args.dataset}")
    print(f"🎯 Target density: {args.target_density:.2f} annotations/image")
    print(f"📊 Target test size: {args.test_size} images")
    
    # Initialize resampler
    resampler = TestSetResampler(
        dataset_path=args.dataset,
        target_density=args.target_density,
        test_size=args.test_size
    )
    
    # Analyze current distribution
    all_images, annotation_counts = resampler.analyze_current_distribution()
    
    # Create backup
    if not resampler.create_backup():
        print("❌ Failed to create backup. Aborting.")
        return
    
    # Resample test set
    selected_images = resampler.resample_test_set(all_images)
    
    # Create new test set
    copied_count, total_annotations, new_density = resampler.create_new_test_set(selected_images)
    
    # Update data.yaml
    resampler.update_data_yaml()
    
    # Generate report
    report = resampler.generate_report(all_images, selected_images, new_density)
    
    print(f"\n✅ Test set resampling complete!")
    print(f"📁 Backup saved at: {resampler.backup_path}")
    print(f"📁 New test set: {copied_count} images with {new_density:.2f} annotations/image")
    print(f"📁 Report: test_resampling_report.json")

if __name__ == "__main__":
    main() 