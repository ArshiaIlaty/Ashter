#!/usr/bin/env python3
"""
Comprehensive dataset analysis for the ShitSpotter pet waste detection dataset.
Analyzes distribution differences, annotation quality, and potential issues.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DatasetAnalyzer:
    def __init__(self, dataset_path="shitspotter_dataset"):
        self.dataset_path = Path(dataset_path)
        self.splits = ['train', 'val', 'test']
        self.results = {}
        
    def analyze_dataset_structure(self):
        """Analyze the overall dataset structure and file counts."""
        print("=" * 60)
        print("DATASET STRUCTURE ANALYSIS")
        print("=" * 60)
        
        structure_info = {}
        
        for split in self.splits:
            split_path = self.dataset_path / split
            if not split_path.exists():
                print(f"❌ {split} directory not found!")
                continue
                
            images_path = split_path / 'images'
            labels_path = split_path / 'labels'
            
            if not images_path.exists() or not labels_path.exists():
                print(f"❌ {split}: images or labels directory missing!")
                continue
            
            # Count files
            image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
            label_files = list(labels_path.glob('*.txt'))
            
            structure_info[split] = {
                'images': len(image_files),
                'labels': len(label_files),
                'image_files': image_files,
                'label_files': label_files
            }
            
            print(f"\n📁 {split.upper()} SPLIT:")
            print(f"   • Images: {len(image_files)}")
            print(f"   • Labels: {len(label_files)}")
            print(f"   • Coverage: {len(label_files)/len(image_files)*100:.1f}% of images have labels")
        
        self.results['structure'] = structure_info
        return structure_info
    
    def analyze_annotations(self):
        """Analyze annotation quality and distribution."""
        print(f"\n" + "=" * 60)
        print("ANNOTATION ANALYSIS")
        print("=" * 60)
        
        annotation_stats = {}
        
        for split in self.splits:
            split_path = self.dataset_path / split
            if not split_path.exists():
                continue
                
            labels_path = split_path / 'labels'
            if not labels_path.exists():
                continue
            
            label_files = list(labels_path.glob('*.txt'))
            
            total_annotations = 0
            annotation_sizes = []
            annotation_positions = []
            class_counts = Counter()
            
            for label_file in label_files:
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    for line in lines:
                        if line.strip():
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                width = float(parts[3])
                                height = float(parts[4])
                                
                                total_annotations += 1
                                class_counts[class_id] += 1
                                annotation_sizes.append((width, height))
                                annotation_positions.append((x_center, y_center))
                                
                except Exception as e:
                    print(f"⚠️ Error reading {label_file}: {e}")
            
            annotation_stats[split] = {
                'total_annotations': total_annotations,
                'avg_annotations_per_image': total_annotations / len(label_files) if label_files else 0,
                'class_distribution': dict(class_counts),
                'annotation_sizes': annotation_sizes,
                'annotation_positions': annotation_positions
            }
            
            print(f"\n📊 {split.upper()} ANNOTATIONS:")
            print(f"   • Total annotations: {total_annotations}")
            print(f"   • Avg per image: {total_annotations / len(label_files):.2f}" if label_files else "   • Avg per image: 0")
            print(f"   • Class distribution: {dict(class_counts)}")
        
        self.results['annotations'] = annotation_stats
        return annotation_stats
    
    def analyze_image_characteristics(self):
        """Analyze image characteristics and quality."""
        print(f"\n" + "=" * 60)
        print("IMAGE CHARACTERISTICS ANALYSIS")
        print("=" * 60)
        
        image_stats = {}
        
        for split in self.splits:
            split_path = self.dataset_path / split
            if not split_path.exists():
                continue
                
            images_path = split_path / 'images'
            if not images_path.exists():
                continue
            
            image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
            
            if not image_files:
                continue
            
            # Sample images for analysis (limit to first 50 for speed)
            sample_files = image_files[:50]
            
            image_sizes = []
            aspect_ratios = []
            brightness_values = []
            
            for img_file in sample_files:
                try:
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        height, width = img.shape[:2]
                        image_sizes.append((width, height))
                        aspect_ratios.append(width / height)
                        
                        # Calculate average brightness
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        brightness_values.append(np.mean(gray))
                        
                except Exception as e:
                    print(f"⚠️ Error reading {img_file}: {e}")
            
            if image_sizes:
                avg_width = np.mean([size[0] for size in image_sizes])
                avg_height = np.mean([size[1] for size in image_sizes])
                avg_aspect = np.mean(aspect_ratios)
                avg_brightness = np.mean(brightness_values)
                
                image_stats[split] = {
                    'avg_width': avg_width,
                    'avg_height': avg_height,
                    'avg_aspect_ratio': avg_aspect,
                    'avg_brightness': avg_brightness,
                    'sample_count': len(sample_files)
                }
                
                print(f"\n🖼️ {split.upper()} IMAGES (sample of {len(sample_files)}):")
                print(f"   • Avg size: {avg_width:.0f}x{avg_height:.0f}")
                print(f"   • Avg aspect ratio: {avg_aspect:.2f}")
                print(f"   • Avg brightness: {avg_brightness:.1f}")
        
        self.results['images'] = image_stats
        return image_stats
    
    def check_label_image_matching(self):
        """Check for mismatches between images and labels."""
        print(f"\n" + "=" * 60)
        print("LABEL-IMAGE MATCHING ANALYSIS")
        print("=" * 60)
        
        matching_issues = {}
        
        for split in self.splits:
            split_path = self.dataset_path / split
            if not split_path.exists():
                continue
                
            images_path = split_path / 'images'
            labels_path = split_path / 'labels'
            
            if not images_path.exists() or not labels_path.exists():
                continue
            
            # Fix: Convert generators to lists before union
            jpg_files = list(images_path.glob('*.jpg'))
            png_files = list(images_path.glob('*.png'))
            image_files = set([f.stem for f in jpg_files + png_files])
            label_files = set([f.stem for f in labels_path.glob('*.txt')])
            
            images_without_labels = image_files - label_files
            labels_without_images = label_files - image_files
            
            matching_issues[split] = {
                'images_without_labels': len(images_without_labels),
                'labels_without_images': len(labels_without_images),
                'orphaned_images': list(images_without_labels)[:10],  # First 10
                'orphaned_labels': list(labels_without_images)[:10]   # First 10
            }
            
            print(f"\n🔗 {split.upper()} MATCHING:")
            print(f"   • Images without labels: {len(images_without_labels)}")
            print(f"   • Labels without images: {len(labels_without_images)}")
            
            if images_without_labels:
                print(f"   • Sample orphaned images: {list(images_without_labels)[:5]}")
            if labels_without_images:
                print(f"   • Sample orphaned labels: {list(labels_without_images)[:5]}")
        
        self.results['matching'] = matching_issues
        return matching_issues
    
    def analyze_distribution_differences(self):
        """Analyze differences between train/val/test distributions."""
        print(f"\n" + "=" * 60)
        print("DISTRIBUTION DIFFERENCES ANALYSIS")
        print("=" * 60)
        
        if 'annotations' not in self.results or 'images' not in self.results:
            print("❌ Need to run annotation and image analysis first!")
            return
        
        # Compare annotation densities
        print("\n📊 ANNOTATION DENSITY COMPARISON:")
        for split in self.splits:
            if split in self.results['annotations']:
                total_annotations = self.results['annotations'][split]['total_annotations']
                total_images = self.results['structure'][split]['images']
                density = total_annotations / total_images if total_images > 0 else 0
                print(f"   • {split}: {density:.2f} annotations per image")
        
        # Compare class distributions
        print("\n🏷️ CLASS DISTRIBUTION COMPARISON:")
        class_distributions = {}
        for split in self.splits:
            if split in self.results['annotations']:
                class_dist = self.results['annotations'][split]['class_distribution']
                class_distributions[split] = class_dist
                print(f"   • {split}: {class_dist}")
        
        # Check for distribution shifts
        if 'train' in class_distributions and 'test' in class_distributions:
            train_dist = class_distributions['train']
            test_dist = class_distributions['test']
            
            print("\n⚠️ POTENTIAL DISTRIBUTION SHIFTS:")
            for class_id in set(train_dist.keys()) | set(test_dist.keys()):
                train_count = train_dist.get(class_id, 0)
                test_count = test_dist.get(class_id, 0)
                
                if train_count > 0 and test_count > 0:
                    ratio = test_count / train_count
                    if ratio < 0.5 or ratio > 2.0:
                        print(f"   • Class {class_id}: Train={train_count}, Test={test_count}, Ratio={ratio:.2f}")
    
    def generate_visualizations(self):
        """Generate visualizations of the dataset analysis."""
        print(f"\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('ShitSpotter Dataset Analysis', fontsize=16, fontweight='bold')
        
        # 1. File counts by split
        if 'structure' in self.results:
            splits = list(self.results['structure'].keys())
            image_counts = [self.results['structure'][s]['images'] for s in splits]
            label_counts = [self.results['structure'][s]['labels'] for s in splits]
            
            x = np.arange(len(splits))
            width = 0.35
            
            axes[0, 0].bar(x - width/2, image_counts, width, label='Images', alpha=0.8)
            axes[0, 0].bar(x + width/2, label_counts, width, label='Labels', alpha=0.8)
            axes[0, 0].set_title('File Counts by Split')
            axes[0, 0].set_xlabel('Split')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(splits)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Annotation counts by split
        if 'annotations' in self.results:
            splits = list(self.results['annotations'].keys())
            annotation_counts = [self.results['annotations'][s]['total_annotations'] for s in splits]
            
            axes[0, 1].bar(splits, annotation_counts, alpha=0.8, color='green')
            axes[0, 1].set_title('Total Annotations by Split')
            axes[0, 1].set_xlabel('Split')
            axes[0, 1].set_ylabel('Annotation Count')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Average annotations per image
        if 'annotations' in self.results:
            splits = list(self.results['annotations'].keys())
            avg_annotations = [self.results['annotations'][s]['avg_annotations_per_image'] for s in splits]
            
            axes[0, 2].bar(splits, avg_annotations, alpha=0.8, color='orange')
            axes[0, 2].set_title('Average Annotations per Image')
            axes[0, 2].set_xlabel('Split')
            axes[0, 2].set_ylabel('Average Annotations')
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Class distribution comparison
        if 'annotations' in self.results:
            splits = list(self.results['annotations'].keys())
            class_data = {}
            
            for split in splits:
                class_dist = self.results['annotations'][split]['class_distribution']
                for class_id, count in class_dist.items():
                    if class_id not in class_data:
                        class_data[class_id] = {}
                    class_data[class_id][split] = count
            
            if class_data:
                class_ids = list(class_data.keys())
                x = np.arange(len(class_ids))
                width = 0.8 / len(splits)
                
                for i, split in enumerate(splits):
                    values = [class_data[class_id].get(split, 0) for class_id in class_ids]
                    axes[1, 0].bar(x + i*width, values, width, label=split, alpha=0.8)
                
                axes[1, 0].set_title('Class Distribution by Split')
                axes[1, 0].set_xlabel('Class ID')
                axes[1, 0].set_ylabel('Count')
                axes[1, 0].set_xticks(x + width * (len(splits)-1) / 2)
                axes[1, 0].set_xticklabels(class_ids)
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Image size distribution
        if 'images' in self.results:
            splits = list(self.results['images'].keys())
            widths = [self.results['images'][s]['avg_width'] for s in splits]
            heights = [self.results['images'][s]['avg_height'] for s in splits]
            
            axes[1, 1].scatter(widths, heights, s=100, alpha=0.8)
            for i, split in enumerate(splits):
                axes[1, 1].annotate(split, (widths[i], heights[i]), xytext=(5, 5), 
                                  textcoords='offset points', fontsize=10)
            axes[1, 1].set_title('Average Image Sizes by Split')
            axes[1, 1].set_xlabel('Width (pixels)')
            axes[1, 1].set_ylabel('Height (pixels)')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Annotation size distribution (if available)
        if 'annotations' in self.results and any('annotation_sizes' in self.results['annotations'][s] for s in self.results['annotations']):
            all_sizes = []
            for split in self.results['annotations']:
                if 'annotation_sizes' in self.results['annotations'][split]:
                    sizes = self.results['annotations'][split]['annotation_sizes']
                    if sizes:
                        widths = [size[0] for size in sizes]
                        heights = [size[1] for size in sizes]
                        all_sizes.extend(list(zip(widths, heights)))
            
            if all_sizes:
                # Sample for visualization
                sample_sizes = all_sizes[:1000] if len(all_sizes) > 1000 else all_sizes
                sample_widths = [size[0] for size in sample_sizes]
                sample_heights = [size[1] for size in sample_sizes]
                
                axes[1, 2].scatter(sample_widths, sample_heights, alpha=0.6, s=20)
                axes[1, 2].set_title('Annotation Size Distribution (Sample)')
                axes[1, 2].set_xlabel('Normalized Width')
                axes[1, 2].set_ylabel('Normalized Height')
                axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations saved as 'dataset_analysis.png'")
    
    def generate_summary_report(self):
        """Generate a summary report of findings."""
        print(f"\n" + "=" * 60)
        print("DATASET ANALYSIS SUMMARY")
        print("=" * 60)
        
        # Key findings
        print("\n🔍 KEY FINDINGS:")
        
        # Check for data imbalance
        if 'structure' in self.results:
            train_images = self.results['structure'].get('train', {}).get('images', 0)
            val_images = self.results['structure'].get('val', {}).get('images', 0)
            test_images = self.results['structure'].get('test', {}).get('images', 0)
            
            total_images = train_images + val_images + test_images
            if total_images > 0:
                train_ratio = train_images / total_images
                val_ratio = val_images / total_images
                test_ratio = test_images / total_images
                
                print(f"   • Train/Val/Test split: {train_ratio:.1%}/{val_ratio:.1%}/{test_ratio:.1%}")
                
                if test_ratio > 0.3:
                    print("   ⚠️ Test set is quite large - may indicate data leakage")
                elif test_ratio < 0.1:
                    print("   ⚠️ Test set is very small - may not be representative")
        
        # Check for annotation issues
        if 'matching' in self.results:
            total_orphaned = 0
            for split, issues in self.results['matching'].items():
                total_orphaned += issues['images_without_labels'] + issues['labels_without_images']
            
            if total_orphaned > 0:
                print(f"   ⚠️ Found {total_orphaned} orphaned files (images without labels or vice versa)")
        
        # Check for class imbalance
        if 'annotations' in self.results:
            all_classes = set()
            for split in self.results['annotations']:
                all_classes.update(self.results['annotations'][split]['class_distribution'].keys())
            
            print(f"   • Number of classes: {len(all_classes)}")
            
            # Check class distribution across splits
            for class_id in all_classes:
                class_counts = []
                for split in self.splits:
                    if split in self.results['annotations']:
                        count = self.results['annotations'][split]['class_distribution'].get(class_id, 0)
                        class_counts.append(count)
                
                if len(class_counts) >= 2:
                    max_count = max(class_counts)
                    min_count = min(class_counts)
                    if max_count > 0 and min_count / max_count < 0.1:
                        print(f"   ⚠️ Class {class_id} is highly imbalanced across splits")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if 'matching' in self.results:
            total_orphaned = sum(issues['images_without_labels'] + issues['labels_without_images'] 
                               for issues in self.results['matching'].values())
            if total_orphaned > 0:
                print("   1. Clean up orphaned files (images without labels or vice versa)")
        
        if 'annotations' in self.results:
            print("   2. Review annotation quality and consistency")
            print("   3. Consider rebalancing class distribution if needed")
        
        print("   4. Implement cross-validation to better assess model performance")
        print("   5. Consider data augmentation to address any imbalances")
        
        # Save detailed results
        with open('dataset_analysis_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n📁 Detailed results saved to 'dataset_analysis_results.json'")
        print(f"📁 Visualizations saved to 'dataset_analysis.png'")

def main():
    """Main analysis function."""
    print("🔍 Starting comprehensive dataset analysis...")
    
    # Analyze both datasets
    datasets = {
        "ShitSpotter": "shitspotter_dataset",
        "Previous": "dataset"
    }
    
    all_results = {}
    
    for dataset_name, dataset_path in datasets.items():
        print(f"\n{'='*80}")
        print(f"ANALYZING {dataset_name.upper()} DATASET")
        print(f"{'='*80}")
        
        analyzer = DatasetAnalyzer(dataset_path)
        
        # Run all analyses
        analyzer.analyze_dataset_structure()
        analyzer.analyze_annotations()
        analyzer.analyze_image_characteristics()
        analyzer.check_label_image_matching()
        analyzer.analyze_distribution_differences()
        
        all_results[dataset_name] = analyzer.results
    
    # Generate comparative visualizations
    generate_comparative_visualizations(all_results)
    
    # Generate comparative summary
    generate_comparative_summary(all_results)
    
    print(f"\n✅ Dataset analysis complete for both datasets!")

def generate_comparative_visualizations(all_results):
    """Generate comparative visualizations between datasets."""
    print(f"\n" + "=" * 60)
    print("GENERATING COMPARATIVE VISUALIZATIONS")
    print("=" * 60)
    
    # Create subplots for comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Dataset Comparison: ShitSpotter vs Previous Dataset', fontsize=16, fontweight='bold')
    
    datasets = list(all_results.keys())
    colors = ['blue', 'orange']
    
    # 1. File counts comparison
    if all('structure' in results for results in all_results.values()):
        splits = ['train', 'val', 'test']
        x = np.arange(len(splits))
        width = 0.35
        
        for i, (dataset_name, results) in enumerate(all_results.items()):
            if 'structure' in results:
                image_counts = [results['structure'].get(s, {}).get('images', 0) for s in splits]
                axes[0, 0].bar(x + i*width, image_counts, width, label=f'{dataset_name} Images', alpha=0.8, color=colors[i])
        
        axes[0, 0].set_title('Image Counts by Split')
        axes[0, 0].set_xlabel('Split')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_xticks(x + width/2)
        axes[0, 0].set_xticklabels(splits)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Annotation counts comparison
    if all('annotations' in results for results in all_results.values()):
        for i, (dataset_name, results) in enumerate(all_results.items()):
            if 'annotations' in results:
                splits = list(results['annotations'].keys())
                annotation_counts = [results['annotations'][s]['total_annotations'] for s in splits]
                axes[0, 1].bar([f"{s}_{dataset_name[:3]}" for s in splits], annotation_counts, 
                              alpha=0.8, color=colors[i], label=dataset_name)
        
        axes[0, 1].set_title('Total Annotations by Split')
        axes[0, 1].set_xlabel('Split')
        axes[0, 1].set_ylabel('Annotation Count')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
    
    # 3. Average annotations per image comparison
    if all('annotations' in results for results in all_results.values()):
        for i, (dataset_name, results) in enumerate(all_results.items()):
            if 'annotations' in results:
                splits = list(results['annotations'].keys())
                avg_annotations = [results['annotations'][s]['avg_annotations_per_image'] for s in splits]
                axes[0, 2].bar([f"{s}_{dataset_name[:3]}" for s in splits], avg_annotations, 
                              alpha=0.8, color=colors[i], label=dataset_name)
        
        axes[0, 2].set_title('Average Annotations per Image')
        axes[0, 2].set_xlabel('Split')
        axes[0, 2].set_ylabel('Average Annotations')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        plt.setp(axes[0, 2].xaxis.get_majorticklabels(), rotation=45)
    
    # 4. Image size comparison
    if all('images' in results for results in all_results.values()):
        for i, (dataset_name, results) in enumerate(all_results.items()):
            if 'images' in results:
                splits = list(results['images'].keys())
                widths = [results['images'][s]['avg_width'] for s in splits]
                heights = [results['images'][s]['avg_height'] for s in splits]
                axes[1, 0].scatter(widths, heights, s=100, alpha=0.8, color=colors[i], label=dataset_name)
                for j, split in enumerate(splits):
                    axes[1, 0].annotate(f"{split}_{dataset_name[:3]}", (widths[j], heights[j]), 
                                      xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        axes[1, 0].set_title('Average Image Sizes by Split')
        axes[1, 0].set_xlabel('Width (pixels)')
        axes[1, 0].set_ylabel('Height (pixels)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Brightness comparison
    if all('images' in results for results in all_results.values()):
        for i, (dataset_name, results) in enumerate(all_results.items()):
            if 'images' in results:
                splits = list(results['images'].keys())
                brightness_values = [results['images'][s]['avg_brightness'] for s in splits]
                axes[1, 1].bar([f"{s}_{dataset_name[:3]}" for s in splits], brightness_values, 
                              alpha=0.8, color=colors[i], label=dataset_name)
        
        axes[1, 1].set_title('Average Brightness by Split')
        axes[1, 1].set_xlabel('Split')
        axes[1, 1].set_ylabel('Brightness')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
    
    # 6. Dataset size comparison
    if all('structure' in results for results in all_results.values()):
        dataset_sizes = []
        dataset_names = []
        
        for dataset_name, results in all_results.items():
            if 'structure' in results:
                total_images = sum(results['structure'][s]['images'] for s in results['structure'])
                dataset_sizes.append(total_images)
                dataset_names.append(dataset_name)
        
        axes[1, 2].pie(dataset_sizes, labels=dataset_names, autopct='%1.1f%%', startangle=90)
        axes[1, 2].set_title('Total Dataset Size Comparison')
    
    plt.tight_layout()
    plt.savefig('dataset_comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Comparative visualizations saved as 'dataset_comparison_analysis.png'")

def generate_comparative_summary(all_results):
    """Generate a comparative summary of both datasets."""
    print(f"\n" + "=" * 60)
    print("COMPARATIVE DATASET ANALYSIS SUMMARY")
    print("=" * 60)
    
    print("\n🔍 KEY DIFFERENCES BETWEEN DATASETS:")
    
    # Compare dataset sizes
    for dataset_name, results in all_results.items():
        if 'structure' in results:
            total_images = sum(results['structure'][s]['images'] for s in results['structure'])
            print(f"   • {dataset_name}: {total_images} total images")
    
    # Compare annotation densities
    print("\n📊 ANNOTATION DENSITY COMPARISON:")
    for dataset_name, results in all_results.items():
        if 'annotations' in results:
            for split in results['annotations']:
                total_annotations = results['annotations'][split]['total_annotations']
                total_images = results['structure'][split]['images']
                density = total_annotations / total_images if total_images > 0 else 0
                print(f"   • {dataset_name} {split}: {density:.2f} annotations per image")
    
    # Compare test set characteristics
    print("\n🧪 TEST SET CHARACTERISTICS:")
    for dataset_name, results in all_results.items():
        if 'structure' in results and 'test' in results['structure']:
            test_images = results['structure']['test']['images']
            test_annotations = results['annotations']['test']['total_annotations'] if 'annotations' in results and 'test' in results['annotations'] else 0
            test_density = test_annotations / test_images if test_images > 0 else 0
            print(f"   • {dataset_name}: {test_images} images, {test_annotations} annotations, {test_density:.2f} density")
    
    # Identify potential issues
    print("\n⚠️ POTENTIAL ISSUES IDENTIFIED:")
    
    # Check for test set size differences
    test_sizes = {}
    for dataset_name, results in all_results.items():
        if 'structure' in results and 'test' in results['structure']:
            test_sizes[dataset_name] = results['structure']['test']['images']
    
    if len(test_sizes) > 1:
        min_test = min(test_sizes.values())
        max_test = max(test_sizes.values())
        if max_test / min_test > 5:
            print(f"   • Large difference in test set sizes: {min_test} vs {max_test} images")
    
    # Check for annotation density differences
    for dataset_name, results in all_results.items():
        if 'annotations' in results and 'test' in results['annotations']:
            test_density = results['annotations']['test']['avg_annotations_per_image']
            train_density = results['annotations']['train']['avg_annotations_per_image'] if 'train' in results['annotations'] else 0
            
            if test_density > train_density * 2:
                print(f"   • {dataset_name}: Test set has much higher annotation density than train set")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("   1. Consider using the larger dataset for training")
    print("   2. Resample test sets to have similar characteristics")
    print("   3. Implement cross-validation across both datasets")
    print("   4. Consider ensemble methods using models trained on both datasets")
    
    # Save comparative results
    with open('dataset_comparison_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n📁 Comparative results saved to 'dataset_comparison_results.json'")
    print(f"📁 Comparative visualizations saved to 'dataset_comparison_analysis.png'")

if __name__ == "__main__":
    main() 