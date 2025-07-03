#!/usr/bin/env python3
"""
Train smaller YOLO models (YOLOv8n and YOLOv8s) on the ShitSpotter dataset
and compare their performance with the existing YOLOv8m model.
"""

import os
import subprocess
import time
import json
from pathlib import Path
import argparse

class SmallerModelTrainer:
    def __init__(self, dataset_path="shitspotter_dataset", epochs=30, batch_size=16):
        self.dataset_path = Path(dataset_path)
        self.epochs = epochs
        self.batch_size = batch_size
        self.models = ['yolov8n.pt', 'yolov8s.pt']
        self.results = {}
        
    def train_model(self, model_name, model_path):
        """Train a specific YOLO model."""
        print(f"\n{'='*60}")
        print(f"TRAINING {model_name.upper()}")
        print(f"{'='*60}")
        
        # Create output directory name
        output_name = f"shitspotter_{model_name.replace('.pt', '')}"
        
        # Build training command
        cmd = [
            'yolo', 'train',
            'model=' + model_path,
            f'data={self.dataset_path}/data.yaml',
            f'epochs={self.epochs}',
            f'batch={self.batch_size}',
            'imgsz=640',
            'device=0',
            'workers=8',
            'project=runs/detect',
            f'name={output_name}',
            'exist_ok=true',
            'pretrained=true',
            'optimizer=auto',
            'verbose=true',
            'seed=42',
            'deterministic=true',
            'single_cls=false',
            'cos_lr=false',
            'close_mosaic=10',
            'amp=true',
            'plots=true',
            'save_period=5',
            'patience=10'
        ]
        
        print(f"🚀 Starting training for {model_name}...")
        print(f"📁 Output directory: runs/detect/{output_name}")
        print(f"⏱️ Expected duration: ~{self.epochs * 2} minutes")
        
        # Record start time
        start_time = time.time()
        
        try:
            # Run training command
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Calculate training time
            training_time = time.time() - start_time
            
            print(f"✅ Training completed successfully!")
            print(f"⏱️ Training time: {training_time/60:.1f} minutes")
            
            # Get the output directory
            output_dir = Path(f"runs/detect/{output_name}")
            
            return {
                'model_name': model_name,
                'output_dir': str(output_dir),
                'training_time': training_time,
                'success': True,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed for {model_name}")
            print(f"Error: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            
            return {
                'model_name': model_name,
                'output_dir': None,
                'training_time': time.time() - start_time,
                'success': False,
                'stdout': e.stdout,
                'stderr': e.stderr
            }
    
    def evaluate_model(self, model_path, model_name):
        """Evaluate a trained model on the test set."""
        print(f"\n🔍 Evaluating {model_name}...")
        
        # Build evaluation command
        cmd = [
            'yolo', 'val',
            f'model={model_path}',
            f'data={self.dataset_path}/data.yaml',
            'split=test',
            'conf=0.001',
            'iou=0.6',
            'max_det=300',
            'save_json=true',
            'save_txt=true',
            'save_conf=true',
            'project=runs/val',
            f'name={model_name}_evaluation',
            'exist_ok=true'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse results from stdout
            evaluation_results = self._parse_evaluation_output(result.stdout)
            
            print(f"✅ Evaluation completed for {model_name}")
            
            return {
                'model_name': model_name,
                'success': True,
                'results': evaluation_results,
                'stdout': result.stdout
            }
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Evaluation failed for {model_name}")
            print(f"Error: {e}")
            
            return {
                'model_name': model_name,
                'success': False,
                'results': None,
                'stdout': e.stdout,
                'stderr': e.stderr
            }
    
    def _parse_evaluation_output(self, stdout):
        """Parse evaluation results from YOLO output."""
        results = {}
        
        # Look for key metrics in the output
        lines = stdout.split('\n')
        for line in lines:
            if 'mAP50' in line and 'all' in line:
                try:
                    # Extract mAP50 value
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'mAP50' and i + 1 < len(parts):
                            results['mAP50'] = float(parts[i + 1])
                            break
                except:
                    pass
            
            elif 'mAP50-95' in line and 'all' in line:
                try:
                    # Extract mAP50-95 value
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'mAP50-95' and i + 1 < len(parts):
                            results['mAP50-95'] = float(parts[i + 1])
                            break
                except:
                    pass
            
            elif 'precision' in line.lower() and 'all' in line:
                try:
                    # Extract precision value
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.lower() == 'precision' and i + 1 < len(parts):
                            results['precision'] = float(parts[i + 1])
                            break
                except:
                    pass
            
            elif 'recall' in line.lower() and 'all' in line:
                try:
                    # Extract recall value
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.lower() == 'recall' and i + 1 < len(parts):
                            results['recall'] = float(parts[i + 1])
                            break
                except:
                    pass
        
        return results
    
    def run_custom_evaluation(self, model_path, model_name):
        """Run custom evaluation using the existing evaluation script."""
        print(f"\n🧪 Running custom evaluation for {model_name}...")
        
        # Temporarily update the model path in the evaluation script
        evaluation_script = "evaluate_on_shitspotter.py"
        
        if not Path(evaluation_script).exists():
            print(f"❌ Evaluation script {evaluation_script} not found!")
            return None
        
        # Read the evaluation script
        with open(evaluation_script, 'r') as f:
            content = f.read()
        
        # Create a temporary script with the new model path
        temp_script = f"temp_eval_{model_name.replace('.pt', '')}.py"
        
        # Replace the model path in the script
        modified_content = content.replace(
            "model_path = 'runs/detect/shitspotter_finetune/weights/best.pt'",
            f"model_path = '{model_path}'"
        )
        
        # Write temporary script
        with open(temp_script, 'w') as f:
            f.write(modified_content)
        
        try:
            # Run the temporary evaluation script
            result = subprocess.run(['python', temp_script], capture_output=True, text=True, check=True)
            
            # Parse the results from stdout
            custom_results = self._parse_custom_evaluation_output(result.stdout)
            
            print(f"✅ Custom evaluation completed for {model_name}")
            
            # Clean up temporary script
            os.remove(temp_script)
            
            return custom_results
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Custom evaluation failed for {model_name}")
            print(f"Error: {e}")
            
            # Clean up temporary script
            if Path(temp_script).exists():
                os.remove(temp_script)
            
            return None
    
    def _parse_custom_evaluation_output(self, stdout):
        """Parse custom evaluation results."""
        results = {}
        
        lines = stdout.split('\n')
        for line in lines:
            if 'Precision:' in line:
                try:
                    results['precision'] = float(line.split(':')[1].strip())
                except:
                    pass
            elif 'Recall:' in line:
                try:
                    results['recall'] = float(line.split(':')[1].strip())
                except:
                    pass
            elif 'F1-Score:' in line:
                try:
                    results['f1_score'] = float(line.split(':')[1].strip())
                except:
                    pass
            elif 'Total Detections:' in line:
                try:
                    results['total_detections'] = int(line.split(':')[1].strip())
                except:
                    pass
            elif 'Total Ground Truth:' in line:
                try:
                    results['total_ground_truth'] = int(line.split(':')[1].strip())
                except:
                    pass
            elif 'True Positives:' in line:
                try:
                    results['true_positives'] = int(line.split(':')[1].strip())
                except:
                    pass
            elif 'False Positives:' in line:
                try:
                    results['false_positives'] = int(line.split(':')[1].strip())
                except:
                    pass
            elif 'False Negatives:' in line:
                try:
                    results['false_negatives'] = int(line.split(':')[1].strip())
                except:
                    pass
        
        return results
    
    def train_all_models(self):
        """Train all smaller models."""
        print("🚀 Starting training of smaller YOLO models...")
        print(f"📁 Dataset: {self.dataset_path}")
        print(f"⏱️ Epochs: {self.epochs}")
        print(f"📦 Batch size: {self.batch_size}")
        
        all_results = {}
        
        for model_name in self.models:
            print(f"\n{'='*80}")
            print(f"PROCESSING MODEL: {model_name}")
            print(f"{'='*80}")
            
            # Train the model
            training_result = self.train_model(model_name, model_name)
            all_results[model_name] = training_result
            
            if training_result['success']:
                # Evaluate the model
                best_model_path = f"{training_result['output_dir']}/weights/best.pt"
                
                # Run YOLO validation
                yolo_eval_result = self.evaluate_model(best_model_path, model_name)
                all_results[model_name]['yolo_evaluation'] = yolo_eval_result
                
                # Run custom evaluation
                custom_eval_result = self.run_custom_evaluation(best_model_path, model_name)
                all_results[model_name]['custom_evaluation'] = custom_eval_result
        
        return all_results
    
    def generate_comparison_report(self, results):
        """Generate a comparison report of all models."""
        print(f"\n{'='*60}")
        print("GENERATING MODEL COMPARISON REPORT")
        print(f"{'='*60}")
        
        # Create comparison table
        comparison_data = []
        
        for model_name, result in results.items():
            if result['success']:
                row = {
                    'model': model_name,
                    'training_time_min': result['training_time'] / 60,
                    'output_dir': result['output_dir']
                }
                
                # Add YOLO evaluation results
                if 'yolo_evaluation' in result and result['yolo_evaluation']['success']:
                    yolo_results = result['yolo_evaluation']['results']
                    row.update({
                        'mAP50': yolo_results.get('mAP50', 'N/A'),
                        'mAP50-95': yolo_results.get('mAP50-95', 'N/A'),
                        'precision_yolo': yolo_results.get('precision', 'N/A'),
                        'recall_yolo': yolo_results.get('recall', 'N/A')
                    })
                
                # Add custom evaluation results
                if 'custom_evaluation' in result and result['custom_evaluation']:
                    custom_results = result['custom_evaluation']
                    row.update({
                        'precision_custom': custom_results.get('precision', 'N/A'),
                        'recall_custom': custom_results.get('recall', 'N/A'),
                        'f1_score': custom_results.get('f1_score', 'N/A'),
                        'total_detections': custom_results.get('total_detections', 'N/A'),
                        'true_positives': custom_results.get('true_positives', 'N/A'),
                        'false_positives': custom_results.get('false_positives', 'N/A'),
                        'false_negatives': custom_results.get('false_negatives', 'N/A')
                    })
                
                comparison_data.append(row)
        
        # Print comparison table
        print("\n📊 MODEL COMPARISON RESULTS")
        print("-" * 120)
        
        if comparison_data:
            # Print header
            headers = ['Model', 'Time(min)', 'mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1-Score']
            header_str = " | ".join(f"{h:>12}" for h in headers)
            print(header_str)
            print("-" * len(header_str))
            
            # Print data
            for row in comparison_data:
                precision = row.get('precision_custom', row.get('precision_yolo', 'N/A'))
                recall = row.get('recall_custom', row.get('recall_yolo', 'N/A'))
                
                data_str = " | ".join([
                    f"{row['model']:>12}",
                    f"{row['training_time_min']:>12.1f}",
                    f"{row.get('mAP50', 'N/A'):>12}",
                    f"{row.get('mAP50-95', 'N/A'):>12}",
                    f"{precision:>12}",
                    f"{recall:>12}",
                    f"{row.get('f1_score', 'N/A'):>12}"
                ])
                print(data_str)
        
        # Save detailed results
        with open('smaller_models_comparison.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📁 Detailed results saved to: smaller_models_comparison.json")
        
        # Generate recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if comparison_data:
            # Find best performing model
            best_f1 = 0
            best_model = None
            
            for row in comparison_data:
                f1_score = row.get('f1_score', 0)
                if f1_score != 'N/A' and f1_score > best_f1:
                    best_f1 = f1_score
                    best_model = row['model']
            
            if best_model:
                print(f"   • Best performing model: {best_model} (F1-Score: {best_f1:.4f})")
            
            # Compare with YOLOv8m
            print(f"   • Compare these results with YOLOv8m performance")
            print(f"   • Consider ensemble methods combining multiple models")
            print(f"   • Optimize confidence thresholds for best performing model")
        
        return comparison_data

def main():
    """Main function to train and compare smaller models."""
    parser = argparse.ArgumentParser(description='Train smaller YOLO models and compare performance')
    parser.add_argument('--dataset', default='shitspotter_dataset', help='Dataset path')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    
    args = parser.parse_args()
    
    print("🚀 Starting smaller model training and comparison...")
    print(f"📁 Dataset: {args.dataset}")
    print(f"⏱️ Epochs: {args.epochs}")
    print(f"📦 Batch size: {args.batch_size}")
    
    # Initialize trainer
    trainer = SmallerModelTrainer(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Train all models
    results = trainer.train_all_models()
    
    # Generate comparison report
    comparison_data = trainer.generate_comparison_report(results)
    
    print(f"\n✅ Smaller model training and comparison complete!")
    print(f"📁 Results saved to: smaller_models_comparison.json")

if __name__ == "__main__":
    main() 