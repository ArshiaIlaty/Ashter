#!/usr/bin/env python3
"""
Hyperparameter Tuning for YOLOv8n Fine-tuned Model
Using Optuna for Bayesian optimization
"""

import os
import optuna
import yaml
import json
from datetime import datetime
from ultralytics import YOLO
import subprocess
import shutil

class HyperparameterTuner:
    def __init__(self, base_model_path, dataset_path, output_dir, n_trials=20):
        self.base_model_path = base_model_path
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.n_trials = n_trials
        self.best_params = None
        self.best_score = 0.0
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def objective(self, trial):
        """Objective function for Optuna optimization"""
        
        # Define hyperparameter search space
        params = {
            'epochs': trial.suggest_int('epochs', 20, 100),
            'batch': trial.suggest_categorical('batch', [8, 16, 32]),
            'imgsz': trial.suggest_categorical('imgsz', [512, 640, 768]),
            'lr0': trial.suggest_float('lr0', 0.0001, 0.01, log=True),
            'lrf': trial.suggest_float('lrf', 0.01, 0.5),
            'momentum': trial.suggest_float('momentum', 0.8, 0.98),
            'weight_decay': trial.suggest_float('weight_decay', 0.0001, 0.001),
            'warmup_epochs': trial.suggest_int('warmup_epochs', 1, 5),
            'box': trial.suggest_float('box', 5.0, 10.0),
            'cls': trial.suggest_float('cls', 0.3, 0.7),
            'dfl': trial.suggest_float('dfl', 1.0, 2.0),
            'patience': trial.suggest_int('patience', 5, 20),
            'hsv_h': trial.suggest_float('hsv_h', 0.0, 0.1),
            'hsv_s': trial.suggest_float('hsv_s', 0.0, 1.0),
            'hsv_v': trial.suggest_float('hsv_v', 0.0, 0.9),
            'degrees': trial.suggest_float('degrees', 0.0, 45.0),
            'translate': trial.suggest_float('translate', 0.0, 0.2),
            'scale': trial.suggest_float('scale', 0.0, 0.9),
            'shear': trial.suggest_float('shear', 0.0, 10.0),
            'perspective': trial.suggest_float('perspective', 0.0, 0.001),
            'flipud': trial.suggest_float('flipud', 0.0, 0.5),
            'mosaic': trial.suggest_float('mosaic', 0.0, 1.0),
            'mixup': trial.suggest_float('mixup', 0.0, 0.3),
            'copy_paste': trial.suggest_float('copy_paste', 0.0, 0.3),
        }
        
        # Create unique trial name
        trial_name = f"trial_{trial.number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        trial_output_dir = os.path.join(self.output_dir, trial_name)
        
        try:
            # Train model with current hyperparameters
            model = YOLO(self.base_model_path)
            
            # Start training
            results = model.train(
                data=os.path.join(self.dataset_path, 'data.yaml'),
                epochs=params['epochs'],
                batch=params['batch'],
                imgsz=params['imgsz'],
                lr0=params['lr0'],
                lrf=params['lrf'],
                momentum=params['momentum'],
                weight_decay=params['weight_decay'],
                warmup_epochs=params['warmup_epochs'],
                box=params['box'],
                cls=params['cls'],
                dfl=params['dfl'],
                patience=params['patience'],
                hsv_h=params['hsv_h'],
                hsv_s=params['hsv_s'],
                hsv_v=params['hsv_v'],
                degrees=params['degrees'],
                translate=params['translate'],
                scale=params['scale'],
                shear=params['shear'],
                perspective=params['perspective'],
                flipud=params['flipud'],
                mosaic=params['mosaic'],
                mixup=params['mixup'],
                copy_paste=params['copy_paste'],
                project=self.output_dir,
                name=trial_name,
                exist_ok=True,
                verbose=False
            )
            
            # Get the best mAP50 score from training
            best_map50 = results.results_dict.get('metrics/mAP50(B)', 0.0)
            
            # Save trial results
            trial_results = {
                'trial_number': trial.number,
                'params': params,
                'best_map50': best_map50,
                'trial_name': trial_name,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save to file
            results_file = os.path.join(trial_output_dir, 'trial_results.json')
            with open(results_file, 'w') as f:
                json.dump(trial_results, f, indent=2)
            
            print(f"Trial {trial.number}: mAP50 = {best_map50:.4f}")
            
            return best_map50
            
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            return 0.0
    
    def run_optimization(self):
        """Run the hyperparameter optimization"""
        print(f"Starting hyperparameter optimization with {self.n_trials} trials...")
        print(f"Base model: {self.base_model_path}")
        print(f"Dataset: {self.dataset_path}")
        print(f"Output directory: {self.output_dir}")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Run optimization
        study.optimize(self.objective, n_trials=self.n_trials)
        
        # Save study results
        self.save_study_results(study)
        
        return study
    
    def save_study_results(self, study):
        """Save comprehensive study results"""
        
        # Best trial info
        best_trial = study.best_trial
        self.best_params = best_trial.params
        self.best_score = best_trial.value
        
        # Create results summary
        results_summary = {
            'optimization_info': {
                'n_trials': self.n_trials,
                'best_score': self.best_score,
                'best_trial_number': best_trial.number,
                'optimization_date': datetime.now().isoformat()
            },
            'best_hyperparameters': self.best_params,
            'all_trials': []
        }
        
        # Collect all trial results
        for trial in study.trials:
            trial_info = {
                'trial_number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name
            }
            results_summary['all_trials'].append(trial_info)
        
        # Save results
        results_file = os.path.join(self.output_dir, 'hyperparameter_optimization_results.json')
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # Create markdown report
        self.create_markdown_report(results_summary)
        
        print(f"\nOptimization completed!")
        print(f"Best mAP50: {self.best_score:.4f}")
        print(f"Best trial: {best_trial.number}")
        print(f"Results saved to: {self.output_dir}")
    
    def create_markdown_report(self, results_summary):
        """Create a markdown report of the optimization results"""
        
        report_content = f"""# Hyperparameter Optimization Report

## Optimization Summary
- **Date**: {results_summary['optimization_info']['optimization_date']}
- **Number of Trials**: {results_summary['optimization_info']['n_trials']}
- **Best mAP50 Score**: {results_summary['optimization_info']['best_score']:.4f}
- **Best Trial Number**: {results_summary['optimization_info']['best_trial_number']}

## Best Hyperparameters

| Parameter | Value |
|-----------|-------|
"""
        
        for param, value in results_summary['best_hyperparameters'].items():
            report_content += f"| {param} | {value} |\n"
        
        report_content += f"""
## Training Command
```bash
yolo detect train \\
  model={self.base_model_path} \\
  data={os.path.join(self.dataset_path, 'data.yaml')} \\
"""
        
        for param, value in results_summary['best_hyperparameters'].items():
            report_content += f"  {param}={value} \\\n"
        
        report_content += """  project=optimized_model \\
  name=best_hyperparameters
```

## Trial Results Summary

| Trial | mAP50 | Status |
|-------|-------|--------|
"""
        
        for trial in results_summary['all_trials']:
            status = "✅" if trial['state'] == 'COMPLETE' else "❌"
            report_content += f"| {trial['trial_number']} | {trial['value']:.4f} | {status} |\n"
        
        # Save report
        report_file = os.path.join(self.output_dir, 'optimization_report.md')
        with open(report_file, 'w') as f:
            f.write(report_content)
    
    def train_best_model(self):
        """Train the final model with the best hyperparameters"""
        if self.best_params is None:
            print("No best parameters found. Run optimization first.")
            return
        
        print(f"\nTraining final model with best hyperparameters...")
        print(f"Expected mAP50: {self.best_score:.4f}")
        
        # Create final model directory
        final_model_dir = os.path.join(self.output_dir, 'best_hyperparameters')
        os.makedirs(final_model_dir, exist_ok=True)
        
        # Train final model
        model = YOLO(self.base_model_path)
        
        results = model.train(
            data=os.path.join(self.dataset_path, 'data.yaml'),
            project=self.output_dir,
            name='best_hyperparameters',
            exist_ok=True,
            **self.best_params
        )
        
        print(f"Final model training completed!")
        print(f"Model saved to: {final_model_dir}")
        
        return results

def main():
    # Configuration
    base_model_path = "/home/ailaty3088@id.sdsu.edu/Ashter/runs/detect/yolov8n_finetune_old_dataset/weights/best.pt"
    dataset_path = "/home/ailaty3088@id.sdsu.edu/Ashter/dataset"
    output_dir = "/home/ailaty3088@id.sdsu.edu/Ashter/hyperparameter_optimization"
    n_trials = 15  # Adjust based on available time
    
    # Initialize tuner
    tuner = HyperparameterTuner(
        base_model_path=base_model_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        n_trials=n_trials
    )
    
    # Run optimization
    study = tuner.run_optimization()
    
    # Train final model with best parameters
    tuner.train_best_model()
    
    print(f"\nHyperparameter optimization completed!")
    print(f"Check results in: {output_dir}")

if __name__ == "__main__":
    main() 