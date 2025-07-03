#!/usr/bin/env python3
"""
Monitor hyperparameter optimization progress
"""

import os
import json
import glob
from datetime import datetime

def monitor_optimization():
    output_dir = "/home/ailaty3088@id.sdsu.edu/Ashter/hyperparameter_optimization"
    
    if not os.path.exists(output_dir):
        print("Optimization directory not found. Has the optimization started?")
        return
    
    # Check for completed trials
    trial_dirs = glob.glob(os.path.join(output_dir, "trial_*"))
    completed_trials = []
    
    for trial_dir in trial_dirs:
        results_file = os.path.join(trial_dir, "trial_results.json")
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
                completed_trials.append(results)
    
    # Sort by trial number
    completed_trials.sort(key=lambda x: x['trial_number'])
    
    print(f"Hyperparameter Optimization Progress")
    print(f"===================================")
    print(f"Output directory: {output_dir}")
    print(f"Completed trials: {len(completed_trials)}/15")
    print(f"Progress: {len(completed_trials)/15*100:.1f}%")
    print()
    
    if completed_trials:
        print("Completed Trials:")
        print("Trial | mAP50  | Status")
        print("------|--------|--------")
        
        best_score = 0
        best_trial = None
        
        for trial in completed_trials:
            score = trial['best_map50']
            if score > best_score:
                best_score = score
                best_trial = trial['trial_number']
            
            print(f"{trial['trial_number']:5d} | {score:.4f} | ✅")
        
        print()
        print(f"Best result so far: Trial {best_trial} with mAP50 = {best_score:.4f}")
        
        # Show best parameters
        if best_trial is not None:
            best_trial_data = next(t for t in completed_trials if t['trial_number'] == best_trial)
            print(f"\nBest hyperparameters (Trial {best_trial}):")
            for param, value in best_trial_data['params'].items():
                print(f"  {param}: {value}")
    
    # Check for final results
    final_results_file = os.path.join(output_dir, "hyperparameter_optimization_results.json")
    if os.path.exists(final_results_file):
        print(f"\n🎉 Optimization completed! Check {final_results_file} for final results.")
    
    # Check for running trials
    running_trials = []
    for trial_dir in trial_dirs:
        if not os.path.exists(os.path.join(trial_dir, "trial_results.json")):
            # Check if training is in progress
            if os.path.exists(os.path.join(trial_dir, "weights")):
                running_trials.append(os.path.basename(trial_dir))
    
    if running_trials:
        print(f"\nCurrently running: {', '.join(running_trials)}")

if __name__ == "__main__":
    monitor_optimization() 