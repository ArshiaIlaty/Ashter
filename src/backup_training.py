import os
import shutil
from datetime import datetime
import yaml
import json

class TrainingBackup:
    def __init__(self, base_dir='src'):
        self.base_dir = base_dir
        self.backup_dir = os.path.join(base_dir, 'backups')
        self.weights_dir = os.path.join(self.backup_dir, 'weights')
        self.results_dir = os.path.join(self.backup_dir, 'results')
        self.configs_dir = os.path.join(self.backup_dir, 'configs')
        
        # Create backup directories if they don't exist
        for dir_path in [self.weights_dir, self.results_dir, self.configs_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def create_backup(self, run_name=None):
        """Create a backup of the current training run"""
        if run_name is None:
            run_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        # Create run-specific directories
        run_weights_dir = os.path.join(self.weights_dir, run_name)
        run_results_dir = os.path.join(self.results_dir, run_name)
        run_configs_dir = os.path.join(self.configs_dir, run_name)
        
        os.makedirs(run_weights_dir, exist_ok=True)
        os.makedirs(run_results_dir, exist_ok=True)
        os.makedirs(run_configs_dir, exist_ok=True)
        
        # Backup model weights
        weights_path = os.path.join(self.base_dir, 'runs/train/pet_waste_detector/weights')
        if os.path.exists(weights_path):
            for file in os.listdir(weights_path):
                if file.endswith('.pt'):  # Backup all model weight files
                    src = os.path.join(weights_path, file)
                    dst = os.path.join(run_weights_dir, file)
                    shutil.copy2(src, dst)
        
        # Backup training results
        results_path = os.path.join(self.base_dir, 'runs/train/pet_waste_detector')
        if os.path.exists(results_path):
            for file in os.listdir(results_path):
                if file.endswith(('.csv', '.png', '.json')):  # Backup results files
                    src = os.path.join(results_path, file)
                    dst = os.path.join(run_results_dir, file)
                    shutil.copy2(src, dst)
        
        # Backup configurations
        configs = {
            'data_yaml': os.path.join(self.base_dir, 'dataset/data.yaml'),
            'training_config': os.path.join(self.base_dir, 'train_waste_detector.py')
        }
        
        for config_name, config_path in configs.items():
            if os.path.exists(config_path):
                dst = os.path.join(run_configs_dir, os.path.basename(config_path))
                shutil.copy2(config_path, dst)
        
        # Create backup metadata
        metadata = {
            'backup_time': datetime.now().isoformat(),
            'run_name': run_name,
            'backup_contents': {
                'weights': os.listdir(run_weights_dir) if os.path.exists(run_weights_dir) else [],
                'results': os.listdir(run_results_dir) if os.path.exists(run_results_dir) else [],
                'configs': os.listdir(run_configs_dir) if os.path.exists(run_configs_dir) else []
            }
        }
        
        # Save metadata
        metadata_path = os.path.join(self.backup_dir, f"{run_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"Backup created successfully: {run_name}")
        return run_name
    
    def list_backups(self):
        """List all available backups"""
        backups = []
        for run_name in os.listdir(self.weights_dir):
            metadata_path = os.path.join(self.backup_dir, f"{run_name}_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    backups.append(metadata)
        
        return backups
    
    def restore_backup(self, run_name):
        """Restore a specific backup"""
        # Verify backup exists
        run_weights_dir = os.path.join(self.weights_dir, run_name)
        if not os.path.exists(run_weights_dir):
            raise ValueError(f"Backup {run_name} not found")
        
        # Restore weights
        weights_path = os.path.join(self.base_dir, 'runs/train/pet_waste_detector/weights')
        os.makedirs(weights_path, exist_ok=True)
        
        for file in os.listdir(run_weights_dir):
            src = os.path.join(run_weights_dir, file)
            dst = os.path.join(weights_path, file)
            shutil.copy2(src, dst)
        
        print(f"Backup {run_name} restored successfully")
        return True

if __name__ == "__main__":
    # Example usage
    backup = TrainingBackup()
    
    # Create a backup
    run_name = backup.create_backup()
    
    # List all backups
    backups = backup.list_backups()
    print("\nAvailable backups:")
    for b in backups:
        print(f"- {b['run_name']} ({b['backup_time']})")
    
    # To restore a backup:
    # backup.restore_backup(run_name) 