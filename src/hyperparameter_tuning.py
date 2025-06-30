import wandb
from train_waste_detector import WasteDetectorTrainer
import yaml

def train_with_config():
    # Initialize wandb
    with wandb.init() as run:
        # Get hyperparameters from wandb config
        config = wandb.config
        
        # Initialize trainer
        trainer = WasteDetectorTrainer(dataset_dir='/home/ailaty3088@id.sdsu.edu/Ashter/dataset/')
        
        # Train model with sweep parameters
        results = trainer.train_model(
            epochs=config.epochs,
            batch_size=config.batch_size,
            img_size=config.img_size
        )
        
        # Log metrics
        wandb.log({
            "mAP50": results.results_dict['metrics/mAP50(B)'],
            "mAP50-95": results.results_dict['metrics/mAP50-95(B)'],
            "precision": results.results_dict['metrics/precision(B)'],
            "recall": results.results_dict['metrics/recall(B)']
        })

def main():
    # Define sweep configuration
    sweep_config = {
        'method': 'bayes',  # Bayesian optimization
        'metric': {
            'name': 'mAP50',
            'goal': 'maximize'
        },
        'parameters': {
            'epochs': {
                'values': [100, 150, 200]
            },
            'batch_size': {
                'values': [8, 16, 32]
            },
            'img_size': {
                'values': [640, 800, 1024]
            },
            'lr0': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-2
            },
            'weight_decay': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-3
            },
            'hsv_h': {
                'distribution': 'uniform',
                'min': 0.0,
                'max': 0.1
            },
            'hsv_s': {
                'distribution': 'uniform',
                'min': 0.0,
                'max': 0.9
            },
            'hsv_v': {
                'distribution': 'uniform',
                'min': 0.0,
                'max': 0.9
            },
            'degrees': {
                'distribution': 'uniform',
                'min': 0.0,
                'max': 30.0
            },
            'translate': {
                'distribution': 'uniform',
                'min': 0.0,
                'max': 0.2
            },
            'scale': {
                'distribution': 'uniform',
                'min': 0.0,
                'max': 0.5
            }
        }
    }

    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="pet-waste-detection")
    
    # Run sweep
    wandb.agent(sweep_id, function=train_with_config, count=20)  # Run 20 trials

if __name__ == "__main__":
    main() 