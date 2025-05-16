import wandb
import glob
from ultralytics.utils.plots import plot_results

# Load and analyze local results
results_file = glob.glob('runs/detect/pet_waste_detector2/results.csv')[0]
wandb.init(project="pet-waste-detection")
wandb.log({"training_results": wandb.Table(data=results_file)})

# Load plots
wandb.log({"metrics_plot": wandb.Image('runs/detect/pet_waste_detector2/results.png')})