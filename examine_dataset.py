from datasets import load_dataset

# Load the dataset
ds = load_dataset("erotemic/shitspotter")

print("Dataset loaded successfully!")
print(f"Available splits: {list(ds.keys())}")

# Examine the first sample from each split
for split_name, dataset_split in ds.items():
    print(f"\n{'='*50}")
    print(f"Split: {split_name}")
    print(f"Number of samples: {len(dataset_split)}")
    
    if len(dataset_split) > 0:
        first_sample = dataset_split[0]
        print(f"Sample keys: {list(first_sample.keys())}")
        print(f"Sample content:")
        for key, value in first_sample.items():
            if key == 'image':
                print(f"  {key}: PIL Image object with size {value.size}")
            else:
                print(f"  {key}: {type(value)} - {value}")
        
        # Try to access image
        try:
            image = first_sample['image']
            print(f"Image size: {image.size}")
            print(f"Image mode: {image.mode}")
        except Exception as e:
            print(f"Error accessing image: {e}")
            
        # Try to access annotations
        try:
            annotations = first_sample.get('objects', {})
            print(f"Annotations: {annotations}")
        except Exception as e:
            print(f"Error accessing annotations: {e}") 