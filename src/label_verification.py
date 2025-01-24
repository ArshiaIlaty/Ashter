# Quick check of a label file
import os

label_path = '/home/ailaty3088@id.sdsu.edu/Ashter/dataset/train/labels'
label_files = os.listdir(label_path)

if label_files:
    with open(os.path.join(label_path, label_files[0])) as f:
        print(f.read())