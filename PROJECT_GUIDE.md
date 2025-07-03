# Waste Detection Project Guide

This guide provides all the necessary commands and steps to work with the Waste Detection project, from setting up the environment to running the mobile application.

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Training the Model](#training-the-model)
3. [Model Conversion](#model-conversion)
4. [Running the Mobile App](#running-the-mobile-app)
5. [Additional Tools](#additional-tools)

## Environment Setup

### Python Environment Setup
```bash
# Create and activate virtual environment
python -m venv myenv
source myenv/bin/activate  # On Linux/Mac
# or
.\myenv\Scripts\activate  # On Windows

# Install requirements
pip install -r requirements.txt
```

### Mobile App Setup
```bash
# Navigate to the mobile app directory
cd WasteDetectionApplication

# Clean npm cache and remove existing node_modules
rm -rf node_modules
npm cache clean --force

# Install dependencies with legacy peer deps to resolve version conflicts
npm install --legacy-peer-deps

# Install Expo CLI globally (if not already installed)
npm install -g expo-cli

# Update npm to latest version (recommended)
npm install -g npm@latest
```

## Training the Model

### Training with YOLOv8
```bash
# Basic training command
python train.py --model yolov8n.pt --data dataset/data.yaml --epochs 100 --imgsz 640

# Training with specific parameters
python train.py --model yolov8n.pt --data dataset/data.yaml --epochs 100 --imgsz 640 --batch 16 --device 0
```

### Monitoring Training
- Training progress can be monitored through Weights & Biases (wandb)
- Check the `runs/` directory for training results and visualizations

## Model Conversion

### Converting to ONNX
```bash
# Convert PyTorch model to ONNX
python -c "from ultralytics import YOLO; model = YOLO('yolo11n.pt'); model.export(format='onnx')"
```

### Converting to TensorFlow.js
```bash
# Convert ONNX model to TensorFlow.js format
tensorflowjs_converter --input_format=onnx yolo11n.onnx yolo11n_web_model
```

## Running the Mobile App

### Development Mode
```bash
# Start the Expo development server
cd WasteDetectionApplication
npx expo start

# If you encounter any issues, try clearing the cache
npx expo start -c

# To run on specific platform
npx expo start --android  # For Android
npx expo start --ios     # For iOS
```

### Using Expo Go
1. Install Expo Go app on your mobile device
2. Scan the QR code shown in the terminal
3. The app will load on your device

### Building for Production
```bash
# Build for Android
eas build -p android

# Build for iOS
eas build -p ios
```

## Additional Tools

### Running Tests
```bash
# Run Python tests
python -m pytest tests/

# Run mobile app tests
cd WasteDetectionApplication
npm test
```

### Data Collection
- Use the crawler scripts in the `Crawler/` directory to collect training data
- Follow the dataset structure in `dataset/` directory

### Model Evaluation
```bash
# Evaluate model performance
python val.py --weights yolo11n.pt --data dataset/data.yaml
```

## Troubleshooting

### Common Issues
1. If you encounter CUDA/GPU issues:
   - Check if CUDA is properly installed
   - Verify GPU compatibility
   - Try running on CPU by adding `--device cpu`

2. Mobile App Issues:
   - Clear Expo cache: `expo start -c`
   - Reset node modules: `rm -rf node_modules && npm install`
   - Check Expo Go app version compatibility

3. Model Conversion Issues:
   - Ensure all dependencies are up to date
   - Check model compatibility with target format
   - Verify input/output shapes match requirements

### Running the Mobile App

#### Troubleshooting Mobile App Issues
1. If you encounter Metro bundler issues:
   ```bash
   # Clean up the project
   rm -rf node_modules package-lock.json
   
   # Install Metro dependencies explicitly
   npm install metro metro-core metro-runtime metro-source-map --save-dev
   
   # Reinstall all dependencies
   npm install --legacy-peer-deps
   
   # Start with a clean cache
   npx expo start --clear
   ```

2. If you see TensorFlow.js compatibility issues:
   - The current setup uses TensorFlow.js 3.11.0 with Expo 49
   - Make sure to use compatible versions of expo-camera and @tensorflow/tfjs-react-native
   - Add resolutions in package.json to force specific versions:
     ```json
     "resolutions": {
       "expo-camera": "~13.4.4"
     }
     ```

3. If the app fails to start:
   ```bash
   # Clear Expo cache
   npx expo start -c
   
   # Reset Metro bundler cache
   npx expo start --clear
   
   # If issues persist, try:
   rm -rf node_modules/.cache
   npx expo start --clear
   ```

4. For persistent Metro bundler issues:
   ```bash
   # Install specific Metro version
   npm install metro@0.76.0 metro-core@0.76.0 metro-runtime@0.76.0 metro-source-map@0.76.0 --save-dev
   
   # Clear watchman watches (if watchman is installed)
   watchman watch-del-all
   
   # Reset Metro bundler
   npx expo start --reset-cache
   ```

## Notes
- Always backup your trained models
- Keep track of model versions and their performance
- Document any custom configurations or changes
- Regular updates to dependencies are recommended

For more detailed information, refer to the project's README.md and documentation in the `docs/` directory. 