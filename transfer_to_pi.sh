#!/bin/bash

# Configuration
PI_USER="pi"  # Change this to your Raspberry Pi username
PI_IP="192.168.1.100"  # Change this to your Raspberry Pi's IP address
PI_DIR="~/waste_detector"

# Create remote directory
ssh $PI_USER@$PI_IP "mkdir -p $PI_DIR"

# Copy files
scp converted_models/model.tflite $PI_USER@$PI_IP:$PI_DIR/
scp src/raspberry_pi_detector.py $PI_USER@$PI_IP:$PI_DIR/detector.py
scp requirements_raspberry_pi.txt $PI_USER@$PI_IP:$PI_DIR/requirements.txt

echo "Files transferred successfully!"
echo "Now connect to your Raspberry Pi and run:"
echo "cd $PI_DIR"
echo "python3 -m venv venv"
echo "source venv/bin/activate"
echo "pip install -r requirements.txt"
echo "python3 detector.py" 