### This is a crawler that downloads all the photos from the internet to compelete our dataset
'''
Ashter/
├── src/               # Main source code
├── data/              # Data storage
│   └── images/        # Image datasets
├── scripts/           # Utility scripts
├── tests/             # Test files
├── docs/              # Documentation
├── requirements.txt   # Project dependencies
└── README.md
'''
# Software Component
## Computer Vision System: 
Framework: YOLOv8 Libraries: - OpenCV 4.8+ -
PyTorch 2.0+ - Ultralytics Hardware Acceleration: CUDA 11.8+ Models: -
Object Detection: YOLOv8m - Pose Detection: YOLOv8-pose - Instance
Segmentation: YOLOv8-seg

## Thermal Processing: 
Hardware: FLIR Lepton 3.5 Libraries: -
PyThermalComfort - thermal_camera_tools Processing: - Thermal Mapping -
Heat Signature Analysis

## Robot Control System: 
Framework: ROS2 Humble Components: - Navigation:
Nav2 - Motion Planning: MoveIt2 - SLAM: RTABMap Hardware Interface: -
Motor Controllers: ROSSerial - Sensor Integration: ros2_control -
Hardware Abstraction: ros2_hardware_interface

## Data Processing: 
Real-time Processing: - Framework: Apache Kafka -
Stream Processing: Apache Flink Storage: - Time Series: InfluxDB -
Object Store: MinIO - Document Store: MongoDB Analytics: - Health
Analytics: TensorFlow - Behavioral Analysis: scikit-learn - Data
Pipeline: Apache Airflow

## Communication: 
### Protocols: 
- MQTT: Eclipse Mosquitto - REST: FastAPI - WebSocket: Socket.IO 
- Security: - Authentication: JWT - Encryption: TLS 1.3 - Access Control: OAuth2

## Frontend Applications: 
Mobile: Framework: Flutter Features: - Real-time
Monitoring - Push Notifications - Health Reports - Remote Control Web
Dashboard: Framework: Next.js Components: - React - TailwindCSS -
Recharts Features: - Analytics Dashboard - Device Management - Health
Monitoring - Configuration Interface

## Development Tools: 
Version Control: Git CI/CD: GitHub Actions
Containerization: - Docker - Kubernetes Monitoring: - Prometheus -
Grafana Testing: - Unit: pytest - Integration: Robot Framework - Load
Testing: k6

# Hardware Specifications

## Core Processing Unit
- NVIDIA Jetson Xavier NX
  - 384 CUDA Cores
  - 48 Tensor Cores
  - 6-core ARM CPU
  - 8GB LPDDR4x memory
  - Power: 10-15W

## Sensing Systems

### Vision System
- Primary Camera: Intel RealSense D455
  - Resolution: 1280x720@90fps
  - FOV: 87°x58°
  - Depth Range: 0.6-6m
- Thermal Camera: FLIR Lepton 3.5
  - Resolution: 160x120
  - Thermal Sensitivity: <50 mK
  - Spectral Range: 8-14μm

### Environmental Sensors
- Gas Sensors
  - MQ-4 Methane Sensor
  - BME688 VOC Sensor
  - Temperature & Humidity
- Position Sensors
  - IMU: BNO055
  - GPS: u-blox ZED-F9P
  - Wheel Encoders: 1024 PPR

## Robotics Components

### Motion System
- Motors
  - 4x Brushless DC Motors
  - 24V, 250W each
  - Integrated encoders
- Wheels
  - All-terrain design
  - 200mm diameter
  - Independent suspension

### Collection Mechanism
- Robotic Arm
  - 6-DOF articulated arm
  - Payload: 2kg
  - Reach: 50cm
- End Effector
  - Vacuum-assisted gripper
  - UV sterilization module
  - Waste storage container: 2L

### Power System
- Battery
  - Li-ion 48V 20Ah
  - Hot-swappable design
  - Runtime: 4-6 hours
- Charging
  - Wireless charging capability
  - Fast charging support
  - Auto-docking system

## Communication
- Wi-Fi: Intel AX210
  - Wi-Fi 6E support
  - Dual-band 2.4/5/6GHz
- Bluetooth 5.2
- 4G/LTE Modem
- LoRaWAN for long-range

## Environmental Protection
- IP65 Rating
- Operating Temperature: -10°C to 45°C
- Humidity: 10-90% non-condensing
- UV-resistant housing
- Anti-corrosion coating

## Physical Specifications
- Dimensions: 600x400x300mm
- Weight: 15kg (without payload)
- Ground Clearance: 100mm
- Turning Radius: 500mm

## Safety Features
- Emergency Stop Button
- Obstacle Detection
- Tip-over Protection
- Battery Protection Circuit
- Thermal Management
- LED Status Indicators
- Audio Warning System