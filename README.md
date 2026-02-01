# Autonomous Platooning System

A small-scale autonomous vehicle platooning system implementing **V2V (Vehicle-to-Vehicle) communication** and **CACC (Cooperative Adaptive Cruise Control)** using MQTT protocol. This project demonstrates how multiple RC cars can maintain safe following distances while synchronizing speeds through wireless communication.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Key Technologies](#key-technologies)
- [Limitations & Future Work](#limitations--future-work)

## 🎯 Overview

### Problem Statement

In logistics and vehicle transportation, maintaining consistent spacing between vehicles is challenging. Common issues include:
- Variable inter-vehicle distances allowing other vehicles to cut in
- Driver fatigue leading to accidents
- Close-following scenarios where sudden braking causes rear-end collisions due to reaction delays

### Solution

This project implements a **platooning system** where multiple vehicles communicate to form a convoy. The system uses:
- **V2V Communication**: MQTT protocol for real-time speed synchronization
- **CACC (Cooperative Adaptive Cruise Control)**: Ultrasonic sensors maintain safe following distances
- **Autonomous Lane Following**: Leader vehicle uses OpenCV for lane detection, followers use infrared sensors

### How It Works

The system consists of **3 RC cars**: 1 leader and 2 followers.

- **Leader Vehicle**: 
  - Uses a camera module to capture lanes and OpenCV for lane detection
  - Measures actual speed using an encoder sensor
  - Publishes speed data to MQTT server
  
- **Follower Vehicles**:
  - Use infrared sensors to stay within lane boundaries
  - Subscribe to leader's speed via MQTT and set it as base speed
  - Use ultrasonic sensors to maintain safe distance from the vehicle ahead
  - Adjust speed (0.7x deceleration when too close, 1.3x acceleration when too far)

**In summary**: Three vehicles autonomously follow lanes while maintaining consistent spacing and synchronized speeds.

## ✨ Features

- **Real-time V2V Communication**: MQTT-based speed synchronization between leader and followers
- **Lane Detection**: OpenCV-based lane detection using Canny edge detection and Hough transform
- **Adaptive Speed Control**: Followers adjust speed based on leader's velocity and distance to preceding vehicle
- **Safe Distance Maintenance**: Ultrasonic sensors ensure minimum safe following distance (25cm target, 10cm emergency stop)
- **PID Steering Control**: Proportional control for smooth lane following
- **Multi-platform Support**: Raspberry Pi 4 (leader) and ESP8266-based WeMos D1 R1 (followers)

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Leader Vehicle                        │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Camera   │→ │ Lane Detect  │→ │ PID Steering │      │
│  │ Module   │  │ (OpenCV)     │  │ Control      │      │
│  └──────────┘  └──────────────┘  └──────────────┘      │
│                                                          │
│  ┌──────────┐  ┌──────────────┐                        │
│  │ Encoder  │→ │ Speed Calc   │→ MQTT Publish          │
│  │ Sensor   │  │ (cm/s)       │  (leader/speed)        │
│  └──────────┘  └──────────────┘                        │
└─────────────────────────────────────────────────────────┘
                        │
                        │ MQTT
                        ▼
              ┌──────────────────┐
              │   MQTT Broker    │
              │  (Wi-Fi Network) │
              └──────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
┌──────────────┐              ┌──────────────┐
│  Follower 1  │              │  Follower 2  │
│  (ESP8266)   │              │  (ESP8266)   │
│              │              │              │
│ MQTT Subscribe              │ MQTT Subscribe
│ → Base Speed │              │ → Base Speed │
│              │              │              │
│ Ultrasonic   │              │ Ultrasonic   │
│ → Distance   │              │ → Distance   │
│              │              │              │
│ IR Sensors   │              │ IR Sensors   │
│ → Lane Keep  │              │ → Lane Keep  │
└──────────────┘              └──────────────┘
```

## 🔧 Hardware Requirements

### Leader Vehicle
- **Raspberry Pi 4** (2GB RAM)
- **Camera Module** (for lane detection)
- **Encoder Sensor** (for speed measurement)
- **DC Motors** with motor driver
- **Battery Pack** (6 batteries recommended)

### Follower Vehicles (2 units)
- **WeMos D1 R1** (ESP8266-based board)
- **Ultrasonic Sensor** (HC-SR04) for distance measurement
- **Infrared Line Tracking Sensors** (2 sensors per vehicle)
- **DC Motors** with motor driver
- **Battery Pack** (4-6 batteries)
- **Reflective Board** (attached to rear for better ultrasonic reflection)

### Common Components
- Wi-Fi router/access point for MQTT communication
- MQTT broker (can run on Raspberry Pi or separate device)

## 💻 Software Requirements

### Leader Vehicle (Raspberry Pi)
- **OS**: Raspberry Pi OS
- **Python 3.x**
- **Libraries**:
  - OpenCV (`opencv-python`)
  - NumPy
  - Paho MQTT Client
  - GPIO Zero
  - Picamera2

### Follower Vehicles (ESP8266)
- **Arduino IDE** or **PlatformIO**
- **Libraries**:
  - ESP8266WiFi
  - PubSubClient (MQTT)

### MQTT Broker
- **Mosquitto** or any MQTT broker

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Autonomous-Platooning-System.git
cd Autonomous-Platooning-System
```

### 2. Install Python Dependencies (Leader Vehicle)

```bash
pip install -r requirements.txt
```

### 3. Install MQTT Broker

On Raspberry Pi or a separate device:

```bash
sudo apt-get update
sudo apt-get install mosquitto mosquitto-clients
```

Start the broker:
```bash
sudo systemctl start mosquitto
sudo systemctl enable mosquitto
```

### 4. Configure ESP8266 Code

1. Open `src/espBlue.cpp` or `src/espWhite.cpp` in Arduino IDE
2. Install ESP8266 board support:
   - File → Preferences → Additional Board Manager URLs: `http://arduino.esp8266.com/stable/package_esp8266com_index.json`
   - Tools → Board → Boards Manager → Search "ESP8266" → Install
3. Install required libraries:
   - Sketch → Include Library → Manage Libraries
   - Search and install: "PubSubClient"
4. Configure Wi-Fi credentials:
   ```cpp
   const char* ssid = "YOUR_WIFI_SSID";
   const char* password = "YOUR_WIFI_PASSWORD";
   const char* mqtt_server = "192.168.0.123";  // MQTT broker IP
   ```
5. Upload to WeMos D1 R1 board

### 5. Configure Leader Vehicle

1. Edit `src/main.py`:
   ```python
   MQTT_BROKER = "localhost"  # or MQTT broker IP
   ```
2. Ensure camera is connected and accessible
3. Connect encoder to GPIO pin 23

## 🚀 Usage

### Starting the System

1. **Start MQTT Broker** (if not running as service):
   ```bash
   mosquitto
   ```

2. **Start Leader Vehicle**:
   ```bash
   cd src
   python main.py
   ```
   
   Controls:
   - `W`: Increase speed
   - `S`: Decrease speed
   - `Space`: Emergency stop
   - `ESC`: Exit

3. **Follower Vehicles**: Automatically connect to Wi-Fi and MQTT on power-up

### System Operation

- Leader vehicle detects lanes and adjusts steering using PID control
- Speed is calculated from encoder pulses and published to MQTT topic `leader/speed`
- Followers subscribe to speed updates and convert to PWM values
- Followers maintain safe distance using ultrasonic sensors:
  - Too close (< 20cm): Decelerate to 70% of base speed
  - Too far (> 30cm): Accelerate to 130% of base speed
  - Safe distance (20-30cm): Maintain base speed
- Emergency stop if distance < 10cm

## 📁 Code Structure

```
Autonomous-Platooning-System/
├── src/
│   ├── main.py              # Leader vehicle main program
│   ├── lane_detect.py       # OpenCV lane detection functions
│   ├── espBlue.cpp          # Follower vehicle 1 code
│   └── espWhite.cpp         # Follower vehicle 2 code
├── docs/
│   └── DETAILS.md           # Detailed algorithm documentation
├── data/
│   ├── sampleimage.jpg      # Sample test image
│   └── sampleroad.mp4       # Sample test video
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

### Key Code Components

#### 1. Lane Detection (`lane_detect.py`)
- `canny_edge()`: Edge detection using Canny algorithm
- `apply_roi()`: Region of Interest (trapezoidal) masking
- `detect_lines_p()`: Hough transform for line detection
- `average_slope_intercept()`: Calculate average left/right lane lines
- `draw_lane_center()`: Calculate steering error (distance from lane center to image center)

#### 2. Leader Control (`main.py`)
- Camera capture and lane detection
- PID steering control: `steering = KP * error`
- Speed calculation from encoder pulses
- MQTT publishing of speed data

#### 3. Follower Control (`espBlue.cpp`, `espWhite.cpp`)
- MQTT subscription to leader speed
- Speed mapping: Convert leader speed (cm/s) to PWM values
- Distance-based ACC: Adjust speed based on ultrasonic sensor readings
- Line tracking: Use IR sensors to stay within lane boundaries

## 🔑 Key Technologies

### Computer Vision
- **Canny Edge Detection**: Detects lane boundaries
- **Hough Transform**: Extracts line segments from edge images
- **ROI (Region of Interest)**: Focuses processing on road area

### Control Systems
- **PID Control**: Proportional control for steering (KP = 0.01)
- **CACC Algorithm**: Cooperative Adaptive Cruise Control for distance maintenance

### Communication
- **MQTT Protocol**: Lightweight publish-subscribe messaging
- **Wi-Fi**: TCP/IP-based communication between vehicles

### Sensor Fusion
- **Camera**: Lane detection (leader)
- **Encoder**: Speed measurement (leader)
- **Ultrasonic**: Distance measurement (followers)
- **Infrared**: Line tracking (followers)

## ⚠️ Limitations & Future Work

### Current Limitations

1. **Motor Characteristics**: Low-cost DC motors have dead zones at very low speeds
   - **Solution**: Increased battery capacity (4→6 batteries) and operated only in stable speed ranges

2. **Hardware Variations**: Different RC car weights and configurations cause speed differences
   - **Solution**: Individual speed mapping calibration for each vehicle

3. **Wireless Network Dependency**: System relies on Wi-Fi connectivity
   - **Note**: Acceptable for indoor small-scale implementation; can be replaced with other communication modules for real-world applications

4. **Environment Assumptions**: Designed for straight sections without obstacles
   - Current implementation assumes controlled indoor environment

### Future Enhancements

1. **ROS 2 Integration**: 
   - Refactor code into ROS 2 node structure
   - Improve real-time performance
   - Enable integration with high-performance sensors (LiDAR, IMU)

2. **Advanced Control**:
   - Full PID control (currently only proportional)
   - Model-based control algorithms
   - Predictive control for smoother following

3. **V2I (Vehicle-to-Infrastructure)**:
   - Extend to V2I communication
   - Integration with traffic lights
   - Smart city applications

4. **Enhanced Perception**:
   - LiDAR integration
   - Multi-sensor fusion
   - Obstacle detection and avoidance

5. **Algorithm Improvements**:
   - Curve handling
   - Lane change capabilities
   - Multi-lane platooning

## 📝 License

[Specify your license here]

## 👥 Contributors

[Add contributor names here]

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- MQTT protocol for lightweight communication
- ESP8266 community for embedded system support

---

**Last Updated**: 2026-02-01
