# V2V-based CACC Platooning System Implementation

A small-scale autonomous vehicle platooning system implementing **V2V (Vehicle-to-Vehicle) communication** and **simplified CACC-based cooperative control** using MQTT protocol. This project demonstrates how a leader vehicle broadcasts its speed via V2V communication, and follower vehicles cooperatively perform longitudinal control through onboard distance sensing.

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
- [Experimental Setup Summary](#experimental-setup-summary)
- [Limitations & Future Work](#limitations--future-work)

## 🎯 Overview

### Problem Statement

Maintaining consistent spacing between vehicles in logistics and transportation is challenging. Common issues include:
- Variable inter-vehicle distances allowing other vehicles to cut in
- Driver fatigue leading to accidents
- Rear-end collisions due to reaction delays during sudden braking

### Solution

This project implements a **simplified CACC-based platooning system** where multiple vehicles communicate to form a convoy. The system uses:
- **V2V Communication**: Wi-Fi based MQTT protocol for real-time speed following
- **Simplified CACC-based Cooperative Control**: Safe following distance maintenance through leader speed broadcast and follower ultrasonic sensor-based distance correction
- **Autonomous Lane Following**: Leader vehicle uses OpenCV for lane detection, followers use infrared sensors

### How It Works

The system consists of **3 RC cars**: 1 leader and 2 followers.

- **Leader Vehicle**: 
  - Uses a camera module to capture lanes and OpenCV for lane detection
  - Measures actual speed using an encoder sensor
  - Publishes speed data to MQTT broker
  
- **Follower Vehicles**:
  - Use infrared sensors to stay within lane boundaries
  - Subscribe to leader's speed via MQTT and set it as base speed
  - Use ultrasonic sensors to maintain safe distance from the vehicle ahead
  - Adjust speed (0.7× deceleration when too close, 1.3× acceleration when too far)

**Summary**: Three vehicles autonomously follow lanes while maintaining consistent spacing and speed synchronized through following-based coordination.

## ✨ Features

- **Real-time V2V Communication**: MQTT-based speed following between leader and followers
- **Lane Detection**: OpenCV-based lane detection using Canny edge detection and Hough transform
- **Adaptive Speed Control**: Followers adjust speed based on leader's velocity and distance to preceding vehicle
- **Safe Distance Maintenance**: Ultrasonic sensors ensure minimum safe following distance (target 25cm, tolerance 20-30cm, emergency stop <10cm)
- **P-based Steering Control**: Proportional control for smooth lane following (`steering = Kp × error`)
- **Multi-platform Support**: Raspberry Pi 4 (leader) and ESP8266-based WeMos D1 R1 (followers)

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Leader Vehicle                       │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Camera   │→ │ Lane Detect  │→ │ P Steering   │      │
│  │ Module   │  │ (OpenCV)      │  │ Control      │      │
│  └──────────┘  └──────────────┘  └──────────────┘      │
│                                                          │
│  ┌──────────┐  ┌──────────────┐                        │
│  │ Encoder  │→ │ Speed Calc   │→ MQTT Publish          │
│  │ Sensor   │  │ (cm/s)        │  (leader/speed)        │
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
3. Connect encoder (example: GPIO pin 23, may vary depending on environment)

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

- Leader vehicle detects lanes and adjusts steering using P control
- Speed is calculated from encoder pulses and published to MQTT topic `leader/speed`
- Followers subscribe to speed updates and convert to PWM values
- Followers maintain safe distance using ultrasonic sensors (rule-based threshold control):
  - Target distance: 25cm, Tolerance: 20-30cm (deadband/hysteresis for sensor noise and control stability)
  - Too close (< 20cm): Decelerate to 70% of base speed
  - Too far (> 30cm): Accelerate to 130% of base speed
  - Safe distance (20-30cm): Maintain base speed
  - *Note: Implemented as rule-based threshold method, not continuous control*
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
- P-based steering control: `steering = Kp * error` (proportional control only)
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
- **P-based Control**: Proportional control for steering (Kp = 0.01, currently only P term used)
  - *Note: Document uses Kp notation, while code variable name is KP.*
- **Simplified CACC-based Algorithm**: Cooperative control through leader speed broadcast and rule-based distance correction

### Communication
- **V2V Communication**: V2V in this project is implemented as Wi-Fi based MQTT messaging in the experimental environment.
- **MQTT Protocol**: Lightweight publish-subscribe messaging
- **Wi-Fi**: TCP/IP-based communication between vehicles

### Sensor Fusion
- **Camera**: Lane detection (leader)
- **Encoder**: Speed measurement (leader)
- **Ultrasonic**: Distance measurement (followers)
- **Infrared**: Line tracking (followers)

## 📊 Experimental Setup Summary

| Item | Value |
|---|---|
| Environment | Indoor test track |
| Communication | Wi-Fi + MQTT |
| Target distance | 25 cm |
| Tolerance (deadband) | 20-30 cm |
| Emergency stop distance | < 10 cm |
| Control method | P-based steering + rule-based ACC |
| Leader platform | Raspberry Pi 4 (2GB) |
| Follower platform | ESP8266 (WeMos D1 R1) |

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
   - Full PID control (currently only P term used)
   - Model-based control algorithms
   - Predictive control for smoother following
   - Continuous control-based CACC implementation (currently rule-based)

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

---

**Last Updated**: 2026-02-01

----
----

# V2V기반 CACC 군집주행 시스템

**V2V (차량 간 통신)** 및 **간소화된 CACC 기반 협동 제어**를 MQTT 프로토콜을 사용하여 구현한 소형 자율 주행 군집주행 시스템입니다. 이 프로젝트는 리더 차량의 속도를 V2V 통신으로 브로드캐스트하고, 팔로워 차량들이 온보드 거리 센싱을 통해 협동적으로 종방향 제어를 수행하는 방법을 보여줍니다.

## 📋 목차

- [개요](#개요)
- [주요 기능](#주요-기능)
- [시스템 아키텍처](#시스템-아키텍처)
- [하드웨어 요구사항](#하드웨어-요구사항)
- [소프트웨어 요구사항](#소프트웨어-요구사항)
- [설치](#설치)
- [사용법](#사용법)
- [코드 구조](#코드-구조)
- [핵심 기술](#핵심-기술)
- [실험 설정 요약](#실험-설정-요약)
- [한계점 및 향후 계획](#한계점-및-향후-계획)

## 🎯 개요

### 문제점

물류 및 차량 운송에서 차량 간 일정한 간격을 유지하는 것은 어려운 과제입니다. 일반적인 문제점은 다음과 같습니다:
- 차량 간 거리가 일정하지 않아 다른 차량이 끼어들 수 있음
- 운전자 졸음으로 인한 사고 발생
- 급제동 시 운전자의 반응 지연으로 인한 추돌 사고

### 해결책

이 프로젝트는 여러 차량이 통신하여 무리를 형성하는 **간소화된 CACC 기반 군집주행 시스템**을 구현합니다. 시스템은 다음을 사용합니다:
- **V2V 통신**: 실시간 속도 추종을 위한 Wi-Fi 기반 MQTT 프로토콜
- **간소화된 CACC 기반 협동 제어**: 리더 속도 브로드캐스트와 팔로워의 초음파 센서 기반 거리 보정을 통한 안전한 추종 거리 유지
- **자율 차선 추종**: 리더 차량은 OpenCV를 사용한 차선 인식, 팔로워는 적외선 센서 사용

### 작동 원리

시스템은 **3대의 RC카**로 구성됩니다: 리더 1대와 팔로워 2대.

- **리더 차량**: 
  - 카메라 모듈로 차선을 촬영하고 OpenCV로 차선 인식
  - 엔코더 센서로 실제 속도 측정
  - MQTT 브로커에 속도 데이터 발행
  
- **팔로워 차량**:
  - 적외선 센서로 차선 경계 내에서 주행
  - MQTT를 통해 리더의 속도를 구독하여 기본 속도로 설정
  - 초음파 센서로 앞 차량과의 안전 거리 유지
  - 속도 조절 (너무 가까우면 0.7배 감속, 너무 멀면 1.3배 가속)

**요약**: 세 대의 차량이 일정한 간격을 유지하고 추종 기반으로 동조된 속도로 차선을 따라 자율 주행합니다.

## ✨ 주요 기능

- **실시간 V2V 통신**: 리더와 팔로워 간 MQTT 기반 속도 추종
- **차선 인식**: Canny 엣지 검출 및 Hough 변환을 사용한 OpenCV 기반 차선 인식
- **적응형 속도 제어**: 팔로워가 리더의 속도와 앞 차량과의 거리에 따라 속도 조절
- **안전 거리 유지**: 초음파 센서로 최소 안전 추종 거리 보장 (목표 25cm, 허용 범위 20-30cm, 비상 정지 <10cm)
- **P 기반 조향 제어**: 부드러운 차선 추종을 위한 비례 제어 (`steering = Kp × error`)
- **다중 플랫폼 지원**: 라즈베리파이 4 (리더) 및 ESP8266 기반 WeMos D1 R1 (팔로워)

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    리더 차량                             │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ 카메라   │→ │ 차선 인식     │→ │ P 조향 제어  │      │
│  │ 모듈     │  │ (OpenCV)      │  │             │      │
│  └──────────┘  └──────────────┘  └──────────────┘      │
│                                                          │
│  ┌──────────┐  ┌──────────────┐                        │
│  │ 엔코더   │→ │ 속도 계산     │→ MQTT 발행             │
│  │ 센서     │  │ (cm/s)        │  (leader/speed)        │
│  └──────────┘  └──────────────┘                        │
└─────────────────────────────────────────────────────────┘
                        │
                        │ MQTT
                        ▼
              ┌──────────────────┐
              │   MQTT 브로커    │
              │  (Wi-Fi 네트워크)│
              └──────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
┌──────────────┐              ┌──────────────┐
│  팔로워 1     │              │  팔로워 2     │
│  (ESP8266)   │              │  (ESP8266)   │
│              │              │              │
│ MQTT 구독     │              │ MQTT 구독     │
│ → 기본 속도   │              │ → 기본 속도   │
│              │              │              │
│ 초음파        │              │ 초음파        │
│ → 거리        │              │ → 거리        │
│              │              │              │
│ 적외선 센서   │              │ 적외선 센서   │
│ → 차선 유지   │              │ → 차선 유지   │
└──────────────┘              └──────────────┘
```

## 🔧 하드웨어 요구사항

### 리더 차량
- **라즈베리파이 4** (2GB RAM)
- **카메라 모듈** (차선 인식용)
- **엔코더 센서** (속도 측정용)
- **DC 모터** 및 모터 드라이버
- **배터리 팩** (6개 배터리 권장)

### 팔로워 차량 (2대)
- **WeMos D1 R1** (ESP8266 기반 보드)
- **초음파 센서** (HC-SR04) - 거리 측정용
- **적외선 라인 트레이싱 센서** (차량당 2개)
- **DC 모터** 및 모터 드라이버
- **배터리 팩** (4-6개 배터리)
- **반사판** (초음파 반사 향상을 위해 후면에 부착)

### 공통 구성 요소
- MQTT 통신을 위한 Wi-Fi 라우터/액세스 포인트
- MQTT 브로커 (라즈베리파이 또는 별도 장치에서 실행 가능)

## 💻 소프트웨어 요구사항

### 리더 차량 (라즈베리파이)
- **OS**: Raspberry Pi OS
- **Python 3.x**
- **라이브러리**:
  - OpenCV (`opencv-python`)
  - NumPy
  - Paho MQTT Client
  - GPIO Zero
  - Picamera2

### 팔로워 차량 (ESP8266)
- **Arduino IDE** 또는 **PlatformIO**
- **라이브러리**:
  - ESP8266WiFi
  - PubSubClient (MQTT)

### MQTT 브로커
- **Mosquitto** 또는 기타 MQTT 브로커

## 📦 설치

### 1. 저장소 클론

```bash
git clone https://github.com/yourusername/Autonomous-Platooning-System.git
cd Autonomous-Platooning-System
```

### 2. Python 의존성 설치 (리더 차량)

```bash
pip install -r requirements.txt
```

### 3. MQTT 브로커 설치

라즈베리파이 또는 별도 장치에서:

```bash
sudo apt-get update
sudo apt-get install mosquitto mosquitto-clients
```

브로커 시작:
```bash
sudo systemctl start mosquitto
sudo systemctl enable mosquitto
```

### 4. ESP8266 코드 설정

1. Arduino IDE에서 `src/espBlue.cpp` 또는 `src/espWhite.cpp` 열기
2. ESP8266 보드 지원 설치:
   - 파일 → 환경설정 → 추가 보드 관리자 URL: `http://arduino.esp8266.com/stable/package_esp8266com_index.json`
   - 도구 → 보드 → 보드 매니저 → "ESP8266" 검색 → 설치
3. 필요한 라이브러리 설치:
   - 스케치 → 라이브러리 포함하기 → 라이브러리 관리
   - "PubSubClient" 검색 및 설치
4. Wi-Fi 자격 증명 설정:
   ```cpp
   const char* ssid = "YOUR_WIFI_SSID";
   const char* password = "YOUR_WIFI_PASSWORD";
   const char* mqtt_server = "192.168.0.123";  // MQTT 브로커 IP
   ```
5. WeMos D1 R1 보드에 업로드

### 5. 리더 차량 설정

1. `src/main.py` 편집:
   ```python
   MQTT_BROKER = "localhost"  # 또는 MQTT 브로커 IP
   ```
2. 카메라가 연결되어 있고 접근 가능한지 확인
3. 엔코더 연결 (예시: GPIO 핀 23, 환경에 따라 변경 가능)

## 🚀 사용법

### 시스템 시작

1. **MQTT 브로커 시작** (서비스로 실행되지 않는 경우):
   ```bash
   mosquitto
   ```

2. **리더 차량 시작**:
   ```bash
   cd src
   python main.py
   ```
   
   조작키:
   - `W`: 속도 증가
   - `S`: 속도 감소
   - `Space`: 비상 정지
   - `ESC`: 종료

3. **팔로워 차량**: 전원 켜면 자동으로 Wi-Fi 및 MQTT에 연결

### 시스템 동작

- 리더 차량이 차선을 인식하고 P 제어로 조향 조절
- 엔코더 펄스로부터 속도를 계산하여 MQTT 토픽 `leader/speed`에 발행
- 팔로워가 속도 업데이트를 구독하고 PWM 값으로 변환
- 팔로워가 초음파 센서를 사용하여 안전 거리 유지 (규칙 기반 임계값 제어):
  - 목표 거리: 25cm, 허용 범위: 20-30cm (센서 노이즈 및 제어 안정화를 위한 데드밴드/히스테리시스)
  - 너무 가까움 (< 20cm): 기본 속도의 70%로 감속
  - 너무 멀음 (> 30cm): 기본 속도의 130%로 가속
  - 안전 거리 (20-30cm): 기본 속도 유지
  - *참고: 연속 제어가 아닌 규칙 기반 임계값 방식으로 구현됨*
- 거리 < 10cm일 경우 비상 정지

## 📁 코드 구조

```
Autonomous-Platooning-System/
├── src/
│   ├── main.py              # 리더 차량 메인 프로그램
│   ├── lane_detect.py       # OpenCV 차선 인식 함수
│   ├── espBlue.cpp          # 팔로워 차량 1 코드
│   └── espWhite.cpp         # 팔로워 차량 2 코드
├── docs/
│   └── DETAILS.md           # 상세 알고리즘 문서
├── data/
│   ├── sampleimage.jpg      # 샘플 테스트 이미지
│   └── sampleroad.mp4       # 샘플 테스트 비디오
├── requirements.txt         # Python 의존성
└── README.md               # 이 파일
```

### 주요 코드 구성 요소

#### 1. 차선 인식 (`lane_detect.py`)
- `canny_edge()`: Canny 알고리즘을 사용한 엣지 검출
- `apply_roi()`: 관심 영역 (사다리꼴) 마스킹
- `detect_lines_p()`: Hough 변환을 사용한 선분 검출
- `average_slope_intercept()`: 좌/우 차선의 평균 계산
- `draw_lane_center()`: 조향 오차 계산 (차선 중심과 이미지 중심 간 거리)

#### 2. 리더 제어 (`main.py`)
- 카메라 캡처 및 차선 인식
- P 기반 조향 제어: `steering = Kp * error` (비례 제어만 사용)
- 엔코더 펄스로부터 속도 계산
- 속도 데이터 MQTT 발행

#### 3. 팔로워 제어 (`espBlue.cpp`, `espWhite.cpp`)
- 리더 속도 MQTT 구독
- 속도 매핑: 리더 속도 (cm/s)를 PWM 값으로 변환
- 거리 기반 ACC: 초음파 센서 판독값에 따라 속도 조절
- 라인 트레이싱: 적외선 센서로 차선 경계 내 유지

## 🔑 핵심 기술

### 컴퓨터 비전
- **Canny 엣지 검출**: 차선 경계 검출
- **Hough 변환**: 엣지 이미지에서 선분 추출
- **ROI (관심 영역)**: 도로 영역에 처리 집중

### 제어 시스템
- **P 기반 제어**: 조향을 위한 비례 제어 (Kp = 0.01, 현재는 P 항만 사용)
  - *참고: 문서에서는 Kp로 표기하며, 코드 변수명은 KP입니다.*
- **간소화된 CACC 기반 알고리즘**: 리더 속도 브로드캐스트와 규칙 기반 거리 보정을 통한 협동 제어

### 통신
- **V2V 통신**: 본 프로젝트의 V2V는 실험 환경에서 Wi-Fi 기반 MQTT 메시징으로 구현되었습니다.
- **MQTT 프로토콜**: 경량 발행-구독 메시징
- **Wi-Fi**: 차량 간 TCP/IP 기반 통신

### 센서 융합
- **카메라**: 차선 인식 (리더)
- **엔코더**: 속도 측정 (리더)
- **초음파**: 거리 측정 (팔로워)
- **적외선**: 라인 트레이싱 (팔로워)

## 📊 실험 설정 요약

| 항목 | 값 |
|---|---|
| 환경 | 실내 테스트 트랙 |
| 통신 | Wi-Fi + MQTT |
| 목표 거리 | 25 cm |
| 허용 범위 (데드밴드) | 20-30 cm |
| 비상 정지 거리 | < 10 cm |
| 제어 방식 | P 기반 조향 + 규칙 기반 ACC |
| 리더 플랫폼 | Raspberry Pi 4 (2GB) |
| 팔로워 플랫폼 | ESP8266 (WeMos D1 R1) |

## ⚠️ 한계점 및 향후 계획

### 현재 한계점

1. **모터 특성**: 저가형 DC 모터는 매우 낮은 속도에서 데드존이 있음
   - **해결책**: 배터리 용량 증가 (4→6개) 및 안정적인 속도 구간에서만 동작

2. **하드웨어 변형**: 서로 다른 RC카의 무게 및 구성으로 인한 속도 차이
   - **해결책**: 각 차량에 대한 개별 속도 매핑 보정

3. **무선 네트워크 의존성**: 시스템이 Wi-Fi 연결에 의존
   - **참고**: 실내 소형 구현에는 적합하며, 실제 응용에서는 다른 통신 모듈로 교체 가능

4. **환경 가정**: 장애물이 없는 직선 구간을 위해 설계됨
   - 현재 구현은 제어된 실내 환경을 가정

### 향후 개선 사항

1. **ROS 2 통합**: 
   - 코드를 ROS 2 노드 구조로 리팩토링
   - 실시간 성능 향상
   - 고성능 센서 (LiDAR, IMU) 통합 가능

2. **고급 제어**:
   - 완전한 PID 제어 (현재는 P 항만 사용)
   - 모델 기반 제어 알고리즘
   - 더 부드러운 추종을 위한 예측 제어
   - 연속 제어 기반 CACC 구현 (현재는 규칙 기반)

3. **V2I (차량-인프라 통신)**:
   - V2I 통신으로 확장
   - 신호등 통합
   - 스마트 시티 응용

4. **향상된 인식**:
   - LiDAR 통합
   - 다중 센서 융합
   - 장애물 감지 및 회피

5. **알고리즘 개선**:
   - 곡선 구간 처리
   - 차선 변경 기능
   - 다중 차선 군집주행


---

**최종 업데이트**: 2026-02-01
