import time
import cv2
import numpy as np
import paho.mqtt.client as mqtt
from gpiozero import Robot, DigitalInputDevice
from picamera2 import Picamera2

from lane_detect import (
    canny_edge, apply_roi, detect_lines_p,
    average_slope_intercept, draw_lane_lines, draw_lane_center
)

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
# [MQTT 설정]
MQTT_BROKER = "localhost"
MQTT_TOPIC = "leader/speed"

# [하드웨어 핀 설정 (BCM 기준)]
# 좌측/우측 모터: 각각 2개 핀(정/역 회전용), 엔코더 입력 핀 23
PIN_MOTOR_LEFT = (17, 18)
PIN_MOTOR_RIGHT = (27, 22)
PIN_ENCODER = 23

# [주행 제어 파라미터]
MAX_SPEED_LIMIT = 0.9  # PWM 최대값 제한(0~1 범위에서 0.9까지만)
SPEED_STEP = 0.05  # 키보드로 속도를 올릴 때 0.05씩 증감
KP = 0.01  # error를 조향량으로 바꾸는 비례상수(너무 크면 흔들림/진동)

# [속도 계산 파라미터]
WHEEL_DIAMETER_CM = 6.5  # 바퀴 지름 6.5cm
PULSE_PER_ROTATION = 20  # 한 바퀴 회전에 엔코더 펄스 20개 발생
CALC_INTERVAL = 0.5  # 0.5초마다 속도 계산 및 MQTT 발행

# ==========================================
# 2. 객체 초기화
# ==========================================

# Robot(모터 제어) 생성
robot = Robot(left=PIN_MOTOR_LEFT,
              right=PIN_MOTOR_RIGHT)  # robot.left_motor.value, robot.right_motor.value에 0~1 값을 넣으면 모터가 그 속도로 돌아감

# 엔코더 펄스 카운팅 준비
encoder = DigitalInputDevice(PIN_ENCODER)
pulse_count = 0


# 펄스 들어올 때마다 실행되는 함수
# 엔코더 신호가 “활성화(상승/입력 발생)”될 때마다 pulse_count가 1씩 증가
# 즉, 바퀴 회전량을 펄스 개수로 누적하는 구조
def count_pulse():
    global pulse_count
    pulse_count += 1


encoder.when_activated = count_pulse

# MQTT 클라이언트 연결
client = mqtt.Client("LeaderCar")
try:
    client.connect(MQTT_BROKER, 1883, 60)
    client.loop_start()
    print(f"[MQTT] Connected to {MQTT_BROKER}")
except Exception as e:
    print(f"[MQTT] Connection Failed: {e}")


# ==========================================
# 3. 메인 루프
# ==========================================
def main():
    global pulse_count

    # [Camera Setup]
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "BGR888", "size": (640, 480)}
    )
    picam2.configure(config)
    picam2.start()

    current_target_speed = 0.0  # 키보드로 바꾸는 목표 PWM 속도 -> 실제 속도가 아닌 모터에 주는 입력값
    last_calc_time = time.time()

    print("=== Leader Vehicle Started (Trapezoid ROI) ===")
    print(" [W]: Speed Up | [S]: Speed Down | [Space]: Stop | [ESC]: Exit")

    try:
        while True:
            # 1) Picamera2 프레임 획득
            frame = picam2.capture_array()
            if frame is None:
                print("프레임을 읽을 수 없습니다.")
                break

            # 2) 차선 인식 처리 (Lane Detection)
            # 320x240 크기로 리사이즈하여 전처리
            edges, frame_small = canny_edge(frame, resize_to=(320, 240))

            h, w = edges.shape[:2]  # h=240, w=320

            # 3) ROI(관심영역)를 사다리꼴로 지정
            # 도로는 화면 아래쪽에 있고, 멀리 갈수록 폭이 좁아지므로 ROI를 사다리꼴로 설정
            # 삼각형 ROI보다 위쪽 폭을 넓혀서 멀리 있는 차선도 덜 놓치게 만든 버전 -> 상단 폭을 넓혀서(w//2 ± 60) 멀리 있는 차선도 놓치지 않게 함
            # 좌표 순서: 좌하단 -> 우하단 -> 우상단 -> 좌상단

            roi_poly = np.array([
                (0, h),  # 좌하단
                (w, h),  # 우하단
                (w // 2 + 80, int(h * 0.55)),  # 우상단 (중앙에서 오른쪽으로 80px)
                (w // 2 - 80, int(h * 0.55))  # 좌상단 (중앙에서 왼쪽으로 80px)
            ])
            edges_roi = apply_roi(edges, roi_poly)

            # 4) 선분 검출 -> 좌/우 차선 대표선 만들기
            # HoughLinesP로 선분을 많이 뽑고, 기울기로 좌/우를 나눈 뒤 평균내서 대표선 2개 생성
            lines = detect_lines_p(edges_roi)
            left_line, right_line = average_slope_intercept(lines, w, h)

            # 5) 시각화 및 error(오차) 추출
            # 차선 2개를 그린 뒤, 차선 중앙과 화면 중앙 차이를 error로 계산
            view = draw_lane_lines(frame_small, left_line, right_line)
            view, error = draw_lane_center(view, left_line, right_line)

            # ROI 영역을 초록색 선으로 그려서 확인 (디버깅용)
            cv2.polylines(view, [roi_poly], True, (0, 255, 0), 1)

            # 6) 키보드로 목표 속도 제어 (실험/데모용 수동 속도 조절)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break
            elif key == ord('w'):  # 목표 속도 증가
                current_target_speed = min(MAX_SPEED_LIMIT, current_target_speed + SPEED_STEP)
                print(f"Target Speed UP: {current_target_speed:.2f}")
            elif key == ord('s'):  # 목표 속도 감소
                current_target_speed = max(0.0, current_target_speed - SPEED_STEP)
                print(f"Target Speed DOWN: {current_target_speed:.2f}")
            elif key == ord(' '):  # space : 즉시 정지
                current_target_speed = 0.0
                print("!!! STOP !!!")

            # 7) 모터 제어 (조향 포함)
            # 정지 조건
            if current_target_speed == 0:
                robot.stop()
            # 주행 중 조향(P 제어)
            else:
                steering = error * KP
                # error = 0 : 좌우 동일 속도 -> 직진
                # error != 0 : 한쪽을 더 빠르게/느리게 -> 회전
                left_motor_speed = current_target_speed + steering
                right_motor_speed = current_target_speed - steering

                # 안전하게 모터 속도 제한
                left_motor_speed = max(0, min(MAX_SPEED_LIMIT, left_motor_speed))
                right_motor_speed = max(0, min(MAX_SPEED_LIMIT, right_motor_speed))

                # 마지막으로 모터에 실제 적용
                robot.left_motor.value = left_motor_speed
                robot.right_motor.value = right_motor_speed

            # 8) 실제 속도 계산 + MQTT 발행 (0.5초마다)
            # 시간 간격 확인
            current_time = time.time()
            dt = current_time - last_calc_time

            # 펄스 -> 이동거리(cm)로 변환
            if dt >= CALC_INTERVAL:
                # pulse_count / PULSE_PER_ROTATION = 바퀴가 몇 바퀴 돌았는지
                # 바퀴 1회전 거리 = 원주 = π × 지름
                # 둘을 곱해서 이동거리(cm)
                distance_cm = (pulse_count / PULSE_PER_ROTATION) * (np.pi * WHEEL_DIAMETER_CM)
                real_speed_cm_s = distance_cm / dt  # 속도(m/s)

                # MQTT 로 속도값 전송 -> 팔로워는 이 속도값을 수신해서 '리더 속도'로 사용
                payload = f"{real_speed_cm_s:.2f}"
                client.publish(MQTT_TOPIC, payload)

                cv2.putText(view, f"Real: {real_speed_cm_s:.1f} cm/s", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(view, f"PWM: {current_target_speed:.2f}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                pulse_count = 0
                last_calc_time = current_time

            # 6) 화면 출력
            cv2.imshow("Lane View", view)
            # cv2.imshow("ROI Debug", edges_roi) # ROI만 보고 싶으면 주석 해제

    finally:
        print("Cleaning up...")
        robot.stop()
        picam2.stop()
        cv2.destroyAllWindows()
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()