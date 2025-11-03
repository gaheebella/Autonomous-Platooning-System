# src/edge_detect.py
import cv2
import numpy as np

def sobel_edge(frame):
    """
    RC카용 경량 Sobel Edge Detection
    - frame: BGR 이미지 (320x240 추천)
    - return: 이진화된 edge 이미지
    """
    # 1. 전처리
    frame = cv2.resize(frame, (320, 240))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Sobel X, Y 계산 (정수 타입)
    sobelx = cv2.Sobel(blur, cv2.CV_16S, 1, 0, ksize=3)
    sobely = cv2.Sobel(blur, cv2.CV_16S, 0, 1, ksize=3)

    abs_grad_x = cv2.convertScaleAbs(sobelx)
    abs_grad_y = cv2.convertScaleAbs(sobely)

    # 3. 두 방향 합성
    edge = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    # 4. 임계값으로 이진화
    _, edge_bin = cv2.threshold(edge, 50, 255, cv2.THRESH_BINARY)

    return edge_bin
