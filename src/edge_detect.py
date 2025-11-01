
import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('./data/3.mp4')
ret, frame = cap.read()

# 1️⃣ 리사이징
frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
height, width = frame.shape[:2]

# 2️⃣ 관심영역 (ROI) 설정 - 필요시 수정 가능
roi = frame[220:height-12, :width, 2]  # Red 채널

# 3️⃣ Sobel 필터 함수 정의
def sobel_xy(img, orient='x', thresh=(20, 100)):
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    else:
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))

    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 255
    return binary_output

# 4️⃣ Sobel X, Y 계산
th_sobelx, th_sobely = (35, 100), (30, 255)
sobel_x = sobel_xy(roi, 'x', th_sobelx)
sobel_y = sobel_xy(roi, 'y', th_sobely)

# 5️⃣ Gradient magnitude 계산
sobelx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
gradmag = np.sqrt(sobelx**2 + sobely**2)
gradmag = np.uint8(255 * gradmag / np.max(gradmag))
gradient_magnitude = np.zeros_like(gradmag)
gradient_magnitude[(gradmag >= 30) & (gradmag <= 255)] = 255

# 6️⃣ Gradient direction 계산
absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
gradient_direction = np.zeros_like(absgraddir)
gradient_direction[(absgraddir >= 0.7) & (absgraddir <= 1.3)] = 255
gradient_direction = gradient_direction.astype(np.uint8)

# 7️⃣ 결과 결합 (원하면)
grad_combine = np.zeros_like(gradient_direction).astype(np.uint8)
grad_combine[((sobel_x > 1) & (gradient_magnitude > 1) & (gradient_direction > 1)) |
             ((sobel_x > 1) & (sobel_y > 1))] = 255

# 결과 확인
plt.imshow(grad_combine, cmap='gray')
plt.title('Sobel Filtered Edges')
plt.show()

