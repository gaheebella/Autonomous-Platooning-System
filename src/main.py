# src/main.py
import cv2
from edge_detect import sobel_edge

def main():
    # --- 1. 영상 불러오기 ---
    cap = cv2.VideoCapture('./data/sampleroad.mp4')  # 또는 0 (USB 카메라)
    
    if not cap.isOpened():
        print("영상 또는 카메라를 열 수 없습니다.")
        return

    # --- 2. 프레임 단위 처리 ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sobel Edge Detection
        edge = sobel_edge(frame)

        # 원본 & 결과 동시 표시
        cv2.imshow("Original", cv2.resize(frame, (320, 240)))
        cv2.imshow("Sobel Edge", edge)

        # ESC키 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
