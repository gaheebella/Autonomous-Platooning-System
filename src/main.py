import cv2
import numpy as np
from lane_detect import (
    canny_edge, apply_roi, detect_lines_p,
    average_slope_intercept, draw_lane_lines, draw_lane_center
)

def main():
    cap = cv2.VideoCapture('./data/sampleroad.mp4')  # 또는 0 (웹캠)
    if not cap.isOpened():
        print("영상 또는 카메라를 열 수 없습니다.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1) Canny
        edges, frame_small = canny_edge(frame, resize_to=(320,240),
                                        blur_ksize=(5,5), canny_low=50, canny_high=150)

        # 2) ROI: 하단 사다리꼴/삼각형 중 택1 (예시는 삼각형)
        h, w = edges.shape[:2]
        roi_poly = np.array([(0,h-1), (w-1,h-1), (w//2, int(h*0.58))])  # 필요 시 조정
        edges_roi = apply_roi(edges, roi_poly)

        # 3) HoughLinesP
        lines = detect_lines_p(edges_roi,
                               rho=1, theta=np.pi/180,
                               threshold=30, min_line_len=30, max_line_gap=5)

        # 4) 좌/우 차선 평균화
        left_line, right_line = average_slope_intercept(lines, img_w=w, img_h=h,
                                                        min_abs_slope=0.5, max_abs_slope=5.0)

        # 5) 시각화
        view = draw_lane_lines(frame_small, left_line, right_line)
        view = draw_lane_center(view, left_line, right_line)

        # 디버그 창
        cv2.imshow("Original", frame_small)
        cv2.imshow("Canny ROI", edges_roi)
        cv2.imshow("Lane View", view)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
