from picamera2 import Picamera2
import cv2
import numpy as np
from lane_detect import (
    canny_edge, apply_roi, detect_lines_p,
    average_slope_intercept, draw_lane_lines, draw_lane_center
)

def main():
    picam2 = Picamera2()

    # 640x480 BGR 포맷 프리뷰 설정
    config = picam2.create_preview_configuration(
        main={"format": "BGR888", "size": (640, 480)}
    )
    picam2.configure(config)
    picam2.start()

    while True:
        # Picamera2 프레임 획득
        frame = picam2.capture_array()
        if frame is None:
            print("프레임을 읽을 수 없습니다.")
            break

        # 1) Canny
        edges, frame_small = canny_edge(
            frame,
            resize_to=(320, 240),
            blur_ksize=(5, 5),
            canny_low=50,
            canny_high=150
        )

        # 2) ROI
        h, w = edges.shape[:2]
        roi_poly = np.array([(0, h - 1), (w - 1, h - 1), (w // 2, int(h * 0.58))])
        edges_roi = apply_roi(edges, roi_poly)

        # 3) HoughLinesP
        lines = detect_lines_p(
            edges_roi,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            min_line_len=30,
            max_line_gap=5
        )

        # 4) 좌/우 차선 평균화
        left_line, right_line = average_slope_intercept(
            lines, img_w=w, img_h=h,
            min_abs_slope=0.5, max_abs_slope=5.0
        )

        # 5) 시각화
        view = draw_lane_lines(frame_small, left_line, right_line)
        view = draw_lane_center(view, left_line, right_line)

        # 디버그 창
        cv2.imshow("Original", frame_small)
        cv2.imshow("Canny ROI", edges_roi)
        cv2.imshow("Lane View", view)

        # 종료 키 (ESC)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
