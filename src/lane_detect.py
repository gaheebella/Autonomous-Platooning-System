# 리더차량

import cv2
import numpy as np

# 라즈베리파이 최적화
cv2.setUseOptimized(True)


def canny_edge(frame, resize_to=(320, 240), blur_ksize=(5, 5), canny_low=50, canny_high=150):
    if resize_to is not None:
        frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, blur_ksize, 0)
    edge = cv2.Canny(blur, canny_low, canny_high)
    return edge, frame


def apply_roi(edge, polygon_pts):
    mask = np.zeros_like(edge)
    cv2.fillPoly(mask, [polygon_pts.astype(np.int32)], 255)
    roi = cv2.bitwise_and(edge, mask)
    return roi


def detect_lines_p(edge_roi, rho=1, theta=np.pi / 180, threshold=30, min_line_len=30, max_line_gap=5):
    lines = cv2.HoughLinesP(edge_roi, rho, theta, threshold,
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines


def average_slope_intercept(lines, img_w, img_h, min_abs_slope=0.5, max_abs_slope=5.0):
    if lines is None:
        return None, None

    left, right = [], []
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        if x2 == x1: continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        if not (min_abs_slope <= abs(slope) <= max_abs_slope): continue

        if slope < 0:
            left.append((slope, intercept))
        else:
            right.append((slope, intercept))

    def make_line(avg_params):
        if not avg_params: return None
        slope = float(np.mean([p[0] for p in avg_params]))
        intercept = float(np.mean([p[1] for p in avg_params]))
        y1 = img_h - 1
        y2 = int(img_h * 0.6)
        if slope == 0: return None
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return (x1, y1, x2, y2)

    return make_line(left), make_line(right)


def draw_lane_lines(frame, left_line, right_line):
    out = frame.copy()
    if left_line: cv2.line(out, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 255, 255), 3)
    if right_line: cv2.line(out, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 255, 0), 3)
    return out


def draw_lane_center(frame, left_line, right_line):
    """
    반환값 수정: (이미지, 오차값 error)
    error < 0 : 왼쪽으로 치우침 (우회전 필요)
    error > 0 : 오른쪽으로 치우침 (좌회전 필요)
    """
    out = frame.copy()
    h, w = out.shape[:2]
    base_y = h - 1
    cx_img = w // 2

    error = 0  # 기본값 (직진)
    xs = []

    for line in (left_line, right_line):
        if line is None: continue
        x1, y1, x2, y2 = line
        if x2 == x1: continue
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        if m == 0: continue
        x_at_base = int((base_y - b) / m)
        xs.append(x_at_base)

    if len(xs) >= 1:
        if len(xs) == 2:
            cx_lane = int(np.mean(xs))
        else:
            cx_lane = xs[0]

        # 오차 계산
        error = cx_lane - cx_img

        cv2.line(out, (cx_lane, base_y), (cx_lane, int(h * 0.6)), (255, 0, 0), 2)
        cv2.line(out, (cx_img, base_y), (cx_img, int(h * 0.6)), (200, 200, 200), 1)
        cv2.putText(out, f"Err: {error}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return out, error