import cv2
import numpy as np

# ------------------------------------------------------
# 1) Canny 에지
# ------------------------------------------------------
def canny_edge(frame, resize_to=(320, 240), blur_ksize=(5, 5), canny_low=50, canny_high=150):
    """
    - frame: BGR 입력
    - return: 8bit 에지 이미지 (0/255)
    """
    if resize_to is not None:
        frame = cv2.resize(frame, resize_to)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, blur_ksize, 0)
    edge = cv2.Canny(blur, canny_low, canny_high)  # 내부에 NMS 포함
    return edge, frame  # 에지, 리사이즈된 원본

# ------------------------------------------------------
# 2) ROI(관심영역) 마스크
# ------------------------------------------------------
def apply_roi(edge, polygon_pts):
    """
    - edge: 에지 이미지
    - polygon_pts: 다각형 꼭짓점 ndarray(shape=(N,2)), 예) np.array([(0,239),(319,239),(160,140)])
    - return: ROI가 적용된 에지
    """
    mask = np.zeros_like(edge)
    cv2.fillPoly(mask, [polygon_pts.astype(np.int32)], 255)
    roi = cv2.bitwise_and(edge, mask)
    return roi

# ------------------------------------------------------
# 3) HoughLinesP로 선 검출
# ------------------------------------------------------
def detect_lines_p(edge_roi,
                   rho=1,
                   theta=np.pi/180,
                   threshold=30,
                   min_line_len=30,
                   max_line_gap=5):
    """
    - return: lines (None 또는 shape=(M,1,4), x1,y1,x2,y2)
    """
    lines = cv2.HoughLinesP(edge_roi, rho, theta, threshold,
                            minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    return lines

# ------------------------------------------------------
# 4) 좌/우 차선 분리 및 평균화
#    (기울기 부호로 좌/우 분리, 각 그룹을 평균 직선으로 표현)
# ------------------------------------------------------
def average_slope_intercept(lines, img_w, img_h, min_abs_slope=0.5, max_abs_slope=5.0):
    """
    - lines: HoughLinesP 결과
    - return: (left_line, right_line) 각 라인은 (x1,y1,x2,y2) 또는 None
    """
    if lines is None:
        return None, None

    left, right = [], []
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        if x2 == 0 and x1 == 0:
            continue
        dx = (x2 - x1)
        if dx == 0:
            continue
        slope = (y2 - y1) / dx
        intercept = y1 - slope * x1

        # 너무 누운 선/너무 급한 선 제외
        if not (min_abs_slope <= abs(slope) <= max_abs_slope):
            continue

        # 화면 중앙 기준 좌/우 분리(기울기 부호 사용)
        if slope < 0:  # 왼쪽 차선(우하향)
            left.append((slope, intercept))
        else:          # 오른쪽 차선(우상향)
            right.append((slope, intercept))

    def make_line(avg_params):
        if len(avg_params) == 0:
            return None
        slope = np.mean([p[0] for p in avg_params])
        intercept = np.mean([p[1] for p in avg_params])
        # y=mx+b -> 두 개의 y를 정해 x 계산하여 라인 구간 설정
        y1 = img_h - 1
        y2 = int(img_h * 0.6)  # 위쪽은 화면의 60% 높이까지
        if slope == 0:
            return None
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return (x1, y1, x2, y2)

    left_line  = make_line(left)
    right_line = make_line(right)
    return left_line, right_line

# ------------------------------------------------------
# 5) 시각화: 차선 및 중앙선
# ------------------------------------------------------
def draw_lane_lines(frame, left_line, right_line, color_left=(0,255,255), color_right=(0,255,0)):
    out = frame.copy()
    if left_line is not None:
        x1,y1,x2,y2 = left_line
        cv2.line(out, (x1,y1), (x2,y2), color_left, 3)
    if right_line is not None:
        x1,y1,x2,y2 = right_line
        cv2.line(out, (x1,y1), (x2,y2), color_right, 3)
    return out

def draw_lane_center(frame, left_line, right_line, color=(255,0,0)):
    """
    좌/우 차선의 하단 교차점 x좌표 평균으로 중앙선을 표시
    """
    out = frame.copy()
    h, w = out.shape[:2]
    base_y = h - 1

    xs = []
    for line in (left_line, right_line):
        if line is None:
            continue
        x1,y1,x2,y2 = line
        # line 방정식으로 y=base_y에서의 x를 구함
        if (y2 - y1) == 0:
            continue
        # 선분이 수평에 가까우면 스킵
        xs.append(x1 if y1 == base_y else (x2 if y2 == base_y else None))
        if xs[-1] is None:
            # 일반식으로 환산: y = m x + b
            dx = (x2 - x1)
            if dx == 0:
                xs.pop(); continue
            m = (y2 - y1) / dx
            b = y1 - m * x1
            if m == 0:
                xs.pop(); continue
            x_at_base = int((base_y - b) / m)
            xs[-1] = x_at_base

    if len(xs) >= 1:
        # 화면 중앙
        cx_img = w // 2
        if len(xs) == 2:
            cx_lane = int(np.mean(xs))
        else:
            cx_lane = xs[0]
        cv2.line(out, (cx_lane, base_y), (cx_lane, int(h*0.6)), color, 2)
        cv2.line(out, (cx_img,  base_y), (cx_img,  int(h*0.6)), (200,200,200), 1)
        cv2.putText(out, f"offset(px): {cx_lane - cx_img}", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out
