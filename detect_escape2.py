import cv2
import numpy as np


def sort_points_clockwise(points):
    """
    점들을 시계방향으로 정렬하여 올바른 다각형을 만듭니다.
    """
    # 중심점 계산
    center_x = sum(p[0] for p in points) / len(points)
    center_y = sum(p[1] for p in points) / len(points)
    center = (center_x, center_y)
    
    # 각 점에서 중심점까지의 각도 계산
    def get_angle(point):
        import math
        return math.atan2(point[1] - center[1], point[0] - center[0])
    
    # 각도 기준으로 정렬 (시계방향)
    sorted_points = sorted(points, key=get_angle)
    return sorted_points


def polygon_selector(temp_img):
    points = []
    
    print("Polygon ROI 설정")
    print("다각형의 꼭짓점 좌표를 입력하세요 (최소 3개)")
    print("좌표 입력 형식: x,y (예: 100,200)")
    print("순서는 상관없습니다. 자동으로 다각형을 만들어줍니다.")
    print("입력을 완료하려면 'done'을 입력하세요")
    print("다시 시작하려면 'reset'을 입력하세요")
    
    while True:
        user_input = input(f"점 {len(points)+1} 좌표 입력: ").strip()
        
        if user_input.lower() == 'done':
            if len(points) >= 3:
                # 점들을 시계방향으로 정렬
                sorted_points = sort_points_clockwise(points)
                print(f"\n다각형 완성: {len(sorted_points)}개 점으로 구성")
                print("정렬된 점들 (시계방향):")
                for i, point in enumerate(sorted_points):
                    print(f"점 {i+1}: ({point[0]}, {point[1]})")
                return np.array(sorted_points, dtype=np.int32)
            else:
                print("최소 3개의 점이 필요합니다. 계속 입력하세요.")
                continue
                
        elif user_input.lower() == 'reset':
            points = []
            print("좌표가 초기화되었습니다. 다시 입력하세요.")
            continue
            
        try:
            x, y = map(int, user_input.split(','))
            points.append((x, y))
            print(f"점 {len(points)} 추가됨: ({x}, {y})")
        except ValueError:
            print("잘못된 형식입니다. x,y 형식으로 입력하세요 (예: 100,200)")
    
    return np.array(points, dtype=np.int32)


# polygon 내에 객체 있는지 확인하는 함수
def is_point_in_polygon(point, polygon):
    """
    point: (x, y) 튜플
    polygon: numpy array of shape (N, 2)
    """
    return cv2.pointPolygonTest(polygon, point, measureDist=False) >= 0


def draw_puttext(img, content, point):
    # cv2.putText(img, content, point, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    print(f"Text: {content} at position {point}")


def detect_escape(boxes, track_ids, frame, frame_cnt, polygon, w, h, polygon_text="Safe Zone"):
    sky_blue = (255, 255, 0)
    
    # 다각형 영역을 alpha 블렌딩으로 채우기
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], color=sky_blue)
    
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # 테두리는 진하게 그리기 (alpha 없이)
    cv2.polylines(frame, [polygon], isClosed=True, color=sky_blue, thickness=3)
    
    # 다각형 위쪽에 텍스트 추가
    # 다각형의 최상단 y 좌표 찾기
    min_y = min(polygon[:, 1])
    # 다각형의 중심 x 좌표 계산
    center_x = int(sum(polygon[:, 0]) / len(polygon))
    
    # 텍스트 위치 (다각형 위쪽 30픽셀)
    text_position = (center_x - 150, min_y)
    
    # 텍스트 배경 사각형 그리기
    text_size = cv2.getTextSize(polygon_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_bg_pt1 = (text_position[0] - 5, text_position[1] - text_size[1] - 5)
    text_bg_pt2 = (text_position[0] + text_size[0] + 5, text_position[1] + 5)
    cv2.rectangle(frame, text_bg_pt1, text_bg_pt2, (0, 0, 0), -1)
    
    # 텍스트 그리기
    cv2.putText(frame, polygon_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    escaped_ids = []
    # obj center point show-up & escaping detection
    for box, id in zip(boxes, track_ids):
        cx, cy, bw, bh = box
        lx, ly = cx - bw/2, cy + bh/2
        rx, ry = cx + bw/2, cy + bh/2 


        if (not is_point_in_polygon((lx, ly), polygon)) or (not is_point_in_polygon((rx, ry), polygon)):
            escaped_ids.append(id)

    return frame, escaped_ids
