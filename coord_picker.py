"""
영상에서 클릭하여 좌표값 추출하는 도구
사용법: python coord_picker.py --input video.mp4
"""
import cv2
import click


# 클릭한 좌표 저장
clicked_points = []


def mouse_callback(event, x, y, flags, param):
    """마우스 클릭 이벤트 콜백"""
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"[점 {len(clicked_points)}] ({x}, {y})")
        print(f"  → 복사용: {x},{y}")


@click.command()
@click.option("--input", required=True, help="비디오 파일 경로")
@click.option("--frame-num", default=0, help="사용할 프레임 번호 (기본: 0번 프레임)")
def main(input, frame_num):
    """영상에서 클릭하여 좌표값 추출"""
    cap = cv2.VideoCapture(input)

    if not cap.isOpened():
        print(f"[ERROR] 영상 열기 실패: {input}")
        return

    # 특정 프레임으로 이동
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()

    if not ret:
        print(f"[ERROR] 프레임 읽기 실패")
        return

    h, w = frame.shape[:2]
    print(f"\n=== 좌표 추출 도구 ===")
    print(f"영상 크기: {w} x {h}")
    print(f"프레임 번호: {frame_num}")
    print(f"\n[사용법]")
    print(f"  - 마우스 클릭: 좌표 추출")
    print(f"  - 'c' 키: 좌표 목록 초기화")
    print(f"  - 'p' 키: 다각형 좌표 출력 (polygon용)")
    print(f"  - 'q' 또는 ESC: 종료")
    print(f"\n클릭한 좌표:")

    cv2.namedWindow("Coordinate Picker")
    cv2.setMouseCallback("Coordinate Picker", mouse_callback)

    while True:
        # 프레임 복사본에 클릭한 점들 표시
        display = frame.copy()

        for i, (px, py) in enumerate(clicked_points):
            # 점 그리기
            cv2.circle(display, (px, py), 5, (0, 0, 255), -1)
            # 좌표 텍스트
            cv2.putText(display, f"{i+1}:({px},{py})", (px+10, py-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 다각형 미리보기 (3개 이상일 때)
        if len(clicked_points) >= 3:
            # numpy 없이 다각형 그리기
            for i in range(len(clicked_points)):
                pt1 = clicked_points[i]
                pt2 = clicked_points[(i + 1) % len(clicked_points)]
                cv2.line(display, pt1, pt2, (255, 255, 0), 2)

        cv2.imshow("Coordinate Picker", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:  # q 또는 ESC
            break
        elif key == ord('c'):  # 초기화
            clicked_points.clear()
            print("\n[좌표 초기화됨]")
            print("클릭한 좌표:")
        elif key == ord('p'):  # 다각형 좌표 출력
            if len(clicked_points) >= 3:
                print(f"\n=== 다각형 좌표 (polygon_selector용) ===")
                for px, py in clicked_points:
                    print(f"{px},{py}")
                print("done")
                print(f"\n=== Python 코드용 ===")
                pts_str = ", ".join([f"({px}, {py})" for px, py in clicked_points])
                print(f"polygon = np.array([{pts_str}], dtype=np.int32)")
            else:
                print("[WARN] 최소 3개 점이 필요합니다.")

    cap.release()
    cv2.destroyAllWindows()

    # 최종 출력
    if clicked_points:
        print(f"\n=== 최종 좌표 목록 ===")
        for i, (px, py) in enumerate(clicked_points):
            print(f"점 {i+1}: ({px}, {py})")


if __name__ == "__main__":
    main()
