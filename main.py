from collections import deque

import av
import click
import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image, ImageDraw, ImageFont
# pilmoji 대신 Windows Segoe UI Emoji 폰트 사용
PILMOJI_AVAILABLE = False  # pilmoji 비활성화
from ultralytics import YOLO

from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from detection import detect_fight, detect_inert, detect_sleep, detect_eat, detect_bathroom
from detection import detect_escape, polygon_selector
from tracking import (
    track_with_botsort,
    track_with_bytetrack,
    track_with_deepsort,
)
from reid import ReIDTracker
from tools import apply_blur, apply_mosaic, apply_black_box


# ============================================================================
# 하드코딩 설정 (여기서 직접 수정하세요)
# ============================================================================

# --- Supervision bbox 설정 ---
CONFIG_USE_SUPERVISION = False                   # supervision 사용 여부 (Pillow 사용 시 False)
CONFIG_SV_BOX_COLOR = (0, 255, 0)               # 박스 색상 (RGB) - 초록
CONFIG_SV_BOX_THICKNESS = 2                      # 박스 두께
CONFIG_SV_BOX_STYLE = "normal"                   # "normal", "round", "corner"
CONFIG_SV_LABEL_SCALE = 0.5                      # 라벨 텍스트 크기
CONFIG_SV_LABEL_COLOR = (255, 255, 255)         # 라벨 텍스트 색상 (RGB) - 흰색
CONFIG_SV_LABEL_BG_COLOR = (0, 255, 0)          # 라벨 배경 색상 (RGB) - 초록
CONFIG_SV_LABEL_PADDING = 5                      # 라벨 패딩
CONFIG_SV_LABEL_POSITION = "TOP_LEFT"            # "TOP_LEFT", "TOP_RIGHT", "TOP_CENTER", "BOTTOM_LEFT", "BOTTOM_RIGHT", "BOTTOM_CENTER"
CONFIG_SV_LABEL_FORMAT = "ID: {id}"              # 라벨 포맷 ({id}, {conf} 사용 가능)
CONFIG_SV_SHOW_TRACE = False                     # 궤적 표시 여부
CONFIG_SV_SHOW_CENTER = True                     # 중심점 표시 여부

# --- Pillow 텍스트 설정 (한글/이모티콘 지원) ---
CONFIG_USE_PILLOW_TEXT = True                    # Pillow 텍스트 사용 여부
CONFIG_PILLOW_FONT_PATH = "./font/Jersey/Jersey20-Regular.ttf"  # 폰트 경로 (None이면 자동 탐색)
CONFIG_PILLOW_FONT_SIZE = 20                     # 폰트 크기
CONFIG_PILLOW_TEXT_COLOR = (0, 0, 0)             # 텍스트 색상 (RGB) - 검정
CONFIG_PILLOW_BG_COLOR = None                    # 배경 색상 (None이면 투명)
CONFIG_PILLOW_PADDING = 1                        # 배경 패딩

# --- 상태 텍스트 커스터마이징 ---
CONFIG_FIGHT_TEXT = "Fight"                      # Fight 감지 시 표시할 텍스트
CONFIG_ESCAPE_TEXT = "Escape"                    # Escape 감지 시 표시할 텍스트
CONFIG_INERT_TEXT = "Inactive"                   # Inactive 감지 시 표시할 텍스트
CONFIG_SLEEP_TEXT = "Sleep"                       # Sleep 감지 시 표시할 텍스트
CONFIG_EAT_TEXT = "Eating"                        # Eating 감지 시 표시할 텍스트
CONFIG_BATHROOM_TEXT = "Bathroom"                  # Bathroom 감지 시 표시할 텍스트
CONFIG_STATE_PREFIX = ""                         # 상태 텍스트 앞에 붙는 접두사
CONFIG_NORMAL_TEXT = ""                          # 아무것도 감지되지 않았을 때 텍스트

# --- 행동별 bbox 색상 설정 ---
CONFIG_NORMAL_BOX_COLOR = (0, 255, 0)            # 정상 상태 bbox 색상 (RGB) - 초록
CONFIG_FIGHT_BOX_COLOR = (255, 0, 0)             # 싸움 감지 bbox 색상 (RGB) - 빨강
CONFIG_ESCAPE_BOX_COLOR = (255, 255, 0)          # 이탈 감지 bbox 색상 (RGB) - 노랑
CONFIG_INERT_BOX_COLOR = (0, 0, 255)             # 무기력 감지 bbox 색상 (RGB) - 파랑
CONFIG_PLAY_BOX_COLOR = (255, 165, 0)            # 활동 감지 bbox 색상 (RGB) - 주황
CONFIG_SLEEP_BOX_COLOR = (128, 0, 128)           # 수면 감지 bbox 색상 (RGB) - 보라
CONFIG_EAT_BOX_COLOR = (255, 105, 180)           # 식사 감지 bbox 색상 (RGB) - 분홍
CONFIG_BATHROOM_BOX_COLOR = (0, 191, 255)        # 배변 감지 bbox 색상 (RGB) - 하늘

# --- 행동별 이모지 설정 ---
CONFIG_FIGHT_EMOJI = "🥊"                        # 싸움 감지 시 ID 옆에 표시할 이모지
CONFIG_ESCAPE_EMOJI = "⚠️"                       # 이탈 감지 시 ID 옆에 표시할 이모지
CONFIG_INERT_EMOJI = "❄️"                        # 무기력 감지 시 ID 옆에 표시할 이모지
CONFIG_NORMAL_EMOJI = ""                         # 정상 상태 이모지 (빈 문자열이면 표시 안 함)
CONFIG_PLAY_EMOJI = "🎾"                         # 활동 감지 시 이모지
CONFIG_SLEEP_EMOJI = "😴"                        # 수면 감지 시 이모지
CONFIG_EAT_EMOJI = "🍽️"                         # 식사 감지 시 이모지
CONFIG_BATHROOM_EMOJI = "🚽"                     # 배변 감지 시 이모지

# --- Pillow 라벨 설정 ---
CONFIG_USE_PILLOW_LABEL = True                   # Pillow로 라벨 그리기 (이모지 지원)
CONFIG_LABEL_FONT_SIZE = 16                      # 라벨 폰트 크기
CONFIG_LABEL_TEXT_COLOR = (255, 255, 255)        # 라벨 텍스트 색상 (RGB) - 흰색
CONFIG_LABEL_BG_PADDING = 3                      # 라벨 배경 패딩

# ============================================================================


def save_video(f, frames, fps):
    container = av.open(f, mode="w")

    stream = container.add_stream("h264", rate=fps)

    first_frame = frames[0]

    stream.height = first_frame.shape[0]
    stream.width = first_frame.shape[1]
    stream.pix_fmt = "yuv420p"

    for img in frames:
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()

    return f


# ============================================================================
# Pillow 텍스트 그리기 함수 (한글/이모티콘 지원)
# ============================================================================
def draw_text_pillow(
    frame: np.ndarray,
    text: str,
    position: tuple,
    font_path: str = None,
    font_size: int = 24,
    color: tuple = (255, 255, 255),  # RGB
    bg_color: tuple = None,  # 배경색 (None이면 투명)
    padding: int = 5,
) -> np.ndarray:
    """
    Pillow를 사용하여 프레임에 텍스트를 그립니다.
    한글, 이모티콘 등 유니코드 문자를 지원합니다.

    Args:
        frame: BGR 형식의 OpenCV 이미지
        text: 그릴 텍스트 (이모티콘 포함 가능, 예: "🐕 강아지 감지!")
        position: (x, y) 텍스트 시작 위치
        font_path: 폰트 파일 경로 (None이면 기본 폰트 사용)
                   예: "C:/Windows/Fonts/malgun.ttf" (윈도우 맑은고딕)
                   예: "/usr/share/fonts/truetype/nanum/NanumGothic.ttf" (리눅스)
        font_size: 폰트 크기
        color: RGB 텍스트 색상 (예: (255, 0, 0) = 빨강)
        bg_color: RGB 배경색 (None이면 배경 없음)
        padding: 배경 패딩

    Returns:
        텍스트가 그려진 BGR 프레임
    """
    # BGR -> RGB 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_image)

    # 폰트 로드
    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            # 기본 폰트 시도 (Windows/Linux/Mac 순서로)
            default_fonts = [
                "C:/Windows/Fonts/malgun.ttf",  # Windows 맑은고딕
                "C:/Windows/Fonts/seguiemj.ttf",  # Windows 이모지 폰트
                "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Linux
                "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # Mac
            ]
            font = None
            for fp in default_fonts:
                try:
                    font = ImageFont.truetype(fp, font_size)
                    break
                except (IOError, OSError):
                    continue
            if font is None:
                font = ImageFont.load_default()
    except (IOError, OSError):
        font = ImageFont.load_default()

    # 텍스트 바운딩 박스 계산
    bbox = draw.textbbox(position, text, font=font)

    # 배경 그리기 (옵션)
    if bg_color is not None:
        bg_bbox = (
            bbox[0] - padding,
            bbox[1] - padding,
            bbox[2] + padding,
            bbox[3] + padding,
        )
        draw.rectangle(bg_bbox, fill=bg_color)

    # 텍스트 그리기
    draw.text(position, text, font=font, fill=color)

    # RGB -> BGR 변환
    frame_result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return frame_result


def get_pillow_font(font_path: str = None, font_size: int = 16):
    """Pillow 폰트를 로드합니다."""
    try:
        if font_path:
            return ImageFont.truetype(font_path, font_size)
        else:
            default_fonts = [
                "C:/Windows/Fonts/malgun.ttf",
                "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            ]
            for fp in default_fonts:
                try:
                    return ImageFont.truetype(fp, font_size)
                except (IOError, OSError):
                    continue
            return ImageFont.load_default()
    except (IOError, OSError):
        return ImageFont.load_default()


def load_emoji_images(emoji_dir: str = None, size: int = 32) -> dict:
    """이모지 PNG 이미지를 로드합니다."""
    import os

    if emoji_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        emoji_dir = os.path.join(script_dir, "assets", "emoji")

    emoji_images = {}

    # 이모지 이름 -> 파일명 매핑
    emoji_files = {
        "fight": ["fight.png", "warning.png", "⚠️.png", "⚠.png"],
        "escape": ["escape.png", "run.png", "🏃.png"],
        "inert": ["inert.png", "snowflake.png", "❄️.png", "❄.png"],
        "play": ["play.png", "tennis.png", "🎾.png"],
        "sleep": ["sleep.png", "zzz.png", "💤.png", "😴.png", "1f4a4.png", "1f634.png"],
        "eat": ["eat.png", "food.png", "🍽️.png", "🍽.png"],
        "bathroom": ["bathroom.png", "toilet.png", "🚽.png"],
    }

    for state, filenames in emoji_files.items():
        for filename in filenames:
            filepath = os.path.join(emoji_dir, filename)
            if os.path.exists(filepath):
                try:
                    img = Image.open(filepath).convert("RGBA")
                    img = img.resize((size, size), Image.Resampling.LANCZOS)
                    emoji_images[state] = img
                    break
                except Exception:
                    continue

    return emoji_images


# 전역 이모지 이미지 캐시
EMOJI_IMAGES = {}


def draw_bbox_with_pillow_label(
    frame: np.ndarray,
    boxes: list,
    track_ids: list,
    id_states: dict,  # {track_id: "fight" | "escape" | "inert" | "normal"}
    font_path: str = None,
    font_size: int = 16,
    label_text_color: tuple = (255, 255, 255),
    label_bg_padding: int = 3,
    box_thickness: int = 2,
    emoji_size: int = 32,
    global_id_map: dict = None,  # {track_id: global_id} 매핑 (ReID full pipeline용)
) -> np.ndarray:
    """
    Pillow를 사용하여 bbox와 라벨(ID)을 그립니다.
    행동 상태에 따라 bbox 색상이 변경되고, 이모지 이미지는 bbox 우측 상단에 표시됩니다.
    global_id_map이 주어지면 "ID:X/G:Y" 형식으로 글로벌 ID도 표시합니다.
    """
    global EMOJI_IMAGES

    if len(boxes) == 0:
        return frame

    # 이모지 이미지 로드 (최초 1회)
    if not EMOJI_IMAGES:
        EMOJI_IMAGES = load_emoji_images(size=emoji_size)
        if EMOJI_IMAGES:
            print(f"[INFO] 이모지 이미지 로드 완료: {list(EMOJI_IMAGES.keys())}")
        else:
            print("[WARN] 이모지 이미지 없음. emoji/ 폴더에 fight.png, escape.png 등 추가하세요.")

    # BGR -> RGB 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb).convert("RGBA")
    draw = ImageDraw.Draw(pil_image)

    # 라벨용 폰트 로드
    font = get_pillow_font(font_path, font_size)

    # 이모지 위치 저장 (나중에 오버레이)
    emoji_overlay_list = []

    for box, track_id in zip(boxes, track_ids):
        x_center, y_center, width, height = box
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # 상태에 따른 색상 결정
        state = id_states.get(track_id, "normal")

        if state == "fight":
            box_color = CONFIG_FIGHT_BOX_COLOR
        elif state == "escape":
            box_color = CONFIG_ESCAPE_BOX_COLOR
        elif state == "inert":
            box_color = CONFIG_INERT_BOX_COLOR
        elif state == "sleep":
            box_color = CONFIG_SLEEP_BOX_COLOR
        elif state == "eat":
            box_color = CONFIG_EAT_BOX_COLOR
        elif state == "bathroom":
            box_color = CONFIG_BATHROOM_BOX_COLOR
        else:
            box_color = CONFIG_NORMAL_BOX_COLOR

        # bbox 그리기
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=box_thickness)

        # 라벨 텍스트 (글로벌 ID가 있으면 함께 표시)
        if global_id_map and track_id in global_id_map:
            label_text = f"ID:{track_id}/G:{global_id_map[track_id]}"
        else:
            label_text = f"ID:{track_id}"

        # 라벨 배경 크기 계산
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # 라벨 배경 그리기 (bbox 좌측 상단)
        label_x1 = x1
        label_y1 = y1 - text_height - label_bg_padding * 2
        label_x2 = x1 + text_width + label_bg_padding * 2
        label_y2 = y1

        # 라벨이 화면 위로 넘어가면 bbox 내부 상단으로
        if label_y1 < 0:
            label_y1 = y1
            label_y2 = y1 + text_height + label_bg_padding * 2

        draw.rectangle([label_x1, label_y1, label_x2, label_y2], fill=box_color)

        # 라벨 텍스트 그리기
        text_x = label_x1 + label_bg_padding
        text_y = label_y1 + label_bg_padding
        draw.text((text_x, text_y), label_text, font=font, fill=label_text_color)

        # 이모지 이미지 위치 저장 (bbox 우측 상단)
        if state != "normal" and state in EMOJI_IMAGES:
            emoji_x = x2 + 2
            emoji_y = y1 - emoji_size // 2
            if emoji_y < 0:
                emoji_y = y1 + 2
            emoji_overlay_list.append((emoji_x, emoji_y, state))

    # 이모지 이미지 오버레이
    for emoji_x, emoji_y, state in emoji_overlay_list:
        emoji_img = EMOJI_IMAGES.get(state)
        if emoji_img:
            pil_image.paste(emoji_img, (emoji_x, emoji_y), emoji_img)

    # RGBA -> RGB -> BGR 변환
    pil_image_rgb = pil_image.convert("RGB")
    frame_result = cv2.cvtColor(np.array(pil_image_rgb), cv2.COLOR_RGB2BGR)
    return frame_result


# ============================================================================
# Supervision Annotator 생성 함수 (supervision 버전 호환)
# ============================================================================
def create_supervision_annotators(
    # === BoxAnnotator 설정 ===
    box_color=None,  # 박스 색상
    box_thickness: int = 2,  # 박스 두께
    # === LabelAnnotator 설정 ===
    label_color=None,  # 라벨 배경 색상
    label_text_color=None,  # 라벨 텍스트 색상
    label_text_scale: float = 0.5,  # 라벨 텍스트 크기
    label_text_thickness: int = 1,  # 라벨 텍스트 두께
    label_text_padding: int = 5,  # 라벨 패딩
    label_position=None,  # 라벨 위치
    # === 추가 Annotator 옵션 ===
    use_round_box: bool = False,  # 둥근 박스 사용 여부
    round_box_radius: int = 10,  # 둥근 박스 반경
    use_corner_box: bool = False,  # 코너만 그리는 박스 사용 여부
    corner_length: int = 15,  # 코너 길이
) -> dict:
    """
    Supervision annotator들을 생성합니다.

    사용 예시:
        annotators = create_supervision_annotators(
            box_color=sv.Color.RED,
            box_thickness=3,
            label_text_scale=0.7,
        )

        # 프레임에 적용
        detections = sv.Detections(...)
        frame = annotators['box'].annotate(frame, detections)
        frame = annotators['label'].annotate(frame, detections, labels)

    Returns:
        dict: {
            'box': BoxAnnotator 또는 RoundBoxAnnotator 또는 BoxCornerAnnotator,
            'label': LabelAnnotator,
            'trace': TraceAnnotator (트랙 궤적용),
        }
    """
    # 기본 색상 설정 (None인 경우, 버전 호환)
    try:
        if box_color is None:
            box_color = sv.Color(r=0, g=255, b=0)
        if label_color is None:
            label_color = sv.Color(r=0, g=255, b=0)
        if label_text_color is None:
            label_text_color = sv.Color(r=255, g=255, b=255)
    except Exception:
        box_color = None
        label_color = None
        label_text_color = None

    # 박스 Annotator 선택 (버전 호환)
    try:
        if use_round_box:
            box_annotator = sv.RoundBoxAnnotator(
                color=box_color,
                thickness=box_thickness,
            )
        elif use_corner_box:
            box_annotator = sv.BoxCornerAnnotator(
                color=box_color,
                thickness=box_thickness,
            )
        else:
            box_annotator = sv.BoxAnnotator(
                color=box_color,
                thickness=box_thickness,
            )
    except Exception:
        box_annotator = sv.BoxAnnotator(thickness=box_thickness)

    # 라벨 Annotator (버전 호환)
    try:
        label_annotator = sv.LabelAnnotator(
            color=label_color,
            text_color=label_text_color,
            text_scale=label_text_scale,
            text_thickness=label_text_thickness,
            text_padding=label_text_padding,
        )
    except Exception:
        label_annotator = None

    # 트랙 궤적 Annotator (버전 호환)
    try:
        trace_annotator = sv.TraceAnnotator(
            color=box_color,
            thickness=box_thickness,
            trace_length=30,
        )
    except Exception:
        trace_annotator = None

    return {
        'box': box_annotator,
        'label': label_annotator,
        'trace': trace_annotator,
    }


def annotate_frame_with_supervision(
    frame: np.ndarray,
    boxes: np.ndarray,  # [[x_center, y_center, width, height], ...]
    track_ids: list,
    confidences: list = None,
    annotators: dict = None,
    label_format: str = "ID: {id}",  # "{id}", "{conf}", "{id} ({conf:.2f})" 등
    class_names: dict = None,  # {class_id: "name"} 매핑
) -> np.ndarray:
    """
    Supervision을 사용하여 프레임에 박스와 라벨을 그립니다.

    Args:
        frame: BGR OpenCV 이미지
        boxes: [N, 4] 형태의 박스 배열 (x_center, y_center, w, h)
        track_ids: 트랙 ID 리스트
        confidences: 신뢰도 리스트 (옵션)
        annotators: create_supervision_annotators()에서 생성한 annotator dict
        label_format: 라벨 포맷 문자열
        class_names: 클래스 이름 매핑

    Returns:
        어노테이션된 프레임
    """
    if len(boxes) == 0:
        return frame

    # 기본 annotator 생성
    if annotators is None:
        annotators = create_supervision_annotators()

    # xywh -> xyxy 변환
    boxes_np = np.array(boxes)
    xyxy = np.zeros((len(boxes_np), 4))
    xyxy[:, 0] = boxes_np[:, 0] - boxes_np[:, 2] / 2  # x1
    xyxy[:, 1] = boxes_np[:, 1] - boxes_np[:, 3] / 2  # y1
    xyxy[:, 2] = boxes_np[:, 0] + boxes_np[:, 2] / 2  # x2
    xyxy[:, 3] = boxes_np[:, 1] + boxes_np[:, 3] / 2  # y2

    # Detections 객체 생성
    detections = sv.Detections(
        xyxy=xyxy,
        tracker_id=np.array(track_ids) if track_ids else None,
        confidence=np.array(confidences) if confidences else None,
    )

    # 라벨 생성
    labels = []
    for i, track_id in enumerate(track_ids):
        conf = confidences[i] if confidences else 0.0
        label = label_format.format(id=track_id, conf=conf)
        labels.append(label)

    # 박스 그리기
    frame = annotators['box'].annotate(frame.copy(), detections)

    # 라벨 그리기
    frame = annotators['label'].annotate(frame, detections, labels)

    return frame

@click.command()
@click.option("--model", default="/home/work/Optim/runs/train3/modelv11x_AdamW_0.0001_freeze5_iou0.4_conf0.001/weights/best.pt")
@click.option("--method", required=True)
@click.option("--input", default="/home/work/Tracker2/output/fight_ces_clip.mp4")
@click.option("--output", required=True)
@click.option("--task-fight", is_flag=True)
@click.option("--task-escape", is_flag=True)
@click.option("--task-inert", is_flag=True)
@click.option("--task-sleep", is_flag=True)
@click.option("--task-eat", is_flag=True)
@click.option("--task-bathroom", is_flag=True)
@click.option("--threshold", default=0.1)
@click.option("--inert-threshold", default=50)
@click.option("--inert-frames", default=100)
@click.option("--sleep-threshold", default=30)
@click.option("--sleep-frames", default=200)
@click.option("--sleep-aspect-ratio", default=1.2)
@click.option("--sleep-area-stability", default=0.15)
@click.option("--eat-iou-threshold", default=0.3)
@click.option("--eat-dwell-frames", default=30)
@click.option("--eat-direction-frames", default=10)
@click.option("--bowl-conf", default=0.5)
@click.option("--bathroom-cls-model", default="weights/bathroom_cls.pt")
@click.option("--bathroom-trigger-frames", default=30)
@click.option("--bathroom-height-drop", default=0.25)
@click.option("--bathroom-cls-conf", default=0.5)
@click.option("--reset-frames", default=60)
@click.option("--flag-frames", default=4)
@click.option("--max-number", default=500)
@click.option("--use-reid", is_flag=True, help="ReID 기반 ID 보정 활성화")
@click.option("--reid-method", default="adaptive", type=click.Choice(["adaptive", "histogram", "mobilenet"]), help="ReID 방식 선택")
@click.option("--reid-threshold", default=0.5, help="ReID 동일 객체 판단 유사도 임계값")
@click.option("--reid-global-id", is_flag=True, help="ReID 글로벌 ID 할당 활성화 (full pipeline)")
@click.option("--privacy", is_flag=True, help="사람 감지 후 프라이버시 필터 적용")
@click.option("--privacy-method", type=click.Choice(["blur", "mosaic", "black"]), default="blur", help="프라이버시 필터 방식")
@click.option("--privacy-model", default="yolo11n.pt", help="사람 감지용 YOLO 모델 경로")
def main(
    model,
    method,
    input,
    output,
    task_fight,
    task_escape,
    task_inert,
    task_sleep,
    task_eat,
    task_bathroom,
    threshold,
    inert_threshold,
    inert_frames,
    sleep_threshold,
    sleep_frames,
    sleep_aspect_ratio,
    sleep_area_stability,
    eat_iou_threshold,
    eat_dwell_frames,
    eat_direction_frames,
    bowl_conf,
    bathroom_cls_model,
    bathroom_trigger_frames,
    bathroom_height_drop,
    bathroom_cls_conf,
    reset_frames,
    flag_frames,
    max_number,
    use_reid,
    reid_method,
    reid_threshold,
    reid_global_id,
    privacy,
    privacy_method,
    privacy_model,
):
    model = YOLO(model)
    cap = cv2.VideoCapture(input)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    max_cosine_distance = 0.5
    nn_budget = None
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget
    )
    tracker = Tracker(metric)

    # ========================================================================
    # ReID 트래커 초기화
    # ========================================================================
    reid_tracker = None
    global_id_map = {}  # track_id -> global_id 매핑 (full pipeline용)
    if use_reid or reid_global_id:
        reid_tracker = ReIDTracker(
            reid_method=reid_method,
            similarity_threshold=reid_threshold,
            correction_enabled=True,
            global_id_enabled=reid_global_id,
        )
        mode = "full pipeline (ID 보정 + 글로벌 ID)" if reid_global_id else "경량 (ID 보정)"
        print(f"[INFO] ReID 활성화 - mode: {mode}, method: {reid_method}, threshold: {reid_threshold}")

    # ========================================================================
    # 프라이버시 필터 초기화
    # ========================================================================
    privacy_yolo = None
    if privacy:
        privacy_yolo = YOLO(privacy_model)
        print(f"[INFO] 프라이버시 필터 활성화 - method: {privacy_method}, model: {privacy_model}")

    # ========================================================================
    # Supervision Annotator 설정 (하드코딩 CONFIG 사용)
    # ========================================================================
    sv_annotators = None
    if CONFIG_USE_SUPERVISION and not CONFIG_USE_PILLOW_LABEL:
        try:
            sv_annotators = create_supervision_annotators(
                box_thickness=CONFIG_SV_BOX_THICKNESS,
                use_round_box=(CONFIG_SV_BOX_STYLE == "round"),
                use_corner_box=(CONFIG_SV_BOX_STYLE == "corner"),
            )
            print(f"[INFO] Supervision annotator 초기화 완료 (스타일: {CONFIG_SV_BOX_STYLE})")
        except Exception as e:
            print(f"[WARN] Supervision 초기화 실패, Pillow 사용: {e}")
            sv_annotators = None

    # ========================================================================
    # Pillow 텍스트 설정 (하드코딩 CONFIG 사용)
    # ========================================================================
    if CONFIG_USE_PILLOW_TEXT:
        print(f"[INFO] Pillow 텍스트 활성화 (한글/이모티콘 지원)")

    frames = []

    inert_coor = {}
    sleep_coor = {}
    sleep_bbox = {}
    eat_coor = {}
    eat_near_count = {}
    bathroom_coor = {}
    bathroom_bbox = {}
    close_count = torch.zeros((max_number, max_number))
    far_count = torch.zeros((max_number, max_number))
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)

    first = True
    frame_cnt = 0

    while cap.isOpened():
        success, frame = cap.read()
        w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if success:
            frame_cnt += 1

            # 프라이버시 필터 적용 (사람 감지 → 블러/모자이크/블랙박스)
            if privacy_yolo is not None:
                person_results = privacy_yolo(frame, conf=0.5, classes=[0], verbose=False)
                if len(person_results[0].boxes) > 0:
                    person_boxes = person_results[0].boxes.xyxy.cpu().numpy().astype(int)
                    for pb in person_boxes:
                        px1, py1, px2, py2 = pb
                        px1, py1 = max(0, px1 - 10), max(0, py1 - 10)
                        px2, py2 = min(w, px2 + 10), min(h, py2 + 10)
                        if privacy_method == "blur":
                            frame = apply_blur(frame, px1, py1, px2, py2)
                        elif privacy_method == "mosaic":
                            frame = apply_mosaic(frame, px1, py1, px2, py2)
                        elif privacy_method == "black":
                            frame = apply_black_box(frame, px1, py1, px2, py2)

            if method == "bytetrack":
                boxes, track_ids, frame = track_with_bytetrack(model, frame)
            elif method == "botsort":
                boxes, track_ids, frame = track_with_botsort(model, frame)
            elif method == "deepsort":
                boxes, track_ids, frame = track_with_deepsort(model, tracker, frame)
            else:
                raise NotImplementedError

            # ReID ID 보정 적용
            if reid_tracker is not None and len(boxes) > 0:
                reid_result = reid_tracker.process(frame, boxes, track_ids)
                track_ids = reid_result['corrected_ids']
                if reid_result['corrections']:
                    for c in reid_result['corrections']:
                        print(f"[ReID] ID 보정: {c['old_id']} -> {c['new_id']} (유사도: {c['similarity']:.3f})")
                # 글로벌 ID 매핑 업데이트 (full pipeline)
                if reid_result.get('global_ids'):
                    for tid, gid in zip(track_ids, reid_result['global_ids']):
                        global_id_map[tid] = gid

            for id in track_ids:
                if id not in inert_coor.keys():
                    inert_coor[id] = deque([], maxlen=inert_frames)
                if id not in sleep_coor.keys():
                    sleep_coor[id] = deque([], maxlen=sleep_frames)
                    sleep_bbox[id] = deque([], maxlen=sleep_frames)
                if id not in eat_coor.keys():
                    eat_coor[id] = deque([], maxlen=eat_direction_frames * 3)
                if id not in bathroom_coor.keys():
                    bathroom_coor[id] = deque([], maxlen=bathroom_trigger_frames * 2)
                    bathroom_bbox[id] = deque([], maxlen=bathroom_trigger_frames * 2)

            id_to_index = {}
            for index, id in enumerate(track_ids):
                id_to_index[id] = index

            if len(boxes) == 0:
                continue

            # 중심점 좌표 수집
            x_centers, y_centers = [], []
            for id, box in zip(track_ids, boxes):
                x_center, y_center, width, height = box
                x_centers.append(x_center)
                y_centers.append(y_center)
                inert_coor[id].append([x_center, y_center])
                sleep_coor[id].append([x_center, y_center])
                sleep_bbox[id].append([width, height])
                eat_coor[id].append([x_center, y_center])
                bathroom_coor[id].append([x_center, y_center])
                bathroom_bbox[id].append([width, height])

            x_centers = np.array(x_centers)
            y_centers = np.array(y_centers)

            # ================================================================
            # 행동 감지 (bbox 그리기 전에 먼저 수행)
            # ================================================================
            state_texts = []  # 감지된 상태 텍스트 리스트
            id_states = {id: "normal" for id in track_ids}  # 각 ID의 상태

            # detect fight
            if task_fight:
                fight_indices = detect_fight(
                    x_centers,
                    y_centers,
                    track_ids,
                    close_count,
                    far_count,
                    threshold,
                    reset_frames,
                    flag_frames,
                    int(width),
                    int(height),
                )

                # 싸움 감지된 ID 상태 업데이트
                for ids in fight_indices:
                    for i in ids:
                        if i.item() in id_to_index.keys():
                            id_states[i.item()] = "fight"
                            print(f"[DEBUG] Fight detected! ID: {i.item()}")  # 디버그

                if len(fight_indices) > 0:
                    print(f"[DEBUG] fight_indices: {fight_indices}, id_states: {id_states}")  # 디버그
                    state_texts.append(CONFIG_FIGHT_TEXT)

            # detect escape
            if task_escape:
                if first:
                    print("[INFO] 마우스로 다각형 ROI를 지정해 주세요.")
                    polygon = polygon_selector(frame)

                first = False

                frame, escaped_ids = detect_escape(
                    boxes, track_ids, frame, frame_cnt, polygon, w, h
                )

                # 이탈 감지된 ID 상태 업데이트
                for id in escaped_ids:
                    if id in id_to_index.keys():
                        id_states[id] = "escape"

                if len(escaped_ids):
                    state_texts.append(CONFIG_ESCAPE_TEXT)

            # detect sleep (inert보다 먼저 — sleep이 우선)
            sleep_indices = []
            if task_sleep:
                sleep_indices = detect_sleep(
                    sleep_coor,
                    sleep_bbox,
                    sleep_threshold,
                    sleep_frames,
                    sleep_aspect_ratio,
                    sleep_area_stability,
                )

                # 수면 감지된 ID 상태 업데이트
                for id in sleep_indices:
                    if id in id_to_index.keys():
                        id_states[id] = "sleep"

                if len(sleep_indices) > 0:
                    state_texts.append(CONFIG_SLEEP_TEXT)

            # detect inert (sleep 감지된 ID 제외)
            if task_inert:
                inert_indices = detect_inert(
                    inert_coor,
                    inert_threshold,
                    inert_frames,
                )

                sleep_set = set(sleep_indices)
                # 무기력 감지된 ID 상태 업데이트 (sleep이 아닌 경우만)
                for id in inert_indices:
                    if id in id_to_index.keys() and id not in sleep_set:
                        id_states[id] = "inert"

                if len(inert_indices) > 0 and len(set(inert_indices) - sleep_set) > 0:
                    state_texts.append(CONFIG_INERT_TEXT)

            # detect eat
            if task_eat:
                # bowl 감지 (class 3, tracking 불필요)
                bowl_results = model.predict(
                    frame, conf=bowl_conf, iou=0.5, classes=[3], verbose=False
                )
                bowl_boxes = []
                if len(bowl_results[0].boxes) > 0:
                    bowl_boxes = bowl_results[0].boxes.xyxy.cpu().numpy()

                eat_indices = detect_eat(
                    eat_coor,
                    eat_near_count,
                    boxes,
                    track_ids,
                    bowl_boxes,
                    eat_iou_threshold,
                    eat_dwell_frames,
                    eat_direction_frames,
                )

                # 식사 감지된 ID 상태 업데이트
                for id in eat_indices:
                    if id in id_to_index.keys():
                        id_states[id] = "eat"

                if len(eat_indices) > 0:
                    state_texts.append(CONFIG_EAT_TEXT)

            # detect bathroom
            if task_bathroom:
                bathroom_indices = detect_bathroom(
                    bathroom_coor,
                    bathroom_bbox,
                    boxes,
                    track_ids,
                    frame,
                    bathroom_cls_model,
                    bathroom_trigger_frames,
                    bathroom_height_drop,
                    cls_confidence=bathroom_cls_conf,
                )

                # 배변 감지된 ID 상태 업데이트
                for id in bathroom_indices:
                    if id in id_to_index.keys():
                        id_states[id] = "bathroom"

                if len(bathroom_indices) > 0:
                    state_texts.append(CONFIG_BATHROOM_TEXT)

            # ================================================================
            # Pillow로 bbox + 라벨(ID + 이모지) 그리기
            # ================================================================
            if CONFIG_USE_PILLOW_LABEL:
                frame = draw_bbox_with_pillow_label(
                    frame=frame,
                    boxes=boxes,
                    track_ids=track_ids,
                    id_states=id_states,
                    font_path=CONFIG_PILLOW_FONT_PATH,
                    font_size=CONFIG_LABEL_FONT_SIZE,
                    label_text_color=CONFIG_LABEL_TEXT_COLOR,
                    label_bg_padding=CONFIG_LABEL_BG_PADDING,
                    box_thickness=CONFIG_SV_BOX_THICKNESS,
                    global_id_map=global_id_map if global_id_map else None,
                )
            # Supervision 사용 (Pillow 라벨 미사용 시)
            elif CONFIG_USE_SUPERVISION and sv_annotators is not None:
                frame = annotate_frame_with_supervision(
                    frame=frame,
                    boxes=boxes,
                    track_ids=track_ids,
                    confidences=None,
                    annotators=sv_annotators,
                    label_format=CONFIG_SV_LABEL_FORMAT,
                )

            # 중심점 표시 (설정에 따라)
            if CONFIG_SV_SHOW_CENTER:
                for id, box in zip(track_ids, boxes):
                    x_center, y_center, _, _ = box
                    cv2.circle(
                        frame, (int(x_center), int(y_center)), 5, (0, 0, 255), 3, cv2.LINE_AA
                    )

            # ================================================================
            # Draw state text (Pillow 또는 cv2) - 하드코딩 설정 사용
            # ================================================================
            state_display = CONFIG_STATE_PREFIX + " ".join(state_texts) if state_texts else CONFIG_STATE_PREFIX + CONFIG_NORMAL_TEXT

            if CONFIG_USE_PILLOW_TEXT:
                # Pillow로 텍스트 그리기 (한글/이모티콘 지원)
                frame = draw_text_pillow(
                    frame,
                    state_display,
                    position=(10, 10),
                    font_path=CONFIG_PILLOW_FONT_PATH,
                    font_size=CONFIG_PILLOW_FONT_SIZE,
                    color=CONFIG_PILLOW_TEXT_COLOR,
                    bg_color=CONFIG_PILLOW_BG_COLOR,
                    padding=CONFIG_PILLOW_PADDING,
                )
            else:
                # 레거시: cv2.putText 사용 (영문만 지원)
                cv2.putText(
                    frame,
                    f"State: {' '.join(state_texts) if state_texts else 'Normal'}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    red,
                    2,
                    cv2.LINE_AA,
                )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        else:
            break

    cap.release()


    save_video(output, frames, fps)


if __name__ == "__main__":
    main()
