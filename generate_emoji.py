"""
Emoji PNG Generator - main.py의 이모지 표시를 위한 PNG 파일 생성

사용법:
    python generate_emoji.py

생성 결과:
    emoji/
    ├── fight.png
    ├── escape.png
    ├── inert.png
    ├── play.png
    ├── sleep.png
    ├── eat.png
    └── bathroom.png
"""

import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# 이모지 매핑: 파일명 -> 유니코드 이모지
EMOJI_MAP = {
    "fight": "\U0001F94A",      # 🥊
    "escape": "\u26A0\uFE0F",   # ⚠️
    "inert": "\u2744\uFE0F",    # ❄️
    "play": "\U0001F3BE",       # 🎾
    "sleep": "\U0001F634",      # 😴
    "eat": "\U0001F37D\uFE0F",  # 🍽️
    "bathroom": "\U0001F6BD",   # 🚽
}

# 이모지 폰트 탐색 경로
EMOJI_FONT_PATHS = [
    # macOS
    "/System/Library/Fonts/Apple Color Emoji.ttc",
    # Linux
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/noto-emoji/NotoColorEmoji.ttf",
    # Windows
    "C:/Windows/Fonts/seguiemj.ttf",
]


def find_emoji_font() -> str:
    """시스템에서 이모지 폰트를 찾습니다."""
    for path in EMOJI_FONT_PATHS:
        if os.path.exists(path):
            return path
    return None


def generate_emoji_pngs(output_dir: str = "emoji", size: int = 64):
    """이모지 PNG 파일을 생성합니다."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    font_path = find_emoji_font()

    if font_path:
        print(f"[INFO] 이모지 폰트: {font_path}")
        try:
            font = ImageFont.truetype(font_path, size)
        except Exception as e:
            print(f"[WARN] 폰트 로드 실패: {e}")
            font = None
    else:
        print("[WARN] 이모지 폰트를 찾을 수 없습니다. 텍스트 기반으로 생성합니다.")
        font = None

    generated = []

    for name, emoji_char in EMOJI_MAP.items():
        filepath = out / f"{name}.png"

        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        if font:
            # 이모지 폰트로 렌더링
            bbox = draw.textbbox((0, 0), emoji_char, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            x = (size - tw) // 2 - bbox[0]
            y = (size - th) // 2 - bbox[1]
            draw.text((x, y), emoji_char, font=font, embedded_color=True)
        else:
            # 폰트 없으면 컬러 원 + 텍스트 약어로 대체
            colors = {
                "fight": (255, 0, 0),
                "escape": (255, 200, 0),
                "inert": (100, 150, 255),
                "play": (0, 200, 0),
                "sleep": (128, 0, 128),
                "eat": (255, 105, 180),
                "bathroom": (0, 191, 255),
            }
            color = colors.get(name, (128, 128, 128))
            margin = 4
            draw.ellipse([margin, margin, size - margin, size - margin], fill=color)
            label = name[0].upper()
            try:
                small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size // 2)
            except (IOError, OSError):
                small_font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), label, font=small_font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text(((size - tw) // 2, (size - th) // 2), label, font=small_font, fill=(255, 255, 255))

        img.save(str(filepath))
        generated.append(name)
        print(f"  {name}.png -> {emoji_char}")

    print(f"\n[OK] {len(generated)}개 이모지 PNG 생성 완료: {out}/")
    return generated


if __name__ == "__main__":
    generate_emoji_pngs()
