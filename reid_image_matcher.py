"""
ReID Image Matcher - Reference 이미지와 Input 이미지 비교 후 Global ID 할당

사용법:
    # 단일 이미지 매칭
    python reid_image_matcher.py --reference ./references --input test.jpg --output result.jpg

    # 폴더 내 모든 이미지 처리
    python reid_image_matcher.py --reference ./references --input-dir ./inputs --output-dir ./outputs

    # 비디오 프레임 처리
    python reid_image_matcher.py --reference ./references --video input.mp4 --output output.mp4
"""

import cv2
import numpy as np
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import click

# YOLO
from ultralytics import YOLO

# ReID 모듈
from reid_features import HybridReID, ColorHistogramReID, create_reid_extractor
from reid_lightweight import AdaptiveReID, FastHistogramReID, create_lightweight_reid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ReferenceGallery:
    """Reference 이미지의 특징 갤러리"""
    global_id: int
    name: str  # 파일명 또는 레이블
    features: np.ndarray  # 대표 특징 벡터 (여러 이미지의 평균)
    image_path: str
    thumbnail: Optional[np.ndarray] = None  # 썸네일 이미지
    feature_count: int = 1  # 특징 추출에 사용된 이미지 수
    all_features: List[np.ndarray] = field(default_factory=list)  # 모든 개별 특징


@dataclass
class MatchResult:
    """매칭 결과"""
    global_id: int
    name: str
    similarity: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    is_unknown: bool = False  # Unknown 객체 여부
    unknown_id: Optional[int] = None  # Unknown 객체 번호
    feature: Optional[np.ndarray] = None  # 특징 벡터 (Unknown 저장용)


class ReIDImageMatcher:
    """
    Reference 이미지와 Input 이미지를 비교하여 Global ID를 할당하는 클래스
    """

    def __init__(self,
                 model_path: str = "weights/modelv11x.pt",
                 reid_method: str = "adaptive",
                 similarity_threshold: float = 0.5,
                 use_deep_model: bool = True):
        """
        Args:
            model_path: YOLO 모델 경로
            reid_method: ReID 방법 ('adaptive', 'histogram', 'hybrid')
            similarity_threshold: 동일 객체 판단 임계값
            use_deep_model: 딥러닝 모델 사용 여부
        """
        self.similarity_threshold = similarity_threshold

        # YOLO 모델 로드
        logger.info(f"YOLO 모델 로드 중: {model_path}")
        self.yolo_model = YOLO(model_path)

        # ReID 특징 추출기 초기화
        logger.info(f"ReID 추출기 초기화: {reid_method}")
        if reid_method == 'adaptive':
            # AdaptiveReID: FastHistogram + MobileNetV3 (상황에 따라 선택)
            self.reid_extractor = create_lightweight_reid('adaptive', use_deep_model=use_deep_model)
        elif reid_method == 'mobilenet':
            # MobileNetV3-Small 단독 사용 (CPU ~10-15ms)
            self.reid_extractor = create_lightweight_reid('mobilenet')
        elif reid_method == 'histogram':
            # FastHistogram 단독 사용 (CPU ~1ms, 가장 빠름)
            self.reid_extractor = create_lightweight_reid('histogram')
        elif reid_method == 'hybrid':
            # OSNet + EfficientNet + Histogram (가장 정확, 느림)
            self.reid_extractor = create_reid_extractor('hybrid', use_osnet=use_deep_model, use_effnet=False)
        else:
            self.reid_extractor = create_lightweight_reid('adaptive', use_deep_model=use_deep_model)

        # Reference 갤러리
        self.galleries: Dict[int, ReferenceGallery] = {}
        self._next_global_id = 1

        # Unknown 객체 관리
        self._next_unknown_id = 1
        self.unknown_names: Dict[int, str] = {}  # unknown_id -> custom name

        # Reference별 최고 매칭 정보 저장
        self.reference_best_matches: Dict[int, Dict] = {}  # global_id -> {'similarity', 'bbox', ...}

        logger.info(f"ReIDImageMatcher 초기화 완료 - threshold: {similarity_threshold}")

    def load_reference_images(self, reference_path: str) -> int:
        """
        Reference 이미지 폴더에서 이미지를 로드하고 특징 추출

        Args:
            reference_path: Reference 이미지 폴더 경로 또는 JSON 파일 경로

        Returns:
            로드된 reference 이미지 수
        """
        ref_path = Path(reference_path)

        if ref_path.is_file() and ref_path.suffix == '.json':
            # JSON 파일에서 로드
            return self._load_from_json(ref_path)
        elif ref_path.is_dir():
            # 폴더에서 이미지 로드
            return self._load_from_directory(ref_path)
        else:
            raise ValueError(f"유효하지 않은 reference 경로: {reference_path}")

    def _load_from_directory(self, dir_path: Path) -> int:
        """폴더에서 reference 이미지 로드"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        loaded_count = 0

        for img_file in sorted(dir_path.iterdir()):
            if img_file.suffix.lower() in image_extensions:
                try:
                    # 이미지 로드
                    image = cv2.imread(str(img_file))
                    if image is None:
                        logger.warning(f"이미지 로드 실패: {img_file}")
                        continue

                    # 한 이미지에서 모든 객체 검출 (동일 객체 여러 장)
                    crops = self._get_all_object_crops(image)

                    if not crops:
                        # 검출 실패시 전체 이미지 사용
                        crops = [image]

                    # 모든 크롭에서 특징 추출 후 평균
                    all_features = []
                    for crop in crops:
                        feat = self.reid_extractor.extract_features(crop)
                        if feat is not None and len(feat) > 0:
                            all_features.append(feat)

                    if not all_features:
                        logger.warning(f"특징 추출 실패: {img_file}")
                        continue

                    # 대표 특징 계산 (평균 후 정규화)
                    representative_feature = self._compute_representative_feature(all_features)

                    # 갤러리에 추가
                    global_id = self._next_global_id
                    self._next_global_id += 1

                    # 파일명에서 이름 추출 (확장자 제외)
                    name = img_file.stem

                    # 첫 번째 크롭으로 썸네일 생성
                    thumbnail = cv2.resize(crops[0], (64, 128))

                    self.galleries[global_id] = ReferenceGallery(
                        global_id=global_id,
                        name=name,
                        features=representative_feature,
                        image_path=str(img_file),
                        thumbnail=thumbnail,
                        feature_count=len(all_features),
                        all_features=all_features
                    )

                    loaded_count += 1
                    logger.info(f"Reference 로드: {name} -> Global ID {global_id} ({len(all_features)}개 이미지에서 특징 추출)")

                except Exception as e:
                    logger.error(f"Reference 이미지 처리 실패 ({img_file}): {e}")

        logger.info(f"총 {loaded_count}개의 reference 이미지 로드 완료")
        return loaded_count

    def _get_all_object_crops(self, image: np.ndarray, min_area: int = 1000) -> List[np.ndarray]:
        """
        이미지에서 모든 객체(pet) 크롭 추출

        한 이미지에 동일 객체의 여러 사진이 있는 경우를 처리

        Args:
            image: 입력 이미지
            min_area: 최소 바운딩 박스 면적 (너무 작은 검출 제외)

        Returns:
            크롭 이미지 리스트
        """
        try:
            # YOLO로 객체 검출 (pet class)
            results = self.yolo_model(image, verbose=False)

            if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
                return []

            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            h, w = image.shape[:2]
            crops = []

            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # 면적 체크
                area = (x2 - x1) * (y2 - y1)
                if area < min_area:
                    continue

                if x2 > x1 and y2 > y1:
                    crop = image[y1:y2, x1:x2]
                    crops.append(crop)

            logger.debug(f"이미지에서 {len(crops)}개의 객체 검출")
            return crops

        except Exception as e:
            logger.warning(f"객체 검출 실패: {e}")
            return []

    def _compute_representative_feature(self, features: List[np.ndarray],
                                          method: str = 'mean') -> np.ndarray:
        """
        여러 특징 벡터에서 대표 특징 계산

        Args:
            features: 특징 벡터 리스트
            method: 'mean' (평균), 'median' (중앙값), 'ema' (지수이동평균)

        Returns:
            대표 특징 벡터
        """
        if not features:
            return None

        if len(features) == 1:
            return features[0]

        features_array = np.array(features)

        if method == 'mean':
            representative = np.mean(features_array, axis=0)
        elif method == 'median':
            representative = np.median(features_array, axis=0)
        elif method == 'ema':
            # 지수이동평균 (최근 특징에 더 높은 가중치)
            alpha = 0.3
            representative = features[0].copy()
            for feat in features[1:]:
                representative = alpha * feat + (1 - alpha) * representative
        else:
            representative = np.mean(features_array, axis=0)

        # L2 정규화
        norm = np.linalg.norm(representative)
        if norm > 1e-7:
            representative = representative / norm

        return representative.astype(np.float32)

    def _load_from_json(self, json_path: Path) -> int:
        """JSON 파일에서 reference 정보 로드"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        loaded_count = 0
        base_dir = json_path.parent

        for item in data.get('references', []):
            try:
                img_path = base_dir / item['image_path']
                image = cv2.imread(str(img_path))

                if image is None:
                    logger.warning(f"이미지 로드 실패: {img_path}")
                    continue

                # 바운딩 박스가 지정된 경우 (여러 개 가능)
                if 'bboxes' in item:
                    # 여러 바운딩 박스
                    crops = []
                    for bbox in item['bboxes']:
                        x1, y1, x2, y2 = bbox
                        crop = image[y1:y2, x1:x2]
                        if crop.size > 0:
                            crops.append(crop)
                elif 'bbox' in item:
                    # 단일 바운딩 박스
                    x1, y1, x2, y2 = item['bbox']
                    crops = [image[y1:y2, x1:x2]]
                else:
                    # 자동 검출
                    crops = self._get_all_object_crops(image)
                    if not crops:
                        crops = [image]

                # 모든 크롭에서 특징 추출
                all_features = []
                for crop in crops:
                    feat = self.reid_extractor.extract_features(crop)
                    if feat is not None and len(feat) > 0:
                        all_features.append(feat)

                if not all_features:
                    logger.warning(f"특징 추출 실패: {img_path}")
                    continue

                # 대표 특징 계산
                representative_feature = self._compute_representative_feature(all_features)

                global_id = item.get('global_id', self._next_global_id)
                if global_id >= self._next_global_id:
                    self._next_global_id = global_id + 1

                name = item.get('name', f"pet_{global_id}")
                thumbnail = cv2.resize(crops[0], (64, 128))

                self.galleries[global_id] = ReferenceGallery(
                    global_id=global_id,
                    name=name,
                    features=representative_feature,
                    image_path=str(img_path),
                    thumbnail=thumbnail,
                    feature_count=len(all_features),
                    all_features=all_features
                )

                loaded_count += 1
                logger.info(f"Reference 로드: {name} -> Global ID {global_id} ({len(all_features)}개 이미지에서 특징 추출)")

            except Exception as e:
                logger.error(f"Reference 항목 처리 실패: {e}")

    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """코사인 유사도 계산"""
        if feat1 is None or feat2 is None:
            return 0.0
        return float(np.dot(feat1, feat2))

    def match_features(self, query_feature: np.ndarray) -> Tuple[Optional[int], str, float]:
        """
        Query 특징과 갤러리 매칭

        Returns:
            (global_id, name, similarity) 또는 (None, "", 0.0)
        """
        if query_feature is None or len(self.galleries) == 0:
            return None, "", 0.0

        best_match = None
        best_name = ""
        best_sim = 0.0

        for global_id, gallery in self.galleries.items():
            sim = self.compute_similarity(query_feature, gallery.features)

            if sim > best_sim:
                best_sim = sim
                best_match = global_id
                best_name = gallery.name

        if best_sim >= self.similarity_threshold:
            return best_match, best_name, best_sim

        return None, "", best_sim

    def set_unknown_name(self, unknown_id: int, name: str):
        """Unknown 객체에 이름 지정"""
        self.unknown_names[unknown_id] = name
        logger.info(f"Unknown {unknown_id} -> '{name}' 이름 지정")

    def get_unknown_name(self, unknown_id: int) -> str:
        """Unknown 객체 이름 반환"""
        return self.unknown_names.get(unknown_id, f"Unknown_{unknown_id}")

    def process_image(self, image: np.ndarray, track_unknown: bool = True) -> Tuple[np.ndarray, List[MatchResult]]:
        """
        Input 이미지 처리 및 Global ID 매칭

        Args:
            image: 입력 이미지 (BGR)
            track_unknown: Unknown 객체에 번호 부여 여부

        Returns:
            (결과 이미지, 매칭 결과 리스트)
        """
        results_list = []
        output_image = image.copy()

        # YOLO로 객체 검출
        results = self.yolo_model(image, verbose=False)

        if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            logger.info("검출된 객체 없음")
            return output_image, results_list

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        h, w = image.shape[:2]

        for i, (box, conf) in enumerate(zip(boxes, confs)):
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # 크롭 및 특징 추출
            crop = image[y1:y2, x1:x2]
            features = self.reid_extractor.extract_features(crop)

            # 갤러리 매칭
            global_id, name, similarity = self.match_features(features)

            # 결과 저장
            if global_id is not None:
                result = MatchResult(
                    global_id=global_id,
                    name=name,
                    similarity=similarity,
                    bbox=(x1, y1, x2, y2),
                    is_unknown=False,
                    feature=features
                )
                results_list.append(result)

                # Reference별 최고 매칭 정보 업데이트
                if global_id not in self.reference_best_matches or \
                   similarity > self.reference_best_matches[global_id]['similarity']:
                    self.reference_best_matches[global_id] = {
                        'similarity': similarity,
                        'bbox': (x1, y1, x2, y2),
                        'crop': crop.copy()
                    }

                # 바운딩 박스 그리기
                color = self._get_color_for_id(global_id)
                cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)

                # 라벨 그리기
                label = f"ID:{global_id} {name} ({similarity:.2f})"
                self._draw_label(output_image, label, (x1, y1), color)
            else:
                # 매칭 실패 - Unknown으로 표시
                if track_unknown:
                    unknown_id = self._next_unknown_id
                    self._next_unknown_id += 1
                    unknown_name = self.get_unknown_name(unknown_id)
                else:
                    unknown_id = None
                    unknown_name = "Unknown"

                result = MatchResult(
                    global_id=-1,  # Unknown은 -1
                    name=unknown_name,
                    similarity=similarity,
                    bbox=(x1, y1, x2, y2),
                    is_unknown=True,
                    unknown_id=unknown_id,
                    feature=features
                )
                results_list.append(result)

                # Unknown 색상 (회색 계열, 번호에 따라 약간 다르게)
                if unknown_id:
                    unknown_color = self._get_color_for_unknown(unknown_id)
                    label = f"{unknown_name} ({similarity:.2f})"
                else:
                    unknown_color = (128, 128, 128)
                    label = f"Unknown ({similarity:.2f})"

                cv2.rectangle(output_image, (x1, y1), (x2, y2), unknown_color, 2)
                self._draw_label(output_image, label, (x1, y1), unknown_color)

        return output_image, results_list

    def _get_color_for_unknown(self, unknown_id: int) -> Tuple[int, int, int]:
        """Unknown ID에 따른 색상 반환 (회색 계열)"""
        gray_colors = [
            (100, 100, 100),
            (150, 100, 100),
            (100, 150, 100),
            (100, 100, 150),
            (150, 150, 100),
            (150, 100, 150),
            (100, 150, 150),
            (180, 180, 180),
        ]
        return gray_colors[unknown_id % len(gray_colors)]

    def process_image_file(self, input_path: str, output_path: str,
                           save_reference_output: bool = True) -> List[MatchResult]:
        """
        이미지 파일 처리

        Args:
            input_path: 입력 이미지 경로
            output_path: 출력 이미지 경로
            save_reference_output: Reference 이미지 출력 저장 여부

        Returns:
            매칭 결과 리스트
        """
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"이미지 로드 실패: {input_path}")

        # 매칭 정보 초기화 (새 이미지마다)
        self.reference_best_matches.clear()

        output_image, results = self.process_image(image)

        cv2.imwrite(output_path, output_image)
        logger.info(f"결과 저장: {output_path} ({len(results)}개 매칭)")

        # Reference 이미지 출력 저장
        if save_reference_output:
            input_name = Path(input_path).stem
            output_dir = Path(output_path).parent
            self.save_reference_outputs(output_dir, input_name)

        return results

    def save_reference_outputs(self, output_dir: Path, input_name: str):
        """
        Reference 이미지에 매칭 정보를 표시하여 저장

        Args:
            output_dir: 출력 디렉토리
            input_name: 입력 이미지 이름 (확장자 제외)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for global_id, gallery in self.galleries.items():
            # Reference 이미지 로드
            ref_image = cv2.imread(gallery.image_path)
            if ref_image is None:
                logger.warning(f"Reference 이미지 로드 실패: {gallery.image_path}")
                continue

            # 객체 검출하여 bbox 표시
            ref_crops_info = self._get_all_object_crops_with_boxes(ref_image)

            # Reference 이미지에 bbox 표시
            color = self._get_color_for_id(global_id)
            for crop_info in ref_crops_info:
                x1, y1, x2, y2 = crop_info['bbox']
                cv2.rectangle(ref_image, (x1, y1), (x2, y2), color, 2)

            # 매칭 정보 표시
            if global_id in self.reference_best_matches:
                match_info = self.reference_best_matches[global_id]
                best_sim = match_info['similarity']

                # 상단에 매칭 정보 표시
                label = f"ID:{global_id} {gallery.name}"
                match_label = f"Best Match: {best_sim:.3f}"

                self._draw_label(ref_image, label, (10, 30), color)
                self._draw_label(ref_image, match_label, (10, 70), (0, 255, 0))

                # 매칭된 크롭 이미지를 우측 하단에 표시
                if 'crop' in match_info:
                    matched_crop = match_info['crop']
                    crop_h, crop_w = matched_crop.shape[:2]

                    # 크롭 크기 조절 (최대 150px)
                    max_size = 150
                    scale = min(max_size / crop_w, max_size / crop_h)
                    new_w, new_h = int(crop_w * scale), int(crop_h * scale)
                    resized_crop = cv2.resize(matched_crop, (new_w, new_h))

                    # 우측 하단에 배치
                    ref_h, ref_w = ref_image.shape[:2]
                    y_offset = ref_h - new_h - 10
                    x_offset = ref_w - new_w - 10

                    # 테두리 추가
                    cv2.rectangle(ref_image, (x_offset - 2, y_offset - 2),
                                  (x_offset + new_w + 2, y_offset + new_h + 2), color, 2)
                    ref_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_crop

                    # "Matched" 라벨
                    self._draw_label(ref_image, "Matched", (x_offset, y_offset), color)
            else:
                # 매칭 없음
                label = f"ID:{global_id} {gallery.name}"
                no_match_label = "No Match Found"

                self._draw_label(ref_image, label, (10, 30), color)
                self._draw_label(ref_image, no_match_label, (10, 70), (0, 0, 255))

            # 저장
            output_filename = f"ref_{input_name}_{gallery.name}.jpg"
            output_path = output_dir / output_filename
            cv2.imwrite(str(output_path), ref_image)
            logger.info(f"Reference 출력 저장: {output_path}")

    def _get_all_object_crops_with_boxes(self, image: np.ndarray,
                                          min_area: int = 1000) -> List[Dict]:
        """
        이미지에서 모든 객체 크롭과 bbox 정보 추출

        Returns:
            [{'crop': np.ndarray, 'bbox': (x1,y1,x2,y2)}, ...]
        """
        try:
            results = self.yolo_model(image, verbose=False)

            if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
                return []

            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            h, w = image.shape[:2]
            crops_info = []

            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                area = (x2 - x1) * (y2 - y1)
                if area < min_area:
                    continue

                if x2 > x1 and y2 > y1:
                    crop = image[y1:y2, x1:x2]
                    crops_info.append({
                        'crop': crop,
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf
                    })

            return crops_info

        except Exception as e:
            logger.warning(f"객체 검출 실패: {e}")
            return []

    def process_video(self, input_path: str, output_path: str) -> Dict:
        """비디오 처리"""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"비디오 열기 실패: {input_path}")

        # 비디오 정보
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 출력 비디오 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        all_results = []

        logger.info(f"비디오 처리 시작: {total_frames} 프레임")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            output_frame, results = self.process_image(frame)
            out.write(output_frame)

            all_results.append({
                'frame': frame_count,
                'matches': [
                    {
                        'global_id': r.global_id,
                        'name': r.name,
                        'similarity': r.similarity,
                        'bbox': r.bbox
                    }
                    for r in results
                ]
            })

            frame_count += 1
            if frame_count % 100 == 0:
                logger.info(f"진행: {frame_count}/{total_frames}")

        cap.release()
        out.release()

        logger.info(f"비디오 처리 완료: {output_path}")

        return {
            'total_frames': frame_count,
            'results': all_results
        }

    def _get_color_for_id(self, global_id: int) -> Tuple[int, int, int]:
        """Global ID에 따른 색상 반환"""
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 255, 0),  # Lime
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
            (0, 128, 255),  # Sky blue
        ]
        return colors[global_id % len(colors)]

    def _draw_label(self, image: np.ndarray, label: str,
                    position: Tuple[int, int], color: Tuple[int, int, int]):
        """라벨 그리기"""
        x, y = position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # 텍스트 크기 계산
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        # 배경 박스
        cv2.rectangle(
            image,
            (x, y - text_height - 10),
            (x + text_width + 10, y),
            color, -1
        )

        # 텍스트
        cv2.putText(
            image, label,
            (x + 5, y - 5),
            font, font_scale, (255, 255, 255), thickness
        )

    def save_galleries(self, output_path: str):
        """갤러리 정보 저장"""
        data = {
            'references': [
                {
                    'global_id': gallery.global_id,
                    'name': gallery.name,
                    'image_path': gallery.image_path,
                    'feature_count': gallery.feature_count
                }
                for gallery in self.galleries.values()
            ]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"갤러리 정보 저장: {output_path}")

    def get_gallery_info(self) -> List[Dict]:
        """갤러리 정보 반환"""
        return [
            {
                'global_id': g.global_id,
                'name': g.name,
                'image_path': g.image_path
            }
            for g in self.galleries.values()
        ]


# CLI 인터페이스
@click.group()
def cli():
    """ReID Image Matcher - Reference 이미지 기반 Global ID 매칭"""
    pass


@cli.command()
@click.option('--reference', '-r', required=True, help='Reference 이미지 폴더 또는 JSON 파일 경로')
@click.option('--input', '-i', 'input_path', required=True, help='입력 이미지 경로')
@click.option('--output', '-o', required=True, help='출력 이미지 경로')
@click.option('--model', '-m', default='weights/modelv11x.pt', help='YOLO 모델 경로')
@click.option('--reid-method', type=click.Choice(['adaptive', 'mobilenet', 'histogram', 'hybrid']),
              default='adaptive', help='ReID 방법 (adaptive: 자동선택, mobilenet: MobileNetV3, histogram: 빠른 히스토그램, hybrid: OSNet+Histogram)')
@click.option('--threshold', '-t', default=0.5, type=float, help='유사도 임계값')
@click.option('--save-ref/--no-save-ref', default=True, help='Reference 출력 이미지 저장 여부')
@click.option('--unknown-names', '-u', default=None, help='Unknown 이름 지정 (JSON 형식: {"1": "새로운개", "2": "흰고양이"})')
def match(reference, input_path, output, model, reid_method, threshold, save_ref, unknown_names):
    """단일 이미지 매칭"""
    matcher = ReIDImageMatcher(
        model_path=model,
        reid_method=reid_method,
        similarity_threshold=threshold
    )

    # Unknown 이름 설정
    if unknown_names:
        try:
            names_dict = json.loads(unknown_names)
            for uid, name in names_dict.items():
                matcher.set_unknown_name(int(uid), name)
        except json.JSONDecodeError:
            logger.warning(f"Unknown 이름 파싱 실패: {unknown_names}")

    # Reference 로드
    matcher.load_reference_images(reference)

    # 이미지 처리
    results = matcher.process_image_file(input_path, output, save_reference_output=save_ref)

    # 결과 출력
    print(f"\n매칭 결과 ({len(results)}개):")
    matched_count = 0
    unknown_count = 0
    for r in results:
        if r.is_unknown:
            unknown_count += 1
            print(f"  - [Unknown {r.unknown_id}] {r.name}, Similarity: {r.similarity:.3f}")
        else:
            matched_count += 1
            print(f"  - [ID:{r.global_id}] {r.name}, Similarity: {r.similarity:.3f}")

    print(f"\n매칭: {matched_count}개, Unknown: {unknown_count}개")


@cli.command()
@click.option('--reference', '-r', required=True, help='Reference 이미지 폴더 또는 JSON 파일 경로')
@click.option('--input-dir', '-i', required=True, help='입력 이미지 폴더')
@click.option('--output-dir', '-o', required=True, help='출력 이미지 폴더')
@click.option('--model', '-m', default='weights/modelv11x.pt', help='YOLO 모델 경로')
@click.option('--reid-method', type=click.Choice(['adaptive', 'mobilenet', 'histogram', 'hybrid']),
              default='adaptive', help='ReID 방법 (adaptive: 자동선택, mobilenet: MobileNetV3, histogram: 빠른 히스토그램, hybrid: OSNet+Histogram)')
@click.option('--threshold', '-t', default=0.5, type=float, help='유사도 임계값')
def match_batch(reference, input_dir, output_dir, model, reid_method, threshold):
    """폴더 내 모든 이미지 매칭"""
    matcher = ReIDImageMatcher(
        model_path=model,
        reid_method=reid_method,
        similarity_threshold=threshold
    )

    # Reference 로드
    matcher.load_reference_images(reference)

    # 출력 폴더 생성
    os.makedirs(output_dir, exist_ok=True)

    # 이미지 처리
    input_path = Path(input_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    all_results = []
    for img_file in sorted(input_path.iterdir()):
        if img_file.suffix.lower() in image_extensions:
            output_path = Path(output_dir) / img_file.name
            try:
                results = matcher.process_image_file(str(img_file), str(output_path))
                all_results.append({
                    'file': str(img_file),
                    'matches': len(results)
                })
            except Exception as e:
                logger.error(f"처리 실패 ({img_file}): {e}")

    print(f"\n처리 완료: {len(all_results)}개 이미지")


@cli.command()
@click.option('--reference', '-r', required=True, help='Reference 이미지 폴더 또는 JSON 파일 경로')
@click.option('--video', '-v', required=True, help='입력 비디오 경로')
@click.option('--output', '-o', required=True, help='출력 비디오 경로')
@click.option('--model', '-m', default='weights/modelv11x.pt', help='YOLO 모델 경로')
@click.option('--reid-method', type=click.Choice(['adaptive', 'mobilenet', 'histogram', 'hybrid']),
              default='adaptive', help='ReID 방법 (adaptive: 자동선택, mobilenet: MobileNetV3, histogram: 빠른 히스토그램, hybrid: OSNet+Histogram)')
@click.option('--threshold', '-t', default=0.5, type=float, help='유사도 임계값')
def match_video(reference, video, output, model, reid_method, threshold):
    """비디오 매칭"""
    matcher = ReIDImageMatcher(
        model_path=model,
        reid_method=reid_method,
        similarity_threshold=threshold
    )

    # Reference 로드
    matcher.load_reference_images(reference)

    # 비디오 처리
    results = matcher.process_video(video, output)

    print(f"\n비디오 처리 완료: {results['total_frames']} 프레임")


@cli.command()
@click.option('--reference', '-r', required=True, help='Reference 이미지 폴더')
@click.option('--output', '-o', required=True, help='출력 JSON 파일 경로')
@click.option('--model', '-m', default='weights/modelv11x.pt', help='YOLO 모델 경로')
def create_gallery(reference, output, model):
    """Reference 갤러리 생성 및 저장"""
    matcher = ReIDImageMatcher(model_path=model)

    # Reference 로드
    count = matcher.load_reference_images(reference)

    # 갤러리 저장
    matcher.save_galleries(output)

    print(f"\n갤러리 생성 완료: {count}개 reference")


if __name__ == "__main__":
    cli()
