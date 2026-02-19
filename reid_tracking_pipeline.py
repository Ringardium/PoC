import cv2
import numpy as np
import torch
from collections import defaultdict, deque
import click
import time
import math
import logging
import threading
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor

from reid_features import create_reid_extractor
from global_id_manager import GlobalIDManager
from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from tracking import (
    track_with_bytetrack,
    track_with_botsort,
    track_with_deepsort
)

def cosine_similarity(vec1, vec2):
    if len(vec1) != len(vec2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = math.sqrt(sum(a * a for a in vec1))
    norm_b = math.sqrt(sum(b * b for b in vec2))
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)

class ReIDTrackingPipeline:
    """
    기존 Tracker + ReID feature Extracter + Consine Similarity 
    """
    def __init__(self, model_path , tracker_type = 'botsort', reid_method = 'histogram', similarity_threshold = 0.5, track_correction_enabled=True, max_track_age = 30) : 
        self.model_path = model_path
        self.tracker_type = tracker_type
        self.reid_method = reid_method
        self.similarity_threshold = similarity_threshold
        self.track_correction_enabled = track_correction_enabled
        self.max_track_age = max_track_age
    
        #Re-ID 특징 추출기 초기화
        if reid_method == "hybrid":
            self.use_osnet= bool(input())
            use_effnet= True, osnet_weight=0.45, effnet_weight = 0.45
            self.reid_extractor = create_reid_extractor(reid_method)

        #Global ID Magnager 초기화
        self.global_id_manager = GlobalIDManager(
            feature_dim=self.reid_extractor.feature_dim,
            similarity_threshold=similarity_threshold
        )
        self.reid_weights = []

        #채널별 트랙 히스토리
        self.track_histories = {}
        self.track_features = {}
        self.track_ages = {}

        #ID보정 히스토리
        self.id_corrections = []

        #채널 설정 및 상태
        self.channels = {}
        self.video_caps = {}
        self.channel_stats = defaultdict(lamdba : {
                'frames_processed' : 0,
                'objects_detected' : 0,
                'processing_fps' : 0,
                'id_corrections' : 0
            })

        #프레임 버퍼(프레임 손실 최소화)
        self.frame_buffer = {}
        self.result_buffer = {}

        # 처리 상태
        self.running = False
        self.should_stop = False
        
        # 통계
        self.total_processing_time = 0
        self.total_frames = 0

        # 멀티스레딩 락
        self.stats_lock = threading.Lock()
        self.results_lock = threading.Lock()

        logging.info(f"======= Re-ID 파이프라인 초기화 ======= \nTracker : {tracker_type}\nRe-ID : {reid_method}")
        

    def extract_reid_features(self, frame, boxes) : 
        """
        2. reid feature 추출
        BBOX기반 crop 추출
        crop별 feature extract
        """
        reid_fetures = []

        for box in boxes :
            try :
                # BBOX 에서 crop image 추출
                try: 
                    x_center, y_center, width, height = box
                    x1 = max(0, int(x_center - width / 2))
                    y1 = max(0, int(y_center - height / 2))
                    x2 = min(frame.shape[1], int(x_center + width / 2))
                    y2 = min(frame.shape[0], int(y_center + height / 2))
                    
                    crop = frame[y1:y2, x1:x2]
                except : 
                    crop = None

                if crop is not None and crop.size > 0 
                    #Re-ID 특징 추출
                    feature = self.reid_extractor.extract_features(crop)
                    reid_features.append(feature)
                else:
                    print("특징 추출 실패")
                    reid_features.append(None)
            except Exception as e:
                logging.warning(f"Re-ID 특징 추출 실패: {e}")
                
        return reid_features

    def apply_reid_correction(self, channel_id, track_ids, reid_features, boxes):
        """3. Cosine Similarity 기반 Re-ID 매칭 + ID 보정"""
        if not self.track_correction_enabled :
            return track_ids, [0.0] * len(track_dis), []

        if channel_id not in self.track_features:
            self.track_feature[channel_id] = {}
            self.track_ages[channel_id] = {}
        
        corrected_ids = track_ids.copy()
        reid_scores = []
        corrections = []
        current_time = time.time()
        
        #기존 track들과 비교
        for existing_id, feature_history in self,track_features[channel_id].items():
            if existing_id == track_id:
                continue #자기 자신은 제외
            if len(feature_history) == 0:
                continue
            
            recent_features = list(feature_history)[-5:] #최근 5개

            if len(recent_features) > 0:
                similarities = cosine_similarity([feature], recent_features)[0]
                avg_similarity = np.mean(similarities)
                max_similarity = np.max(similarities)
                
                # 가중 평균 사용
                combined_similarity = 0.7 * avg_similarity + 0.3 * max_similarity

            if combined_similarity > self.similarity_threshold and combined_similarity > best_similarity:
                best_similarity = combined_similarity
                best_match_id = existing_id
            
                reid_scores.append(best_similarity)
            
            #ID 보정 수행
            if best_match_id is not None and best_match_id != track_id:
                old_id = track_id
                new_id = best_match_id

                correction = {
                    'old_id': old_id,
                    'new_id': new_id,
                    'similarity': best_similarity,
                    'reason': 'reid_matching'
                }
                corrections.append(correction)

                # 전역 보정 히스토리에 추가
                self.id_corrections.append((old_id, new_id, channel_id, current_time, 'reid_matching'))

                # ID 업데이트
                corrected_ids[i] = new_id
                
                logging.info(f"채널 {channel_id}: ID 보정 {old_id} → {new_id} (유사도: {best_similarity:.3f})")
        
        return corrected_ids, reid_scores, corrections

    def assign_global_ids(self, channel_id, track_ids, reid_features, boxes):
        """4. 글로벌 ID 할당"""
        global_ids = []
        
        for track_id, feature, box in zip(track_ids, reid_features, boxes):
            # 글로벌 ID 매니저를 통해 글로벌 ID 할당
            global_id = self.global_id_manager.get_global_id(
                channel_id, track_id, feature, box
            )
            global_ids.append(global_id)
        
        return global_ids

    def update_track_histories(self, channel_id, track_ids, reid_features, boxes):
        """5. 트랙 히스토리 업데이트"""
        current_time = time.time()
        
        # 채널 초기화
        if channel_id not in self.track_features:
            self.track_features[channel_id] = {}
            self.track_ages[channel_id] = {}
        
        # 현재 프레임의 트랙들 업데이트
        active_tracks = set()
        
        for track_id, feature, box in zip(track_ids, reid_features, boxes):
            active_tracks.add(track_id)
            
            # 특징 히스토리 업데이트
            if track_id not in self.track_features[channel_id]:
                self.track_features[channel_id][track_id] = deque(maxlen=20)  # 최근 20개
            
            self.track_features[channel_id][track_id].append(feature)
            
            # 트랙 기간 초기화
            self.track_ages[channel_id][track_id] = 0
        
        # 비활성 트랙들의 기간 증가 및 정리
        tracks_to_remove = []
        for track_id in list(self.track_ages[channel_id].keys()):
            if track_id not in active_tracks:
                self.track_ages[channel_id][track_id] += 1
                
                # 최대 기간 초과시 제거
                if self.track_ages[channel_id][track_id] > self.max_track_age:
                    tracks_to_remove.append(track_id)
        
        # 오래된 트랙 제거
        for track_id in tracks_to_remove:
            if track_id in self.track_features[channel_id]:
                del self.track_features[channel_id][track_id]
            if track_id in self.track_ages[channel_id]:
                del self.track_ages[channel_id][track_id]
            
            # 글로벌 ID 매니저에서도 제거
            self.global_id_manager.remove_local_id(channel_id, track_id)
    
    def visualize_results(self, frame, boxes, track_ids, global_ids, reid_scores, corrections):
        """
        6. 결과 시각화
        BBOX 그리기 + ID 표시
        """
        annotated_frame = frame.copy()
        
        # 바운딩 박스 및 ID 표시
        for i, (box, track_id, global_id, score) in enumerate(zip(boxes, track_ids, global_ids, reid_scores)):
            x_center, y_center, width, height = box
            x1, y1 = int(x_center - width/2), int(y_center - height/2)
            x2, y2 = int(x_center + width/2), int(y_center + height/2)
            
            # 바운딩 박스 색상
            color = (0, 255, 0)  # 높은 신뢰도 - 초록색
            
            # 바운딩 박스 그리기
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # ID 정보 표시
            id_text = f"L:{track_id}/G:{global_id}"
            cv2.putText(annotated_frame, id_text, (x1, y1-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        
        # ID 보정 정보 표시
        if corrections:
            y_offset = 30
            for correction in corrections[-3:]:  # 최근 3개만 표시
                correction_text = f"ID 보정: {correction['old_id']} → {correction['new_id']} ({correction['similarity']:.2f})"
                cv2.putText(annotated_frame, correction_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += 25
        
        return annotated_frame

    def get_statistics(self):
        """시스템 통계 반환"""
        total_tracks = sum(len(tracks) for tracks in self.track_features.values())
        total_corrections = len(self.id_corrections)
        
        # 최근 보정률 계산 (지난 1분간)
        recent_time = time.time() - 60
        recent_corrections = sum(1 for _, _, _, timestamp, _ in self.id_corrections 
                               if timestamp > recent_time)
        
        # 글로벌 ID 통계
        global_stats = self.global_id_manager.get_stats()
        
        return {
            'tracker_type': self.tracker_type,
            'reid_method': self.reid_method,
            'total_active_tracks': total_tracks,
            'total_id_corrections': total_corrections,
            'recent_corrections_per_min': recent_corrections,
            'similarity_threshold': self.similarity_threshold,
            'global_id_stats': global_stats,
            'cross_channel_objects': len(self.global_id_manager.get_cross_channel_objects())
        }
    
    def get_correction_history(self, limit=50):
        """ID 보정 히스토리 반환"""
        return self.id_corrections[-limit:]
    
    def cleanup_old_data(self):
        """오래된 데이터 정리"""
        self.global_id_manager.cleanup_old_data()
        
        # 오래된 보정 히스토리 정리 (1시간 이상 된 것)
        cutoff_time = time.time() - 3600
        self.id_corrections = [
            correction for correction in self.id_corrections
            if correction[3] > cutoff_time
        ]

    def process_batch_with_yolo(self, frames_batch):
        """YOLO로 프레임 배치 병렬 처리 (GPU 활용도 80-95%)"""
        if len(frames_batch) == 0:
            return []
        
        try:
            # YOLO 배치 추론 (GPU 병렬 처리)
            start_time = time.time()
            results = self.model(frames_batch, verbose=False)
            processing_time = time.time() - start_time
            
            # 통계 업데이트
            self.total_processing_time += processing_time
            self.total_frames += len(frames_batch)
            
            return results
            
        except Exception as e:
            logging.error(f"YOLO 배치 처리 실패: {str(e)}")
            return []
    
    def process_frame(self, frame, channel_id) : 
        """
        단일 프레임 처리 - 전체 파이프라인 실행
        
        Args:
            frame: 입력 프레임
            channel_id: 채널 ID
            
        Returns:
            results: {
                'boxes': 바운딩 박스 리스트,
                'track_ids': 로컬 트랙 ID 리스트,
                'global_ids': 글로벌 트랙 ID 리스트,
                'reid_scores': Re-ID 신뢰도 점수,
                'corrected_ids': 보정된 ID 정보,
                'frame': 시각화된 프레임
            }
        """

        # 1. YOLO 적용 + tracking
        model = YOLO(self.model_path)

        if self.tracker_type == "bytetrack" : 
            boxes, track_ids, frame = track_with_bytetrack(model, frame)
        elif self.tracker_type == "bytetrack" : 
            boxes, track_ids, frame = track_with_botsort(model, frame)
        elif self.tracker_type == "bytetrack" : 
            boxes, track_ids, frame = track_with_deepsort(model, tracker, frame)
        else:
            raise NotImplementedError
        
        if len(boxes) == 0:
            return {
                'boxes': [],
                'track_ids': [],
                'global_ids': [],
                'reid_scores': [],
                'corrected_ids': [],
                'frame': frame
            }
        
        # 2. Re-ID 특징 추출
        reid_features = self.extract_reid_features(frame, boxes)

        # 3. Cosine Similarity 기반 Re-ID 매칭 및 ID 보정
        corrected_track_ids, reid_scores, corrections = self.apply_reid_correction(channel_id, track_ids, reid_features, boxes)

        # 4. Global ID 할당
        global_ids = self.assign_global_ids(channel_id, corrected_track_ids, reid_features, boxes)

        # 5. Track histroy update
        self.update_track_histories(channel_id, corrected_track_ids, reid_features, boxes)

        # 6. 결과 시각화
        annotated_frame = self.visualize_results(frame, boxes, corrected_track_ids, global_ids, reid_scores, corrections)

        return {
            'boxes': boxes,
            'track_ids': corrected_track_ids,
            'global_ids': global_ids,
            'reid_scores': reid_scores,
            'corrected_ids': corrections,
            'frame': annotated_frame}

    def add_channel(self, channel_id, config):
        """채널 추가"""
        try:
            input_source = config.get('input', 'play.mp4')
            cap = cv2.VideoCapture(input_source)
            
            if not cap.isOpened():
                raise ValueError(f"비디오 소스를 열 수 없습니다: {input_source}")
            
            self.channels[channel_id] = config
            self.video_caps[channel_id] = cap
            self.frame_buffer[channel_id] = deque(maxlen=10)
            self.result_buffer[channel_id] = deque(maxlen=10)
            
            logging.info(f"채널 {channel_id} 추가 성공: {input_source}")
            return True, "채널 추가 성공"
            
        except Exception as e:
            logging.error(f"채널 {channel_id} 추가 실패: {str(e)}")
            return False, f"채널 추가 실패: {str(e)}"
    
    def remove_channel(self, channel_id):
        """채널 제거"""
        if channel_id in self.video_caps:
            self.video_caps[channel_id].release()
            del self.video_caps[channel_id]
        
        for attr in [self.channels, self.frame_buffer, self.result_buffer, 
                     self.channel_pipelines]:
            if channel_id in attr:
                del attr[channel_id]
        
        logging.info(f"채널 {channel_id} 제거 완료")
    
    def process_channel_worker(self, channel_id):
        """단일 채널 워커 스레드"""
        try:
            #1. 프레임 읽기 (임)
            success, frame = self.video_caps[channel_id].read()
            w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if success:
                frame_cnt += 1
                tracking_results = process_frame(frame, channel_id)
            
                with self.stats_lock:
                    self.channel_stats[channel_id]['objects_detected'] += len(boxes)
                    self.channel_stats[channel_id]['id_corrections'] += len(corrections)
                
                return result
                
        except Exception as e:
            logging.warning(f"채널 {channel_id} 처리 실패: {str(e)}")
            return None
    
    def process_parallel(self, video_paths_dict=None, num_frames=None):
        """
        멀티스레딩 병렬 처리 메인 루프
        
        Args:
            video_paths_dict: (선택) {channel_id: video_path} 형태의 딕셔너리
                             지정하면 기존 채널 대신 이 경로들을 사용
            num_frames: (선택) 처리할 프레임 수, 지정하면 무한 루프 대신 고정 횟수 실행
        """
        temp_caps = {}
        use_temp_caps = video_path_dict is not None

        try:
            self.running = True
            self.should_stop = False
            
            if use_temp_caps:
                for channel_id, video_path in video_paths_dict.items():
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpend():
                        logging.warning(f"비디오 열기 실패 : {video_path}")
                        continue
                    
                    temp_caps[channel_id] = cap
                channel_to_use = temp_caps
                logging.info(f"사용자 입력 {len(temp_caps)}개 비디오로 멀티 스레딩 병렬 처리 시작")
            
            else:
                channels_to_use = self.video_caps
                logging.info(f"기존 {len(self.channels)}개 채널로 멀티 스레딩 병렬 처리 시작")
            
            fps_counter = 0
            fps_start_time = time.time()
            frame_cout = 0
            
            #무한 루프 or 고정 횟수 처리
            while self.running and not self.should_stop:
                if num_frames is not None and frame_count >= num_frames :
                    break
                
                # ThreadPoolExecutor로 모든 채널을 동시에 처리
                with ThreadPoolExecutor(max_workers=len(self.channels_to_use)) as executor:
                    # 모든 채널에 대해 비동기 작업 제출
                    if use_temp_caps:
                        #사용자 입력 비디오 경로
                        futures ={
                            executor.submit(self.process_channel_worker, channel_id) : channel_id 
                            for channel_id, cap in channels_to_use.items()
                        }
                    else:
                        #기존 채널 사용

                        channel_ids = list(self.channels.keys())
                        futures = {
                            executor.submit(self.process_channel_worker, channel_id): channel_id 
                            for channel_id in channel_ids
                        }
                        
                    # 결과 수집
                    tracking_results = {}
                    for future in futures:
                        channel_id = futures[future]
                        try:
                            result = future.result(timeout=5.0)  # 5초 타임아웃
                            if result:
                                tracking_results[channel_id] = result
                        except Exception as e:
                            logging.warning(f"채널 {channel_id} 처리 타임아웃: {str(e)}")
                
                # 스레드 안전한 결과 저장
                with self.results_lock:
                    for channel_id, result in tracking_results.items():
                        self.result_buffer[channel_id].append(result)
                
                # FPS 계산 및 통계 업데이트
                fps_counter += len(tracking_results)

                if fps_counter >= 30:
                    current_time = time.time()
                    processing_fps = fps_counter / (current_time - fps_start_time)
                    
                    with self.stats_lock:
                        # 각 채널의 FPS 업데이트
                        for channel_id in tracking_results.keys():
                            self.channel_stats[channel_id]['processing_fps'] = processing_fps
                            self.channel_stats[channel_id]['frames_processed'] += 1
                    
                    fps_start_time = current_time
                    fps_counter = 0
                
                # 주기적으로 오래된 데이터 정리
                if self.total_frames % 1000 == 0:
                    self.global_id_manager.cleanup_old_data()
                    for pipeline in self.channel_pipelines.values():
                        pipeline.cleanup_old_data()
                        
                self.total_frames += len(tracking_results)
                
                # 잠시 대기 (CPU 과부하 방지)
                time.sleep(0.01)
        
        except Exception as e:
            logging.error(f"멀티스레딩 처리 중 오류: {str(e)}")
        finally:
            self.running = False


    def start_processing(self):
        """처리 시작"""
        if len(self.channels) == 0:
            return False, "처리할 채널이 없습니다"
        
        # 처리 스레드 시작
        self.processing_thread = threading.Thread(target=self.process_parallel)
        self.processing_thread.start()
        
        logging.info("배치 병렬 처리 시작됨")
        return True, "처리 시작됨"
    
    def stop_processing(self):
        """처리 중단"""
        self.should_stop = True
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=5)
        
        logging.info("배치 병렬 처리 중단됨")
    
    def get_status(self):
        """상태 조회"""
        avg_fps = (self.total_frames / self.total_processing_time) if self.total_processing_time > 0 else 0
        
        # 채널별 통계 수집
        total_corrections = sum(stats['id_corrections'] for stats in self.channel_stats.values())
        total_objects = sum(stats['objects_detected'] for stats in self.channel_stats.values())
        
        return {
            'running': self.running,
            'channels': len(self.channels),
            'total_frames': self.total_frames,
            'avg_processing_fps': avg_fps,
            'batch_size': self.batch_size,
            'total_objects_detected': total_objects,
            'total_id_corrections': total_corrections,
            'channel_stats': dict(self.channel_stats),
            'global_stats': self.global_id_manager.get_stats(),
            'cross_channel_objects': len(self.global_id_manager.get_cross_channel_objects())
        }
    
    def get_latest_results(self, channel_id=None):
        """최신 처리 결과 조회"""
        if channel_id:
            if channel_id in self.result_buffer and self.result_buffer[channel_id]:
                return self.result_buffer[channel_id][-1]
            return None
        else:
            results = {}
            for ch_id, buffer in self.result_buffer.items():
                if buffer:
                    results[ch_id] = buffer[-1]
            return results
    
    def shutdown(self):
        """시스템 종료"""
        self.stop_processing()
        
        # 비디오 캡처 해제
        for cap in self.video_caps.values():
            cap.release()
        
        logging.info("멀티채널 Re-ID 처리 시스템 종료 완료")