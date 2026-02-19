"""
Track Event System

트랙 상태 변화 이벤트 시스템 - Observer 패턴
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Callable, Optional
from enum import Enum, auto
import time
import logging
from collections import defaultdict
import threading


class TrackEventType(Enum):
    """트랙 이벤트 타입"""
    TRACK_CREATED = auto()       # 새 트랙 생성
    TRACK_UPDATED = auto()       # 트랙 업데이트
    TRACK_LOST = auto()          # 트랙 소실 (일시적)
    TRACK_RECOVERED = auto()     # 트랙 복구
    TRACK_DELETED = auto()       # 트랙 삭제
    ID_SWITCHED = auto()         # ID 스위칭 감지
    ID_CORRECTED = auto()        # ID 보정됨
    FEATURE_UPDATED = auto()     # 특징 업데이트
    MATCHING_FAILED = auto()     # 매칭 실패
    OCCLUSION_START = auto()     # 가림 시작
    OCCLUSION_END = auto()       # 가림 종료
    BEHAVIOR_DETECTED = auto()   # 행동 감지


@dataclass
class TrackEvent:
    """트랙 이벤트"""
    event_type: TrackEventType
    track_id: int
    timestamp: float = field(default_factory=time.time)
    frame_idx: int = 0
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = ""  # 이벤트 발생 소스

    def __repr__(self):
        return f"TrackEvent({self.event_type.name}, track={self.track_id}, frame={self.frame_idx})"


class TrackEventHandler(ABC):
    """이벤트 핸들러 인터페이스"""

    @abstractmethod
    def handle(self, event: TrackEvent):
        """이벤트 처리"""
        pass

    @property
    def interested_events(self) -> List[TrackEventType]:
        """관심 있는 이벤트 타입 (기본: 전체)"""
        return list(TrackEventType)


class TrackEventBus:
    """
    이벤트 버스 - 이벤트 발행 및 구독 관리

    Thread-safe 구현
    """

    def __init__(self, async_mode: bool = False):
        self.handlers: Dict[TrackEventType, List[TrackEventHandler]] = defaultdict(list)
        self.callbacks: Dict[TrackEventType, List[Callable]] = defaultdict(list)
        self.async_mode = async_mode
        self._lock = threading.Lock()
        self.logger = logging.getLogger(self.__class__.__name__)

        # 이벤트 히스토리 (선택적)
        self.history: List[TrackEvent] = []
        self.max_history = 1000

        # 통계
        self._stats = defaultdict(int)

    def subscribe(self, handler: TrackEventHandler):
        """핸들러 구독"""
        with self._lock:
            for event_type in handler.interested_events:
                if handler not in self.handlers[event_type]:
                    self.handlers[event_type].append(handler)
                    self.logger.debug(f"Subscribed {handler.__class__.__name__} to {event_type.name}")

    def subscribe_callback(self, event_type: TrackEventType,
                          callback: Callable[[TrackEvent], None]):
        """콜백 함수 구독"""
        with self._lock:
            self.callbacks[event_type].append(callback)

    def unsubscribe(self, handler: TrackEventHandler):
        """핸들러 구독 해제"""
        with self._lock:
            for event_type in handler.interested_events:
                if handler in self.handlers[event_type]:
                    self.handlers[event_type].remove(handler)

    def publish(self, event: TrackEvent):
        """이벤트 발행"""
        self._stats[event.event_type] += 1

        # 히스토리 저장
        if len(self.history) >= self.max_history:
            self.history.pop(0)
        self.history.append(event)

        # 핸들러 호출
        with self._lock:
            handlers = list(self.handlers.get(event.event_type, []))
            callbacks = list(self.callbacks.get(event.event_type, []))

        for handler in handlers:
            try:
                handler.handle(event)
            except Exception as e:
                self.logger.error(f"Handler {handler.__class__.__name__} failed: {e}")

        for callback in callbacks:
            try:
                callback(event)
            except Exception as e:
                self.logger.error(f"Callback failed: {e}")

    def get_stats(self) -> Dict[str, int]:
        """이벤트 통계"""
        return {k.name: v for k, v in self._stats.items()}

    def get_history(self, event_type: TrackEventType = None,
                   track_id: int = None,
                   limit: int = 100) -> List[TrackEvent]:
        """이벤트 히스토리 조회"""
        filtered = self.history

        if event_type:
            filtered = [e for e in filtered if e.event_type == event_type]

        if track_id is not None:
            filtered = [e for e in filtered if e.track_id == track_id]

        return filtered[-limit:]

    def clear_history(self):
        """히스토리 초기화"""
        self.history.clear()


class LoggingEventHandler(TrackEventHandler):
    """로깅 이벤트 핸들러"""

    def __init__(self, log_level: int = logging.INFO):
        self.logger = logging.getLogger("TrackEvents")
        self.log_level = log_level

    def handle(self, event: TrackEvent):
        message = f"[{event.event_type.name}] Track {event.track_id} at frame {event.frame_idx}"

        if event.data:
            message += f" - {event.data}"

        self.logger.log(self.log_level, message)


class IDSwitchHandler(TrackEventHandler):
    """ID 스위칭 감지 및 처리 핸들러"""

    def __init__(self, correction_callback: Callable = None):
        self.correction_callback = correction_callback
        self.switch_history: Dict[int, List[Dict]] = defaultdict(list)

    @property
    def interested_events(self) -> List[TrackEventType]:
        return [TrackEventType.ID_SWITCHED, TrackEventType.ID_CORRECTED]

    def handle(self, event: TrackEvent):
        self.switch_history[event.track_id].append({
            'event_type': event.event_type.name,
            'timestamp': event.timestamp,
            'frame_idx': event.frame_idx,
            'data': event.data
        })

        if event.event_type == TrackEventType.ID_SWITCHED and self.correction_callback:
            self.correction_callback(event)

    def get_switch_count(self, track_id: int = None) -> int:
        """ID 스위칭 횟수"""
        if track_id is not None:
            return len([h for h in self.switch_history[track_id]
                       if h['event_type'] == 'ID_SWITCHED'])
        return sum(len([h for h in history if h['event_type'] == 'ID_SWITCHED'])
                   for history in self.switch_history.values())


class OcclusionHandler(TrackEventHandler):
    """가림 이벤트 처리 핸들러"""

    def __init__(self):
        self.occluded_tracks: Dict[int, Dict] = {}

    @property
    def interested_events(self) -> List[TrackEventType]:
        return [TrackEventType.OCCLUSION_START, TrackEventType.OCCLUSION_END,
                TrackEventType.TRACK_LOST, TrackEventType.TRACK_RECOVERED]

    def handle(self, event: TrackEvent):
        if event.event_type in [TrackEventType.OCCLUSION_START, TrackEventType.TRACK_LOST]:
            self.occluded_tracks[event.track_id] = {
                'start_frame': event.frame_idx,
                'start_time': event.timestamp
            }
        elif event.event_type in [TrackEventType.OCCLUSION_END, TrackEventType.TRACK_RECOVERED]:
            if event.track_id in self.occluded_tracks:
                start_info = self.occluded_tracks.pop(event.track_id)
                duration_frames = event.frame_idx - start_info['start_frame']
                duration_time = event.timestamp - start_info['start_time']
                # 가림 지속 시간 등 통계 수집 가능

    def get_occluded_tracks(self) -> List[int]:
        """현재 가려진 트랙 목록"""
        return list(self.occluded_tracks.keys())


class BehaviorEventHandler(TrackEventHandler):
    """행동 감지 이벤트 핸들러"""

    def __init__(self, behavior_callback: Callable = None):
        self.behavior_callback = behavior_callback
        self.detected_behaviors: Dict[int, List[Dict]] = defaultdict(list)

    @property
    def interested_events(self) -> List[TrackEventType]:
        return [TrackEventType.BEHAVIOR_DETECTED]

    def handle(self, event: TrackEvent):
        behavior_type = event.data.get('behavior_type', 'unknown')

        self.detected_behaviors[event.track_id].append({
            'behavior': behavior_type,
            'frame_idx': event.frame_idx,
            'timestamp': event.timestamp,
            'confidence': event.data.get('confidence', 1.0)
        })

        if self.behavior_callback:
            self.behavior_callback(event)

    def get_behaviors(self, track_id: int = None) -> Dict[int, List[Dict]]:
        """감지된 행동 조회"""
        if track_id is not None:
            return {track_id: self.detected_behaviors.get(track_id, [])}
        return dict(self.detected_behaviors)


class TrackStateManager:
    """
    트랙 상태 관리자

    이벤트 기반 트랙 상태 추적 및 관리
    """

    def __init__(self, event_bus: TrackEventBus = None):
        self.event_bus = event_bus or TrackEventBus()
        self.tracks: Dict[int, Dict] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

        # 이벤트 버스에 자체 핸들러 등록
        self.event_bus.subscribe(self._InternalHandler(self))

    class _InternalHandler(TrackEventHandler):
        def __init__(self, manager: 'TrackStateManager'):
            self.manager = manager

        def handle(self, event: TrackEvent):
            self.manager._update_state(event)

    def _update_state(self, event: TrackEvent):
        """이벤트에 따른 상태 업데이트"""
        track_id = event.track_id

        if event.event_type == TrackEventType.TRACK_CREATED:
            self.tracks[track_id] = {
                'created_at': event.timestamp,
                'created_frame': event.frame_idx,
                'last_seen': event.timestamp,
                'last_frame': event.frame_idx,
                'status': 'active',
                'history': []
            }

        elif track_id in self.tracks:
            track = self.tracks[track_id]

            if event.event_type == TrackEventType.TRACK_UPDATED:
                track['last_seen'] = event.timestamp
                track['last_frame'] = event.frame_idx
                track['status'] = 'active'

            elif event.event_type == TrackEventType.TRACK_LOST:
                track['status'] = 'lost'

            elif event.event_type == TrackEventType.TRACK_RECOVERED:
                track['status'] = 'active'

            elif event.event_type == TrackEventType.TRACK_DELETED:
                track['status'] = 'deleted'

            track['history'].append({
                'event': event.event_type.name,
                'frame': event.frame_idx
            })

    def create_track(self, track_id: int, frame_idx: int = 0, **data):
        """새 트랙 생성 이벤트 발행"""
        self.event_bus.publish(TrackEvent(
            event_type=TrackEventType.TRACK_CREATED,
            track_id=track_id,
            frame_idx=frame_idx,
            data=data,
            source='TrackStateManager'
        ))

    def update_track(self, track_id: int, frame_idx: int = 0, **data):
        """트랙 업데이트 이벤트 발행"""
        self.event_bus.publish(TrackEvent(
            event_type=TrackEventType.TRACK_UPDATED,
            track_id=track_id,
            frame_idx=frame_idx,
            data=data,
            source='TrackStateManager'
        ))

    def mark_lost(self, track_id: int, frame_idx: int = 0, **data):
        """트랙 소실 이벤트 발행"""
        self.event_bus.publish(TrackEvent(
            event_type=TrackEventType.TRACK_LOST,
            track_id=track_id,
            frame_idx=frame_idx,
            data=data,
            source='TrackStateManager'
        ))

    def mark_recovered(self, track_id: int, frame_idx: int = 0, **data):
        """트랙 복구 이벤트 발행"""
        self.event_bus.publish(TrackEvent(
            event_type=TrackEventType.TRACK_RECOVERED,
            track_id=track_id,
            frame_idx=frame_idx,
            data=data,
            source='TrackStateManager'
        ))

    def delete_track(self, track_id: int, frame_idx: int = 0, **data):
        """트랙 삭제 이벤트 발행"""
        self.event_bus.publish(TrackEvent(
            event_type=TrackEventType.TRACK_DELETED,
            track_id=track_id,
            frame_idx=frame_idx,
            data=data,
            source='TrackStateManager'
        ))

    def get_active_tracks(self) -> List[int]:
        """활성 트랙 목록"""
        return [tid for tid, info in self.tracks.items()
                if info['status'] == 'active']

    def get_track_info(self, track_id: int) -> Optional[Dict]:
        """트랙 정보 조회"""
        return self.tracks.get(track_id)

    def cleanup_deleted(self):
        """삭제된 트랙 정리"""
        deleted = [tid for tid, info in self.tracks.items()
                   if info['status'] == 'deleted']
        for tid in deleted:
            del self.tracks[tid]
