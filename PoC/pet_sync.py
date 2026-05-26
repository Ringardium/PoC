"""시설 등록 강아지 자동 동기화 — backend dogs API → PetProfileStore → ReID gallery.

흐름:
    1. 설정된 streams 의 stream_id 에서 facility user_slug 들을 unique 추출
       (예: "facility-bouncedog04_gmail-every1" → "bouncedog04_gmail")
    2. 주기적(default 60s)으로 backend GET /api/ai/dogs/facility/by-slug/{slug} 호출
    3. 응답으로 받은 강아지 메타 + profileImageUrl 을 다운로드
       → references/images/{slug}_{dog_id}.jpg
    4. PetProfileStore 에 global_id = dogs.id 로 등록 (backend 와 ID 일관)
    5. 변경 발생 시 on_change 콜백 호출 → stream_processor 가 ReID gallery 재로딩

설계 원칙:
    - 메인 처리 루프 영향 없음 (백그라운드 데몬 스레드)
    - 인증 없음 (/api/ai/* 패턴, 임시)
    - 강아지 추가/제거 모두 자동 반영. 다음 sync 에서 빠진 강아지는 store 에서 제거.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable, Dict, List, Optional, Set
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)


def extract_facility_slugs(stream_ids: List[str]) -> Set[str]:
    """stream_id 들에서 facility user_slug 들을 unique 추출."""
    slugs: Set[str] = set()
    for sid in stream_ids:
        parts = sid.split("-", 2)
        if len(parts) >= 3 and parts[0] == "facility":
            slugs.add(parts[1])
    return slugs


class PetSync:
    """주기적으로 backend 에서 시설 강아지를 가져와 ReID gallery 동기화."""

    def __init__(
        self,
        base_url: str,
        stream_ids: List[str],
        references_dir: str = "references",
        interval_sec: float = 60.0,
        on_change: Optional[Callable[[], None]] = None,
        request_timeout: float = 10.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.stream_ids = list(stream_ids)
        self.references_dir = references_dir
        self.interval_sec = interval_sec
        self.on_change = on_change
        self.request_timeout = request_timeout

        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # (slug, dog_id) 와 그 last_image_url 캐시 — 변경 감지용
        self._known: Dict[str, Dict[int, str]] = {}   # slug -> {dog_id: image_url}

    # ------------------------------------------------------------------
    # 라이프사이클
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        os.makedirs(os.path.join(self.references_dir, "images"), exist_ok=True)
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="PetSync")
        self._thread.start()
        slugs = extract_facility_slugs(self.stream_ids)
        logger.info(
            f"PetSync started → {self.base_url} (interval={self.interval_sec}s, "
            f"facilities={sorted(slugs)})"
        )

    def stop(self, timeout: float = 5.0) -> None:
        if self._thread is None:
            return
        self._running.clear()
        self._thread.join(timeout=timeout)
        self._thread = None
        logger.info("PetSync stopped")

    # ------------------------------------------------------------------
    # 메인 루프
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        # 시작 즉시 1회, 그 다음 interval 마다
        while self._running.is_set():
            try:
                changed = self._sync_once()
                if changed and self.on_change is not None:
                    try:
                        self.on_change()
                    except Exception as e:
                        logger.warning(f"PetSync on_change callback failed: {e}")
            except Exception as e:
                logger.warning(f"PetSync error: {e}")
            # interval 대기 (중간 종료 시 빠르게 빠져나오기)
            slept = 0.0
            step = 0.5
            while slept < self.interval_sec and self._running.is_set():
                time.sleep(step)
                slept += step

    # ------------------------------------------------------------------
    # 1회 동기화
    # ------------------------------------------------------------------

    def _sync_once(self) -> bool:
        """모든 facility slug 순회. 변경 발생 여부 반환."""
        # PetProfileStore 는 매번 fresh 로 만들어 backend 가 곧 진실 (현재 체류 강아지) 가 되게.
        from tools.pet_profiles import PetProfileStore
        store = PetProfileStore(self.references_dir)
        # 기존 pets.json 로드 (이미지 경로 유지용)
        try:
            store.load()
        except Exception:
            pass

        slugs = extract_facility_slugs(self.stream_ids)
        if not slugs:
            return False

        new_known: Dict[str, Dict[int, str]] = {}
        any_change = False

        for slug in slugs:
            try:
                dogs = self._fetch_dogs(slug)
            except Exception as e:
                logger.warning(f"PetSync fetch failed (slug={slug}): {e}")
                # 실패 시 기존 정보 유지
                if slug in self._known:
                    new_known[slug] = self._known[slug]
                continue

            slug_dogs: Dict[int, str] = {}
            for dog in dogs:
                try:
                    dog_id = int(dog["id"])
                except (KeyError, TypeError, ValueError):
                    continue
                name = dog.get("name") or ""
                breed = dog.get("breed") or ""
                image_url = dog.get("profileImageUrl") or ""
                slug_dogs[dog_id] = image_url

                # PetProfileStore 에 등록 (없으면 추가, 있으면 이름/품종 갱신)
                if dog_id not in store.profiles:
                    store.add_pet(name=name, species="dog", breed=breed, global_id=dog_id)
                else:
                    existing = store.profiles[dog_id]
                    if existing.name != name or existing.breed != breed:
                        store.update_pet(dog_id, name=name, breed=breed)

                # 이미지 다운로드 (이전 URL 과 다를 때만)
                prev_url = self._known.get(slug, {}).get(dog_id)
                if image_url and image_url != prev_url:
                    local = self._download_image(slug, dog_id, image_url)
                    if local:
                        try:
                            store.add_reference_image(dog_id, local)
                        except Exception as e:
                            logger.debug(f"add_reference_image failed (gid={dog_id}): {e}")

            # 사라진 강아지(체크아웃 또는 삭제) → store 에서 제거
            prev_ids = set(self._known.get(slug, {}).keys())
            current_ids = set(slug_dogs.keys())
            for removed in prev_ids - current_ids:
                if removed in store.profiles:
                    try:
                        store.remove_pet(removed)
                    except Exception as e:
                        logger.debug(f"remove_pet failed (gid={removed}): {e}")

            if prev_ids != current_ids:
                any_change = True
            new_known[slug] = slug_dogs

        # 변경 있을 때만 저장 (디스크 I/O 절약)
        if any_change or new_known != self._known:
            try:
                store.save()
            except Exception as e:
                logger.warning(f"PetProfileStore save failed: {e}")
            self._known = new_known
            logger.info(
                f"PetSync updated — facilities={list(new_known)} "
                f"dogs={sum(len(v) for v in new_known.values())}"
            )
            return True
        return False

    # ------------------------------------------------------------------
    # backend 호출 + 이미지 다운로드
    # ------------------------------------------------------------------

    def _fetch_dogs(self, slug: str) -> List[dict]:
        url = f"{self.base_url}/api/ai/dogs/facility/by-slug/{slug}"
        resp = requests.get(url, timeout=self.request_timeout)
        resp.raise_for_status()
        body = resp.json()
        return body.get("dogs", []) if isinstance(body, dict) else []

    def _download_image(self, slug: str, dog_id: int, url: str) -> Optional[str]:
        try:
            ext = os.path.splitext(urlparse(url).path)[1] or ".jpg"
            local = os.path.join(self.references_dir, "images", f"{slug}_{dog_id}{ext}")
            r = requests.get(url, timeout=self.request_timeout, stream=True)
            r.raise_for_status()
            with open(local, "wb") as f:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        f.write(chunk)
            return local
        except Exception as e:
            logger.debug(f"image download failed ({slug}/{dog_id}): {e}")
            return None
