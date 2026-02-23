"""
Pet Profile Store - JSON 기반 펫 프로필 관리

펫의 이름, 종, 이미지 등 정보를 references/pets.json에 저장하고
ReID / Global ID 시스템과 연동하여 사용.

사용법:
    from pet_profiles import PetProfileStore

    store = PetProfileStore("references")
    gid = store.add_pet(name="뽀삐", species="dog", breed="골든 리트리버")
    store.add_reference_image(gid, "path/to/poppi.jpg")
    store.save()

    # ReID Image Matcher와 연동
    refs = store.to_reid_references()
"""

import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


@dataclass
class PetProfile:
    """단일 펫 프로필"""
    global_id: int
    name: str
    species: str = ""           # dog, cat 등
    breed: str = ""
    color: str = ""
    notes: str = ""
    reference_images: List[str] = field(default_factory=list)  # images/ 하위 상대 경로
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PetProfile":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class PetProfileStore:
    """
    JSON 파일 기반 펫 프로필 CRUD 관리

    디렉토리 구조:
        {base_dir}/
        ├── pets.json
        └── images/
            ├── poppi_001.jpg
            └── mimi_001.jpg
    """

    def __init__(self, base_dir: str = "references"):
        self.base_dir = Path(base_dir)
        self.images_dir = self.base_dir / "images"
        self.json_path = self.base_dir / "pets.json"
        self.profiles: Dict[int, PetProfile] = {}
        self._next_id = 1

        # 디렉토리 생성
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

        # 기존 데이터 로드
        if self.json_path.exists():
            self.load()

    def load(self) -> List[PetProfile]:
        """pets.json에서 프로필 로드"""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.profiles.clear()
            for pet_data in data.get("pets", []):
                profile = PetProfile.from_dict(pet_data)
                self.profiles[profile.global_id] = profile
                if profile.global_id >= self._next_id:
                    self._next_id = profile.global_id + 1

            logger.info(f"{len(self.profiles)}개 펫 프로필 로드 완료: {self.json_path}")
            return list(self.profiles.values())

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"pets.json 파싱 실패: {e}")
            return []

    def save(self):
        """현재 프로필을 pets.json에 저장"""
        data = {
            "version": "1.0",
            "pets": [p.to_dict() for p in sorted(self.profiles.values(), key=lambda x: x.global_id)]
        }
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"{len(self.profiles)}개 펫 프로필 저장 완료: {self.json_path}")

    def add_pet(self, name: str, species: str = "", breed: str = "",
                color: str = "", notes: str = "") -> int:
        """펫 추가, global_id 반환"""
        now = datetime.now().isoformat(timespec='seconds')
        global_id = self._next_id
        self._next_id += 1

        profile = PetProfile(
            global_id=global_id,
            name=name,
            species=species,
            breed=breed,
            color=color,
            notes=notes,
            reference_images=[],
            created_at=now,
            updated_at=now,
        )
        self.profiles[global_id] = profile
        logger.info(f"펫 추가: {name} (Global ID: {global_id})")
        return global_id

    def update_pet(self, global_id: int, **kwargs):
        """펫 정보 수정"""
        if global_id not in self.profiles:
            raise KeyError(f"Global ID {global_id} 없음")
        profile = self.profiles[global_id]
        for key, value in kwargs.items():
            if hasattr(profile, key) and key not in ('global_id', 'created_at'):
                setattr(profile, key, value)
        profile.updated_at = datetime.now().isoformat(timespec='seconds')

    def remove_pet(self, global_id: int):
        """펫 삭제 (이미지 파일은 유지)"""
        if global_id not in self.profiles:
            raise KeyError(f"Global ID {global_id} 없음")
        name = self.profiles[global_id].name
        del self.profiles[global_id]
        logger.info(f"펫 삭제: {name} (Global ID: {global_id})")

    def add_reference_image(self, global_id: int, image_path: str) -> str:
        """
        레퍼런스 이미지 추가 (images/ 디렉토리로 복사)

        Returns:
            images/ 하위 상대 경로
        """
        if global_id not in self.profiles:
            raise KeyError(f"Global ID {global_id} 없음")

        src = Path(image_path)
        if not src.exists():
            raise FileNotFoundError(f"이미지 없음: {image_path}")

        profile = self.profiles[global_id]
        # 파일명: {name}_{순번}.{ext}
        safe_name = profile.name.replace(" ", "_")
        idx = len(profile.reference_images) + 1
        dst_name = f"{safe_name}_{idx:03d}{src.suffix}"
        dst = self.images_dir / dst_name

        shutil.copy2(str(src), str(dst))

        rel_path = f"images/{dst_name}"
        profile.reference_images.append(rel_path)
        profile.updated_at = datetime.now().isoformat(timespec='seconds')
        logger.info(f"레퍼런스 이미지 추가: {rel_path} -> {profile.name}")
        return rel_path

    def get_pet(self, global_id: int) -> Optional[PetProfile]:
        """단일 펫 조회"""
        return self.profiles.get(global_id)

    def list_pets(self) -> List[PetProfile]:
        """전체 펫 목록"""
        return sorted(self.profiles.values(), key=lambda x: x.global_id)

    def to_reid_references(self) -> dict:
        """
        reid_image_matcher.py의 _load_from_json()과 호환되는 형식으로 변환

        Returns:
            {"references": [{"name": ..., "image_path": ..., "global_id": ...}, ...]}
        """
        references = []
        for profile in self.profiles.values():
            for img_path in profile.reference_images:
                references.append({
                    "name": profile.name,
                    "image_path": img_path,
                    "global_id": profile.global_id,
                })
        return {"references": references}

    def get_name_map(self) -> Dict[int, str]:
        """global_id -> name 매핑 딕셔너리 반환 (시각화용)"""
        return {p.global_id: p.name for p in self.profiles.values()}
