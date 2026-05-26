"""facility/cameras 계층 config → 기존 streams 평탄화.

새 포맷 (간결):
    {
      "defaults": { ...공통 stream 설정...,
                    "tasks": {fight, escape, inert, sleep, eat, bathroom, active},
                    "yolo": {conf, iou, classes, ...},
                    "adaptive_fps": {enabled, max, idle, min, displacement_low, displacement_high},
                    "reid": {use_reid, method, threshold, global_id, freeze_registered, every_n_frames} },
      "facilities": [
        {
          "user_type": "facility", "user_slug": "bouncedog04_gmail",
          "defaults": { ...시설별 override... },
          "cameras": [
            {"name": "every1"},
            {"name": "every2", "overrides": {"yolo_conf": 0.2}}
          ]
        }
      ],
      "srs": {"host": "118.41.173.32", "port": 2985, "vhost": "kr"}
    }

기존 포맷 (장황):
    { "streams": [{stream_id, input_source, method, target_fps, ...30+ 필드}, ...] }

이 모듈은 새 포맷을 기존 streams 배열로 펼쳐서 SystemConfig.from_dict 가 그대로 받게 한다.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List

# 그룹별 → flat 필드 prefix 매핑
_NESTED_PREFIX = {
    "tasks": "task_",            # tasks.fight → task_fight
    "yolo": "yolo_",             # yolo.conf  → yolo_conf
    "adaptive_fps": "adaptive_fps_",
    # reid 는 use_reid 만 특수 (prefix 없음), 나머지는 reid_*
    "reid": "reid_",
}

_DEFAULT_SRS = {"host": "118.41.173.32", "port": 2985, "vhost": "kr"}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any] | None) -> Dict[str, Any]:
    """dict 깊은 병합. override 값이 우선. dict 끼리는 재귀, 그 외엔 교체."""
    if not override:
        return copy.deepcopy(base)
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _flatten_group(group_key: str, group_val: Dict[str, Any]) -> Dict[str, Any]:
    """nested dict → flat. 특수 케이스(use_reid)는 prefix 없음."""
    out: Dict[str, Any] = {}
    if not isinstance(group_val, dict):
        return out
    prefix = _NESTED_PREFIX[group_key]
    for sub_k, sub_v in group_val.items():
        # reid.use_reid 만 prefix 없는 use_reid 로
        if group_key == "reid" and sub_k == "use_reid":
            out["use_reid"] = sub_v
        else:
            out[f"{prefix}{sub_k}"] = sub_v
    return out


def flatten_stream(merged: Dict[str, Any]) -> Dict[str, Any]:
    """merged dict 에서 nested 그룹을 flat 필드로 변환."""
    out: Dict[str, Any] = {}
    for k, v in merged.items():
        if k in _NESTED_PREFIX:
            out.update(_flatten_group(k, v))
        else:
            out[k] = v
    return out


def _build_input_source(srs: Dict[str, Any], stream_id: str) -> str:
    host = srs.get("host", _DEFAULT_SRS["host"])
    port = srs.get("port", _DEFAULT_SRS["port"])
    vhost = srs.get("vhost", _DEFAULT_SRS["vhost"])
    return f"whep://{host}:{port}/live/{stream_id}?vhost={vhost}"


def expand_facilities_to_streams(data: Dict[str, Any]) -> Dict[str, Any]:
    """new 포맷 dict 를 받아 streams 배열을 만들어 반환. 원본을 변경하지 않음."""
    data = copy.deepcopy(data)

    global_defaults: Dict[str, Any] = data.pop("defaults", {}) or {}
    facilities: List[Dict[str, Any]] = data.pop("facilities", []) or []
    srs: Dict[str, Any] = data.pop("srs", {}) or _DEFAULT_SRS

    streams: List[Dict[str, Any]] = list(data.get("streams", []) or [])  # 혼합 사용도 허용

    for fac in facilities:
        user_type = fac.get("user_type", "facility")
        user_slug = fac["user_slug"]
        fac_defaults: Dict[str, Any] = fac.get("defaults", {}) or {}

        for cam in fac.get("cameras", []) or []:
            name = cam["name"]
            overrides: Dict[str, Any] = cam.get("overrides", {}) or {}

            # 우선순위: global_defaults < facility_defaults < camera_overrides
            merged = deep_merge(deep_merge(global_defaults, fac_defaults), overrides)

            stream_id = cam.get("stream_id") or f"{user_type}-{user_slug}-{name}"
            input_source = cam.get("input_source") or _build_input_source(srs, stream_id)

            stream = flatten_stream(merged)
            stream["stream_id"] = stream_id
            stream["input_source"] = input_source
            # output_path 기본 None
            stream.setdefault("output_path", None)

            streams.append(stream)

    data["streams"] = streams
    return data
