# WebRTC / WHEP Integration

GPU 서버가 SRS에서 WebRTC(WHEP)로 영상을 받아 처리하고, bbox/global_id/track_id/행동 메타데이터만 모바일로 쏴서 클라이언트가 오버레이하도록 하는 경로.

## 아키텍처

```
[카메라] ──SRT──▶ [SRS]  ──WebRTC / HLS──▶ [모바일 영상 재생]
                   │
                   └─WHEP(WebRTC pull)──▶ [GPU 서버]
                                              │
                                              ├─YOLO + Behavior 감지
                                              │
                                              └─WebSocket(JSON)──▶ [모바일 오버레이]
```

**핵심**:
- SRS가 media authority (GPU 죽어도 영상은 계속 재생됨)
- GPU 서버는 WebRTC로 pull해서 YOLO/ReID/행동 감지만 수행
- 메타데이터만 WebSocket으로 분리 송출 → 모바일에서 bbox/라벨 그림
- 영상과 메타는 각자 다른 경로로 도착 → pts/wallclock 기반 sync

## SRS 연결 정보 (ddnapet 환경 기준)

| 항목 | 값 |
|---|---|
| 서버 1 API | `http://118.41.173.32:1985` |
| 서버 2 API | `http://118.41.173.32:2985` |
| RTMP 수신 | 1935 |
| WebRTC UDP (서버1) | 8000 |
| WebRTC UDP (서버2) | 28000 |
| WHEP 엔드포인트 | `/rtc/v1/whep/?app=<app>&stream=<stream>&vhost=<vhost>` |
| vhost 예시 | `kr` (SRT tcUrl `srt://kr/live` 기준) |

스트림 목록 확인:
```bash
curl -sL http://118.41.173.32:2985/api/v1/streams | python3 -m json.tool
```

## GPU 서버 의존성

```bash
pip install aiortc aiohttp
```

- `aiortc 1.14+` (pure-python WebRTC)
- `aiohttp 3.x` (WHEP signaling + 메타데이터 WS 서버 공용)
- `av` (PyAV — aiortc가 자동으로 pull, H.264 소프트웨어 디코딩 사용)

> **주의**: aiortc는 CPU 디코딩이라 1080p 초과 해상도 4개 이상 동시 처리 시 병목 가능. 640p는 문제 없음.

## 연결 테스트

독립 스크립트로 signaling / 미디어 경로 검증:

```bash
python tools/test_whep.py \
    --url http://118.41.173.32:2985 \
    --app live \
    --stream facility-ddnapet_gmail-every1 \
    --vhost kr \
    --duration 10
```

성공 출력 예:
```
[signaling] OK (82ms)
[track] received: kind=video
[track] received: kind=audio
[track] first frame — 640x360, pts=54000, time_base=1/90000, ttff=1099ms
[result] frames received: 140
[result] effective fps:   15.58
[result] first_pts_sec:   0.600s
```

`frames received: 0` 이면:
- Publisher 비활성 (`publish.active: false`) → 카메라 송출 확인
- UDP 28000 outbound 차단 → GPU 서버 방화벽 확인

## 파이프라인 통합

### 설정 (JSON)

`sample_config.json`의 `input_source`를 `whep://` 스킴으로 교체:

```json
{
  "streams": [{
    "stream_id": "every1",
    "input_source": "whep://118.41.173.32:2985/live/facility-ddnapet_gmail-every1?vhost=kr",
    "method": "bytetrack",
    "task_fight": true,
    "task_sleep": true,
    "task_active": true,
    "target_fps": 15,
    "yolo_classes": [1]
  }],
  "metadata_ws_enabled": true,
  "metadata_ws_host": "0.0.0.0",
  "metadata_ws_port": 8766,
  "metadata_ws_path": "/ws/metadata"
}
```

**URL 포맷**: `whep://HOST:PORT/APP/STREAM[?vhost=X&mode=whep|srs&https=true]`

- `vhost` — SRS의 SRT tcUrl이 `srt://kr/live` 이면 `kr`
- `mode=srs` — SRS 구버전의 proprietary `/rtc/v1/play/` JSON API 강제
- `https=true` — TLS signaling (기본 http)

### 실행

```bash
cd PoC
python main.py run --config whep_test_config.json
```

- 기존 `rtsp://` / `rtmp://` 설정은 **그대로 동작** (WHEP 분기는 URL 스킴 기반)
- `metadata_ws_enabled=false`면 메타 송출 전부 no-op (zero overhead)

## 메타데이터 WebSocket 프로토콜

양방향 프로토콜. 서버 → 클라이언트는 프레임/스냅샷/제어 응답을 보내고,
클라이언트 → 서버는 구독·스냅샷 요청·핑을 보낸다.

### 연결

```
ws://<gpu-host>:8766/ws/metadata
```

연결 직후 서버가 `hello` 컨트롤 메시지로 현재 활성 stream_id 목록을 알려준다:
```json
{"type":"hello","streams":["every1","facility-ddnapet_gmail-every2"],"ts":1712345678.1}
```

### 서버 → 클라이언트 메시지

| `type` | 용도 |
|---|---|
| `frame_metadata` | 매 프레임 bbox/track/behavior 페이로드 (아래 스키마) |
| `hello` | 연결 직후 1회. 활성 stream_id 목록 |
| `ack` | subscribe/unsubscribe 처리 결과 (`stream_ids` 필드에 현재 구독 상태) |
| `snapshot` | `request_snapshot` 응답. `{stream_id, payload}` (payload=null 가능) |
| `pong` | `ping` 응답. `{ts}` |
| `error` | 잘못된 클라이언트 메시지 |

#### `frame_metadata` 페이로드

```json
{
  "type": "frame_metadata",
  "stream_id": "every1",
  "ts": 1712345678.123,
  "tracks": [
    {
      "tid": 12,
      "gid": 3,
      "pet_name": "뽀삐",
      "bbox_xywh": [cx, cy, w, h],
      "behavior": "sleeping"
    }
  ],
  "person_boxes": [[x1, y1, x2, y2], ...],
  "privacy_method": "blur"
}
```

### 클라이언트 → 서버 메시지

| 메시지 | 효과 |
|---|---|
| `{"type":"subscribe","stream_ids":["s1","s2"]}` | 해당 stream_id만 받음. **빈 리스트** 또는 필드 생략 시 모든 스트림 구독(기본값) |
| `{"type":"unsubscribe","stream_ids":["s1"]}` | 일부 스트림 구독 해제 (남은 구독이 비면 더 이상 frame_metadata를 받지 않음) |
| `{"type":"unsubscribe_all"}` | 필터 해제, 다시 모든 스트림 수신 |
| `{"type":"request_snapshot"}` | 현재 구독 중인(또는 모든) 스트림의 가장 최신 frame_metadata를 즉시 응답 |
| `{"type":"ping"}` | `{"type":"pong","ts":...}` 응답 |

> **모바일 권장 패턴**: 화면 진입 시 `subscribe`로 해당 카메라의 stream_id만 받고,
> 즉시 `request_snapshot`을 보내 첫 박스를 빠르게 그린다. 화면 이탈 시 `unsubscribe_all` 후 close.

| 필드 | 의미 |
|---|---|
| `type` | 항상 `"frame_metadata"` (추후 타입 확장 여지) |
| `stream_id` | config의 `streams[].stream_id` |
| `ts` | 소스 wallclock (WHEP 사용 시 RTP pts 기반, 아니면 SyncClock unix time) |
| `tracks[].tid` | ByteTrack/BoT-SORT가 부여한 ID (재접속 시 리셋됨) |
| `tracks[].gid` | ReID Global ID — 스트림 넘어서 안정. 미등록 시 `null` |
| `tracks[].pet_name` | pet_profiles에 등록된 이름 |
| `tracks[].bbox_xywh` | **center-xywh** 좌표 (소스 원본 픽셀 기준) |
| `tracks[].behavior` | `normal` · `fight` · `escape` · `sleeping` · `bathroom` · `feeding` · `playing` · `inactive` |
| `person_boxes` | (optional) `sc.privacy=true`일 때만. 사람 bbox (**corner-xyxy**, pad 10px 적용). 모바일에서 클라이언트 사이드 blur/모자이크 적용 |
| `privacy_method` | (optional) `"blur"` / `"mosaic"` / `"black"` — 모바일 렌더링 힌트 |

### 모바일 오버레이 구현 가이드

1. **영상 재생**: 기존 SRS WebRTC/HLS URL 그대로 사용
2. **메타 구독**: 위 WebSocket 연결
3. **sync offset 계산**:
   - WebRTC: player의 `currentTime` 가져와서 `ts`와 비교
   - HLS: `#EXT-X-PROGRAM-DATE-TIME` 기반 player 재생 시각 비교
4. **버퍼링**: 메타데이터가 영상보다 먼저 도착하므로 짧은 (200~500ms) ring buffer 유지하고, 영상 재생 시각에 맞는 메타를 꺼내 그림
5. **bbox 변환**: `bbox_xywh = [cx, cy, w, h]` → corner 좌표
   ```
   x1 = cx - w/2, y1 = cy - h/2
   x2 = cx + w/2, y2 = cy + h/2
   ```
   소스 해상도(640x360) → 플레이어 화면 해상도 비율로 스케일

## 코드 구성

| 파일 | 역할 |
|---|---|
| [PoC/whep_reader.py](../PoC/whep_reader.py) | cv2.VideoCapture 호환 WHEP 어댑터. 백그라운드 쓰레드에서 aiortc 돌리고 thread-safe queue로 frame 전달 |
| [PoC/metadata_sender.py](../PoC/metadata_sender.py) | aiohttp WebSocket broadcast 서버. 독립 asyncio 쓰레드 |
| [PoC/config.py](../PoC/config.py) | `metadata_ws_*` 필드 4개 추가 (모두 optional, default off) |
| [PoC/stream_processor.py](../PoC/stream_processor.py) | `_do_open_capture`에 `whep://` 분기, `_run_behavior_detection` 끝에 metadata push 훅 |
| [tools/test_whep.py](../tools/test_whep.py) | 독립 WHEP 연결 테스터 |

## Troubleshooting

### Signaling 실패 (`[signaling] FAILED`)

- HTTP 응답 상태 확인 (404 → 엔드포인트 경로 틀림, 400/500 → vhost/스트림 키 오류)
- `curl http://<srs>:1985/api/v1/versions` 로 API 살아있는지 확인
- 스트림 존재/활성 확인: `curl http://<srs>:1985/api/v1/streams`

### Signaling OK, frames=0

- `publish.active: false` → 카메라가 실제로 SRT/RTMP 밀어넣고 있는지
- 서버 send bytes 확인: `curl .../streams` → `send_30s` 0이면 SRS가 forwarding 안 함 (vhost 미스매치 가능성 높음)
- UDP outbound 차단 여부: aiortc 로깅 켜서 ICE candidate pair가 `SUCCEEDED`로 가는지 확인
  ```python
  logging.getLogger("aioice").setLevel(logging.INFO)
  ```

### Codec 에러

- 현재 aiortc가 지원하는 코덱: H.264, VP8, VP9
- H.265 (HEVC) 소스면 SRS에서 H.264로 트랜스코드 필요하거나, PyAV HEVC 지원 빌드 필요

### metadata_sender 포트 충돌

- `metadata_ws_port`를 다른 값으로 변경
- 헬스체크: `curl http://<gpu-host>:8766/health` → `{"clients": N, "queued": N}`

### pts/wallclock 드리프트

- `whep_reader.py`는 첫 프레임의 pts를 `time.time()` 앵커에 고정 → 이후 pts 차이로 wallclock 추론
- RTP clock (90kHz)이 실제 벽시계와 다르면 drift 발생 (분 단위로 무시 가능 수준)
- 정밀 동기화 필요 시 RTCP Sender Report 파싱으로 재보정 필요 (현재 미구현)

## 향후 개선

- [ ] RTCP SR 기반 pts→wallclock 동기화 (aiortc에서 SR 노출 후)
- [ ] H.265 인그레스 지원 (PyAV HEVC 빌드)
- [x] 메타데이터 WS에 `client->server` 메시지 추가 (스트림 구독/언구독)
- [x] stream_id별 메타 필터링 (현재 모든 클라이언트에 모든 스트림 전송)
- [ ] 행동 이벤트 시작/종료(Onset/Offset) 별도 타입 추가
