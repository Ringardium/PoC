# 모바일 앱 AI 오버레이 구현 계획

MungAI 앱이 PoC AI 서버로부터 JSON 메타데이터(bbox / track_id / global_id / behavior)를 WebSocket으로 받아 WebRTC 스트림 위에 시각화하기 위한 계획.

서버 측(PoC)은 이미 메타데이터 송신 구조가 완성되어 있으므로, 본 문서는 **모바일 앱 측 구현**에 집중한다.

---

## 1. 기존 PoC 자산 (서버 측, 구현 불필요)

| 항목 | 위치 |
|---|---|
| WebSocket 송신 서버 | [PoC/metadata_sender.py](../PoC/metadata_sender.py) — `ws://<server>:8766/ws/metadata` |
| JSON 스키마 정의 | 같은 파일 상단 docstring (line 7–26) |
| 행동별 bbox 색상 상수 | [main.py:64-71](../main.py) |
| 행동별 이모지 PNG 생성기 | [tools/generate_emoji.py:26-34](../tools/generate_emoji.py) |
| OpenCV 기준 그리기 로직 (시각 레퍼런스) | [main.py:280-370](../main.py) |
| Privacy 마스킹 메타데이터 | `person_boxes`, `privacy_method` 필드 |

---

## 2. 확정 스키마

`metadata_sender.py`가 송신하는 프레임 페이로드:

```json
{
  "type": "frame_metadata",
  "stream_id": "facility-ddnapet_gmail-every1",
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

- `bbox_xywh`는 **원본 해상도 픽셀, center-xywh**.
- `behavior` 값: `normal | fight | escape | sleeping | bathroom | feeding | playing | inactive`.
- `person_boxes`는 **corner-xyxy**. privacy 설정이 켜진 스트림에서만 존재.

> 주의: 모바일 앱의 기존 `BoundingBoxOverlay.tsx`는 좌상단-xywh 전제라 **재사용이 어렵다**. 새 컴포넌트로 작성하는 것이 깔끔하다.

---

## 3. 전달 채널

`WebSocket` 사용 (PoC 서버가 이미 `/ws/metadata`로 제공).

- 앱에서 직접 8766에 접속할지, backend(Express)를 프록시로 두고 인증 붙일지 결정 필요.
- 재연결은 exponential backoff (1s → 30s 상한).
- 같은 소켓에 여러 카메라가 섞여 오므로 **`stream_id` 필터링 필수**.

### stream_id 매칭 규약 (확인 필요)

서버 기준 `facility-{user}-{camera}` 포맷. 앱의 `cameraName`을 여기에 어떻게 매핑할지 AI 서버 config 쪽과 확정해야 한다.

---

## 4. 행동 정규화 + 스타일 매핑

메타데이터의 behavior 값(`sleeping/feeding/playing/inactive`)이 PoC 이모지/색상 키(`sleep/eat/play/inert`)와 이름이 달라 정규화가 필요. 앱에서는 **한 테이블로 통합**한다.

```ts
// MungAI/src/config/behaviorStyle.ts
export const BEHAVIOR_STYLE = {
  normal:   { color: '#00FF00', emoji: null  },
  fight:    { color: '#FF0000', emoji: '🥊' },
  escape:   { color: '#FFFF00', emoji: '⚠️' },
  inactive: { color: '#0000FF', emoji: '❄️' },
  playing:  { color: '#FFA500', emoji: '🎾' },
  sleeping: { color: '#800080', emoji: '😴' },
  feeding:  { color: '#FF69B4', emoji: '🍽️' },
  bathroom: { color: '#00BFFF', emoji: '🚽' },
} as const;
```

색상 값은 PoC `main.py:64-71`의 RGB 상수와 1:1 일치시킨 것.

> React Native `<Text>`는 시스템 이모지 폰트로 컬러 이모지를 바로 렌더하므로 **PNG 에셋 번들링 불필요**. PoC가 PNG를 쓰는 이유는 OpenCV가 이모지를 못 그리기 때문이다.

---

## 5. 앱 파일 구조

```
MungAI/src/
├── config/
│   └── behaviorStyle.ts        색상/이모지 테이블
├── types/
│   └── aiMetadata.ts           FrameMetadata, Track 타입
├── services/
│   └── aiMetadataSocket.ts     WS 구독, stream_id 필터, 재연결
├── hooks/
│   └── useAIMetadata.ts        useAIMetadata(streamId) -> {tracks, personBoxes, ts}
├── utils/
│   └── videoCoord.ts           cover letterbox 보정 + center-xywh → 화면 xyxy
└── components/
    └── AIOverlay.tsx           SVG 기반 오버레이
```

---

## 6. 렌더 레이어 순서

`CameraDetailScreen.tsx`의 `videoWrapper` 안에서, `WebRTCPlayer` 위에 쌓는 순서:

1. `RTCView` (비디오)
2. `AIOverlay` 내부:
   - (a) **person_boxes 마스킹** — `privacy_method`에 따라 SVG `<Rect fill="black">` / 모자이크 / (블러는 MVP 제외)
   - (b) bbox — `<Rect stroke={color} strokeWidth=3 fill="none">`
   - (c) 좌상단 라벨 — `pet_name` 있으면 `{pet_name} · id:{tid}`, 없으면 `ID:{tid}/G:{gid}` (PoC `main.py:331` 포맷 일치)
   - (d) 우상단 이모지 — `<Text style={{fontSize:22}}>` (behavior ≠ normal일 때만)

**Privacy 박스는 반드시 AI 감지 박스보다 먼저 그려서 아래 깔리도록 하면 안 된다. 순서상 (a) 먼저 검정 채움 → 그 위에 bbox/라벨.**

---

## 7. 좌표 변환 (필수)

### 문제

- 서버 bbox: 원본 해상도 pixel, center-xywh
- `WebRTCPlayer`: `objectFit="cover"` → letterbox crop 발생
- 기존 `CameraDetailScreen.tsx:88-117`의 단순 스케일 변환은 **cover를 반영하지 않음** → 그대로 쓰면 박스가 화면과 어긋난다

### 해결

`utils/videoCoord.ts`에서 cover fit 반영:

```
scale = max(containerW / srcW, containerH / srcH)
renderedW = srcW * scale
renderedH = srcH * scale
offsetX = (containerW - renderedW) / 2    // 좌우 크롭이면 음수
offsetY = (containerH - renderedH) / 2    // 상하 크롭이면 음수

screenX = cx * scale + offsetX - (w * scale) / 2
screenY = cy * scale + offsetY - (h * scale) / 2
screenW = w * scale
screenH = h * scale
```

`videoResolution`은 이미 `CameraDetailScreen.tsx:150`에서 추적 중이므로 그대로 사용. Escape zone도 동일 유틸로 통합하면 일관됨.

---

## 8. 동기화 전략

WebRTC 영상과 AI 메타 도착 시각이 달라 그대로 그리면 박스가 앞서/뒤처질 수 있음.

- **MVP**: 최신 tracks만 그리기 (best-effort). 대부분 체감 문제없음.
- **개선 (필요 시)**: 300ms 버퍼에 메타 프레임 쌓고 `onProgress` 재생 시각과 `ts` 매칭해 pick.
- **떨림 완화**: bbox 위치/크기에 200ms LERP (Reanimated `withTiming`).

---

## 9. 성능

- 렌더 쓰로틀: AI가 30fps로 와도 UI는 10–15fps 충분.
- 박스 컴포넌트 `React.memo`, key=`tid`.
- 화면 이탈 시 WS close — `WebRTCPlayer` mount/unmount 패턴을 따라간다.

---

## 10. 구현 순서 (체크리스트)

- [ ] stream_id 규약 확정 (서버 config ↔ 앱 cameraName)
- [ ] `types/aiMetadata.ts` — 스키마 타입
- [ ] `config/behaviorStyle.ts` — 매핑 테이블
- [ ] `utils/videoCoord.ts` — cover letterbox 보정
- [ ] `services/aiMetadataSocket.ts` — WS 클라이언트 + 재연결
- [ ] `hooks/useAIMetadata.ts`
- [ ] `components/AIOverlay.tsx` — SVG 기반
- [ ] `CameraDetailScreen.tsx:394` 에 `<AIOverlay>` 삽입
- [ ] 기존 `BoundingBoxOverlay.tsx` / `yoloUtils.ts` 제거 또는 on-device 용도로 명시 분리
- [ ] (선택) ts 기반 동기화 버퍼 추가
- [ ] (선택) bbox 보간 (Reanimated)

---

## 11. 열린 결정 사항

1. **WS 노출 경로**: 앱 → 8766 직접 vs backend Express 프록시 (+ JWT 검증)
2. **privacy_method=blur 처리**: MVP는 검정 박스, 이후 `@react-native-community/blur`로 업그레이드 여부
3. **behavior 다국어**: 이모지 + 라벨 텍스트의 i18n 여부 (`t('behavior.sleeping')` 등)
4. **이벤트 토스트 연동**: `CameraDetailScreen.showToastNotification`이 이미 있음. 특정 behavior 최초 감지 시 자동 호출할지, bbox만 표시할지
