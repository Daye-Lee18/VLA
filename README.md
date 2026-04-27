# VLA — myCobot 280 Pick & Place

목표: myCobot 280 + RealSense로 pick & place 시연 데이터 50개 수집 → 서버(RTX 2080 Ti)에서 ACT policy fine-tuning → 실제 로봇 평가

---

## 파일 안내

| 파일 | 언제 보는가 |
|------|------------|
| `README.md` | 실제로 실행할 때 — 커맨드 복붙용 치트시트 |
| `vla_roadmap.qmd` | 실험 설계 / 배경 이해할 때 — JD 연결, 전체 파이프라인 흐름, 각 Phase 상세 설명, 진행 로그 |
| `lerobot_models.qmd` | 모델 선택 고민할 때 — 전체 38개 모델 목록, 프로젝트 추천 순위, 모델 교체 근거 |

**서버:** `ssh team2@100.66.177.119` (RTX 2080 Ti, 11GB VRAM)  
**로봇:** myCobot 280 for Arduino · Raspberry Pi 5 내장  
**카메라:** Intel RealSense (USB, top-down 고정)  
**연결:** `/dev/ttyJETCOBOT` · Baud `1000000`

---

## 진행 순서

| Phase | 내용 | 환경 | 상태 |
|---|---|---|---|
| 0 | 서버 연결 (Tailscale + SSH) | 로컬 | ✅ |
| 1 | 서버 환경 확인 (lerobot, CUDA) | 서버 | ✅ |
| 2 | myCobot + RealSense 연결 확인 | Raspberry Pi | 🔲 |
| 3 | 데이터 수집 → 서버 전송 → LeRobot 포맷 변환 | Raspberry Pi → 서버 | 🔲 |
| 4 | ACT fine-tuning | 서버 | 🔲 |
| 5 | 실제 로봇 평가 | 서버 + 로봇 | 🔲 |

---

## 장치 연결 구조

| 장치 | 연결 방식 | 인식 | 포트 |
|---|---|---|---|
| myCobot 280 | Raspberry Pi USB-A 포트 (CH340 칩) | tty 계열 | `/dev/ttyUSB0` |
| myCobot 280 | GPIO UART 핀 직접 연결 | tty 계열 | `/dev/ttyAMA0` |
| Intel RealSense | Raspberry Pi USB-A 포트 | video/bulk (tty 아님) | lsusb에서 확인 |

- **현재 구성:** myCobot + RealSense 모두 Raspberry Pi USB-A 포트에 연결
- myCobot은 USB 연결이므로 포트는 `/dev/ttyUSB0` (udev symlink → `/dev/ttyJETCOBOT`)
- RealSense는 `pyrealsense2`가 직접 USB 접근 → 경로 지정 불필요

---

## Phase 2 — Raspberry Pi 연결 확인

### 장치 확인

```bash
ls /dev/tty*
lsusb
dmesg | tail -20
```

### 기존 udev rule 확인

```bash
ls /etc/udev/rules.d/
```

**있으면 — 내용 확인**
```bash
cat /etc/udev/rules.d/99-jetcobot.rules

# symlink 실제로 걸려있는지도 확인
ls -l /dev/ttyJETCOBOT
```

**없으면 — rule 작성**
```bash
udevadm info -a -n /dev/ttyUSB0 | grep -E "idVendor|idProduct"
sudo nano /etc/udev/rules.d/99-mycobot.rules
# SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", SYMLINK+="ttyJETCOBOT"

sudo udevadm control --reload-rules && sudo udevadm trigger
ls -l /dev/ttyJETCOBOT
```

### 연결 구성

```
myCobot  → USB → /dev/ttyUSB0 → udev symlink → /dev/ttyJETCOBOT → pymycobot
RealSense → USB → pyrealsense2 자동 인식 (경로 지정 불필요)
```

### pymycobot + RealSense 설치

```bash
pip install pymycobot pyrealsense2
```

### myCobot 연결 테스트

```bash
python -c "
from pymycobot.mycobot280 import MyCobot280
mc = MyCobot280('/dev/ttyJETCOBOT', 1000000)
mc.thread_lock = True
print('관절값:', mc.get_angles())
"
```

### RealSense 연결 테스트

```bash
python -c "
import pyrealsense2 as rs, numpy as np
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
pipeline.start(cfg)
img = np.asanyarray(pipeline.wait_for_frames().get_color_frame().get_data())
print('frame shape:', img.shape)  # (480, 640, 3) 나와야 함
pipeline.stop()
"
```

### 관절 움직임 테스트

```bash
python -c "
from pymycobot.mycobot280 import MyCobot280; import time
mc = MyCobot280('/dev/ttyJETCOBOT', 1000000)
mc.send_angles([0, 0, 0, 0, 0, 0], 30)
time.sleep(3)
print('현재 각도:', mc.get_angles())
"
```

---

## 모델별 필요 데이터

| 모델 계열 | 이미지 | 관절값 | 언어 명령 | 비고 |
|-----------|--------|--------|-----------|------|
| **ACT** | ✅ RGB (top-down) | ✅ 6-DOF 관절 각도 | ❌ | 현재 프로젝트 선택 |
| **Diffusion Policy** | ✅ RGB | ✅ 관절 각도 | ❌ |  |
| **π0 / SmolVLA / xVLA** | ✅ RGB | ✅ 관절 각도 | ✅ 텍스트 지시문 필요 | "pick up the red block" 같은 언어 라벨 추가 수집 필요 |

> ACT와 Diffusion Policy는 **이미지 + 관절값**만으로 학습 가능.  
> π0·SmolVLA 계열(VLA)은 "Vision-Language-Action" — 에피소드마다 텍스트 지시문도 수집해야 함.

---

## 스크립트 실행 시 출력

### `phase3_collect.py` (Raspberry Pi)

```
~/data/pickplace/
├── dataset_meta.json          ← 전체 수집 설정 (robot, hz, shape 등)
├── episode_000000/
│   ├── images.npy             ← (T, 480, 640, 3) uint8  — RealSense 프레임
│   ├── joint_states.npy       ← (T, 6) float32          — 현재 관절 각도 (deg)
│   ├── actions.npy            ← (T, 6) float32          — 1-step ahead 관절 각도
│   ├── timestamps.npy         ← (T,)   float64          — Unix 타임스탬프
│   └── meta.json              ← episode별 프레임 수, 수집 시간
├── episode_000001/
└── ...
```

### `phase3_convert.py` (서버)

```
data/mycobot_lerobot/
├── meta_data/
│   ├── info.json              ← 데이터셋 스펙 (fps, shape, feature 정의)
│   ├── stats.json             ← 정규화용 평균/표준편차 (학습 시 자동 사용)
│   └── episodes.jsonl         ← episode별 프레임 수 목록
├── data/
│   └── train-00000-of-00001.parquet   ← 전체 프레임 테이블 (관절값 + 액션)
└── videos/
    └── observation.images.top/
        ├── episode_000000.mp4 ← LeRobot이 학습 시 프레임 단위로 읽는 영상
        └── ...
```

### Phase 4 학습 (`lerobot_train`)

```
outputs/act_mycobot_YYYYMMDD_HHMM/
├── checkpoints/
│   ├── last/                  ← 마지막 epoch 체크포인트
│   └── best/                  ← validation loss 최저 체크포인트
└── train.log                  ← 학습 로그 (wandb에도 동시 전송)
```

### Phase 5 평가 (`phase5_compare.py`)

```
results/
└── final_report_YYYYMMDD.txt  ← 성공률, 평균 소요 시간 등
```

---

## Phase 3 — 데이터 수집

### scripts/ 를 Raspberry Pi로 전송

```bash
scp scripts/phase3_collect.py pi@<raspberry-pi-ip>:~/
```

### 데이터 수집 (Raspberry Pi)

```bash
python phase3_collect.py --n-episodes 50 --output-dir ~/data/pickplace
```

- `Enter` → 모터 OFF + 기록 시작
- 손으로 pick & place 시연
- `Enter` → 기록 종료 + 저장

### 서버로 전송

```bash
scp -r ~/data/pickplace team2@100.66.177.119:~/dev_ws/daye_vla/data/
```

### LeRobot 포맷 변환 (서버)

```bash
source ~/venv/vision/bin/activate
cd ~/dev_ws/daye_vla
pip install datasets imageio[ffmpeg] pyarrow

python scripts/phase3_convert.py \
    --input-dir  data/pickplace \
    --output-dir data/mycobot_lerobot
```

---

## Phase 4 — Fine-tuning (서버)

```bash
source ~/venv/vision/bin/activate
cd ~/dev_ws/daye_vla

python -m lerobot.scripts.lerobot_train \
  --policy.type=act \
  --dataset.repo_id=data/mycobot_lerobot \
  --output_dir=outputs/act_mycobot_$(date +%Y%m%d_%H%M) \
  --training.num_epochs=100 \
  --wandb.enable=true \
  --wandb.project=vla_mycobot
```

---

## Phase 5 — 실제 로봇 평가 (서버)

```bash
python scripts/phase5_compare.py \
  --checkpoint outputs/act_mycobot_<날짜>/checkpoints/last \
  --n-episodes 20 \
  --robot-port /dev/ttyJETCOBOT
```
