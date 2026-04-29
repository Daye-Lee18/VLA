# VLA — myCobot 280 Pick & Place

목표: myCobot 280 + RealSense로 pick & place 시연 데이터 50개 수집 → 서버(RTX 2080 Ti)에서 ACT policy fine-tuning → 실제 로봇 평가

---

## 파일 안내

| 파일 | 언제 보는가 |
|------|------------|
| `README.md` | 실제로 실행할 때 — 커맨드 복붙용 치트시트 |
| `vla_roadmap.qmd` | 실험 설계 / 배경 이해할 때 — JD 연결, 전체 파이프라인 흐름, 각 Phase 상세 설명, 진행 로그 |
| `lerobot_models.qmd` | 모델 선택 고민할 때 — 전체 38개 모델 목록, 프로젝트 추천 순위, 모델 교체 근거 |
| `dataset_collection_guideline_ACT.md` | **Phase 3 데이터 수집 직전** — 카메라 세팅, 시연 방법, 주의사항, 품질 체크리스트 |

### scripts/ 안내

| 스크립트 | 실행 환경 | 설명 |
|----------|-----------|------|
| `camera_server.py` | Raspberry Pi | RealSense + wrist 카메라 프레임을 소켓으로 전송 |
| `camera_viewer.py` | Ubuntu | RPi 카메라 서버에 연결해 실시간으로 모니터에 표시 |
| `phase3_collect.py` | Raspberry Pi / 단일 PC | Leader-Follower 방식 데이터 수집 (단일 PC 구성) |
| `phase3_collect_distributed.py` | Ubuntu | 분산 구성 데이터 수집 (카메라는 RPi, 로봇은 Ubuntu) |
| `phase3_convert.py` | 서버 | 수집 데이터 → LeRobot 포맷 변환 |
| `phase4_train.sh` | 서버 | ACT fine-tuning 실행 |
| `phase5_compare.py` | 서버 + 로봇 | 학습된 policy 실제 로봇 평가 |
| `phase2_baseline.py` | Raspberry Pi | Phase 2 기본 연결 확인 |
| `phase2_check_env.py` | Raspberry Pi | 환경 설정 확인 |

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

### `phase3_collect.py` (수집 PC — Mac / Linux / Raspberry Pi)

```
~/data/pickplace/
├── dataset_meta.json          ← 전체 수집 설정 (robot, hz, shape 등)
├── episode_000000/
│   ├── images_top.npy         ← (T, 480, 640, 3) uint8  — RealSense top-down 프레임
│   ├── images_wrist.npy       ← (T, 480, 640, 3) uint8  — wrist 카메라 프레임
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
    ├── observation.images.top/
    │   ├── episode_000000.mp4 ← top-down 카메라 영상
    │   └── ...
    └── observation.images.wrist/
        ├── episode_000000.mp4 ← wrist 카메라 영상
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

> **현재 구성 (분산):**
> - Raspberry Pi: top 카메라(RealSense) + wrist 카메라 → `camera_server.py` 실행
> - Ubuntu: 리더팔 + 팔로워팔 → `phase3_collect_distributed.py` 실행
> - 모든 데이터는 Ubuntu 한 곳에 저장됨

### Step 1 — RPi IP 확인

```bash
# RPi에서 실행
hostname -I
# → 192.168.x.x 형태로 출력됨 (첫 번째 숫자가 IP)
```

### Step 2 — RPi에 카메라 서버 올리기

```bash
# Ubuntu → RPi로 스크립트 전송
scp scripts/camera_server.py pi@<rpi-ip>:~/

# RPi에서 실행
python camera_server.py
# → "대기 중 — 포트 5000 (Ubuntu 연결 기다리는 중...)" 출력 후 대기
```

### Step 3 — (선택) Ubuntu에서 카메라 실시간 모니터링

데이터 수집 중 RealSense + wrist 카메라를 Ubuntu 모니터에서 실시간으로 확인할 수 있음.  
**별도 터미널**에서 Step 4 실행 전에 미리 띄워둘 것.

```bash
# Ubuntu에서 실행 (Step 4와 별도 터미널)
python scripts/camera_viewer.py --host <rpi-ip>
# 포트 변경 시: --port 5001
```

| 키 | 동작 |
|----|------|
| `q` | 종료 |
| `s` | 현재 프레임 스크린샷 저장 (`viewer_screenshots/`) |
| `space` | 일시정지 / 재개 |

> `camera_server.py`가 RPi에서 실행 중이어야 연결 가능.  
> `camera_viewer.py`와 `phase3_collect_distributed.py`는 동시에 실행하지 말 것 — 둘 다 같은 소켓에 연결을 시도해 충돌 발생.  
> **모니터링 목적이면** 뷰어만 단독으로 띄우거나, 수집 스크립트 실행 후 로그로 상태 확인 권장.

---

### Step 4 — Ubuntu에서 수집 실행

```bash
python scripts/phase3_collect_distributed.py \
    --rpi-ip <Step 1에서 확인한 RPi IP> \
    --n-episodes 100 \
    --output-dir ~/data/pickplace
# 로봇팔 포트 미지정 시 자동 감지 후 선택
# → RPi 콘솔에 "연결됨" 출력되면 정상
```

---

### scripts/ 를 수집 PC로 전송 (단일 PC 구성 시)

```bash
# Raspberry Pi로 전송하는 경우
scp scripts/phase3_collect.py pi@<raspberry-pi-ip>:~/

# 노트북에서 직접 실행하는 경우 — 전송 불필요
```

### 데이터 수집

```bash
# 포트 미지정 → 자동 감지 후 입력 요청 (Mac/Linux 모두 동작)
python phase3_collect.py \
    --n-episodes 100 \
    --output-dir ~/data/pickplace

# 포트 미리 아는 경우 직접 지정
# Linux:  --leader-port /dev/ttyUSB0   --follower-port /dev/ttyUSB1
# Mac:    --leader-port /dev/cu.usbserial-XXXX  --follower-port /dev/cu.usbserial-YYYY
```

- `Enter` → 리더 모터 OFF + 기록 시작 (팔로워 자동 미러링)
- 리더팔 손으로 잡고 pick & place 시연 (카메라는 팔로워만 촬영)
- `Enter` → 기록 종료 + 두 팔 홈 복귀

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
