# VLA — myCobot 280 Pick & Place

목표: myCobot 280 + RealSense로 pick & place 시연 데이터 50개 수집 → 서버(RTX 2080 Ti)에서 ACT policy fine-tuning → 실제 로봇 평가

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
