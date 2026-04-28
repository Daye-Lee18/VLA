"""
Phase 3 — Leader-Follower 데이터 수집

구조:
    리더팔  (/dev/ttyUSB0) — 사람이 손으로 잡고 움직임, 카메라 프레임 밖에 위치
    팔로워팔 (/dev/ttyUSB1) — 리더 각도 실시간 미러링, 카메라가 촬영

카메라:
    top   : RealSense (팔로워 작업공간 top-down 고정)
    wrist : USB 웹캠  (팔로워 wrist에 부착)

실행 방법:
    python phase3_collect.py --n-episodes 100 --output-dir ~/data/pickplace

USB 포트 확인:
    ls /dev/ttyUSB*      # 두 개 보여야 함
    # 어느 쪽이 어느 팔인지 모르면:
    python -c "from pymycobot.mycobot280 import MyCobot280; mc=MyCobot280('/dev/ttyUSB0',1000000); print(mc.get_angles())"
    # 리더팔을 손으로 움직여서 각도가 바뀌는 포트 = 리더

서버로 전송:
    scp -r ~/data/pickplace team2@100.66.177.119:~/dev_ws/daye_vla/data/
"""

import json
import time
import threading
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

import cv2
from pymycobot.mycobot280 import MyCobot280
import pyrealsense2 as rs

LEADER_PORT   = '/dev/ttyUSB0'
FOLLOWER_PORT = '/dev/ttyUSB1'
BAUD          = 1000000
REC_HZ        = 30
IMG_W         = 640
IMG_H         = 480
WRIST_CAM_IDX = 0    # 팔로워 wrist USB 웹캠 인덱스
MIRROR_SPEED  = 80   # 팔로워 이동 속도 (0–100), 너무 낮으면 lag 심해짐


# ── top 카메라 (RealSense) ────────────────────────────────

def setup_realsense():
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, IMG_W, IMG_H, rs.format.rgb8, 30)
    pipeline.start(cfg)
    time.sleep(1)
    return pipeline

def get_top_frame(pipeline):
    return np.asanyarray(
        pipeline.wait_for_frames().get_color_frame().get_data()
    )


# ── wrist 카메라 (팔로워 wrist USB 웹캠) ─────────────────

def setup_wrist_camera(cam_idx):
    cap = cv2.VideoCapture(cam_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  IMG_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError(f"wrist 카메라 열기 실패 (index={cam_idx})")
    time.sleep(1)
    return cap

def get_wrist_frame(cap):
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("wrist 카메라 프레임 읽기 실패")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ── 에피소드 수집 ─────────────────────────────────────────

def record_episode(leader, follower, top_pipeline, wrist_cap, ep_idx):
    print(f"\n{'='*50}")
    print(f"  Episode {ep_idx:03d}")
    print("  준비: 리더팔 + 팔로워팔 모두 홈 포지션 확인 후 Enter")
    print("  시연: Enter 누르면 리더 모터 OFF → 리더팔 손으로 움직이기 시작")
    print("       팔로워팔이 자동으로 미러링됨")
    print("  종료: pick & place 완료 후 Enter")
    print(f"{'='*50}")
    input("  준비됐으면 Enter... ")

    follower.focus_all_servos()
    leader.release_all_servos()    # 리더 모터 OFF → 손으로 자유롭게
    print("  ▶ 기록 중... (종료: Enter)")

    top_images   = []
    wrist_images = []
    joint_states = []
    timestamps   = []
    stop_flag    = threading.Event()

    def record_loop():
        interval = 1.0 / REC_HZ
        while not stop_flag.is_set():
            t0 = time.time()

            # 리더 각도 읽기 → 팔로워 미러링
            angles = leader.get_angles()
            if angles:
                follower.send_angles(angles, MIRROR_SPEED)

            # 팔로워 실제 각도 기록 (관측값)
            follower_angles = follower.get_angles()

            # 카메라 프레임
            top_img   = get_top_frame(top_pipeline)
            wrist_img = get_wrist_frame(wrist_cap)

            timestamps.append(t0)
            top_images.append(top_img)
            wrist_images.append(wrist_img)
            joint_states.append(follower_angles if follower_angles else [0]*6)

            time.sleep(max(0, interval - (time.time() - t0)))

    t = threading.Thread(target=record_loop, daemon=True)
    t.start()
    input()
    stop_flag.set()
    t.join()

    leader.focus_all_servos()      # 리더 모터 복귀
    n = len(timestamps)
    print(f"  ✅ {n} frames ({n / REC_HZ:.1f}초) 수집 완료")

    return {
        "episode_idx" : ep_idx,
        "timestamps"  : np.array(timestamps,    dtype=np.float64),
        "top_images"  : np.array(top_images,    dtype=np.uint8),
        "wrist_images": np.array(wrist_images,  dtype=np.uint8),
        "joint_states": np.array(joint_states,  dtype=np.float32),
    }


# ── 저장 ─────────────────────────────────────────────────

def save_episode(ep, output_dir):
    ep_dir = Path(output_dir) / f"episode_{ep['episode_idx']:06d}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    states  = ep["joint_states"]
    actions = np.concatenate([states[1:], states[-1:]], axis=0)

    np.save(ep_dir / "images_top.npy",   ep["top_images"])
    np.save(ep_dir / "images_wrist.npy", ep["wrist_images"])
    np.save(ep_dir / "joint_states.npy", states)
    np.save(ep_dir / "actions.npy",      actions)
    np.save(ep_dir / "timestamps.npy",   ep["timestamps"])

    meta = {
        "episode_idx": ep["episode_idx"],
        "n_frames"   : len(states),
        "duration_s" : float(ep["timestamps"][-1] - ep["timestamps"][0]),
        "record_hz"  : REC_HZ,
    }
    with open(ep_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  저장: {ep_dir}")
    return str(ep_dir)


# ── 메인 ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes",    type=int, default=100)
    parser.add_argument("--output-dir",    type=str, default="~/data/pickplace")
    parser.add_argument("--leader-port",   type=str, default=LEADER_PORT)
    parser.add_argument("--follower-port", type=str, default=FOLLOWER_PORT)
    parser.add_argument("--wrist-cam-idx", type=int, default=WRIST_CAM_IDX)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"리더팔 연결 중... ({args.leader_port})")
    leader = MyCobot280(args.leader_port, BAUD)
    leader.thread_lock = True
    time.sleep(1)
    print(f"  리더 현재 관절값: {leader.get_angles()}")

    print(f"팔로워팔 연결 중... ({args.follower_port})")
    follower = MyCobot280(args.follower_port, BAUD)
    follower.thread_lock = True
    time.sleep(1)
    print(f"  팔로워 현재 관절값: {follower.get_angles()}")

    print("RealSense (top) 초기화 중...")
    top_pipeline = setup_realsense()
    print(f"  top 카메라 OK — {get_top_frame(top_pipeline).shape}")

    print(f"wrist 카메라 초기화 중 (index={args.wrist_cam_idx})...")
    wrist_cap = setup_wrist_camera(args.wrist_cam_idx)
    print(f"  wrist 카메라 OK — {get_wrist_frame(wrist_cap).shape}")

    print("\n두 팔 홈 포지션으로 이동 중...")
    leader.send_angles([0, 0, 0, 0, 0, 0], 30)
    follower.send_angles([0, 0, 0, 0, 0, 0], 30)
    time.sleep(3)

    collected = []
    for ep_idx in range(args.n_episodes):
        ep_data = record_episode(leader, follower, top_pipeline, wrist_cap, ep_idx)
        ep_path = save_episode(ep_data, output_dir)
        collected.append(ep_path)

        print(f"  진행: {ep_idx+1}/{args.n_episodes}")
        if ep_idx < args.n_episodes - 1:
            ans = input("  다음 episode? (Enter=계속 / q=종료): ").strip().lower()
            if ans == 'q':
                break

        leader.send_angles([0, 0, 0, 0, 0, 0], 30)
        follower.send_angles([0, 0, 0, 0, 0, 0], 30)
        time.sleep(2)

    top_pipeline.stop()
    wrist_cap.release()

    dataset_meta = {
        "task"          : "pick_and_place",
        "robot"         : "mycobot280_arduino",
        "collection"    : "leader_follower",
        "cameras"       : ["realsense_top", "usb_wrist"],
        "n_episodes"    : len(collected),
        "record_hz"     : REC_HZ,
        "img_shape"     : [IMG_H, IMG_W, 3],
        "state_dim"     : 6,
        "action_dim"    : 6,
        "mirror_speed"  : MIRROR_SPEED,
        "timestamp"     : datetime.now().isoformat(),
    }
    with open(output_dir / "dataset_meta.json", "w") as f:
        json.dump(dataset_meta, f, indent=2)

    print(f"\n{'='*50}")
    print(f"수집 완료: {len(collected)} episodes")
    print(f"저장 경로: {output_dir}")
    print(f"\n서버로 전송:")
    print(f"  scp -r {output_dir} team2@100.66.177.119:~/dev_ws/daye_vla/data/")
    print('='*50)


if __name__ == "__main__":
    main()
