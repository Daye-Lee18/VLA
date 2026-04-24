"""
Phase 3 — myCobot 280 pick & place 시연 데이터 수집 (Kinesthetic Teaching)

실행 환경: Raspberry Pi 5 (JetCobot)
실행 방법: python phase3_collect.py --n-episodes 50 --output-dir ~/data/pickplace

사전 설치:
    pip install pymycobot pyrealsense2 numpy

데이터 수집 방식 (Kinesthetic Teaching):
    1. Enter → 모터 OFF (팔이 자유롭게 움직임)
    2. 손으로 팔을 움직여 pick & place 시연
    3. Enter → 기록 종료 + 모터 ON

서버로 전송:
    scp -r ~/data/pickplace team2@100.66.177.119:~/dev_ws/daye_vla/data/
"""

import os
import sys
import json
import time
import threading
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

from pymycobot.mycobot280 import MyCobot280
import pyrealsense2 as rs

PORT     = '/dev/ttyJETCOBOT'
BAUD     = 1000000
REC_HZ   = 10   # 기록 주기 (Hz) — pick & place는 10Hz로 충분
IMG_W    = 640
IMG_H    = 480


# ── 카메라 ────────────────────────────────────────────────

def setup_realsense():
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, IMG_W, IMG_H, rs.format.rgb8, 30)
    pipeline.start(cfg)
    time.sleep(1)  # 워밍업
    return pipeline

def get_frame(pipeline):
    frames = pipeline.wait_for_frames()
    color = frames.get_color_frame()
    return np.asanyarray(color.get_data())  # (H, W, 3) uint8


# ── 데이터 수집 ───────────────────────────────────────────

def record_episode(mc, pipeline, ep_idx):
    print(f"\n{'='*50}")
    print(f"  Episode {ep_idx:03d}")
    print("  준비: 팔을 시작 위치(홈)로 놓은 뒤 Enter")
    print("  시연: Enter 누르면 모터 OFF → 손으로 움직이기 시작")
    print("  종료: 태스크 완료 후 Enter")
    print(f"{'='*50}")
    input("  준비됐으면 Enter... ")

    mc.release_all_servos()
    print("  ▶ 기록 중... (종료: Enter)")

    images      = []
    joint_states = []
    timestamps  = []
    stop_flag   = threading.Event()

    def record_loop():
        interval = 1.0 / REC_HZ
        while not stop_flag.is_set():
            t0 = time.time()
            img    = get_frame(pipeline)
            angles = mc.get_angles()   # [j1..j6] degrees
            timestamps.append(t0)
            images.append(img)
            joint_states.append(angles if angles else [0]*6)
            time.sleep(max(0, interval - (time.time() - t0)))

    t = threading.Thread(target=record_loop, daemon=True)
    t.start()
    input()          # Enter 대기
    stop_flag.set()
    t.join()

    mc.focus_all_servos()
    n = len(timestamps)
    print(f"  ✅ {n} frames ({n / REC_HZ:.1f}초) 수집 완료")

    return {
        "episode_idx"  : ep_idx,
        "timestamps"   : np.array(timestamps,               dtype=np.float64),
        "images"       : np.array(images,                   dtype=np.uint8),    # (T, H, W, 3)
        "joint_states" : np.array(joint_states,             dtype=np.float32),  # (T, 6)
    }


# ── 저장 ─────────────────────────────────────────────────

def save_episode(ep, output_dir):
    ep_dir = Path(output_dir) / f"episode_{ep['episode_idx']:06d}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    states  = ep["joint_states"]
    # action = 다음 타임스텝의 관절값 (Behavior Cloning 표준)
    actions = np.concatenate([states[1:], states[-1:]], axis=0)  # (T, 6)

    np.save(ep_dir / "images.npy",       ep["images"])
    np.save(ep_dir / "joint_states.npy", states)
    np.save(ep_dir / "actions.npy",      actions)
    np.save(ep_dir / "timestamps.npy",   ep["timestamps"])

    # episode 메타
    meta = {
        "episode_idx" : ep["episode_idx"],
        "n_frames"    : len(states),
        "duration_s"  : float(ep["timestamps"][-1] - ep["timestamps"][0]),
        "record_hz"   : REC_HZ,
    }
    with open(ep_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  저장: {ep_dir}")
    return str(ep_dir)


# ── 메인 ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes",  type=int, default=50)
    parser.add_argument("--output-dir",  type=str, default="~/data/pickplace")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── myCobot 연결
    print("myCobot 연결 중...")
    mc = MyCobot280(PORT, BAUD)
    mc.thread_lock = True
    time.sleep(1)
    angles = mc.get_angles()
    print(f"  현재 관절값: {angles}")

    # ── RealSense 연결
    print("RealSense 초기화 중...")
    pipeline = setup_realsense()
    img_test = get_frame(pipeline)
    print(f"  카메라 OK — frame shape: {img_test.shape}")

    # ── 홈 포지션
    print("\n홈 포지션으로 이동 중...")
    mc.send_angles([0, 0, 0, 0, 0, 0], 30)
    time.sleep(3)

    # ── 수집 루프
    collected = []
    for ep_idx in range(args.n_episodes):
        ep_data  = record_episode(mc, pipeline, ep_idx)
        ep_path  = save_episode(ep_data, output_dir)
        collected.append(ep_path)

        print(f"  진행: {ep_idx+1}/{args.n_episodes}")
        if ep_idx < args.n_episodes - 1:
            ans = input("  다음 episode? (Enter=계속 / q=종료): ").strip().lower()
            if ans == 'q':
                break

        # episode 사이 홈 복귀
        mc.send_angles([0, 0, 0, 0, 0, 0], 30)
        time.sleep(2)

    pipeline.stop()

    # ── 전체 메타데이터
    dataset_meta = {
        "task"       : "pick_and_place",
        "robot"      : "mycobot280_arduino",
        "camera"     : "realsense_top",
        "n_episodes" : len(collected),
        "record_hz"  : REC_HZ,
        "img_shape"  : [IMG_H, IMG_W, 3],
        "state_dim"  : 6,
        "action_dim" : 6,
        "timestamp"  : datetime.now().isoformat(),
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
