import json
import time
import threading
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

import cv2
import pyrealsense2 as rs


# ===== 설정 =====
REC_HZ        = 10
IMG_W         = 424
IMG_H         = 240
WRIST_CAM_IDX = 0


# ===== 전역 버퍼 =====
latest_top = None
latest_wrist = None
lock = threading.Lock()


# ===== RealSense =====
def setup_realsense():
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, IMG_W, IMG_H, rs.format.rgb8, 15)
    pipeline.start(cfg)
    time.sleep(1)
    return pipeline


def realsense_loop(pipeline):
    global latest_top
    while True:
        try:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            color = frames.get_color_frame()
            if not color:
                continue

            img = np.asanyarray(color.get_data())

            with lock:
                latest_top = img

        except Exception as e:
            continue


# ===== Wrist Cam =====
def setup_wrist_camera(idx):
    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_H)

    if not cap.isOpened():
        raise RuntimeError("웹캠 열기 실패")

    time.sleep(1)
    return cap


def wrist_loop(cap):
    global latest_wrist
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with lock:
            latest_wrist = img


# ===== 녹화 =====
def record_episode(ep_idx):
    global latest_top, latest_wrist

    print(f"\n{'='*50}")
    print(f"Episode {ep_idx:03d}")
    print("카메라 확인 후 Enter → 녹화 시작")
    print("녹화 중 Enter → 종료")
    print(f"{'='*50}")

    input("준비되면 Enter...")

    top_images = []
    wrist_images = []
    timestamps = []

    stop_flag = threading.Event()

    def record_loop():
        while not stop_flag.is_set():
            t = time.time()

            with lock:
                if latest_top is None or latest_wrist is None:
                    continue

                top_images.append(latest_top.copy())
                wrist_images.append(latest_wrist.copy())

            timestamps.append(t)
            time.sleep(1.0 / REC_HZ)

    t = threading.Thread(target=record_loop)
    t.start()

    input()  # 종료 대기
    stop_flag.set()
    t.join()

    print(f"수집 완료: {len(timestamps)} frames")

    return {
        "episode_idx": ep_idx,
        "timestamps": np.array(timestamps, dtype=np.float64),
        "top_images": np.array(top_images, dtype=np.uint8),
        "wrist_images": np.array(wrist_images, dtype=np.uint8),
    }


# ===== 저장 =====
def save_episode(ep, output_dir):
    ep_dir = Path(output_dir) / f"episode_{ep['episode_idx']:06d}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    np.save(ep_dir / "images_top.npy", ep["top_images"])
    np.save(ep_dir / "images_wrist.npy", ep["wrist_images"])
    np.save(ep_dir / "timestamps.npy", ep["timestamps"])

    print(f"저장 완료: {ep_dir}")


# ===== main =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="~/data/camera_only")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("RealSense 초기화 중...")
    pipeline = setup_realsense()
    print("RealSense OK")

    print("웹캠 초기화 중...")
    cap = setup_wrist_camera(WRIST_CAM_IDX)
    print("웹캠 OK")

    # ===== 카메라 스레드 시작 =====
    threading.Thread(target=realsense_loop, args=(pipeline,), daemon=True).start()
    threading.Thread(target=wrist_loop, args=(cap,), daemon=True).start()

    time.sleep(2)  # 버퍼 안정화

    collected = []

    for ep_idx in range(args.n_episodes):
        ep = record_episode(ep_idx)
        save_episode(ep, output_dir)
        collected.append(ep_idx)

        if ep_idx < args.n_episodes - 1:
            ans = input("다음 녹화? (Enter / q 종료): ").strip().lower()
            if ans == 'q':
                break

    pipeline.stop()
    cap.release()

    print("\n수집 완료")


if __name__ == "__main__":
    main()