"""
Phase 3 — Leader-Follower 분산 데이터 수집 (Ubuntu 실행)

구성:
    Raspberry Pi  — camera_server.py 실행 (top + wrist 카메라 서빙)
    Ubuntu (여기) — 로봇팔 2대 연결 + RPi에서 카메라 프레임 수신 → 저장

실행 순서:
    1. [RPi]    python camera_server.py
    2. [Ubuntu] python phase3_collect_distributed.py --rpi-ip <RPi IP>

RPi IP 확인 (RPi에서):
    hostname -I

포트 직접 지정 시:
    python phase3_collect_distributed.py --rpi-ip 192.168.1.10 --rpi-port 5000

로봇팔 포트 미지정 시 자동 감지:
    python phase3_collect_distributed.py --rpi-ip 192.168.1.10

서버로 전송:
    scp -r ~/data/pickplace team2@100.66.177.119:~/dev_ws/daye_vla/data/
"""

import argparse
import json
import pickle
import socket
import struct
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import serial.tools.list_ports
from pymycobot.mycobot280 import MyCobot280

BAUD         = 1000000
REC_HZ       = 30
IMG_W        = 640
IMG_H        = 480
MIRROR_SPEED = 80


# ── RPi 카메라 클라이언트 ─────────────────────────────────

class CameraClient:
    def __init__(self, host, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        print(f"  RPi 카메라 서버 연결됨: {host}:{port}")

    def grab(self):
        self.sock.sendall(b'GRAB')
        raw_len = self._recvall(4)
        msg_len = struct.unpack('>I', raw_len)[0]
        data    = pickle.loads(self._recvall(msg_len))
        return data['top'], data['wrist'], data['ts']

    def _recvall(self, n):
        buf = b''
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise RuntimeError("RPi 소켓 연결 끊김")
            buf += chunk
        return buf

    def close(self):
        try:
            self.sock.sendall(b'STOP')
        except Exception:
            pass
        self.sock.close()


# ── 포트 자동 감지 ────────────────────────────────────────

def resolve_port(label, arg_value):
    if arg_value:
        return arg_value
    ports = sorted(p.device for p in serial.tools.list_ports.comports())
    if not ports:
        raise RuntimeError("연결된 시리얼 포트가 없습니다.")
    print(f"\n감지된 시리얼 포트:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p}")
    val = input(f"{label} 포트 번호 또는 경로 입력: ").strip()
    return ports[int(val)] if val.isdigit() else val


# ── 에피소드 수집 ─────────────────────────────────────────

def record_episode(leader, follower, cam_client, ep_idx):
    print(f"\n{'='*50}")
    print(f"  Episode {ep_idx:03d}")
    print("  준비: 두 팔 홈 포지션 확인 후 Enter")
    print("  시연: Enter → 리더 모터 OFF → 리더팔 손으로 조종")
    print("        팔로워가 미러링 + RPi 카메라 자동 기록")
    print("  종료: pick & place 완료 후 Enter")
    print(f"{'='*50}")
    input("  준비됐으면 Enter... ")

    follower.focus_all_servos()
    leader.release_all_servos()
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

            # 카메라 프레임 (RPi에서 수신)
            top_img, wrist_img, cam_ts = cam_client.grab()

            # 리더 → 팔로워 미러링
            angles = leader.get_angles()
            if angles:
                follower.send_angles(angles, MIRROR_SPEED)

            # 팔로워 실제 관절값 기록
            follower_angles = follower.get_angles()

            timestamps.append(cam_ts)          # 카메라 타임스탬프 기준
            top_images.append(top_img)
            wrist_images.append(wrist_img)
            joint_states.append(follower_angles if follower_angles else [0]*6)

            time.sleep(max(0, interval - (time.time() - t0)))

    t = threading.Thread(target=record_loop, daemon=True)
    t.start()
    input()
    stop_flag.set()
    t.join()

    leader.focus_all_servos()
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
    parser.add_argument("--rpi-ip",        type=str, required=True,
                        help="Raspberry Pi IP 주소 (예: 192.168.1.10)")
    parser.add_argument("--rpi-port",      type=int, default=5000)
    parser.add_argument("--n-episodes",    type=int, default=100)
    parser.add_argument("--output-dir",    type=str, default="~/data/pickplace")
    parser.add_argument("--leader-port",   type=str, default=None,
                        help="리더팔 시리얼 포트 (미지정 시 자동 감지)")
    parser.add_argument("--follower-port", type=str, default=None,
                        help="팔로워팔 시리얼 포트 (미지정 시 자동 감지)")
    parser.add_argument("--mirror-speed",  type=int, default=MIRROR_SPEED)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # RPi 카메라 서버 연결
    print(f"RPi 카메라 서버 연결 중... ({args.rpi_ip}:{args.rpi_port})")
    cam_client = CameraClient(args.rpi_ip, args.rpi_port)

    # 로봇팔 연결
    leader_port   = resolve_port("리더팔",   args.leader_port)
    follower_port = resolve_port("팔로워팔", args.follower_port)

    print(f"\n리더팔 연결 중... ({leader_port})")
    leader = MyCobot280(leader_port, BAUD)
    leader.thread_lock = True
    time.sleep(1)
    print(f"  리더 현재 관절값: {leader.get_angles()}")

    print(f"팔로워팔 연결 중... ({follower_port})")
    follower = MyCobot280(follower_port, BAUD)
    follower.thread_lock = True
    time.sleep(1)
    print(f"  팔로워 현재 관절값: {follower.get_angles()}")

    # 카메라 테스트
    print("\n카메라 프레임 테스트...")
    top_test, wrist_test, _ = cam_client.grab()
    print(f"  top   frame: {top_test.shape}")
    print(f"  wrist frame: {wrist_test.shape}")

    # 홈 포지션
    print("\n두 팔 홈 포지션으로 이동 중...")
    leader.send_angles([0, 0, 0, 0, 0, 0], 30)
    follower.send_angles([0, 0, 0, 0, 0, 0], 30)
    time.sleep(3)

    collected = []
    for ep_idx in range(args.n_episodes):
        ep_data = record_episode(leader, follower, cam_client, ep_idx)
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

    cam_client.close()

    dataset_meta = {
        "task"         : "pick_and_place",
        "robot"        : "mycobot280_arduino",
        "collection"   : "leader_follower_distributed",
        "cameras"      : ["realsense_top", "usb_wrist"],
        "camera_host"  : f"{args.rpi_ip}:{args.rpi_port}",
        "n_episodes"   : len(collected),
        "record_hz"    : REC_HZ,
        "img_shape"    : [IMG_H, IMG_W, 3],
        "state_dim"    : 6,
        "action_dim"   : 6,
        "mirror_speed" : args.mirror_speed,
        "timestamp"    : datetime.now().isoformat(),
    }
    with open(output_dir / "dataset_meta.json", "w") as f:
        json.dump(dataset_meta, f, indent=2)

    print(f"\n{'='*50}")
    print(f"수집 완료: {len(collected)} episodes → {output_dir}")
    print(f"\n서버로 전송:")
    print(f"  scp -r {output_dir} team2@100.66.177.119:~/dev_ws/daye_vla/data/")
    print('='*50)


if __name__ == '__main__':
    main()
