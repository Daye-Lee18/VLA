"""
카메라 서버 — Raspberry Pi에서 실행

Ubuntu의 phase3_collect_distributed.py 가 연결하면
요청마다 top(RealSense) + wrist(USB 웹캠) 프레임을 전송.

사전 설치:
    pip install pyrealsense2 numpy opencv-python

실행:
    python camera_server.py

    # 포트 변경 시:
    python camera_server.py --port 5001

    # wrist 카메라 인덱스 변경 시:
    python camera_server.py --wrist-cam-idx 2
"""

import argparse
import pickle
import socket
import struct
import time

import cv2
import numpy as np
import pyrealsense2 as rs

IMG_W = 640
IMG_H = 480


# ── 카메라 초기화 ─────────────────────────────────────────

def setup_realsense():
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, IMG_W, IMG_H, rs.format.rgb8, 30)
    pipeline.start(cfg)
    time.sleep(1)
    return pipeline

def setup_wrist(cam_idx):
    cap = cv2.VideoCapture(cam_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  IMG_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError(f"wrist 카메라 열기 실패 (index={cam_idx})")
    time.sleep(1)
    return cap

def grab_frames(pipeline, cap):
    top   = np.asanyarray(pipeline.wait_for_frames().get_color_frame().get_data())
    ret, wrist = cap.read()
    if not ret:
        raise RuntimeError("wrist 카메라 프레임 읽기 실패")
    wrist = cv2.cvtColor(wrist, cv2.COLOR_BGR2RGB)
    return top, wrist


# ── 소켓 유틸 ─────────────────────────────────────────────

def send_msg(conn, data):
    payload = pickle.dumps(data, protocol=4)
    conn.sendall(struct.pack('>I', len(payload)) + payload)


# ── 메인 서버 루프 ────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",          type=int, default=5000)
    parser.add_argument("--wrist-cam-idx", type=int, default=0)
    args = parser.parse_args()

    print("카메라 초기화 중...")
    pipeline = setup_realsense()
    cap      = setup_wrist(args.wrist_cam_idx)
    print(f"  top (RealSense) OK")
    print(f"  wrist (cam idx={args.wrist_cam_idx}) OK")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(('0.0.0.0', args.port))
        srv.listen(1)
        print(f"\n대기 중 — 포트 {args.port} (Ubuntu 연결 기다리는 중...)")

        while True:
            conn, addr = srv.accept()
            print(f"연결됨: {addr}")
            try:
                with conn:
                    while True:
                        cmd = conn.recv(4)
                        if not cmd or cmd == b'STOP':
                            print("연결 종료")
                            break
                        if cmd == b'GRAB':
                            top, wrist = grab_frames(pipeline, cap)
                            send_msg(conn, {
                                'top':   top,
                                'wrist': wrist,
                                'ts':    time.time(),
                            })
            except (ConnectionResetError, BrokenPipeError):
                print("연결 끊김 — 다음 연결 대기 중...")

    pipeline.stop()
    cap.release()


if __name__ == '__main__':
    main()
