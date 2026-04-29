"""
카메라 실시간 뷰어 — Ubuntu에서 실행

Raspberry Pi의 camera_server.py에 연결해서
top (RealSense) + wrist (USB 웹캠) 프레임을 실시간으로 표시.

사전 설치:
    pip install opencv-python numpy

실행:
    python camera_viewer.py --host <라즈베리파이 IP>
    python camera_viewer.py --host 100.xx.xx.xx --port 5001  # 포트 변경 시

조작:
    q     — 종료
    s     — 현재 프레임 스크린샷 저장 (viewer_top_NNNN.png / viewer_wrist_NNNN.png)
    space — 일시정지 / 재개
"""

import argparse
import pickle
import socket
import struct
import time
from pathlib import Path

import cv2
import numpy as np


# ── 소켓 유틸 ─────────────────────────────────────────────

def recv_exactly(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("서버 연결이 끊겼습니다.")
        buf += chunk
    return buf

def recv_msg(sock):
    raw_len = recv_exactly(sock, 4)
    msg_len = struct.unpack('>I', raw_len)[0]
    payload = recv_exactly(sock, msg_len)
    return pickle.loads(payload)


# ── 뷰어 ─────────────────────────────────────────────────

def run_viewer(host, port):
    print(f"연결 중... {host}:{port}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((host, port))
        print("연결됨! (q: 종료 / s: 스크린샷 / space: 일시정지)")

        paused    = False
        save_idx  = 0
        last_top  = None
        last_wrist = None
        fps_t     = time.time()
        frame_cnt = 0

        while True:
            if not paused:
                sock.sendall(b'GRAB')
                data = recv_msg(sock)

                top   = data['top']    # (H, W, 3) RGB
                wrist = data['wrist']  # (H, W, 3) RGB
                ts    = data['ts']

                # RGB → BGR for OpenCV display
                top_bgr   = cv2.cvtColor(top,   cv2.COLOR_RGB2BGR)
                wrist_bgr = cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR)

                # FPS 계산
                frame_cnt += 1
                elapsed = time.time() - fps_t
                if elapsed >= 1.0:
                    fps = frame_cnt / elapsed
                    frame_cnt = 0
                    fps_t = time.time()
                else:
                    fps = frame_cnt / max(elapsed, 1e-6)

                # 오버레이: FPS + 타임스탬프
                label = f"FPS: {fps:.1f}  ts: {ts:.3f}"
                cv2.putText(top_bgr,   label, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(wrist_bgr, "wrist", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                last_top   = top_bgr
                last_wrist = wrist_bgr

                cv2.imshow("top (RealSense)", top_bgr)
                cv2.imshow("wrist (USB cam)", wrist_bgr)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("종료")
                sock.sendall(b'STOP')
                break

            elif key == ord('s') and last_top is not None:
                Path("viewer_screenshots").mkdir(exist_ok=True)
                top_path   = f"viewer_screenshots/top_{save_idx:04d}.png"
                wrist_path = f"viewer_screenshots/wrist_{save_idx:04d}.png"
                cv2.imwrite(top_path,   last_top)
                cv2.imwrite(wrist_path, last_wrist)
                print(f"저장: {top_path}, {wrist_path}")
                save_idx += 1

            elif key == ord(' '):
                paused = not paused
                status = "일시정지" if paused else "재개"
                print(f"{status}")

    cv2.destroyAllWindows()


# ── 메인 ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, required=True,
                        help="Raspberry Pi IP 주소 (예: 100.66.x.x)")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    run_viewer(args.host, args.port)


if __name__ == "__main__":
    main()
