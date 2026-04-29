"""
에피소드 .npy 파일 확인 스크립트

수집된 데이터를 비디오로 재생하고 관절값 그래프를 출력.

사전 설치:
    pip install numpy opencv-python matplotlib

실행:
    # 특정 에피소드 확인
    python check_episode.py --episode-dir ~/data/pickplace/episode_000000

    # 전체 데이터셋 순서대로 확인
    python check_episode.py --dataset-dir ~/data/pickplace

    # 저장 없이 빠르게 훑기
    python check_episode.py --dataset-dir ~/data/pickplace --no-plot

키 조작 (비디오 재생 중):
    Space  — 일시정지 / 재개
    →      — 다음 프레임
    ←      — 이전 프레임
    n      — 다음 에피소드
    q      — 종료
    s      — 현재 프레임 PNG 저장
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


# ── 로드 ─────────────────────────────────────────────────

def load_episode(ep_dir: Path):
    ep_dir = Path(ep_dir)
    data = {}

    for key, fname in [
        ("top",    "images_top.npy"),
        ("wrist",  "images_wrist.npy"),
        ("joints", "joint_states.npy"),
        ("actions","actions.npy"),
        ("ts",     "timestamps.npy"),
    ]:
        path = ep_dir / fname
        data[key] = np.load(path) if path.exists() else None

    meta_path = ep_dir / "meta.json"
    data["meta"] = json.load(open(meta_path)) if meta_path.exists() else {}

    return data


# ── 관절값 그래프 ─────────────────────────────────────────

def plot_joints(data, ep_dir):
    joints  = data["joints"]
    actions = data["actions"]
    ts      = data["ts"]
    if joints is None:
        print("  joint_states.npy 없음 — 그래프 스킵")
        return

    rel_ts = ts - ts[0] if ts is not None else np.arange(len(joints))
    fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"{ep_dir.name}  ({len(joints)} frames)", fontsize=13)

    for i, ax in enumerate(axes):
        ax.plot(rel_ts, joints[:, i],  label=f"joint_{i+1} state",  linewidth=1.5)
        if actions is not None:
            ax.plot(rel_ts, actions[:, i], label=f"joint_{i+1} action",
                    linewidth=1, linestyle='--', alpha=0.7)
        ax.set_ylabel(f"J{i+1} (°)", fontsize=8)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("time (s)")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


# ── 비디오 재생 ───────────────────────────────────────────

def play_episode(data, ep_dir):
    top   = data["top"]
    wrist = data["wrist"]
    ts    = data["ts"]
    meta  = data["meta"]

    if top is None:
        print("  images_top.npy 없음")
        return "next"

    n_frames = len(top)
    fps      = meta.get("record_hz", 10)
    duration = meta.get("duration_s", n_frames / fps)

    print(f"\n  에피소드: {ep_dir.name}")
    print(f"  프레임수: {n_frames}  |  FPS: {fps}  |  길이: {duration:.1f}초")
    print(f"  이미지  : {top.shape}  dtype={top.dtype}")
    if wrist is not None:
        print(f"  wrist   : {wrist.shape}")
    print("  Space=일시정지  →/←=프레임이동  n=다음  q=종료  s=저장")

    frame_idx = 0
    paused    = False
    delay     = max(1, int(1000 / fps))

    while True:
        top_bgr = cv2.cvtColor(top[frame_idx], cv2.COLOR_RGB2BGR)

        if wrist is not None:
            wrist_bgr = cv2.cvtColor(wrist[frame_idx], cv2.COLOR_RGB2BGR)
            # top과 wrist 높이 맞춰 좌우로 붙이기
            if top_bgr.shape[0] != wrist_bgr.shape[0]:
                wrist_bgr = cv2.resize(wrist_bgr,
                    (int(wrist_bgr.shape[1] * top_bgr.shape[0] / wrist_bgr.shape[0]),
                     top_bgr.shape[0]))
            frame_show = np.hstack([top_bgr, wrist_bgr])
            label_top   = f"TOP   {top_bgr.shape[1]}x{top_bgr.shape[0]}"
            label_wrist = f"WRIST {wrist_bgr.shape[1]}x{wrist_bgr.shape[0]}"
            cv2.putText(frame_show, label_top,   (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.putText(frame_show, label_wrist, (top_bgr.shape[1]+10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        else:
            frame_show = top_bgr

        # 프레임 번호 / 시간 오버레이
        t_str = f"{ts[frame_idx] - ts[0]:.2f}s" if ts is not None else ""
        info  = f"[{frame_idx+1}/{n_frames}] {t_str}  {'PAUSE' if paused else 'PLAY'}"
        cv2.putText(frame_show, info, (10, frame_show.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

        cv2.imshow("Episode Viewer", frame_show)
        key = cv2.waitKey(0 if paused else delay) & 0xFF

        if key == ord('q'):
            cv2.destroyAllWindows()
            return "quit"
        elif key == ord('n'):
            cv2.destroyAllWindows()
            return "next"
        elif key == ord('s'):
            save_path = ep_dir / f"frame_{frame_idx:04d}.png"
            cv2.imwrite(str(save_path), frame_show)
            print(f"  저장: {save_path}")
        elif key == ord(' '):
            paused = not paused
        elif key == 83 or key == ord('d'):   # →
            frame_idx = min(frame_idx + 1, n_frames - 1)
        elif key == 81 or key == ord('a'):   # ←
            frame_idx = max(frame_idx - 1, 0)
        elif not paused:
            frame_idx += 1
            if frame_idx >= n_frames:
                frame_idx = 0   # 루프

    cv2.destroyAllWindows()
    return "next"


# ── 요약 출력 ─────────────────────────────────────────────

def print_dataset_summary(ep_dirs):
    print(f"\n{'='*50}")
    print(f"데이터셋 요약 — 총 {len(ep_dirs)} episodes")
    print(f"{'='*50}")
    total_frames = 0
    for ep_dir in ep_dirs:
        meta_path = ep_dir / "meta.json"
        if meta_path.exists():
            meta = json.load(open(meta_path))
            n    = meta.get("n_frames", "?")
            dur  = meta.get("duration_s", "?")
            hz   = meta.get("record_hz", "?")
            total_frames += n if isinstance(n, int) else 0
            print(f"  {ep_dir.name}: {n} frames  {dur:.1f}s  @ {hz}Hz")
        else:
            top_path = ep_dir / "images_top.npy"
            n = len(np.load(top_path)) if top_path.exists() else "?"
            total_frames += n if isinstance(n, int) else 0
            print(f"  {ep_dir.name}: {n} frames")
    print(f"  ─────────────────────────")
    print(f"  합계: {total_frames} frames")
    print(f"{'='*50}\n")


# ── 메인 ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--episode-dir",  type=str,
                       help="단일 에피소드 디렉토리")
    group.add_argument("--dataset-dir",  type=str,
                       help="전체 데이터셋 디렉토리 (episode_* 하위 폴더)")
    parser.add_argument("--no-plot", action="store_true",
                        help="관절값 그래프 출력 안 함 (빠른 확인)")
    parser.add_argument("--start",   type=int, default=0,
                        help="시작 에피소드 번호 (기본 0)")
    args = parser.parse_args()

    if args.episode_dir:
        ep_dirs = [Path(args.episode_dir).expanduser()]
    else:
        base    = Path(args.dataset_dir).expanduser()
        ep_dirs = sorted(base.glob("episode_*"))[args.start:]

    if not ep_dirs:
        print("에피소드 디렉토리를 찾을 수 없습니다.")
        return

    print_dataset_summary(ep_dirs)

    for ep_dir in ep_dirs:
        data = load_episode(ep_dir)

        if not args.no_plot:
            plot_joints(data, ep_dir)

        result = play_episode(data, ep_dir)

        if not args.no_plot:
            plt.close('all')

        if result == "quit":
            break

    print("확인 완료")


if __name__ == "__main__":
    main()
