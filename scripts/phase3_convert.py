"""
Phase 3 Convert — 수집 데이터(numpy) → LeRobot 0.5.1 학습 포맷 변환

실행 환경: 서버 (team2@100.66.177.119)
실행 방법:
    python phase3_convert.py \
        --input-dir  ~/dev_ws/daye_vla/data/pickplace \
        --output-dir ~/dev_ws/daye_vla/data/mycobot_lerobot

사전 설치 (서버):
    pip install datasets imageio[ffmpeg] pyarrow

결과 구조 (LeRobot 0.5.1 포맷):
    mycobot_lerobot/
    ├── meta_data/
    │   ├── info.json        ← 데이터셋 메타
    │   ├── stats.json       ← 정규화용 평균/표준편차
    │   └── episodes.jsonl   ← episode별 frame 수
    ├── data/
    │   └── train-00000-of-00001.parquet
    └── videos/
        ├── observation.images.top/
        │   ├── episode_000000.mp4
        │   └── ...
        └── observation.images.wrist/
            ├── episode_000000.mp4
            └── ...
"""

import json
import argparse
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import imageio.v2 as imageio
from pathlib import Path


FPS         = 30
IMG_H       = 480
IMG_W       = 640
STATE_DIM   = 6
ACTION_DIM  = 6
JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

VIDEO_FEATURE = {
    "dtype": "video",
    "shape": [IMG_H, IMG_W, 3],
    "names": ["height", "width", "channel"],
    "video_info": {
        "video.fps":          FPS,
        "video.codec":        "libx264",
        "video.pix_fmt":      "yuv420p",
        "video.is_depth_map": False,
    },
}


# ── episode 로드 ──────────────────────────────────────────

def load_episode(ep_dir: Path):
    top_images   = np.load(ep_dir / "images_top.npy")   # (T, H, W, 3)
    wrist_images = np.load(ep_dir / "images_wrist.npy") # (T, H, W, 3)
    joint_states = np.load(ep_dir / "joint_states.npy") # (T, 6)
    actions      = np.load(ep_dir / "actions.npy")      # (T, 6)
    timestamps   = np.load(ep_dir / "timestamps.npy")   # (T,)
    return top_images, wrist_images, joint_states, actions, timestamps


# ── 영상 저장 ─────────────────────────────────────────────

def save_video(images, video_path: Path, fps=FPS):
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(video_path), fps=fps, codec="libx264",
                                output_params=["-crf", "23", "-pix_fmt", "yuv420p"])
    for frame in images:
        writer.append_data(frame)
    writer.close()


# ── stats 계산 ────────────────────────────────────────────

def compute_stats(all_states, all_actions):
    states  = np.concatenate(all_states,  axis=0)
    actions = np.concatenate(all_actions, axis=0)

    def stat(arr):
        return {
            "mean": arr.mean(axis=0).tolist(),
            "std":  arr.std(axis=0).clip(1e-6).tolist(),
            "min":  arr.min(axis=0).tolist(),
            "max":  arr.max(axis=0).tolist(),
        }

    image_stat = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
                  "min":  [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}
    return {
        "observation.state":        stat(states),
        "action":                   stat(actions),
        "observation.images.top":   image_stat,
        "observation.images.wrist": image_stat,
    }


# ── 메인 변환 ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",  type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    input_dir  = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    ep_dirs = sorted(input_dir.glob("episode_*"))
    if not ep_dirs:
        raise FileNotFoundError(f"episode_* 디렉토리가 없습니다: {input_dir}")
    print(f"변환 대상: {len(ep_dirs)} episodes → {output_dir}")

    (output_dir / "meta_data").mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    top_video_dir   = output_dir / "videos" / "observation.images.top"
    wrist_video_dir = output_dir / "videos" / "observation.images.wrist"
    top_video_dir.mkdir(parents=True, exist_ok=True)
    wrist_video_dir.mkdir(parents=True, exist_ok=True)

    rows          = []
    episodes_meta = []
    all_states    = []
    all_actions   = []
    global_idx    = 0

    for ep_idx, ep_dir in enumerate(ep_dirs):
        print(f"  [{ep_idx+1}/{len(ep_dirs)}] {ep_dir.name} ...", end=" ")

        top_imgs, wrist_imgs, states, actions, timestamps = load_episode(ep_dir)
        T = len(timestamps)
        rel_ts = (timestamps - timestamps[0]).astype(np.float32)

        save_video(top_imgs,   top_video_dir   / f"episode_{ep_idx:06d}.mp4")
        save_video(wrist_imgs, wrist_video_dir / f"episode_{ep_idx:06d}.mp4")

        for f in range(T):
            rows.append({
                "index":                    global_idx + f,
                "episode_index":            ep_idx,
                "frame_index":              f,
                "timestamp":                float(rel_ts[f]),
                "next.done":                (f == T - 1),
                "observation.state":        states[f].tolist(),
                "action":                   actions[f].tolist(),
            })

        episodes_meta.append({
            "episode_index": ep_idx,
            "tasks":         ["pick_and_place"],
            "length":        T,
        })

        all_states.append(states)
        all_actions.append(actions)
        global_idx += T
        print(f"{T} frames")

    table = pa.Table.from_pylist(rows, schema=pa.schema([
        pa.field("index",             pa.int64()),
        pa.field("episode_index",     pa.int64()),
        pa.field("frame_index",       pa.int64()),
        pa.field("timestamp",         pa.float32()),
        pa.field("next.done",         pa.bool_()),
        pa.field("observation.state", pa.list_(pa.float32())),
        pa.field("action",            pa.list_(pa.float32())),
    ]))
    pq.write_table(table, output_dir / "data" / "train-00000-of-00001.parquet")
    print(f"parquet 저장 완료: {global_idx} frames total")

    stats = compute_stats(all_states, all_actions)
    with open(output_dir / "meta_data" / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    with open(output_dir / "meta_data" / "episodes.jsonl", "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")

    info = {
        "codebase_version": "v2.0",
        "robot_type":       "mycobot280_arduino",
        "total_episodes":   len(ep_dirs),
        "total_frames":     global_idx,
        "fps":              FPS,
        "video":            True,
        "features": {
            "observation.images.top":   VIDEO_FEATURE,
            "observation.images.wrist": VIDEO_FEATURE,
            "observation.state": {
                "dtype": "float32",
                "shape": [STATE_DIM],
                "names": JOINT_NAMES,
            },
            "action": {
                "dtype": "float32",
                "shape": [ACTION_DIM],
                "names": JOINT_NAMES,
            },
        },
    }
    with open(output_dir / "meta_data" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n✅ 변환 완료: {output_dir}")
    print(f"\nLeRobot 데이터셋 확인:")
    print(f"  python -c \"")
    print(f"  from lerobot.datasets.lerobot_dataset import LeRobotDataset")
    print(f"  ds = LeRobotDataset('{output_dir}')")
    print(f"  print('episodes:', ds.num_episodes, 'frames:', len(ds))\"")


if __name__ == "__main__":
    main()
