"""
Phase 5 — Baseline vs Fine-tuned 결과 비교
실행: python phase5_compare.py --checkpoint outputs/<your_run>/checkpoints/last

결과를 results/final_report.txt 에 저장
"""

import os
import json
import argparse
import datetime
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True,
                    help="fine-tuned checkpoint 경로 (outputs/.../checkpoints/last)")
parser.add_argument("--n_episodes", type=int, default=50)
parser.add_argument("--env", type=str, default="pusht")
args = parser.parse_args()

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
report_path = os.path.join(RESULTS_DIR, f"final_report_{timestamp}.txt")

def log(msg):
    print(msg)
    with open(report_path, "a") as f:
        f.write(msg + "\n")

# baseline 결과 불러오기
baseline_path = os.path.join(RESULTS_DIR, "baseline.json")
if os.path.exists(baseline_path):
    with open(baseline_path) as f:
        baseline = json.load(f)
    log(f"Baseline 결과 불러옴: {baseline_path}")
else:
    log("⚠️  baseline.json 없음 — phase2_baseline.py 먼저 실행 필요")
    baseline = None

log("=" * 60)
log(f"Phase 5 — Fine-tuned Policy 평가")
log(f"checkpoint : {args.checkpoint}")
log(f"episodes   : {args.n_episodes}")
log("=" * 60)

# fine-tuned policy 로드
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    from lerobot.policies.factory import make_policy
    policy = make_policy(hydra_cfg=None, pretrained_policy_name_or_path=args.checkpoint)
    policy = policy.to(device)
    policy.eval()
    log(f"Fine-tuned policy 로드 완료 (device: {device})")
except Exception as e:
    log(f"ERROR loading policy: {e}")
    raise SystemExit(1)

# 평가 루프
from lerobot.envs.factory import make_env
import time

env = make_env(env_type=args.env, n_envs=1, seed=0)

successes = []
episode_lengths = []
inference_times = []

for ep in range(args.n_episodes):
    obs, _ = env.reset(seed=ep + 1000)
    done = False
    steps = 0
    success = False

    while not done:
        t0 = time.time()
        with torch.no_grad():
            obs_tensor = {k: torch.tensor(v).unsqueeze(0).to(device)
                          for k, v in obs.items() if isinstance(v, np.ndarray)}
            action = policy.select_action(obs_tensor)
            action = action.cpu().numpy().squeeze(0)
        inference_times.append((time.time() - t0) * 1000)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        if info.get("is_success", False) or reward > 0.9:
            success = True

    successes.append(success)
    episode_lengths.append(steps)
    status = "✅" if success else "❌"
    log(f"  ep {ep+1:02d}/{args.n_episodes} {status} steps={steps}")

env.close()

sr      = sum(successes) / args.n_episodes * 100
avg_s   = np.mean(episode_lengths)
avg_ms  = np.mean(inference_times)

log("")
log("=" * 60)
log("최종 비교 결과")
log("=" * 60)
log(f"{'지표':<20} {'Baseline':>12} {'Fine-tuned':>12} {'목표':>10}")
log("-" * 60)

if baseline:
    log(f"{'성공률 (%)':<20} {baseline['success_rate']:>11.1f}% {sr:>11.1f}% {'> 80%':>10}")
    log(f"{'평균 steps':<20} {baseline['avg_steps']:>12.1f} {avg_s:>12.1f} {'':>10}")
    log(f"{'추론 시간 (ms)':<20} {baseline['avg_infer_ms']:>11.1f}  {avg_ms:>11.1f}  {'< 100ms':>10}")
    improvement = sr - baseline['success_rate']
    log(f"\n  성공률 개선: {improvement:+.1f}%p")
else:
    log(f"  성공률     : {sr:.1f}%")
    log(f"  평균 steps : {avg_s:.1f}")
    log(f"  추론 시간  : {avg_ms:.1f} ms/frame")

log(f"\n  리포트 저장: {report_path}")
log("=" * 60)

# 이력서 문장 출력
if baseline:
    log("\n[ 이력서 문장 템플릿 ]")
    log(f"Fine-tuned a Diffusion Policy (LeRobot 0.5.1) on {args.n_episodes} sim")
    log(f"demonstrations (pusht task), improving task success rate from")
    log(f"{baseline['success_rate']:.0f}% (pre-trained baseline) to {sr:.0f}%,")
    log(f"with inference latency of {avg_ms:.0f}ms/frame on RTX 2080 Ti.")

# JSON 저장
with open(os.path.join(RESULTS_DIR, "finetuned.json"), "w") as f:
    json.dump({
        "timestamp": timestamp,
        "checkpoint": args.checkpoint,
        "success_rate": sr,
        "avg_steps": avg_s,
        "avg_infer_ms": avg_ms,
    }, f, indent=2)
