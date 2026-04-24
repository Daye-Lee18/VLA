"""
Phase 2 — Step 2: Pre-trained policy baseline 평가
phase2_check_env.py 실행 후 사용 가능한 env 확인하고 나서 실행

실행 방법:
    python phase2_baseline.py

결과 저장:
    results/phase2_baseline.txt
"""

import os
import json
import time
import datetime
import torch
import numpy as np

# ── 설정 (여기만 바꾸면 됨) ──────────────────────────
ENV_NAME        = "pusht"        # "pusht" 또는 "metaworld/..."
N_EPISODES      = 20             # 일단 20개로 빠르게 확인 (나중에 50으로)
PRETRAINED_ID   = "lerobot/diffusion_pusht"   # HuggingFace model id
RESULTS_DIR     = "results"
# ──────────────────────────────────────────────────────

os.makedirs(RESULTS_DIR, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
result_file = os.path.join(RESULTS_DIR, f"phase2_baseline_{timestamp}.txt")

def log(msg):
    print(msg)
    with open(result_file, "a") as f:
        f.write(msg + "\n")

log("=" * 60)
log(f"Phase 2 — Baseline Evaluation")
log(f"시각       : {timestamp}")
log(f"env        : {ENV_NAME}")
log(f"model      : {PRETRAINED_ID}")
log(f"episodes   : {N_EPISODES}")
log(f"GPU        : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
log("=" * 60)

# ── 1. Policy 로드 ─────────────────────────────────────
log("\n[1/4] Pre-trained policy 로드 중...")
try:
    from lerobot.policies.factory import make_policy

    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = make_policy(
        hydra_cfg=None,
        pretrained_policy_name_or_path=PRETRAINED_ID,
    )
    policy = policy.to(device)
    policy.eval()
    log(f"  policy 로드 완료 (device: {device})")

except Exception as e:
    log(f"  ERROR: {e}")
    log("")
    log("  → lerobot CLI 방식으로 시도:")
    log(f"    python -m lerobot.scripts.lerobot_eval \\")
    log(f"      --policy.path={PRETRAINED_ID} \\")
    log(f"      --env.type={ENV_NAME} \\")
    log(f"      --eval.n_episodes={N_EPISODES}")
    raise SystemExit(1)

# ── 2. 환경 생성 ────────────────────────────────────────
log("\n[2/4] 환경 생성 중...")
try:
    from lerobot.envs.factory import make_env

    env = make_env(
        env_type=ENV_NAME,
        n_envs=1,
        seed=42,
    )
    log(f"  환경 생성 완료: {ENV_NAME}")
except Exception as e:
    log(f"  ERROR: {e}")
    raise SystemExit(1)

# ── 3. 평가 루프 ────────────────────────────────────────
log(f"\n[3/4] 평가 시작 ({N_EPISODES} episodes)...")

successes = []
episode_lengths = []
inference_times = []

for ep in range(N_EPISODES):
    obs, _ = env.reset(seed=ep)
    done = False
    steps = 0
    success = False

    while not done:
        start = time.time()

        with torch.no_grad():
            obs_tensor = {k: torch.tensor(v).unsqueeze(0).to(device)
                          for k, v in obs.items() if isinstance(v, np.ndarray)}
            action = policy.select_action(obs_tensor)
            action = action.cpu().numpy().squeeze(0)

        inference_ms = (time.time() - start) * 1000
        inference_times.append(inference_ms)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

        if info.get("is_success", False) or reward > 0.9:
            success = True

    successes.append(success)
    episode_lengths.append(steps)

    status = "✅ success" if success else "❌ fail"
    log(f"  ep {ep+1:02d}/{N_EPISODES} | {status} | steps: {steps}")

env.close()

# ── 4. 결과 저장 ────────────────────────────────────────
success_rate    = sum(successes) / N_EPISODES * 100
avg_steps       = np.mean(episode_lengths)
avg_infer_ms    = np.mean(inference_times)

log("")
log("=" * 60)
log("BASELINE 결과 (Phase 5 비교용으로 반드시 보관)")
log("=" * 60)
log(f"  성공률     : {success_rate:.1f}%  ({sum(successes)}/{N_EPISODES})")
log(f"  평균 steps : {avg_steps:.1f}")
log(f"  추론 시간  : {avg_infer_ms:.1f} ms/frame")
log(f"  목표 기준  : 성공률 >80%, 추론 <100ms")
log(f"\n  결과 파일  : {result_file}")
log("=" * 60)

# JSON으로도 저장 (Phase 5에서 자동 비교용)
json_path = os.path.join(RESULTS_DIR, "baseline.json")
with open(json_path, "w") as f:
    json.dump({
        "timestamp"     : timestamp,
        "model"         : PRETRAINED_ID,
        "env"           : ENV_NAME,
        "n_episodes"    : N_EPISODES,
        "success_rate"  : success_rate,
        "avg_steps"     : avg_steps,
        "avg_infer_ms"  : avg_infer_ms,
    }, f, indent=2)

log(f"  JSON 저장  : {json_path}")
