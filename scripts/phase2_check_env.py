"""
Phase 2 — Step 1: 사용 가능한 sim 환경 확인
결과 보고 phase2_baseline.py 에서 쓸 env 결정
"""

import subprocess
import sys

print("=" * 50)
print("Step 1. CUDA / torch")
print("=" * 50)
import torch
print(f"  torch      : {torch.__version__}")
print(f"  CUDA 사용  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU        : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM       : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

print()
print("=" * 50)
print("Step 2. lerobot envs.factory import")
print("=" * 50)
try:
    from lerobot.envs.factory import make_env
    print("  make_env import : OK")
except Exception as e:
    print(f"  ERROR: {e}")

print()
print("=" * 50)
print("Step 3. sim 환경 설치 여부")
print("=" * 50)

envs_to_check = {
    "gym_pusht"  : "pusht (T블록 밀기 — 가장 가벼움)",
    "metaworld"  : "metaworld (로봇팔 조작)",
    "mujoco"     : "mujoco (물리 시뮬레이터)",
    "libero"     : "libero (조립 태스크)",
}

available = []
for pkg, desc in envs_to_check.items():
    try:
        __import__(pkg)
        print(f"  ✅  {pkg:15s} — {desc}")
        available.append(pkg)
    except ImportError:
        print(f"  ❌  {pkg:15s} — {desc}")

print()
print("=" * 50)
print("Step 4. lerobot 내장 rl 모듈")
print("=" * 50)
try:
    import lerobot.rl
    import os
    rl_path = os.path.dirname(lerobot.rl.__file__)
    files = [f for f in os.listdir(rl_path) if f.endswith(".py") and not f.startswith("_")]
    print(f"  rl 모듈 파일: {files}")
except Exception as e:
    print(f"  ERROR: {e}")

print()
print("=" * 50)
print("결론")
print("=" * 50)
if "gym_pusht" in available:
    print("  → pusht 사용 가능. phase2_baseline.py 바로 실행 가능.")
elif "metaworld" in available:
    print("  → metaworld 사용 가능. env_name을 metaworld로 설정 필요.")
else:
    print("  → 설치된 sim 없음. 아래 명령어로 설치:")
    print()
    print("     pip install gym_pusht")
    print("     # 또는")
    print("     pip install metaworld")
