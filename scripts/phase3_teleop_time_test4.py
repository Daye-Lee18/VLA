import time
import csv
import os
import threading
import serial.tools.list_ports
from pymycobot.mycobot280 import MyCobot280
from pynput import keyboard

# ---------------- 설정 ----------------
BAUD = 1000000

REC_HZ = 6          # CSV 기록 Hz / 팔로워 조인트 읽기 Hz
LEADER_HZ = 6       # 리더 읽기 Hz
CONTROL_HZ = 30     # 팔로워 제어 Hz

MIRROR_SPEED = 100
ANGLE_THRESHOLD = 1.0  # deg

OUTPUT_FILE = "follower_joint_data.csv"

# ---------------- 공유 변수 ----------------
latest_leader_angles = None      # 팔로워 제어용: 리더 조인트 값
latest_follower_angles = None    # CSV 저장용: 팔로워 실제 조인트 값

latest_gripper = 100
gripper_changed = False
running = True

lock = threading.Lock()


# ---------------- 키보드 ----------------
def on_press(key):
    global latest_gripper, gripper_changed

    try:
        if key.char == 'o':
            with lock:
                latest_gripper = 100
                gripper_changed = True
            print("[KEY] gripper open")

        elif key.char == 'c':
            with lock:
                latest_gripper = 0
                gripper_changed = True
            print("[KEY] gripper close")

    except AttributeError:
        pass


# ---------------- Leader 읽기: 팔로워 제어용 ----------------
def leader_thread_func(leader):
    global latest_leader_angles, running

    interval = 1.0 / LEADER_HZ
    next_time = time.perf_counter()

    while running:
        try:
            angles = leader.get_angles()

            if angles and len(angles) == 6:
                with lock:
                    latest_leader_angles = angles.copy()

        except Exception as e:
            print(f"[WARN] leader read 실패: {e}")

        next_time += interval
        sleep = next_time - time.perf_counter()

        if sleep > 0:
            time.sleep(sleep)
        else:
            next_time = time.perf_counter()


# ---------------- Follower 실제 조인트 읽기: CSV 저장용 ----------------
def follower_logger_thread_func(follower):
    global latest_follower_angles, running

    interval = 1.0 / REC_HZ
    next_time = time.perf_counter()

    while running:
        try:
            angles = follower.get_angles()

            if angles and len(angles) == 6:
                with lock:
                    latest_follower_angles = angles.copy()

        except Exception as e:
            print(f"[WARN] follower read 실패: {e}")

        next_time += interval
        sleep = next_time - time.perf_counter()

        if sleep > 0:
            time.sleep(sleep)
        else:
            next_time = time.perf_counter()


# ---------------- Follower 제어 ----------------
def control_thread_func(follower):
    global latest_leader_angles, latest_gripper, gripper_changed, running

    prev_angles = None

    interval = 1.0 / CONTROL_HZ
    next_time = time.perf_counter()

    while running:
        with lock:
            angles = latest_leader_angles.copy() if latest_leader_angles else None
            gripper = latest_gripper
            do_gripper = gripper_changed

            if gripper_changed:
                gripper_changed = False

        # ---- 각도 제어: 리더 값을 팔로워에 전송 ----
        if angles and len(angles) == 6:
            send = False

            if prev_angles is None:
                send = True
            else:
                for a, b in zip(angles, prev_angles):
                    if abs(a - b) > ANGLE_THRESHOLD:
                        send = True
                        break

            if send:
                try:
                    follower.send_angles(angles, MIRROR_SPEED)
                    prev_angles = angles.copy()
                except Exception as e:
                    print(f"[WARN] send_angles 실패: {e}")

        # ---- 그리퍼 제어 ----
        if do_gripper:
            try:
                follower.set_gripper_value(gripper, 50)
            except Exception as e:
                print(f"[WARN] gripper 실패: {e}")

        next_time += interval
        sleep = next_time - time.perf_counter()

        if sleep > 0:
            time.sleep(sleep)
        else:
            next_time = time.perf_counter()


# ---------------- 메인 ----------------
def main():
    global running

    ports = sorted(p.device for p in serial.tools.list_ports.comports())

    if not ports:
        print("❌ 포트를 찾을 수 없습니다.")
        return

    print(f"포트 목록: {ports}")

    l_idx = int(input("리더(Leader) 번호: "))
    f_idx = int(input("팔로워(Follower) 번호: "))

    leader = MyCobot280(ports[l_idx], BAUD)
    follower = MyCobot280(ports[f_idx], BAUD)

    try:
        follower.set_fresh_mode(1)
        print("✅ follower fresh mode ON")
    except Exception as e:
        print(f"[WARN] follower fresh mode 설정 실패: {e}")

    try:
        leader.set_fresh_mode(1)
        print("✅ leader fresh mode ON")
    except Exception as e:
        print(f"[WARN] leader fresh mode 설정 실패: {e}")

    # 키보드 리스너
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # 상태 설정
    try:
        follower.focus_all_servos()
    except Exception as e:
        print(f"[WARN] follower focus 실패: {e}")

    try:
        leader.release_all_servos()
    except Exception as e:
        print(f"[WARN] leader release 실패: {e}")

    # CSV 준비
    file_exists = os.path.isfile(OUTPUT_FILE)
    f_csv = open(OUTPUT_FILE, mode="a", newline="")
    writer = csv.writer(f_csv)

    if not file_exists:
        writer.writerow([
            "timestamp",
            "follower_j1", "follower_j2", "follower_j3",
            "follower_j4", "follower_j5", "follower_j6",
            "gripper",
            "status"
        ])

    # 리더 읽기 스레드: 팔로워 제어용
    leader_thread = threading.Thread(
        target=leader_thread_func,
        args=(leader,),
        daemon=True
    )
    leader_thread.start()

    # 팔로워 제어 스레드
    control_thread = threading.Thread(
        target=control_thread_func,
        args=(follower,),
        daemon=True
    )
    control_thread.start()

    # 팔로워 실제 조인트 읽기 스레드: CSV 저장용
    follower_logger_thread = threading.Thread(
        target=follower_logger_thread_func,
        args=(follower,),
        daemon=True
    )
    follower_logger_thread.start()

    print(f"\n✅ Leader read: {LEADER_HZ}Hz")
    print(f"✅ Follower control: {CONTROL_HZ}Hz")
    print(f"✅ Follower logging: {REC_HZ}Hz")
    print("o: open / c: close / Ctrl+C 종료")
    print(f"CSV 저장 파일: {OUTPUT_FILE}")

    # ---- CSV 기록 루프 ----
    interval = 1.0 / REC_HZ
    next_time = time.perf_counter()
    row_count = 0

    try:
        while True:
            unix_ts = time.time()

            with lock:
                follower_angles = latest_follower_angles.copy() if latest_follower_angles else None
                gripper = latest_gripper

            if follower_angles and len(follower_angles) == 6:
                row = [unix_ts] + follower_angles + [gripper, "OK"]
            else:
                row = [unix_ts] + ["FAIL"] * 6 + [gripper, "FOLLOWER_READ_FAIL"]

            writer.writerow(row)
            row_count += 1

            # 너무 자주 flush하지 않도록 1초에 한 번 정도 저장
            if row_count % REC_HZ == 0:
                f_csv.flush()

            next_time += interval
            sleep = next_time - time.perf_counter()

            if sleep > 0:
                time.sleep(sleep)
            else:
                next_time = time.perf_counter()

    except KeyboardInterrupt:
        print("\n🛑 종료 중...")

    finally:
        running = False
        time.sleep(0.3)

        f_csv.flush()
        f_csv.close()

        try:
            leader.focus_all_servos()
        except:
            pass

        try:
            listener.stop()
        except:
            pass

        print(f"✅ 저장 완료: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()