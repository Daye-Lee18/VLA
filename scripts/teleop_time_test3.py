import time
import csv
import os
import threading
import serial.tools.list_ports
from pymycobot.mycobot280 import MyCobot280
from pynput import keyboard

# ---------------- 설정 ----------------
BAUD = 1000000

REC_HZ = 6       # 리더 읽기 Hz
CONTROL_HZ = 30    # 팔로워 제어 Hz

MIRROR_SPEED = 100
ANGLE_THRESHOLD = 1.0  # deg

OUTPUT_FILE = "joint_data.csv"

# ---------------- 공유 변수 ----------------
latest_angles = None
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

        elif key.char == 'c':
            with lock:
                latest_gripper = 0
                gripper_changed = True

    except AttributeError:
        pass


# ---------------- Leader 읽기 ----------------
def leader_thread_func(leader):
    global latest_angles, running

    interval = 1.0 / REC_HZ
    next_time = time.perf_counter()

    while running:
        try:
            angles = leader.get_angles()

            if angles and len(angles) == 6:
                with lock:
                    latest_angles = angles

        except Exception as e:
            print(f"[WARN] leader read 실패: {e}")

        next_time += interval
        sleep = next_time - time.perf_counter()

        if sleep > 0:
            time.sleep(sleep)
        else:
            next_time = time.perf_counter()


# ---------------- Follower 제어 ----------------
def control_thread_func(follower):
    global latest_angles, latest_gripper, gripper_changed, running

    prev_angles = None

    interval = 1.0 / CONTROL_HZ
    next_time = time.perf_counter()

    while running:
        with lock:
            angles = latest_angles.copy() if latest_angles else None
            gripper = latest_gripper
            do_gripper = gripper_changed
            if gripper_changed:
                gripper_changed = False

        # print(f"Debug!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print(angles)
        # print(f"Debug!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


        # ---- 각도 제어 (변화 있을 때만) ----
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
                    prev_angles = angles
                except Exception as e:
                    print(f"[WARN] send_angles 실패: {e}")

        # ---- 그리퍼 ----
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
        print(":x: 포트를 찾을 수 없습니다.")
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

    # 키보드
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # 상태 설정
    follower.focus_all_servos()
    leader.release_all_servos()

    # CSV
    file_exists = os.path.isfile(OUTPUT_FILE)
    f_csv = open(OUTPUT_FILE, mode="a", newline="")
    writer = csv.writer(f_csv)

    if not file_exists:
        writer.writerow([
            "timestamp",
            "j1", "j2", "j3", "j4", "j5", "j6",
            "gripper",
            "status"
        ])

    # 스레드 시작
    leader_thread = threading.Thread(
        target=leader_thread_func,
        args=(leader,),
        daemon=True
    )
    leader_thread.start()

    control_thread = threading.Thread(
        target=control_thread_func,
        args=(follower,),
        daemon=True
    )
    control_thread.start()

    print(f"\n:white_check_mark: Leader {REC_HZ}Hz / Follower {CONTROL_HZ}Hz")
    print("o: open / c: close / Ctrl+C 종료")

    # ---- CSV 기록 루프 ----
    interval = 1.0 / REC_HZ
    next_time = time.perf_counter()

    try:
        while True:
            unix_ts = time.time()

            with lock:
                angles = latest_angles.copy() if latest_angles else None
                gripper = latest_gripper

            if angles and len(angles) == 6:
                row = [unix_ts] + angles + [gripper, "OK"]
            else:
                row = [unix_ts] + ["FAIL"] * 6 + [gripper, "READ_FAIL"]

            writer.writerow(row)

            # flush 최소화
            if int(unix_ts) % 1 == 0:
                f_csv.flush()

            next_time += interval
            sleep = next_time - time.perf_counter()

            if sleep > 0:
                time.sleep(sleep)
            else:
                next_time = time.perf_counter()

    except KeyboardInterrupt:
        print("\n:octagonal_sign: 종료 중...")

    finally:
        running = False
        time.sleep(0.2)

        f_csv.flush()
        f_csv.close()

        try:
            leader.focus_all_servos()
        except:
            pass

        listener.stop()
        print(f":white_check_mark: 저장 완료: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()