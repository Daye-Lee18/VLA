import cv2
import time
import os
import csv
import numpy as np

cams = {
    "C920_1": "/dev/video2",
    "USB_Cam": "/dev/video4",
    "C920_2": "/dev/video6",
}

SAVE_DIR = "camera_log"
FPS = 6
WIDTH = 640
HEIGHT = 480

os.makedirs(SAVE_DIR, exist_ok=True)

for name in cams.keys():
    os.makedirs(os.path.join(SAVE_DIR, name), exist_ok=True)

csv_path = os.path.join(SAVE_DIR, "log.csv")

caps = {}

for name, dev in cams.items():
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    if cap.isOpened():
        print(f"[OK] {name}: {dev}")
        caps[name] = cap
    else:
        print(f"[FAIL] {name}: {dev}")

if len(caps) != 3:
    print("[ERROR] 카메라 3대가 모두 열리지 않았습니다.")
    exit()

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "frame_id",
        "sync_unix_time",
        "camera_name",
        "device",
        "image_path"
    ])

    frame_id = 0
    print("[START] Logging started. Press 'q' to stop.")

    while True:
        sync_unix_time = time.time()
        frames = []

        for name, cap in caps.items():
            ret, frame = cap.read()

            if not ret:
                frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
                image_path = "READ_FAIL"
            else:
                filename = f"{frame_id:06d}_{name}_{sync_unix_time:.6f}.jpg"
                image_path = os.path.join(SAVE_DIR, name, filename)
                cv2.imwrite(image_path, frame)

                cv2.putText(frame, name, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"unix: {sync_unix_time:.6f}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"frame: {frame_id}", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            writer.writerow([
                frame_id,
                f"{sync_unix_time:.6f}",
                name,
                cams[name],
                image_path
            ])

            view = cv2.resize(frame, (426, 320))
            frames.append(view)

        f.flush()

        combined = np.hstack(frames)
        cv2.imshow("3 Camera Sync Logging", combined)

        frame_id += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

for cap in caps.values():
    cap.release()

cv2.destroyAllWindows()
print(f"[DONE] Saved to: {SAVE_DIR}")