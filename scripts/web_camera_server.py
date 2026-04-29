import cv2
import numpy as np
from flask import Flask, Response
import pyrealsense2 as rs
import threading

app = Flask(__name__)

# 카메라 초기화 (기존 로직 유지)
def setup_cameras():
    # RealSense 설정
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    
    # USB 웹캠 설정
    cap = cv2.VideoCapture(0)
    return pipeline, cap

pipeline, cap = setup_cameras()

def generate_frames(camera_type):
    while True:
        if camera_type == 'top':
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            frame = np.asanyarray(color_frame.get_data())
        else:
            ret, frame = cap.read()
            if not ret: continue

        # JPEG로 인코딩
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # 스트리밍 데이터 형식으로 반환
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_top')
def video_top():
    return Response(generate_frames('top'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_wrist')
def video_wrist():
    return Response(generate_frames('wrist'), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':
    print("웹 서버 시작: http://192.168.6.1:5000")
    # 호스트를 0.0.0.0으로 설정해야 외부(노트북)에서 접속 가능
    app.run(host='0.0.0.0', port=5000, threaded=True)