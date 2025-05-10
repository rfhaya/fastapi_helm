import cv2
import time
import os
import json
import sqlite3
import numpy as np
import imageio.v2 as imageio
from datetime import datetime
from threading import Thread, Lock
from ultralytics import YOLO

model = YOLO("app/data/model/best.pt")

camera_configs = {
    "simpang_dharma3": "https://cctv-stream.bandaacehkota.info/memfs/1e560ac1-8b57-416a-b64e-d4190ff83f88_output_0.m3u8",
    "simpang_dharma4": "https://cctv-stream.bandaacehkota.info/memfs/f9444904-ad31-4401-9643-aee6e33b85c7_output_0.m3u8",
    "katamso_aniidrus": "https://atcsdishub.pemkomedan.go.id/camera/KATAMSOANIIDRUS.m3u8",
    "katamso_masjidraya": "https://atcsdishub.pemkomedan.go.id/camera/KATAMSOMASJIDRAYA.m3u8",
    "gelora1": "https://cctv.balitower.co.id/Gelora-017-700470_3/tracks-v1/index.fmp4.m3u8",
    "gelora2": "https://cctv.balitower.co.id/Gelora-017-700470_4/tracks-v1/index.fmp4.m3u8",
    "simpang_lr20": "https://stream.denava.id/stream/a64fdc17-7a69-43b1-a26f-232c3df82641/channel/0/hls/live/index.m3u8"
}

camera_streams = {}
detections_per_camera = {}
DETECTIONS_FILE = "app/static/detections.json"
os.makedirs("app/static/detected", exist_ok=True)

DB_PATH = "app/data/detections.db"

def insert_detection(frame, photo, label, time_str, confidence):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            INSERT INTO detections (frame, photo, class, date, confidence)
            VALUES (?, ?, ?, ?, ?)
        """, (frame, photo, label, time_str, confidence))
        conn.commit()
        conn.close()
    except Exception as e:
        print("DB Error:", e)

def save_detections_to_file():
    with open(DETECTIONS_FILE, "w") as f:
        json.dump(detections_per_camera, f)

def load_detections_from_file():
    global detections_per_camera
    if os.path.exists(DETECTIONS_FILE):
        with open(DETECTIONS_FILE, "r") as f:
            detections_per_camera = json.load(f)

last_boxes = []
IOU_THRESHOLD = 0.5

def iou(b1, b2):
    xA = max(b1['x1'], b2['x1'])
    yA = max(b1['y1'], b2['y1'])
    xB = min(b1['x2'], b2['x2'])
    yB = min(b1['y2'], b2['y2'])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (b1['x2'] - b1['x1']) * (b1['y2'] - b1['y1'])
    boxBArea = (b2['x2'] - b2['x1']) * (b2['y2'] - b2['y1'])
    union = boxAArea + boxBArea - interArea
    return interArea / union if union else 0

def is_duplicate(bbox):
    for prev in last_boxes:
        if iou(prev, bbox) > IOU_THRESHOLD:
            return True
    return False

def save_detection(label, conf, crop_img, bbox, camera_id, frame_count):
    folder = f"app/static/detected/{camera_id}"
    os.makedirs(folder, exist_ok=True)

    filename = f"{int(time.time()*1000)}.jpg"
    abs_path = os.path.join(folder, filename)
    rel_path = f"detected/{camera_id}/{filename}"

    saved = cv2.imwrite(abs_path, crop_img)
    if not saved:
        return

    if camera_id not in detections_per_camera:
        detections_per_camera[camera_id] = []

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    detections_per_camera[camera_id].append({
        "frame": frame_count,
        "class": label,
        "confidence": round(conf, 2),
        "time": timestamp,
        "img": rel_path
    })

    detections_per_camera[camera_id] = detections_per_camera[camera_id][-100:]
    save_detections_to_file()
    insert_detection(frame_count, rel_path, label, timestamp, round(conf, 2))

def stream_reader(camera_id, url):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"❌ Failed to open stream {camera_id}")
        return
    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(0.5)
            continue
        with camera_streams[camera_id]["lock"]:
            camera_streams[camera_id]["frame"] = frame.copy()
        time.sleep(0.03)

def initialize_streams():
    for cam_id, url in camera_configs.items():
        camera_streams[cam_id] = {"frame": None, "lock": Lock()}
        t = Thread(target=stream_reader, args=(cam_id, url), daemon=True)
        t.start()
        camera_streams[cam_id]["thread"] = t

def generate_live_frames(camera_id, apply_clahe=False, clip_limit=2.0, tile_size=8):
    frame_count = 0
    FRAME_SKIP = 2
    global last_boxes
    while True:
        with camera_streams[camera_id]["lock"]:
            frame = camera_streams[camera_id]["frame"]
            if frame is None:
                continue
            frame = frame.copy()

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        frame = cv2.resize(frame, (640, 360))
        if apply_clahe:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        try:
            results = model.predict(frame, imgsz=640, conf=0.1, verbose=False, device=0)
            result = results[0]
        except:
            continue

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h))
            y2 = max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue
            crop_img = frame[y1:y2, x1:x2]
            bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            if is_duplicate(bbox):
                continue
            save_detection(label, conf, crop_img, bbox, camera_id, frame_count)
            last_boxes.append(bbox)

        annotated = result.plot()
        ret, buffer = cv2.imencode(".jpg", annotated)
        if not ret:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def save_uploaded_video_and_process(original_path, clip_limit, tile_size):
    cap = cv2.VideoCapture(original_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    output_filename = f"app/static/results/result_{int(time.time())}.mp4"
    writer = imageio.get_writer(output_filename, fps=fps, codec="libx264")
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        results = model.predict(enhanced, imgsz=640, conf=0.25, verbose=False)
        annotated = results[0].plot()
        writer.append_data(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

    cap.release()
    writer.close()
    return output_filename
