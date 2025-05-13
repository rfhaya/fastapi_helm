# main.py

from fastapi import FastAPI, Request, UploadFile, File, Form, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.yoloutils import (
    generate_live_frames, detections_per_camera,
    camera_configs, initialize_streams,
    save_uploaded_video_and_process, model
)

import os
import uuid
import shutil
import io
import base64
import sqlite3
from threading import Thread
from PIL import Image
import numpy as np
import cv2

# === Setup ===
app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# === Startup ===
@app.on_event("startup")
def startup_event():
    initialize_streams()

def get_db_connection():
    return sqlite3.connect("app/data/detections.db")

# === Page Routes ===
@app.get("/", response_class=HTMLResponse)
def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/upload_image", response_class=HTMLResponse)
def upload_image_form(request: Request):
    return templates.TemplateResponse("input_image.html", {"request": request})

@app.get("/upload_video", response_class=HTMLResponse)
def upload_video_form(request: Request):
    return templates.TemplateResponse("input_record.html", {"request": request})

@app.get("/cctv/{camera_id}", response_class=HTMLResponse)
def camera_view(request: Request, camera_id: str):
    data = detections_per_camera.get(camera_id, [])
    return templates.TemplateResponse(f"cctv_{camera_id}.html", {
        "request": request,
        "table_data": data
    })

@app.get("/data", response_class=HTMLResponse)
def data(request: Request):
    return templates.TemplateResponse("table_all.html", {"request": request})

# === Video Streaming ===
@app.get("/video_feed/{camera_id}")
def video_feed(camera_id: str, clip: float = 2.0, tile: int = 8):
    return StreamingResponse(
        generate_live_frames(camera_id, apply_clahe=False, clip_limit=clip, tile_size=tile),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/video_feed_clahe/{camera_id}")
def video_feed_clahe(camera_id: str, clip: float = 2.0, tile: int = 8):
    return StreamingResponse(
        generate_live_frames(camera_id, apply_clahe=True, clip_limit=clip, tile_size=tile),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# === Image/Video Upload ===
@app.post("/upload_image", response_class=HTMLResponse)
def upload_image(request: Request, image: UploadFile = File(...)):
    image_stream = image.file.read()
    pil_img = Image.open(io.BytesIO(image_stream)).convert('RGB')
    image_np = np.array(pil_img)

    results = model.predict(image_np, imgsz=640, conf=0.25, verbose=False)
    annotated = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    _, buffer = cv2.imencode(".jpg", annotated_rgb)
    img_b64 = base64.b64encode(buffer).decode("utf-8")

    return templates.TemplateResponse("input_image.html", {
        "request": request,
        "original_b64": base64.b64encode(image_stream).decode("utf-8"),
        "prediction_b64": img_b64
    })

@app.post("/upload_video", response_class=HTMLResponse)
def upload_video(request: Request, video: UploadFile = File(...), clip: float = Form(2.0), tile: int = Form(8)):
    filename = f"video_{uuid.uuid4()}.mp4"
    upload_path = os.path.join("app/static/uploads", filename)
    os.makedirs("app/static/uploads", exist_ok=True)

    with open(upload_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    Thread(target=save_uploaded_video_and_process, args=(upload_path, clip, tile), daemon=True).start()

    return templates.TemplateResponse("input_record.html", {
        "request": request,
        "original_video": upload_path.split("app/static/")[1],
        "predicted_video": None
    })

# === Detection Data Endpoints ===
@app.get("/detections/{camera_id}")
def get_detections(camera_id: str):
    return JSONResponse(detections_per_camera.get(camera_id, []))

@app.get("/detections_count/{camera_id}")
def get_count_by_camera(camera_id: str):
    query = "SELECT COUNT(*) FROM detections WHERE photo LIKE ?"
    like_value = f"%{camera_id}%"
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute(query, (like_value,))
        count = c.fetchone()[0]
    return {"camera_id": camera_id, "count": count}

@app.get("/detections_by_location/{location}", response_class=JSONResponse)
def get_detections_by_location(location: str):
    location_map = {
        "banda_aceh": ["simpang_dharma3", "simpang_dharma4"],
        "medan": ["katamso_aniidrus", "katamso_masjidraya"],
        "jakarta": ["gelora1", "gelora2"],
        "pematang_siantar": ["simpang_lr20"]
    }

    camera_names = {
        "simpang_dharma3": "Banda Aceh - Simpang Dharma 3",
        "simpang_dharma4": "Banda Aceh - Simpang Dharma 4",
        "katamso_aniidrus": "Medan - Katamso Ani Idrus",
        "katamso_masjidraya": "Medan - Katamso Masjid Raya",
        "gelora1": "Jakarta - Gelora 1",
        "gelora2": "Jakarta - Gelora 2",
        "simpang_lr20": "Pematang Siantar - Sp. Lorong 20"
    }

    folders = location_map.get(location.lower())
    if not folders:
        return []

    query = f"""
        SELECT frame, photo, class, date, confidence, clahe
        FROM detections
        WHERE {" OR ".join(["photo LIKE ?"] * len(folders))}
        ORDER BY frame ASC
    """
    like_values = [f"%{f}%" for f in folders]

    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute(query, like_values)
        rows = c.fetchall()

    data_clahe = {}
    data_nonclahe = {}

    for row in rows:
        frame, photo, cls, date, conf, clahe = row
        cam_code = photo.split("/")[1] if "/" in photo else "unknown"
        key = (frame, cls, photo)
        entry = {
            "photo": photo,
            "class": cls,
            "date": date,
            "confidence": conf,
            "location": camera_names.get(cam_code, "Tidak diketahui")
        }
        if clahe == 1:
            data_clahe[key] = entry
        else:
            data_nonclahe[key] = entry

    all_keys = sorted(set(data_clahe.keys()) | set(data_nonclahe.keys()))
    result = []

    for i, key in enumerate(all_keys, 1):
        frame, cls, _ = key
        n = data_nonclahe.get(key)
        y = data_clahe.get(key)

        result.append({
            "No": i,
            "frame": frame,
            "Gambar_Tanpa_CLAHE": n["photo"] if n else "-",
            "Gambar_Dengan_CLAHE": y["photo"] if y else "-",
            "class": cls,
            "confidence": f'{n["confidence"] if n else "-"} / {y["confidence"] if y else "-"}',
            "Status_Deteksi": (
                "Terdeteksi di keduanya" if n and y else
                "Hanya di CLAHE" if y else
                "Hanya di Non-CLAHE"
            ),
            "date": y["date"] if y else (n["date"] if n else "-"),
            "location": y["location"] if y else (n["location"] if n else "Tidak diketahui")
        })

    return result

# === Realtime Detection Comparison & Count ===
@app.get("/realtime_count_nonclahe/{camera_id}")
def realtime_count_nonclahe(camera_id: str):
    data = detections_per_camera.get(camera_id, [])
    count = {"helm": 0, "no_helm": 0}
    for det in data:
        if det.get("clahe") == 0:
            cls = det["class"].lower()
            if cls in count:
                count[cls] += 1
    return count

@app.get("/realtime_count_clahe/{camera_id}")
def realtime_count_clahe(camera_id: str):
    data = detections_per_camera.get(camera_id, [])
    count = {"helm": 0, "no_helm": 0}
    for det in data:
        if det.get("clahe") == 1:  # filter hanya CLAHE
            cls = det["class"].lower()
            if cls in count:
                count[cls] += 1
    return count


@app.get("/realtime_comparison/{camera_id}")
def realtime_comparison(camera_id: str):
    data = detections_per_camera.get(camera_id, [])
    result_dict = {}

    for d in data:
        key = (d['frame'], d['class'])
        if key not in result_dict:
            result_dict[key] = {
                "frame": d['frame'],
                "class": d['class'],
                "Gambar_Tanpa_CLAHE": "-",
                "Gambar_Dengan_CLAHE": "-",
                "Confidence": "-",
                "Status_Deteksi": ""
            }

        if d.get("clahe"):
            result_dict[key]["Gambar_Dengan_CLAHE"] = d["img"]
            result_dict[key]["Confidence"] = f"{result_dict[key]['Confidence'].split(' / ')[0] if ' / ' in result_dict[key]['Confidence'] else '-'} / {d['confidence']}"
        else:
            result_dict[key]["Gambar_Tanpa_CLAHE"] = d["img"]
            result_dict[key]["Confidence"] = f"{d['confidence']} / {result_dict[key]['Confidence'].split(' / ')[1] if ' / ' in result_dict[key]['Confidence'] else '-'}"

        result_dict[key]["Status_Deteksi"] = (
            "Terdeteksi di keduanya" if result_dict[key]["Gambar_Tanpa_CLAHE"] != "-" and result_dict[key]["Gambar_Dengan_CLAHE"] != "-" else
            "Hanya di CLAHE" if result_dict[key]["Gambar_Tanpa_CLAHE"] == "-" else
            "Hanya di Non-CLAHE"
        )

    result = []
    for i, v in enumerate(result_dict.values(), 1):
        v["No"] = i
        result.append(v)

    return result

# === CLAHE Comparison from DB ===
@app.get("/perbandingan_clahe", response_class=JSONResponse)
def get_comparison_table():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT frame, photo, class, confidence, clahe FROM detections ORDER BY frame")
        rows = cursor.fetchall()

    data_clahe = {}
    data_nonclahe = {}

    for row in rows:
        frame, photo, cls, conf, clahe = row
        key = (frame, cls)
        entry = {"photo": photo, "confidence": conf}
        (data_clahe if clahe == 1 else data_nonclahe)[key] = entry

    keys = sorted(set(data_clahe.keys()) | set(data_nonclahe.keys()))
    table = []

    for i, key in enumerate(keys, 1):
        frame, cls = key
        n = data_nonclahe.get(key)
        y = data_clahe.get(key)

        table.append({
            "No": i,
            "Frame": frame,
            "Gambar_Tanpa_CLAHE": n["photo"] if n else "-",
            "Gambar_Dengan_CLAHE": y["photo"] if y else "-",
            "Class": cls,
            "Confidence": f'{n["confidence"] if n else "-"} / {y["confidence"] if y else "-"}',
            "Status_Deteksi": (
                "Terdeteksi di keduanya" if n and y else
                "Hanya di CLAHE" if y else
                "Hanya di Non-CLAHE"
            )
        })

    return table

# === API: Stats and Summary ===
@app.get("/stats/{location}")
def get_stats(location: str):
    location_map = {
        "banda_aceh": ["simpang_dharma3", "simpang_dharma4"],
        "medan": ["katamso_aniidrus", "katamso_masjidraya"],
        "jakarta": ["gelora1", "gelora2"],
        "pematang_siantar": ["simpang_lr20"]
    }

    folders = location_map.get(location.lower())
    if not folders:
        return {"labels": [], "helm": [], "no_helm": []}

    query = f"""
        SELECT substr(date, 1, 10) as tanggal, class, COUNT(*) 
        FROM detections
        WHERE {" OR ".join(["photo LIKE ?"] * len(folders))}
        GROUP BY tanggal, class
        ORDER BY tanggal
    """
    like_values = [f"%{folder}%" for folder in folders]

    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute(query, like_values)
        rows = c.fetchall()

    # Kumpulkan per tanggal
    date_counts = {}
    for tanggal, cls, count in rows:
        if tanggal not in date_counts:
            date_counts[tanggal] = {"helm": 0, "no_helm": 0}
        if cls.lower() == "helm":
            date_counts[tanggal]["helm"] += count
        else:
            date_counts[tanggal]["no_helm"] += count

    labels = sorted(date_counts.keys())
    helm = [date_counts[t]["helm"] for t in labels]
    no_helm = [date_counts[t]["no_helm"] for t in labels]

    return {"labels": labels, "helm": helm, "no_helm": no_helm}

@app.get("/summary/{location}")
def get_summary(location: str):
    location_map = {
        "banda_aceh": ["simpang_dharma3", "simpang_dharma4"],
        "medan": ["katamso_aniidrus", "katamso_masjidraya"],
        "jakarta": ["gelora1", "gelora2"],
        "pematang_siantar": ["simpang_lr20"]
    }

    folders = location_map.get(location.lower())
    if not folders:
        return {"helm": 0, "non_helm": 0}

    query = f"""
        SELECT class, COUNT(*) 
        FROM detections
        WHERE {" OR ".join(["photo LIKE ?"] * len(folders))}
        GROUP BY class
    """
    like_values = [f"%{folder}%" for folder in folders]

    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute(query, like_values)
        rows = c.fetchall()

    result = {"helm": 0, "non_helm": 0}
    for cls, count in rows:
        if cls.lower() == "helm":
            result["helm"] += count
        else:
            result["non_helm"] += count

    return result

@app.post("/reset_comparison/{camera_id}")
def reset_comparison(camera_id: str):
    detections_per_camera[camera_id] = []
    return {"status": "reset successful"}
