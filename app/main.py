# main.py
from fastapi import FastAPI, Request, UploadFile, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.yoloutils import (
    generate_live_frames,
    detections_per_camera,
    camera_configs,
    initialize_streams,
    save_uploaded_video_and_process
)
import uuid
import shutil
import os

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.on_event("startup")
def startup_event():
    initialize_streams()

@app.get("/", response_class=HTMLResponse)
def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/cctv/{camera_id}", response_class=HTMLResponse)
def camera_view(request: Request, camera_id: str):
    data = detections_per_camera.get(camera_id, [])
    return templates.TemplateResponse(f"cctv_{camera_id}.html", {"request": request, "table_data": data})

@app.get("/video_feed/{camera_id}")
def video_feed(camera_id: str):
    return StreamingResponse(generate_live_frames(camera_id), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/video_feed_clahe/{camera_id}")
def video_feed_clahe(camera_id: str, clip: float = 2.0, tile: int = 8):
    return StreamingResponse(generate_live_frames(camera_id, apply_clahe=True, clip_limit=clip, tile_size=tile),
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/detections/{camera_id}")
def get_detections(camera_id: str):
    return JSONResponse(detections_per_camera.get(camera_id, []))

@app.get("/upload_image", response_class=HTMLResponse)
def upload_image_form(request: Request):
    return templates.TemplateResponse("input_image.html", {"request": request})

@app.post("/upload_image", response_class=HTMLResponse)
def upload_image(request: Request, image: UploadFile):
    from PIL import Image
    import io
    import numpy as np
    import cv2
    import base64
    from app.yoloutils import model

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

@app.get("/upload_video", response_class=HTMLResponse)
def upload_video_form(request: Request):
    return templates.TemplateResponse("input_record.html", {"request": request})

@app.post("/upload_video", response_class=HTMLResponse)
def upload_video(request: Request, video: UploadFile, clip: float = Form(2.0), tile: int = Form(8)):
    filename = f"video_{uuid.uuid4()}.mp4"
    upload_path = os.path.join("app/static/uploads", filename)
    os.makedirs("app/static/uploads", exist_ok=True)
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    predicted_rel_path = save_uploaded_video_and_process(upload_path, clip, tile)

    return templates.TemplateResponse("input_record.html", {
        "request": request,
        "original_video": upload_path.split("app/static/")[1],
        "predicted_video": predicted_rel_path.split("app/static/")[1]
    })
