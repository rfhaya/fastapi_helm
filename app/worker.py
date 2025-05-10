from celery import Celery
from app.yoloutils import process_video_with_yolo

celery = Celery("worker", broker="redis://localhost:6379/0")
celery.conf.task_routes = {"app.worker.*": {"queue": "default"}}

@celery.task
def process_video_task(video_path, clip_limit, tile_size):
    output_path = video_path.replace("uploads", "results").replace(".mp4", "_out.mp4")
    process_video_with_yolo(video_path, output_path, clip_limit, tile_size)
    return output_path