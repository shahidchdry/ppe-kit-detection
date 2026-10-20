"""
SafeGuard AI - Web Server & Inference Dispatcher
Handles ephemeral video processing, background inference jobs, range-based streaming,
model weights loading, and live violation alerts without persistent disk accumulation.
"""

import os
import sys
import json
import time
import uuid
import mimetypes
import tempfile
import threading
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import urllib.parse

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
WEBSITE_DIR = BASE_DIR / "website"
STATIC_DIR = WEBSITE_DIR / "static"
MODEL_DIR = BASE_DIR / "model"

# Use OS temporary directory for ephemeral video processing
# Videos are NEVER permanently stored in a database or project folder
TEMP_DIR = Path(tempfile.gettempdir()) / "safeguard_ppe_temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Add base directory to path so inference module can be imported
sys.path.append(str(BASE_DIR))

# In-memory Task State Tracker
tasks = {}


def cleanup_temp_files(specific_task_id=None, max_age_seconds=900):
    """
    Purges temporary video files to prevent accumulation on disk.
    If specific_task_id is given, removes files associated with that task immediately.
    Otherwise, cleans files older than max_age_seconds.
    """
    now = time.time()
    try:
        for f in TEMP_DIR.glob("*"):
            if not f.is_file():
                continue
            if specific_task_id and specific_task_id in f.name:
                try:
                    f.unlink(missing_ok=True)
                except Exception:
                    pass
            elif max_age_seconds is not None:
                try:
                    if (now - f.stat().st_mtime) > max_age_seconds:
                        f.unlink(missing_ok=True)
                except Exception:
                    pass
    except Exception as e:
        print(f"[WARN] Error during temp file cleanup: {e}")


def start_janitor_thread():
    """Background janitor thread that regularly cleans expired temp files."""
    def janitor_loop():
        while True:
            time.sleep(300)  # Check every 5 minutes
            cleanup_temp_files(max_age_seconds=900)  # Evict files older than 15 min
    t = threading.Thread(target=janitor_loop, daemon=True)
    t.start()

start_janitor_thread()


def get_detector(conf_thresh=0.45, require_hardhat=True, require_vest=True, require_mask=False):
    """Safely loads the PPE detector."""
    try:
        from inference.ppe_detector import PPEDetector
        # Look for custom model weights across model/ and root directory
        candidate_weights = [
            MODEL_DIR / "best.pt",
            MODEL_DIR / "best_ppe_yolov8.pt",
            BASE_DIR / "best_ppe_yolov8.pt",
            BASE_DIR / "best.pt"
        ]
        valid_custom = next((p for p in candidate_weights if p.exists()), None)
        weights_str = str(valid_custom) if valid_custom else None

        detector = PPEDetector(
            weights_path=weights_str,
            conf_thresh=conf_thresh,
            require_hardhat=require_hardhat,
            require_vest=require_vest,
            require_mask=require_mask
        )
        return detector
    except Exception as e:
        print(f"[WARN] Could not instantiate PPEDetector: {e}")
        return None


def run_video_job(task_id, input_video_path, output_video_path, rules):
    """Executes the video inference task in a background worker thread."""
    task = tasks[task_id]
    task["status"] = "PROCESSING"
    task["start_time"] = time.time()

    detector = get_detector(
        conf_thresh=rules.get("conf_thresh", 0.45),
        require_hardhat=rules.get("require_hardhat", True),
        require_vest=rules.get("require_vest", True),
        require_mask=rules.get("require_mask", False)
    )

    if detector is None:
        task["status"] = "FAILED"
        task["error"] = "PPE Detector model could not be loaded."
        return

    def progress_callback(pct, frame_idx, total_frames, fps, frame_data):
        task["progress"] = pct
        task["frame"] = frame_idx
        task["total_frames"] = total_frames
        task["fps"] = round(fps, 1)
        if frame_data and frame_data.get("alerts"):
            for a in frame_data["alerts"]:
                task["recent_alerts"].append(a)
        task["alerts_count"] = len(task["recent_alerts"])

    try:
        results = detector.process_video(
            input_path=str(input_video_path),
            output_path=str(output_video_path),
            progress_callback=progress_callback
        )

        task["status"] = "COMPLETED"
        task["progress"] = 100
        task["alerts"] = results.get("alerts", [])
        task["compliance_rate"] = results.get("compliance_rate", 85.0)
        task["total_violations"] = results.get("total_violations", len(task["alerts"]))
        task["total_persons"] = max(1, len(set(a.get("person_id") for a in task["alerts"])) if task["alerts"] else 1)
        task["video_url"] = f"/api/video/{task_id}"
        print(f"[SUCCESS] Video processing task {task_id} completed successfully!")
    except Exception as e:
        print(f"[ERROR] Error during video processing: {e}")
        task["status"] = "FAILED"
        task["error"] = str(e)
    finally:
        # Immediately delete the raw uploaded input video - never retain on disk
        try:
            in_p = Path(input_video_path)
            if in_p.exists():
                in_p.unlink(missing_ok=True)
                print(f"[CLEANUP] Deleted ephemeral input video {in_p.name}")
        except Exception as e:
            print(f"[WARN] Could not delete input video: {e}")


class SafeGuardRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEBSITE_DIR), **kwargs)

    def do_GET(self):
        url_parts = urllib.parse.urlparse(self.path)
        path = url_parts.path

        if path.startswith("/api/status/"):
            task_id = path.replace("/api/status/", "").strip()
            self.send_task_status(task_id)
            return

        elif path.startswith("/api/video/"):
            task_id = path.replace("/api/video/", "").strip()
            self.serve_task_video(task_id)
            return

        elif path == "/api/model-status":
            self.send_model_status()
            return

        elif path.startswith("/static/"):
            self.serve_file_with_range(WEBSITE_DIR / path.lstrip("/"))
            return

        # Serve static web frontend
        if path == "/" or path == "":
            self.path = "/index.html"

        return super().do_GET()

    def do_POST(self):
        url_parts = urllib.parse.urlparse(self.path)
        path = url_parts.path

        if path == "/api/upload":
            self.handle_video_upload()
        elif path == "/api/cleanup":
            self.handle_cleanup()
        elif path == "/api/upload-weights":
            self.handle_weights_upload()
        else:
            self.send_error(404, "Endpoint not found")

    def send_json(self, data, status_code=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_task_status(self, task_id):
        if task_id not in tasks:
            self.send_json({"error": "Task not found"}, 404)
            return
        task = tasks[task_id]
        response = {
            "task_id": task_id,
            "status": task.get("status", "QUEUED"),
            "progress": task.get("progress", 0),
            "frame": task.get("frame", 0),
            "total_frames": task.get("total_frames", 0),
            "fps": task.get("fps", 0),
            "alerts_count": len(task.get("recent_alerts", [])),
            "recent_alerts": task.get("recent_alerts", []),
            "video_url": task.get("video_url"),
            "compliance_rate": task.get("compliance_rate"),
            "total_violations": task.get("total_violations"),
            "total_persons": task.get("total_persons"),
            "alerts": task.get("alerts", []),
            "error": task.get("error")
        }
        self.send_json(response)

    def send_model_status(self):
        candidate_weights = [
            MODEL_DIR / "best.pt",
            MODEL_DIR / "best_ppe_yolov8.pt",
            BASE_DIR / "best_ppe_yolov8.pt",
            BASE_DIR / "best.pt"
        ]
        best_pt = next((p for p in candidate_weights if p.exists()), None)
        is_custom = best_pt is not None
        model_name = best_pt.name if is_custom else "YOLOv8 Base (PPE Compliance Heuristics)"
        self.send_json({
            "is_custom": is_custom,
            "model_name": model_name,
            "path": str(best_pt) if is_custom else None
        })

    def handle_video_upload(self):
        """Parses multipart upload, saves to ephemeral temp storage, and starts inference job."""
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self.send_json({"error": "Expected multipart/form-data"}, 400)
            return

        boundary = content_type.split("boundary=")[-1].strip().encode("utf-8")
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length)

        # Simple multipart parser
        parts = raw_body.split(b"--" + boundary)
        video_data = None
        filename = "uploaded_video.mp4"
        rules = {"require_hardhat": True, "require_vest": True, "require_mask": False, "conf_thresh": 0.45}

        for part in parts:
            if b'name="video"' in part and b"filename=" in part:
                headers_and_body = part.split(b"\r\n\r\n", 1)
                if len(headers_and_body) == 2:
                    h, body = headers_and_body
                    video_data = body.rstrip(b"\r\n")
                    for line in h.decode("latin1", errors="ignore").split("\r\n"):
                        if "filename=" in line:
                            filename = line.split("filename=")[-1].strip('"\';')
            elif b'name="require_hardhat"' in part:
                rules["require_hardhat"] = b"true" in part.lower()
            elif b'name="require_vest"' in part:
                rules["require_vest"] = b"true" in part.lower()
            elif b'name="require_mask"' in part:
                rules["require_mask"] = b"true" in part.lower()
            elif b'name="conf_thresh"' in part:
                try:
                    val = part.split(b"\r\n\r\n")[-1].decode().strip()
                    rules["conf_thresh"] = float(val)
                except Exception:
                    pass

        if not video_data:
            self.send_json({"error": "No video data found in request"}, 400)
            return

        # Clean previous session temp files to prevent accumulation
        cleanup_temp_files(max_age_seconds=180)

        task_id = str(uuid.uuid4())
        ext = os.path.splitext(filename)[-1] or ".mp4"
        temp_input_path = TEMP_DIR / f"temp_in_{task_id}{ext}"
        temp_output_path = TEMP_DIR / f"temp_out_{task_id}.mp4"

        with open(temp_input_path, "wb") as f:
            f.write(video_data)

        tasks[task_id] = {
            "task_id": task_id,
            "status": "QUEUED",
            "progress": 0,
            "frame": 0,
            "total_frames": 0,
            "fps": 0,
            "recent_alerts": [],
            "input_path": str(temp_input_path),
            "output_path": str(temp_output_path)
        }

        thread = threading.Thread(
            target=run_video_job,
            args=(task_id, temp_input_path, temp_output_path, rules),
            daemon=True
        )
        thread.start()

        self.send_json({
            "task_id": task_id,
            "status": "QUEUED"
        })

    def handle_cleanup(self):
        """Immediately cleans up temporary video files when requested by client."""
        content_length = int(self.headers.get("Content-Length", 0))
        task_id = None
        if content_length > 0:
            try:
                body = self.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
                task_id = data.get("task_id")
            except Exception:
                pass

        if task_id:
            cleanup_temp_files(specific_task_id=task_id)
            tasks.pop(task_id, None)
            print(f"[CLEANUP] Purged temporary video for task: {task_id}")
        else:
            cleanup_temp_files(max_age_seconds=0)

        self.send_json({"success": True})

    def handle_weights_upload(self):
        """Saves custom weights uploaded from browser into model/best.pt."""
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self.send_json({"error": "Expected multipart/form-data"}, 400)
            return

        boundary = content_type.split("boundary=")[-1].strip().encode("utf-8")
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length)

        parts = raw_body.split(b"--" + boundary)
        weights_data = None
        for part in parts:
            if b'name="weights"' in part and b"filename=" in part:
                headers_and_body = part.split(b"\r\n\r\n", 1)
                if len(headers_and_body) == 2:
                    weights_data = headers_and_body[1].rstrip(b"\r\n")

        if weights_data:
            dest = MODEL_DIR / "best.pt"
            with open(dest, "wb") as f:
                f.write(weights_data)
            self.send_json({"success": True, "path": str(dest)})
        else:
            self.send_json({"error": "No weights file received"}, 400)

    def serve_task_video(self, task_id):
        """Streams the temporary annotated video file for task_id using HTTP Range support."""
        task = tasks.get(task_id)
        if task and "output_path" in task:
            local_path = Path(task["output_path"])
        else:
            local_path = TEMP_DIR / f"temp_out_{task_id}.mp4"

        if not local_path.exists():
            self.send_error(404, "Temporary video not found or expired")
            return

        self.serve_file_with_range(local_path)

    def serve_file_with_range(self, local_path):
        """Serves video files with HTTP Range 206 Partial Content support for native browser seeking."""
        local_path = Path(local_path)
        if not local_path.exists():
            self.send_error(404, "File not found")
            return

        file_size = local_path.stat().st_size
        range_header = self.headers.get("Range")

        mime_type, _ = mimetypes.guess_type(str(local_path))
        if not mime_type:
            mime_type = "video/mp4"

        if range_header:
            range_val = range_header.replace("bytes=", "").strip()
            parts = range_val.split("-")
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if parts[1] else file_size - 1

            if start >= file_size or end >= file_size:
                self.send_response(416, "Requested Range Not Satisfiable")
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                return

            length = end - start + 1

            self.send_response(206, "Partial Content")
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

            with open(local_path, "rb") as f:
                f.seek(start)
                self.wfile.write(f.read(length))
        else:
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

            with open(local_path, "rb") as f:
                self.wfile.write(f.read())


def start_server(port=8000):
    server_address = ("127.0.0.1", port)
    httpd = ThreadingHTTPServer(server_address, SafeGuardRequestHandler)
    print("=" * 65)
    print(f"[SAFEGUARD AI] PPE Kit Detection & Live Alert Web Server")
    print(f"[RUNNING] Running locally at: http://localhost:{port}")
    print(f"[ROOT]    Website Root:     {WEBSITE_DIR}")
    print(f"[STORAGE] Ephemeral Temp:   {TEMP_DIR} (Auto-cleaned, no permanent storage)")
    print("=" * 65)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        cleanup_temp_files(max_age_seconds=0)
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    start_server(port)
