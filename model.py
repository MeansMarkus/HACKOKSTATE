# yolo_parking_tracker.py
from ultralytics import YOLO
import cv2, time, numpy as np
import os

# ------------------------
# CONFIG
# ------------------------
SOURCE = 0                           # 0 = default webcam; or "/path/to/video.mp4"; or "http://<phone-ip>:8080/video"
MODEL  = "yolov8n.pt"                # or your custom Roboflow/YOLO weights .pt
OUT    = "annotated.mp4"             # output video file
IMG_SIZE = 832                       # detector input size (832/1024 helps with small cars)
CONF    = 0.25
IOU     = 0.6
FRAME_SKIP = 1                       # analyze every Nth frame; 1 = every frame, 5 = every 5th frame
MAX_FRAMES = None                    # e.g., 300 to stop early; None to process all
TTL_SEC = 2.5                        # seconds to keep a track id after last seen
SHOW_EVERY_N_ANALYZED = 1            # display every Nth analyzed frame (viz throttling)

# ROI polygon in *video coordinates* (clockwise or ccw). Edit to your view.
# Example below puts a wide box near the bottom; adjust for your scene or set ROI=None to disable.
ROI = np.array([[100, 550], [1220, 550], [1220, 980], [100, 980]], dtype=np.int32)
USE_ROI = True                       # set False to count anywhere in frame

# ------------------------
# ENV/Display handling
# ------------------------
IS_COLAB = False
try:
    import google.colab  # type: ignore
    from google.colab.patches import cv2_imshow
    IS_COLAB = True
except Exception:
    pass

def show_frame(win_name, frame):
    if IS_COLAB:
        # In Colab: display stills; no live window support
        from google.colab.patches import cv2_imshow  # safe import
        cv2_imshow(frame)
    else:
        cv2.imshow(win_name, frame)

def key_pressed():
    # Return True if ESC pressed (local). In Colab always False.
    if IS_COLAB:
        return False
    return (cv2.waitKey(1) & 0xFF) == 27  # ESC

# ------------------------
# Helpers
# ------------------------
def inside(pt, poly):
    # pt=(x,y); poly=Nx2 array
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0

# ------------------------
# Initialize
# ------------------------
model = YOLO(MODEL)

cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    raise RuntimeError(f"❌ Could not open video source: {SOURCE}")

w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1280
h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
fps= cap.get(cv2.CAP_PROP_FPS) or 30

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUT, fourcc, fps, (w, h))

seen = {}   # track_id -> last_seen_time
frame_idx = 0
analyzed_counter = 0

print("▶️ Starting. Press ESC to stop (local runs).")
start_time = time.time()

try:
    # Use Ultralytics internal reader with tracker + frame skipping
    for r in model.track(
            source=SOURCE,
            tracker="bytetrack.yaml",
            conf=CONF,
            iou=IOU,
            persist=True,
            stream=True,
            imgsz=IMG_SIZE,
            vid_stride=max(1, FRAME_SKIP)):  # skip frames efficiently
        # r contains results for the current processed frame
        frame_idx += 1
        analyzed_counter += 1

        frame = r.orig_img.copy()
        now = time.time()

        # Draw ROI
        if USE_ROI and ROI is not None:
            cv2.polylines(frame, [ROI], True, (0, 255, 0), 2)

        # Update 'seen' dict with active tracks that fall inside ROI (or whole frame)
        if r.boxes.id is not None:
            ids  = r.boxes.id.int().cpu().tolist()
            xyxy = r.boxes.xyxy.cpu().numpy()
            for tid, box in zip(ids, xyxy):
                x1, y1, x2, y2 = box.astype(int)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                in_region = True
                if USE_ROI and ROI is not None:
                    in_region = inside((cx, cy), ROI)

                if in_region:
                    seen[tid] = now
                    # draw box + id
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID {tid}", (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Expire old IDs
        for tid in list(seen.keys()):
            if now - seen[tid] > TTL_SEC:
                del seen[tid]

        count = len(seen)
        cv2.putText(frame, f"Cars in lot: {count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        writer.write(frame)

        # Display (throttled)
        if (analyzed_counter % SHOW_EVERY_N_ANALYZED) == 0:
            disp = frame if (w <= 1280) else cv2.resize(frame, (1280, int(1280 * h / w)))
            show_frame("YOLO Parking Counter", disp)

        # Early stop conditions
        if MAX_FRAMES is not None and analyzed_counter >= MAX_FRAMES:
            print(f"⏹️ Stopped after {analyzed_counter} analyzed frames.")
            break
        if key_pressed():  # ESC locally
            print("⏹️ Stopped by user (ESC).")
            break

finally:
    writer.release()
    cap.release()
    if not IS_COLAB:
        cv2.destroyAllWindows()

elapsed = time.time() - start_time
print(f"✅ Done. Saved annotated video to: {os.path.abspath(OUT)}  |  Elapsed: {elapsed:.1f}s")
