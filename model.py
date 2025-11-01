# sign_live_tracker.py
import cv2, time, math, os
import numpy as np

# ------------------------
# CONFIG (edit as needed)
# ------------------------
CAMERA_INDEX = 0          # 0 = default webcam; try 1/2 if you have multiple
FRAME_SKIP = 1            # analyze every Nth frame (1 = every frame, 2 = every other, etc.)
MAX_MATCH_DIST = 80       # px: nearest-neighbor hand ID matching threshold
TRACK_TTL_SEC = 1.2       # seconds to keep a hand track after last seen
DRAW_ROI = False          # set True and adjust ROI if you want a specific region
ROI = np.array([[200, 200], [1100, 200], [1100, 700], [200, 700]], dtype=np.int32)
WINDOW_NAME = "Sign (Hands) Live"

# Try to load an optional classifier for letters/keywords (joblib: (clf, LABELS))
LOAD_CLASSIFIER = True
CLASSIFIER_PATH = "sign_clf.pkl"  # produced by your own trainer (optional)

# ------------------------
# Helpers
# ------------------------
def inside(pt, poly):
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0

def l2(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])

# ------------------------
# Init: Camera
# ------------------------
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"❌ Could not open camera index {CAMERA_INDEX}")

w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1280
h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
fps = cap.get(cv2.CAP_PROP_FPS) or 30
print(f"🎥 Camera opened: {w}x{h} @ {fps:.1f} FPS")

# ------------------------
# Init: MediaPipe Hands
# ------------------------
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=2,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)
except Exception as e:
    raise RuntimeError("⚠️ Install mediapipe: pip install mediapipe") from e

# ------------------------
# Optional classifier
# ------------------------
clf = None
LABELS = None
if LOAD_CLASSIFIER:
    try:
        from joblib import load
        clf, LABELS = load(CLASSIFIER_PATH)  # expects (sklearn_model, list_of_label_names)
        print(f"✅ Loaded classifier: {CLASSIFIER_PATH}")
    except Exception:
        print("ℹ️ No classifier found (sign_clf.pkl). Landmarks will be drawn without labels.")

# ------------------------
# Simple hand tracker (nearest-neighbor data association)
# tracks: id -> {center, last, handed, landmarks_norm}
# ------------------------
from collections import OrderedDict
hand_tracks = OrderedDict()
next_id = 1

print("▶️ Running. Press ESC to quit.")

try:
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("⚠️ Camera read failed.")
            break
        frame_idx += 1
        if FRAME_SKIP > 1 and (frame_idx % FRAME_SKIP != 0):
            # skip processing this frame to save compute
            cv2.imshow(WINDOW_NAME, frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
            continue

        view = frame.copy()
        now = time.time()

        # Optional ROI drawing/masking
        if DRAW_ROI and ROI is not None:
            cv2.polylines(view, [ROI], True, (0, 255, 0), 2)
            mask = np.zeros_like(view)
            cv2.fillPoly(mask, [ROI], (255, 255, 255))
            proc = cv2.bitwise_and(view, mask)
        else:
            proc = view

        # ---- MediaPipe Hands ----
        current_hands = []  # list of {center:(x,y), landmarks_norm:(21,2) in [0,1], handed:str}
        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            handedness_list = []
            if getattr(res, "multi_handedness", None):
                handedness_list = [h.classification[0].label for h in res.multi_handedness]  # "Left"/"Right"
            else:
                handedness_list = ["Unknown"] * len(res.multi_hand_landmarks)

            for idx, handlms in enumerate(res.multi_hand_landmarks):
                # Normalized coords in [0,1]
                pts = np.array([[lm.x, lm.y] for lm in handlms.landmark], dtype=np.float32)
                # Pixel coords for drawing
                pts_px = pts.copy()
                pts_px[:, 0] *= view.shape[1]
                pts_px[:, 1] *= view.shape[0]
                pts_px = pts_px.astype(int)

                cx, cy = int(pts_px[:, 0].mean()), int(pts_px[:, 1].mean())
                handed = handedness_list[idx] if idx < len(handedness_list) else "Unknown"

                # Draw landmarks
                mp_draw.draw_landmarks(view, handlms, mp.solutions.hands.HAND_CONNECTIONS)
                cv2.circle(view, (cx, cy), 6, (255, 0, 0), -1)

                # Optional ROI filter
                if DRAW_ROI and ROI is not None and inside((cx, cy), ROI) < 0:
                    continue

                current_hands.append({
                    "center": (cx, cy),
                    "landmarks_norm": pts,
                    "handed": handed
                })

        # ---- Associate to tracks (greedy nearest neighbor + handedness) ----
        used_tracks = set()
        for hcur in current_hands:
            best_tid, best_d = None, 1e9
            for tid, tinfo in hand_tracks.items():
                # Prefer matching same handedness if known
                if tinfo.get("handed") != "Unknown" and hcur["handed"] != "Unknown":
                    if tinfo["handed"] != hcur["handed"]:
                        continue
                d = l2(hcur["center"], tinfo["center"])
                if d < best_d and d <= MAX_MATCH_DIST:
                    best_tid, best_d = tid, d

            if best_tid is not None and best_tid not in used_tracks:
                # Update existing
                hand_tracks[best_tid]["center"] = hcur["center"]
                hand_tracks[best_tid]["last"] = now
                hand_tracks[best_tid]["handed"] = hcur["handed"]
                hand_tracks[best_tid]["landmarks_norm"] = hcur["landmarks_norm"]
                used_tracks.add(best_tid)
            else:
                # New track
                hand_tracks[next_id] = {
                    "center": hcur["center"],
                    "last": now,
                    "handed": hcur["handed"],
                    "landmarks_norm": hcur["landmarks_norm"]
                }
                used_tracks.add(next_id)
                next_id += 1

        # Expire old tracks
        for tid in list(hand_tracks.keys()):
            if now - hand_tracks[tid]["last"] > TRACK_TTL_SEC:
                del hand_tracks[tid]

        # ---- Optional: classify each hand (requires sign_clf.pkl)
        if clf is not None and LABELS is not None:
            for tid, tinfo in hand_tracks.items():
                pts = tinfo["landmarks_norm"].copy()  # (21,2) in [0,1]
                # Normalize: translate to wrist (landmark 0) and scale by max distance
                anchor = pts[0].copy()                # wrist
                pts -= anchor
                scale = np.linalg.norm(pts, axis=1).max() + 1e-6
                pts /= scale
                feat = pts.flatten()[None, :]         # (1,42)

                try:
                    pred = clf.predict(feat)[0]
                    label = LABELS[pred] if isinstance(pred, (int, np.integer)) else str(pred)
                except Exception:
                    label = "?"
                x, y = map(int, tinfo["center"])
                cv2.putText(view, f"{label}", (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Visualize track IDs and count
        for tid, tinfo in hand_tracks.items():
            x, y = map(int, tinfo["center"])
            cv2.putText(view, f"Hand ID {tid} ({tinfo['handed']})", (x + 10, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        count = len(hand_tracks)
        cv2.putText(view, f"Hands tracked: {count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Show
        cv2.imshow(WINDOW_NAME, view)
        if (cv2.waitKey(1) & 0xFF) == 27:  # ESC to quit
            break

except KeyboardInterrupt:
    print("\n🛑 Interrupted by user.")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Clean exit.")
