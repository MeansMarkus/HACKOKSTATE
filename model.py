# sign_live_simple.py  — no tracking IDs, live webcam only
import cv2, time, os
import numpy as np

# -------- CONFIG --------
CAMERA_INDEX = 0
FRAME_SKIP   = 1          # 1 = every frame, 2 = every other, etc.
DRAW_ROI     = False
ROI = np.array([[200, 200], [1100, 200], [1100, 700], [200, 700]], dtype=np.int32)
WINDOW_NAME  = "Sign (Hands) Live"

LOAD_CLASSIFIER = True
CLASSIFIER_PATH = "sign_clf.pkl"  # expects (clf, LABELS) via joblib

# -------- Helpers --------
def inside(pt, poly):
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0

# -------- Camera --------
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"❌ Could not open camera index {CAMERA_INDEX}")

w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1280
h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
fps = cap.get(cv2.CAP_PROP_FPS) or 30
print(f"🎥 Camera opened: {w}x{h} @ {fps:.1f} FPS")

# -------- MediaPipe Hands --------
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

# -------- Optional classifier --------
clf, LABELS = None, None
if LOAD_CLASSIFIER:
    try:
        from joblib import load
        clf, LABELS = load(CLASSIFIER_PATH)  # (sklearn_model, list_of_label_names)
        print(f"✅ Loaded classifier: {CLASSIFIER_PATH}")
    except Exception:
        print("ℹ️ No classifier found (sign_clf.pkl). Will draw landmarks only.")

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
            cv2.imshow(WINDOW_NAME, frame)
            if (cv2.waitKey(1) & 0xFF) == 27: break
            continue

        view = frame.copy()

        # Optional ROI
        if DRAW_ROI and ROI is not None:
            cv2.polylines(view, [ROI], True, (0, 255, 0), 2)
            mask = np.zeros_like(view)
            cv2.fillPoly(mask, [ROI], (255, 255, 255))
            proc = cv2.bitwise_and(view, mask)
        else:
            proc = view

        # ---- Hands ----
        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        hands_count = 0
        if res.multi_hand_landmarks:
            # Handedness (Left/Right) list if available
            handedness_list = []
            if getattr(res, "multi_handedness", None):
                handedness_list = [h.classification[0].label for h in res.multi_handedness]
            else:
                handedness_list = ["Unknown"] * len(res.multi_hand_landmarks)

            for idx, handlms in enumerate(res.multi_hand_landmarks):
                # 21 normalized landmarks (x,y) in [0,1]
                pts = np.array([[lm.x, lm.y] for lm in handlms.landmark], dtype=np.float32)

                # Pixel coords for drawing & center
                pts_px = pts.copy()
                pts_px[:, 0] *= view.shape[1]
                pts_px[:, 1] *= view.shape[0]
                pts_px = pts_px.astype(int)
                cx, cy = int(pts_px[:, 0].mean()), int(pts_px[:, 1].mean())

                # Optional ROI filter
                if DRAW_ROI and ROI is not None and inside((cx, cy), ROI) < 0:
                    continue

                hands_count += 1

                # Draw landmarks
                mp_draw.draw_landmarks(view, handlms, mp.solutions.hands.HAND_CONNECTIONS)
                cv2.circle(view, (cx, cy), 6, (255, 0, 0), -1)

                # Optional classification
                if clf is not None and LABELS is not None:
                    # Normalize landmarks: translate to wrist (id=0), scale by max distance
                    pts_norm = pts.copy()
                    anchor   = pts_norm[0].copy()           # wrist
                    pts_norm -= anchor
                    scale    = np.linalg.norm(pts_norm, axis=1).max() + 1e-6
                    pts_norm /= scale
                    feat = pts_norm.flatten()[None, :]      # (1,42)

                    try:
                        pred  = clf.predict(feat)[0]
                        label = LABELS[pred] if isinstance(pred, (int, np.integer)) else str(pred)
                    except Exception:
                        label = "?"
                    cv2.putText(view, f"{label}", (cx + 10, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # HUD
        cv2.putText(view, f"Hands: {hands_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Show
        cv2.imshow(WINDOW_NAME, view)
        if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
            break

except KeyboardInterrupt:
    print("\n🛑 Interrupted by user.")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Clean exit.")
