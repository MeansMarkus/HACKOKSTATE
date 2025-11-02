# sign_to_text.py — Sign language recognition with text output
import cv2, time, os
import numpy as np
from collections import deque

# -------- CONFIG --------
CAMERA_INDEX = 0
FRAME_SKIP   = 1
DRAW_ROI     = False
ROI = np.array([[200, 200], [1100, 200], [1100, 700], [200, 700]], dtype=np.int32)
WINDOW_NAME  = "Sign Language to Text"

LOAD_CLASSIFIER = True
CLASSIFIER_PATH = "sign_clf.pkl"

# Text recognition settings
STABILITY_FRAMES = 15        # Frames sign must be stable before adding to text (reduced for testing)
WORD_TIMEOUT = 2.0          # Seconds without signs = add space
CLEAR_GESTURE = None        # Set to a sign label to clear text (e.g., "CLEAR")

# -------- Helpers --------
def inside(pt, poly):
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0

# -------- Text Manager --------
class TextBuilder:
    def __init__(self):
        self.text = ""
        self.current_sign = None
        self.stable_count = 0
        self.last_sign_time = 0
        self.last_added_sign = None
        self.recent_history = deque(maxlen=20)  # Track recent predictions
        
    def add_prediction(self, label):
        """Process a new sign prediction"""
        current_time = time.time()
        self.recent_history.append(label)
        
        # Check if sign changed
        if label != self.current_sign:
            self.current_sign = label
            self.stable_count = 1
            print(f"[TextBuilder] New sign: {label}")
        else:
            self.stable_count += 1
        
        # Add space if timeout occurred
        if self.last_sign_time > 0 and current_time - self.last_sign_time > WORD_TIMEOUT:
            if self.text and not self.text.endswith(" "):
                self.text += " "
                self.last_added_sign = None
                print("[TextBuilder] Added space (timeout)")
        
        # If sign is stable enough and different from last added
        if self.stable_count >= STABILITY_FRAMES and label != self.last_added_sign:
            # Check for clear gesture
            if CLEAR_GESTURE and label == CLEAR_GESTURE:
                self.text = ""
                self.last_added_sign = None
                print("[TextBuilder] Cleared text")
            else:
                self.text += label
                self.last_added_sign = label
                print(f"[TextBuilder] Added '{label}' to text. Current text: '{self.text}'")
            
            self.stable_count = 0  # Reset counter after adding
        
        self.last_sign_time = current_time
        return self.text
    
    def no_hands_detected(self):
        """Call when no hands are detected"""
        self.current_sign = None
        self.stable_count = 0
    
    def backspace(self):
        """Remove last character"""
        if self.text:
            self.text = self.text[:-1]
    
    def add_space(self):
        """Manually add space"""
        if self.text and not self.text.endswith(" "):
            self.text += " "
    
    def clear(self):
        """Clear all text"""
        self.text = ""
        self.last_added_sign = None

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
        clf, LABELS = load(CLASSIFIER_PATH)
        print(f"✅ Loaded classifier: {CLASSIFIER_PATH}")
        print(f"   Labels: {LABELS}")
        print(f"   Classifier type: {type(clf)}")
        
        # Test the classifier with dummy data
        test_feat = np.random.rand(1, 42)
        test_pred = clf.predict(test_feat)
        print(f"   Test prediction works: {test_pred}")
        
    except Exception as e:
        print(f"❌ Classifier loading failed: {e}")
        print(f"   Current directory: {os.getcwd()}")
        print(f"   Looking for: {CLASSIFIER_PATH}")
        print("   Will draw landmarks only.")
        clf, LABELS = None, None

# -------- Text Builder --------
text_builder = TextBuilder()

print("\n" + "="*60)
print("CONTROLS:")
print("  ESC      - Quit")
print("  SPACE    - Add space to text")
print("  BACKSPACE- Delete last character")
print("  C        - Clear all text")
print("="*60)
print("▶️ Running...\n")

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
            key = cv2.waitKey(1) & 0xFF
            if key == 27: break
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

        # ---- Hands Detection ----
        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        hands_count = 0
        current_prediction = None
        
        if res.multi_hand_landmarks:
            print(f"[DEBUG Frame {frame_idx}] Found {len(res.multi_hand_landmarks)} hands")
            for idx, handlms in enumerate(res.multi_hand_landmarks):
                # 21 normalized landmarks
                pts = np.array([[lm.x, lm.y] for lm in handlms.landmark], dtype=np.float32)

                # Pixel coords
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

                        # Classification
                if clf is not None and LABELS is not None:
                    # Normalize landmarks
                    pts_norm = pts.copy()
                    anchor   = pts_norm[0].copy()
                    pts_norm -= anchor
                    scale    = np.linalg.norm(pts_norm, axis=1).max() + 1e-6
                    pts_norm /= scale
                    feat = pts_norm.flatten()[None, :]

                    try:
                        pred  = clf.predict(feat)[0]
                        label = LABELS[pred] if isinstance(pred, (int, np.integer)) else str(pred)
                        current_prediction = label
                        
                        # DEBUG: Print to console
                        if frame_idx % 10 == 0:  # Every 10 frames
                            print(f"Pred: {label}, Stable: {text_builder.stable_count}/{STABILITY_FRAMES}")
                        
                        # Display current sign
                        cv2.putText(view, f"{label}", (cx + 10, cy - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                    except Exception as e:
                        print(f"Classification error: {e}")
                        label = "?"
        
        # Update text builder
        if current_prediction:
            print(f"[DEBUG Frame {frame_idx}] Sending prediction to TextBuilder: {current_prediction}")
            text_builder.add_prediction(current_prediction)
        else:
            text_builder.no_hands_detected()
            if frame_idx % 30 == 0:
                print(f"[DEBUG Frame {frame_idx}] No hands detected")

        # ---- Draw UI ----
        # Text output box (top of screen)
        box_height = 100
        overlay = view.copy()
        cv2.rectangle(overlay, (0, 0), (view.shape[1], box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, view, 0.3, 0, view)
        
        # Display accumulated text
        text_to_display = text_builder.text if text_builder.text else "[No text yet]"
        # Wrap text if too long
        max_chars = 80
        if len(text_to_display) > max_chars:
            text_to_display = "..." + text_to_display[-max_chars:]
        
        cv2.putText(view, text_to_display, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Status info
        status = f"Current: {current_prediction or 'None'} | Stability: {text_builder.stable_count}/{STABILITY_FRAMES}"
        cv2.putText(view, status, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Hand count
        cv2.putText(view, f"Hands: {hands_count}", (view.shape[1] - 180, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Show
        cv2.imshow(WINDOW_NAME, view)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # SPACE
            text_builder.add_space()
        elif key == 8:  # BACKSPACE
            text_builder.backspace()
        elif key == ord('c') or key == ord('C'):  # C
            text_builder.clear()

except KeyboardInterrupt:
    print("\n🛑 Interrupted by user.")
finally:
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final text
    print("\n" + "="*60)
    print("FINAL TEXT:")
    print(text_builder.text if text_builder.text else "[No text captured]")
    print("="*60)
    print("✅ Clean exit.")