# sign_collector.py — Collect training data for sign language recognition
import cv2
import numpy as np
import os
import json
from datetime import datetime

# -------- CONFIG --------
CAMERA_INDEX = 0
WINDOW_NAME = "Sign Language Collector"
DATA_DIR = "sign_data"
SAMPLES_PER_SIGN = 200  # Increased from 100 - more data = better accuracy
DATA_AUGMENTATION = True  # Add slight variations to improve robustness

# Signs to collect (customize this list!)
SIGNS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
         'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# -------- Setup --------
os.makedirs(DATA_DIR, exist_ok=True)

# -------- MediaPipe Hands --------
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False,
                          max_num_hands=1,  # One hand at a time for training
                          min_detection_confidence=0.7,
                          min_tracking_confidence=0.7)
except Exception as e:
    raise RuntimeError("⚠️ Install mediapipe: pip install mediapipe") from e

# -------- Camera --------
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"❌ Could not open camera index {CAMERA_INDEX}")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
print(f"🎥 Camera opened: {w}x{h}")

# -------- Collection State --------
current_sign_idx = 0
collecting = False
collected_count = 0
all_data = []

def save_data():
    """Save collected data to file"""
    if not all_data:
        print("No data to save!")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(DATA_DIR, f"dataset_{timestamp}.json")
    
    with open(filename, 'w') as f:
        json.dump(all_data, f)
    
    print(f"\n✅ Saved {len(all_data)} samples to {filename}")
    return filename

print("\n" + "="*70)
print("SIGN LANGUAGE DATA COLLECTOR")
print("="*70)
print(f"Will collect {SAMPLES_PER_SIGN} samples for each of {len(SIGNS)} signs")
print("\nCONTROLS:")
print("  SPACE     - Start/Stop collecting current sign")
print("  N         - Next sign (skip current)")
print("  P         - Previous sign")
print("  S         - Save data and quit")
print("  ESC/Q     - Quit without saving")
print("="*70)
print(f"\nReady to collect sign: {SIGNS[current_sign_idx]}")
print("Press SPACE to start collecting...\n")

try:
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("⚠️ Camera read failed.")
            break
        frame_idx += 1

        view = frame.copy()
        
        # Detect hands
        rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        hand_detected = False
        landmarks_data = None

        if res.multi_hand_landmarks:
            for handlms in res.multi_hand_landmarks:
                hand_detected = True
                
                # Extract normalized landmarks
                pts = np.array([[lm.x, lm.y] for lm in handlms.landmark], dtype=np.float32)
                
                # Normalize: translate to wrist, scale by max distance
                pts_norm = pts.copy()
                anchor = pts_norm[0].copy()  # wrist
                pts_norm -= anchor
                scale = np.linalg.norm(pts_norm, axis=1).max() + 1e-6
                pts_norm /= scale
                
                landmarks_data = pts_norm.flatten().tolist()  # 42 features
                
                # Draw landmarks
                mp_draw.draw_landmarks(view, handlms, mp.solutions.hands.HAND_CONNECTIONS)
                
                # Draw center
                pts_px = pts.copy()
                pts_px[:, 0] *= view.shape[1]
                pts_px[:, 1] *= view.shape[0]
                pts_px = pts_px.astype(int)
                cx, cy = int(pts_px[:, 0].mean()), int(pts_px[:, 1].mean())
                cv2.circle(view, (cx, cy), 8, (0, 255, 0), -1)
                
                break  # Only process first hand

        # Collect data if in collection mode
        if collecting and hand_detected and landmarks_data:
            current_sign = SIGNS[current_sign_idx]
            
            # Store original sample
            all_data.append({
                'sign': current_sign,
                'landmarks': landmarks_data
            })
            collected_count += 1
            
            # Optional: Add augmented samples (small random variations)
            if DATA_AUGMENTATION and collected_count % 3 == 0:  # Every 3rd sample
                # Create 2 augmented versions with small noise
                for _ in range(2):
                    noise = np.random.normal(0, 0.02, len(landmarks_data))  # Small noise
                    augmented = [float(x + n) for x, n in zip(landmarks_data, noise)]
                    all_data.append({
                        'sign': current_sign,
                        'landmarks': augmented
                    })
            
            # Auto-stop when enough samples collected
            if collected_count >= SAMPLES_PER_SIGN:
                collecting = False
                print(f"✅ Collected {collected_count} samples for '{current_sign}'")
                collected_count = 0
                
                # Auto-advance to next sign
                if current_sign_idx < len(SIGNS) - 1:
                    current_sign_idx += 1
                    print(f"\n➡️  Next sign: {SIGNS[current_sign_idx]}")
                    print("Press SPACE to start collecting...")
                else:
                    print("\n🎉 All signs collected! Press S to save.")

        # ---- Draw UI ----
        # Status bar
        bar_height = 120
        overlay = view.copy()
        cv2.rectangle(overlay, (0, 0), (view.shape[1], bar_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, view, 0.2, 0, view)
        
        # Current sign (large)
        current_sign = SIGNS[current_sign_idx]
        cv2.putText(view, f"Sign: {current_sign}", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        
        # Progress
        progress_text = f"{current_sign_idx + 1}/{len(SIGNS)}"
        cv2.putText(view, progress_text, (view.shape[1] - 150, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Collection status
        if collecting:
            status = f"COLLECTING: {collected_count}/{SAMPLES_PER_SIGN}"
            color = (0, 255, 0)
            # Progress bar
            bar_width = 400
            bar_x = (view.shape[1] - bar_width) // 2
            cv2.rectangle(view, (bar_x, 85), (bar_x + bar_width, 105), (100, 100, 100), -1)
            fill_width = int((collected_count / SAMPLES_PER_SIGN) * bar_width)
            cv2.rectangle(view, (bar_x, 85), (bar_x + fill_width, 105), (0, 255, 0), -1)
        else:
            status = "Ready - Press SPACE to collect"
            color = (200, 200, 200)
        
        cv2.putText(view, status, (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Hand detection indicator
        hand_status = "✓ Hand detected" if hand_detected else "✗ No hand"
        hand_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        cv2.putText(view, hand_status, (view.shape[1] - 220, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)
        
        # Total samples collected
        cv2.putText(view, f"Total samples: {len(all_data)}", (20, view.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow(WINDOW_NAME, view)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27 or key == ord('q') or key == ord('Q'):  # ESC or Q
            print("\n❌ Quit without saving")
            break
            
        elif key == ord(' '):  # SPACE - toggle collection
            if not collecting:
                collecting = True
                collected_count = 0
                print(f"📹 Collecting '{current_sign}'... Hold your sign steady!")
            else:
                collecting = False
                print(f"⏸️  Paused at {collected_count} samples")
                
        elif key == ord('n') or key == ord('N'):  # Next sign
            if current_sign_idx < len(SIGNS) - 1:
                collecting = False
                collected_count = 0
                current_sign_idx += 1
                print(f"\n➡️  Skipped to: {SIGNS[current_sign_idx]}")
                
        elif key == ord('p') or key == ord('P'):  # Previous sign
            if current_sign_idx > 0:
                collecting = False
                collected_count = 0
                current_sign_idx -= 1
                print(f"\n⬅️  Back to: {SIGNS[current_sign_idx]}")
                
        elif key == ord('s') or key == ord('S'):  # Save and quit
            filename = save_data()
            print("\n✅ Data saved! You can now run the trainer.")
            break

except KeyboardInterrupt:
    print("\n🛑 Interrupted by user.")
    save_data()
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Clean exit.")