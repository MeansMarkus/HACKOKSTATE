# sign_trainer.py — Train classifier from collected data
import json
import numpy as np
import os
from glob import glob

# -------- CONFIG --------
DATA_DIR = "sign_data"
OUTPUT_FILE = "sign_clf.pkl"
TEST_SPLIT = 0.2  # 20% for testing

print("="*70)
print("SIGN LANGUAGE CLASSIFIER TRAINER")
print("="*70)

# -------- Load Data --------
print("\n📂 Looking for training data...")
data_files = glob(os.path.join(DATA_DIR, "dataset_*.json"))

if not data_files:
    print(f"❌ No training data found in {DATA_DIR}/")
    print("   Please run sign_collector.py first to collect training data.")
    exit(1)

print(f"Found {len(data_files)} dataset file(s):")
for f in data_files:
    print(f"  - {f}")

# Load all data
all_samples = []
for data_file in data_files:
    with open(data_file, 'r') as f:
        samples = json.load(f)
        all_samples.extend(samples)
        print(f"  Loaded {len(samples)} samples from {os.path.basename(data_file)}")

print(f"\n✅ Total samples loaded: {len(all_samples)}")

# -------- Prepare Dataset --------
X = []  # Features (landmarks)
y = []  # Labels (signs)

for sample in all_samples:
    X.append(sample['landmarks'])
    y.append(sample['sign'])

X = np.array(X, dtype=np.float32)
y = np.array(y)

print(f"\nDataset shape: {X.shape}")
print(f"Feature dimensions: {X.shape[1]}")

# Get unique labels
unique_labels = sorted(list(set(y)))
print(f"\nSigns in dataset: {unique_labels}")
print(f"Number of classes: {len(unique_labels)}")

# Show class distribution
print("\nSamples per sign:")
for label in unique_labels:
    count = np.sum(y == label)
    print(f"  {label}: {count} samples")

# -------- Split Train/Test --------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SPLIT, random_state=42, stratify=y
)

print(f"\n📊 Train/Test Split:")
print(f"  Training samples: {len(X_train)}")
print(f"  Testing samples: {len(X_test)}")

# -------- Train Classifier --------
print("\n🤖 Training classifier...")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Train Random Forest
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train, y_train)
print("✅ Training complete!")

# -------- Evaluate --------
print("\n📈 Evaluating on test set...")

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🎯 Test Accuracy: {accuracy*100:.2f}%")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# -------- Save Model --------
print(f"\n💾 Saving model to {OUTPUT_FILE}...")

try:
    from joblib import dump
    # Save as tuple: (classifier, label_list)
    dump((clf, unique_labels), OUTPUT_FILE)
    print(f"✅ Model saved successfully!")
    print(f"\nYou can now use this classifier with sign_to_text.py")
except Exception as e:
    print(f"❌ Error saving model: {e}")
    print("   Install joblib: pip install joblib")

# -------- Feature Importance (optional) --------
print("\n📊 Top 10 Most Important Landmarks:")
importances = clf.feature_importances_
# Each landmark has 2 features (x, y), so we sum pairs
landmark_importance = []
for i in range(0, len(importances), 2):
    landmark_importance.append(importances[i] + importances[i+1])

landmark_names = [
    "Wrist", "Thumb_CMC", "Thumb_MCP", "Thumb_IP", "Thumb_Tip",
    "Index_MCP", "Index_PIP", "Index_DIP", "Index_Tip",
    "Middle_MCP", "Middle_PIP", "Middle_DIP", "Middle_Tip",
    "Ring_MCP", "Ring_PIP", "Ring_DIP", "Ring_Tip",
    "Pinky_MCP", "Pinky_PIP", "Pinky_DIP", "Pinky_Tip"
]

top_landmarks = sorted(zip(landmark_names, landmark_importance), 
                       key=lambda x: x[1], reverse=True)[:10]

for name, importance in top_landmarks:
    print(f"  {name:15s}: {importance:.4f}")

print("\n" + "="*70)
print("✅ Training complete! Your classifier is ready to use.")
print("="*70)