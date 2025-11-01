from flask import Flask, jsonify, request
from flask_cors import CORS
import firebase_admin
from firebase_admin import firestore
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Initialize Firebase (if credentials available)
try:
    if os.path.exists('serviceAccountKey.json'):
        cred = firebase_admin.credentials.Certificate('serviceAccountKey.json')
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("Firebase initialized successfully")
    else:
        print("Firebase credentials not found - running in mock mode")
        db = None
except Exception as e:
    print(f"Firebase initialization failed: {e}")
    db = None

# Basic health check route
@app.route('/')
def home():
    return jsonify({
        "message": "🚗 Parking Helper API is running!",
        "status": "success",
        "endpoints": {
            "health": "/api/health",
            "parking_spots": "/api/parking-spots"
        }
    })

@app.route('/api/health')
def health_check():
    return jsonify({"status": "healthy", "service": "parking-helper-api"})

# Mock parking spots data
@app.route('/api/parking-spots')
def get_parking_spots():
    mock_spots = [
        {"id": 1, "number": 1, "occupied": False, "confidence": 0.95},
        {"id": 2, "number": 2, "occupied": True, "confidence": 0.87},
        {"id": 3, "number": 3, "occupied": False, "confidence": 0.92},
        {"id": 4, "number": 4, "occupied": True, "confidence": 0.88},
    ]
    return jsonify({
        "spots": mock_spots,
        "total": len(mock_spots),
        "available": len([s for s in mock_spots if not s['occupied']])
    })

# Endpoint for CNN model to update spots
@app.route('/api/update-spots', methods=['POST'])
def update_spots():
    try:
        data = request.get_json()
        # This is where your CNN model will send updates
        print("Received spot updates:", data)
        return jsonify({"status": "success", "message": "Spots updated"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)