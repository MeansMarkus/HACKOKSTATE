# app.py
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
import math
import requests

app = Flask(__name__)

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# --- Helper function: Haversine distance (km)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat/2)**2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(d_lon/2)**2)
    return 2 * R * math.asin(math.sqrt(a))

# 1️⃣ Get all parking spots
@app.route("/spots", methods=["GET"])
def get_spots():
    spots_ref = db.collection("parking_spots").stream()
    spots = [doc.to_dict() for doc in spots_ref]
    return jsonify(spots)

# 2️⃣ Get nearest empty spot
@app.route("/spots/nearest", methods=["GET"])
def get_nearest_spot():
    lat = request.args.get("lat", type=float)
    lng = request.args.get("lng", type=float)
    if not lat or not lng:
        return jsonify({"error": "Missing lat/lng"}), 400

    spots_ref = db.collection("parking_spots").where("isOccupied", "==", False).stream()
    empty_spots = [doc.to_dict() for doc in spots_ref]

    if not empty_spots:
        return jsonify({"error": "No empty spots found"}), 404

    nearest_spot = min(empty_spots, key=lambda s: haversine(lat, lng, s["coordinates"]["lat"], s["coordinates"]["lng"]))
    return jsonify(nearest_spot)

# 3️⃣ Get route (optional - uses Google Maps Directions API)
@app.route("/route", methods=["GET"])
def get_route():
    lat = request.args.get("lat", type=float)
    lng = request.args.get("lng", type=float)
    dest_lat = request.args.get("dest_lat", type=float)
    dest_lng = request.args.get("dest_lng", type=float)
    api_key = "YOUR_GOOGLE_MAPS_API_KEY"

    url = f"https://maps.googleapis.com/maps/api/directions/json?origin={lat},{lng}&destination={dest_lat},{dest_lng}&key={api_key}"
    r = requests.get(url)
    data = r.json()

    if data["status"] != "OK":
        return jsonify({"error": "Route not found", "details": data}), 400

    return jsonify(data["routes"][0])

if __name__ == "__main__":
    app.run(debug=True)
