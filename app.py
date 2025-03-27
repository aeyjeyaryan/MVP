from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
from PIL import Image
import google.generativeai as genai
import base64
import io
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import mediapipe as mp
from typing import Optional
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Global camera instance
cap = None

def initialize_camera():
    global cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open camera")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def cleanup_resources():
    global cap
    if cap is not None:
        cap.release()
    hands.close()
    cv2.destroyAllWindows()

def process_hand_region(frame):
    if frame is None or frame.size == 0:
        return None, None
    
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        if not results.multi_hand_landmarks:
            return mask, frame
        
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            points = []
            for landmark in hand_landmarks.landmark:
                h, w, _ = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                points.append([x, y])
            
            if points:
                hull = cv2.convexHull(np.array(points))
                cv2.fillConvexPoly(mask, hull, 255)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        combined_mask = cv2.bitwise_and(mask, skin_mask)
        
        kernel = np.ones((5,5), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
        combined_mask = cv2.erode(combined_mask, kernel, iterations=1)
        combined_mask = cv2.GaussianBlur(combined_mask, (5,5), 50)
        
        return combined_mask, frame
    except Exception as e:
        return None, None

def get_gemini_prediction(image):
    try:
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        prompt = """
        Analyze this Indian Sign Language gesture image. Focus on:
        1. Hand shape and finger positions
        2. Orientation of the palm
        3. Any distinct patterns or configurations
        
        and only return the word you predict, nothing else please.
        """
        response = model.generate_content([prompt, image_pil])
        return response.text
    except Exception as e:
        return f"Error in prediction: {str(e)}"

def process_frame(frame, roi_coords):
    roi_top, roi_bottom, roi_left, roi_right = roi_coords
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)
    
    roi = frame[roi_top:roi_bottom, roi_left:roi_right]
    if roi is None or roi.size == 0:
        return frame, roi, None
        
    processed_mask, annotated_roi = process_hand_region(roi)
    
    if processed_mask is not None and annotated_roi is not None:
        mask_indices = processed_mask > 0
        if np.any(mask_indices):
            display_roi = roi.copy()
            green_overlay = np.full_like(roi, [0, 255, 0])
            display_roi[mask_indices] = cv2.addWeighted(
                roi[mask_indices], 0.7,
                green_overlay[mask_indices], 0.3,
                0
            )
            frame[roi_top:roi_bottom, roi_left:roi_right] = display_roi
    
    return frame, roi, processed_mask

# HTML for WebSocket client
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>ISL Detection</title>
    </head>
    <body>
        <h1>Indian Sign Language Detection</h1>
        <img id="videoFeed" src="">
        <p>Prediction: <span id="prediction"></span></p>
        <button onclick="capture()">Capture</button>
        <script>
            const ws = new WebSocket(`ws://${window.location.host}/ws`);
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                document.getElementById('videoFeed').src = 'data:image/jpeg;base64,' + data.frame;
                document.getElementById('prediction').textContent = data.prediction || '';
            };
            
            function capture() {
                ws.send('capture');
            }
        </script>
    </body>
</html>
"""

@app.on_event("startup")
async def startup_event():
    initialize_camera()

@app.on_event("shutdown")
async def shutdown_event():
    cleanup_resources()

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    roi_coords = (80, 400, 200, 520)
    previous_prediction = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame, roi, processed_mask = process_frame(frame, roi_coords)
            
            # Convert frame to base64
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            await websocket.send_json({
                "frame": frame_base64,
                "prediction": previous_prediction
            })
            
            # Check for capture command
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                if data == "capture" and roi is not None and roi.size > 0:
                    prediction = get_gemini_prediction(roi)
                    previous_prediction = prediction
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"capture_{prediction}_{timestamp}.jpg", roi)
            except asyncio.TimeoutError:
                pass
                
            await asyncio.sleep(0.03)  # Control frame rate
            
    except WebSocketDisconnect:
        pass

@app.post("/predict")
async def predict_image(file: bytes):
    """Endpoint to predict ISL from a single uploaded image"""
    try:
        image = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)
        _, roi, _ = process_frame(image, (80, 400, 200, 520))
        if roi is not None and roi.size > 0:
            prediction = get_gemini_prediction(roi)
            return {"prediction": prediction}
        return {"error": "No valid hand region detected"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)