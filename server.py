import os
import cv2
import mediapipe as mp
import base64
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# --- AYARLAR ---
DATA_PATH = "veriseti"
# Klasörleri alfabetik sıraya göre al (Eğitimdeki sırayla aynı olması için)
actions = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])

# Modeli yükle
model = load_model('isaret_dili_modeli.h5')

sequence = []
threshold = 0.8  # Güven eşiğini biraz artırdık (Daha kesin sonuçlar için)

# MediaPipe Kurulumu
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract_landmarks(results):
    lh, rh = np.zeros(21*3), np.zeros(21*3)
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[i].classification[0].label
            coords = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
            if label == 'Left': 
                lh = coords
            else: 
                rh = coords
    return np.concatenate([lh, rh])

@app.route("/predict", methods=["POST"])
def predict():
    global sequence
    try:
        data = request.json["image"]
        encoded_data = data.split(",")[1]
        np_arr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Görüntü işleme
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        sequence.append(extract_landmarks(results))
        sequence = sequence[-30:] # Son 30 kareyi tut

        result_text = "İşaret Bekleniyor..."

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            max_idx = np.argmax(res)
            confidence = res[max_idx]
            predicted_action = actions[max_idx]

            # --- "BOS" VE GÜVEN KONTROLÜ ---
            if predicted_action != "bos" and confidence > threshold:
                result_text = predicted_action.upper()
            else:
                # Model 'bos' diyorsa veya emin değilse varsayılan metne dön
                result_text = "İşaret Bekleniyor..."

        return jsonify({"prediction": result_text})
    
    except Exception as e:
        print(f"Hata oluştu: {e}")
        return jsonify({"prediction": "Hata!", "error": str(e)})

if __name__ == "__main__":
    print(f"Sistem başlatıldı. Tanınan kelimeler: {actions}")
    app.run(port=5000, debug=False)